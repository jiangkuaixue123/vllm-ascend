# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from weakref import ref

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from vllm.distributed import get_tp_group
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode

_TP_SIZE = 8
_DEFAULT_VERIFY_ROWS = 6
_BROADCAST_ROWS = 1024


@triton.jit
def _pack_winner(score, index, packet, OFFSET: tl.constexpr):
    row = tl.program_id(0)
    value = tl.load(score + row)
    tl.store(packet + row * 2, value.to(tl.int32, bitcast=True))
    tl.store(packet + row * 2 + 1, tl.load(index + row).to(tl.int32) + OFFSET)


@triton.jit
def _select_winner(packets, output, ROWS: tl.constexpr, RANKS: tl.constexpr):
    row = tl.program_id(0)
    rank = tl.arange(0, RANKS)
    offset = (rank * ROWS + row) * 2
    values = tl.load(packets + offset).to(tl.float32, bitcast=True)
    ids = tl.load(packets + offset + 1)
    nan = values != values
    has_nan = tl.max(nan.to(tl.int32), 0) != 0
    maximum = tl.max(tl.where(nan, -float("inf"), values), 0)
    # Full-vocabulary argmax chooses the first NaN or the first maximal score.
    keep = tl.where(has_nan, nan, values == maximum)
    tl.store(output + row, tl.min(tl.where(keep, ids, 2147483647), 0).to(tl.int64))


class LocalVocabulary(nn.Module):
    """Replicate ordinary input embeddings and keep a rank-local FP32 projection."""

    def __init__(
        self, embedding, head, sharded_verification=False, copy_group=None, *, max_decode_tokens=_DEFAULT_VERIFY_ROWS
    ):
        super().__init__()
        self.max_decode_tokens = max_decode_tokens
        self.embedding = ref(embedding)
        self.head = ref(head)
        self.sharded_verification = sharded_verification
        self.copy_group = copy_group
        self.register_buffer("table", None, persistent=False)
        self.register_buffer("projection", None, persistent=False)
        self.start = embedding.shard_indices.org_vocab_start_index

    @staticmethod
    def supports(embedding, head):
        return (
            getattr(embedding, "parallel_mode", None) is VocabParallelMode.STANDARD
            and getattr(head, "parallel_mode", None) is VocabParallelMode.STANDARD
            and embedding.tp_size == head.tp_size == _TP_SIZE
            and embedding.shard_indices == head.shard_indices
            and embedding.org_vocab_size == embedding.num_embeddings == embedding.num_embeddings_padded
            and head.org_vocab_size == head.num_embeddings == embedding.org_vocab_size
            and type(embedding.quant_method) is UnquantizedEmbeddingMethod
            and isinstance(head.quant_method, (UnquantizedLinearMethod, UnquantizedEmbeddingMethod))
            and embedding.weight.dtype == head.weight.dtype == torch.bfloat16
            and embedding.weight.shape == head.weight.shape
            and embedding.shard_indices.org_vocab_start_index == embedding.tp_rank * embedding.weight.shape[0]
            and embedding.weight.shape[0] * _TP_SIZE == embedding.org_vocab_size
        )

    @torch.no_grad()
    def process_weights_after_loading(self):
        embedding, head = self.embedding(), self.head()
        if embedding.weight.device.type != "npu":
            return
        group = get_tp_group()
        local_rows, width = embedding.weight.shape
        if self.table is None:
            self.table = embedding.weight.new_empty((embedding.org_vocab_size, width))
            self.projection = torch.empty_like(head.weight, dtype=torch.float32, memory_format=torch.contiguous_format)
        self.projection.copy_(head.weight)
        self.table.narrow(0, self.start, local_rows).copy_(embedding.weight)
        for rank, source in enumerate(group.ranks):
            shard = self.table.narrow(0, rank * local_rows, local_rows)
            for chunk in shard.split(_BROADCAST_ROWS):
                dist.broadcast(chunk, src=source, group=group.device_group)
        logger.info_once("HYV4 Flash prepared the replicated token embeddings and local FP32 vocabulary shard.")

    def can_embed(self, ids):
        return self.table is not None and ids.ndim == 1 and 0 < ids.numel() <= self.max_decode_tokens

    def embed(self, ids):
        # Negative async slots use token zero. Discard values above the real vocabulary as in sharded lookup.
        size = self.table.shape[0]
        rows = F.embedding(ids.clamp(0, size - 1).long(), self.table)
        return rows.masked_fill_((ids >= size)[:, None], 0)

    def supports_greedy(self, processor):
        return (
            self.projection is not None
            and processor.head_dtype == torch.float32
            and processor.soft_cap is None
            and processor.scale == 1
            and not processor.logits_as_input
            and processor.org_vocab_size == self.table.shape[0]
        )

    def project(self, hidden):
        if self.sharded_verification and hidden.dtype == torch.bfloat16:
            return torch.mm(hidden, self.head().weight.T, out_dtype=torch.float32)
        return F.linear(hidden.float(), self.projection)

    def greedy(self, hidden):
        scores = self.project(hidden)
        value, index = scores.max(dim=-1)
        rows = hidden.shape[0]
        packet = scores.new_empty(rows * 2, dtype=torch.int32)
        gathered = packet.new_empty(_TP_SIZE * packet.numel())
        _pack_winner[(rows,)](value, index, packet, self.start, num_warps=1)
        group = get_tp_group().device_group if self.copy_group is None else self.copy_group
        dist.all_gather_into_tensor(gathered, packet, group=group)
        if self.copy_group is not None:
            logger.info_once("HYV4 Flash gathers draft vocabulary winners with CCU.")
        winner = torch.empty_like(index)
        _select_winner[(rows,)](gathered, winner, rows, _TP_SIZE, num_warps=1)
        return winner
