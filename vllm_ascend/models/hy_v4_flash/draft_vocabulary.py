# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
import torch_npu
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessor, VocabParallelMode
from vllm_ascend.quantization.nz import fp4_packed_nz

_MX_GROUP_SIZE = 32
_VOCABULARY_SHARD_SIZE = 60_416
_VOCABULARY_SHARDS = 2
_SHORTLIST_SIZE = 32


def _rerank(hidden: torch.Tensor, weight: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    rows = weight.index_select(0, tokens.flatten()).view(*tokens.shape, hidden.shape[-1])
    scores = torch.bmm(rows.float(), hidden.float().unsqueeze(-1)).squeeze(-1)
    # Ties choose the lowest token, as in full-vocabulary argmax.
    highest = scores.amax(dim=-1, keepdim=True)
    selected = (scores == highest) | (scores.isnan() & highest.isnan())
    return tokens.masked_fill(~selected, weight.shape[0]).amin(dim=-1)


class MXFP4DraftVocabulary:
    """Generate approximate proposals with a full-precision shortlist rerank."""

    @staticmethod
    def supports(head, processor) -> bool:
        return (
            isinstance(processor, AscendLogitsProcessor)
            and processor.use_native_bf16_fp32_output
            and processor.head_dtype == torch.float32
            and processor.scale == 1
            and processor.soft_cap is None
            and not processor.logits_as_input
            and head.tp_size == 1
            and head.parallel_mode is VocabParallelMode.STANDARD
            and isinstance(head.quant_method, (UnquantizedLinearMethod, UnquantizedEmbeddingMethod))
            and getattr(head, "bias", None) is None
            and head.weight.device.type == "npu"
            and head.weight.dtype == torch.bfloat16
            and head.weight.shape[0] == processor.org_vocab_size == _VOCABULARY_SHARD_SIZE * _VOCABULARY_SHARDS
            and head.weight.shape[1] % _MX_GROUP_SIZE == 0
        )

    @torch.no_grad()
    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        shards = []
        for part in weight.split(_VOCABULARY_SHARD_SIZE, dim=0):
            packed, scales = torch_npu.npu_dynamic_mx_quant(part, dst_type=torch_npu.float4_e2m1fn_x2)
            matrix = fp4_packed_nz().cast(packed.view(torch.uint8)).t()
            shards.append((matrix, scales.view(torch.uint8).transpose(0, 1)))
        self.shards = tuple(shards)

    def greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        activation, activation_scale = torch_npu.npu_dynamic_mx_quant(hidden, dst_type=torch.float8_e4m3fn)
        proposals = []
        for index, (weight, scales) in enumerate(self.shards):
            scores = torch_npu.npu_quant_matmul(
                activation,
                weight,
                scales,
                pertoken_scale=activation_scale,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[0, 0, _MX_GROUP_SIZE],
                output_dtype=torch.bfloat16,
                x2_dtype=torch_npu.float4_e2m1fn_x2,
            )
            proposals.append(scores.topk(_SHORTLIST_SIZE, dim=-1).indices + index * _VOCABULARY_SHARD_SIZE)
        return _rerank(hidden, self.weight, torch.cat(proposals, dim=-1))
