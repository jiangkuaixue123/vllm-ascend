# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the local attention core of complete DSpark proposal blocks."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from vllm.utils.torch_utils import async_tensor_h2d

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

DRAFT_ROWS = 5
HIDDEN_SIZE = 5120
HEAD_DIM = 512
# The checkpoint has 64 global heads; TP4 owns 16 heads per worker.
LOCAL_HEADS = 16
WINDOW = 128
ND_FORMAT = 2
SUPPORTED_BATCHES = (1, 4)


def supports_draft_attention_graph(attention, hidden, positions, metadata):
    """Qualify using host metadata and tensor geometry; never read a device scalar."""
    if (
        hidden.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or attention.training
        or metadata is None
        or hidden.ndim != 2
        or hidden.shape[1] != HIDDEN_SIZE
        or hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or hidden.is_neg()
        or hidden.is_conj()
        or positions.shape != (hidden.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != hidden.device
        or not positions.is_contiguous()
        or positions.is_neg()
        or positions.is_conj()
        or len(metadata.requests) not in SUPPORTED_BATCHES
        or hidden.shape[0] != len(metadata.requests) * DRAFT_ROWS
        or metadata.num_actual_tokens != hidden.shape[0]
        or attention.config.dspark_block_size != DRAFT_ROWS
        or attention.config.sliding_window != WINDOW
        or attention.cache.block_size != WINDOW
        or attention.dim != HEAD_DIM
        or attention.heads != LOCAL_HEADS
        or attention.dsa_cp
        or getattr(attention.wq_b, "gather_output", False)
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return False
    batch = len(metadata.requests)
    table = metadata.block_table_tensor
    return (
        table.ndim == 2
        and table.shape[0] >= batch
        and table.dtype in (torch.int32, torch.int64)
        and table.device == hidden.device
        and table.stride(1) == 1
        and not table.is_neg()
        and not table.is_conj()
        and all(
            span.begin == row * DRAFT_ROWS
            and span.end == (row + 1) * DRAFT_ROWS
            and type(span.start_pos) is int
            and span.start_pos >= WINDOW
            and table.shape[1] > (span.start_pos - 1) // WINDOW
            for row, span in enumerate(metadata.requests)
        )
    )


def draft_attention_core(attention, hidden, positions, table, start):
    """Keep the parent's operations and full-ring logical ordering unchanged."""
    qr = attention.q_norm(attention.wq_a(hidden)[0])
    q = attention.wq_b(qr)[0].view(-1, attention.heads, attention.dim)
    attention.rotate(q.unsqueeze(0), positions)
    query_kv = attention.project_kv(hidden, positions)
    slots = torch.arange(WINDOW, device=hidden.device)
    batch = start.shape[0]
    if batch == 1:
        history = start - 1 - (start - 1 - slots).remainder(WINDOW)
        metadata = SimpleNamespace(block_table_tensor=table)
        context = attention.cache.read(metadata, 0, history, 0, 0, HEAD_DIM).to(hidden.dtype)
        kv = torch.cat((context, query_kv), dim=0).unsqueeze(0)
    else:
        history = start[:, None] - 1 - (start[:, None] - 1 - slots).remainder(WINDOW)
        pages = table.gather(1, history // WINDOW).long()
        cache_slots = pages * WINDOW + history % WINDOW
        context = attention.cache.field(0, 0, HEAD_DIM)[cache_slots]
        kv = torch.cat((context, query_kv.view(batch, DRAFT_ROWS, HEAD_DIM)), dim=1)
    indices = torch.arange(WINDOW + DRAFT_ROWS, dtype=torch.int32, device=hidden.device)
    indices = indices.view(1, 1, -1).expand(batch, DRAFT_ROWS, -1)
    attended = attention.attention_kernel(
        q.view(batch, DRAFT_ROWS, LOCAL_HEADS, HEAD_DIM),
        kv,
        attention.attn_sink,
        indices,
        HEAD_DIM**-0.5,
        mode="SWA draft bidirectional",
    ).view(-1, LOCAL_HEADS, HEAD_DIM)
    attention.rotate(attended.unsqueeze(0), positions, inverse=True)
    return attended


@dataclass
class _AttentionGraphEntry:
    hidden: torch.Tensor
    positions: torch.Tensor
    table: torch.Tensor
    start: torch.Tensor
    graph: object
    output: torch.Tensor
    signature: tuple


class DraftAttentionGraph:
    """Bound each layer to two graphs; output projection and communication stay outside."""

    def __init__(self):
        self.entries = {}

    def __call__(self, attention, hidden, positions, metadata):
        if not supports_draft_attention_graph(attention, hidden, positions, metadata):
            return None
        if torch.npu.is_current_stream_capturing():
            return None
        from torch_npu import get_npu_format

        field = attention.cache.field(0, 0, HEAD_DIM)
        if (
            field.ndim != 2
            or field.shape[1] != HEAD_DIM
            or field.stride(1) != 1
            or field.dtype != torch.bfloat16
            or field.device != hidden.device
            or field.is_neg()
            or field.is_conj()
            or any(get_npu_format(tensor) != ND_FORMAT for tensor in (hidden, positions, field))
        ):
            return None
        batch = len(metadata.requests)
        table = metadata.block_table_tensor[:batch]
        tensors = [attention.attn_sink, attention.frequencies]
        for module in (attention.wq_a, attention.wq_b, attention.wkv, attention.q_norm, attention.kv_norm):
            tensors.extend(module.parameters())
            tensors.extend(module.buffers())
        if any(tensor.device != hidden.device for tensor in tensors):
            return None
        signature = (
            hidden.device,
            positions.dtype,
            torch.npu.current_stream(hidden.device).npu_stream,
            field.data_ptr(),
            tuple(field.shape),
            field.stride(),
            tuple(table.shape),
            table.dtype,
            tuple(
                (
                    tensor.data_ptr(),
                    tensor.dtype,
                    tuple(tensor.shape),
                    tensor.stride(),
                    tensor.is_neg(),
                    tensor.is_conj(),
                )
                for tensor in tensors
            ),
            bool(torch.npu.matmul.allow_hf32),
        )
        entry = self.entries.get(batch)
        if entry is not None and entry.signature != signature:
            return None
        if entry is None:
            static_hidden, static_positions, static_table = hidden.clone(), positions.clone(), table.clone()
            start = async_tensor_h2d(
                [span.start_pos for span in metadata.requests], device=hidden.device, dtype=torch.int64
            )
            # Warm lazy kernels and native attention metadata before recording once.
            draft_attention_core(attention, static_hidden, static_positions, static_table, start)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = draft_attention_core(attention, static_hidden, static_positions, static_table, start)
            entry = _AttentionGraphEntry(static_hidden, static_positions, static_table, start, graph, output, signature)
            self.entries[batch] = entry
        else:
            entry.hidden.copy_(hidden)
            entry.positions.copy_(positions)
            entry.table.copy_(table)
            if batch == 1:
                entry.start.fill_(metadata.requests[0].start_pos)
            else:
                entry.start.copy_(
                    async_tensor_h2d(
                        [span.start_pos for span in metadata.requests], device=hidden.device, dtype=torch.int64
                    )
                )
        entry.graph.replay()
        return entry.output
