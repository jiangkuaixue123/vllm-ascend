# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the local attention core of one complete DSpark proposal block."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

DRAFT_ROWS = 5
HIDDEN_SIZE = 5120
HEAD_DIM = 512
LOCAL_HEADS = 64
WINDOW = 128
ND_FORMAT = 2


def supports_single_attention_graph(attention, hidden, positions, metadata):
    """Qualify using host metadata and tensor geometry; never read a device scalar."""
    if (
        hidden.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or attention.training
        or metadata is None
        or hidden.shape != (DRAFT_ROWS, HIDDEN_SIZE)
        or hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or hidden.is_neg()
        or hidden.is_conj()
        or positions.shape != (DRAFT_ROWS,)
        or positions.dtype != torch.int64
        or positions.device != hidden.device
        or not positions.is_contiguous()
        or positions.is_neg()
        or positions.is_conj()
        or metadata.num_actual_tokens != DRAFT_ROWS
        or len(metadata.requests) != 1
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
    span = metadata.requests[0]
    table = metadata.block_table_tensor
    return (
        span.begin == 0
        and span.end == DRAFT_ROWS
        and type(span.start_pos) is int
        and span.start_pos >= WINDOW
        and table.ndim == 2
        and table.shape[0] >= 1
        and table.shape[1] > (span.start_pos - 1) // WINDOW
        and table.dtype in (torch.int32, torch.int64)
        and table.device == hidden.device
        and table.stride(1) == 1
    )


def single_attention_core(attention, hidden, positions, table, start):
    """Keep the parent's operations and full-ring logical ordering unchanged."""
    qr = attention.q_norm(attention.wq_a(hidden)[0])
    q = attention.wq_b(qr)[0].view(-1, attention.heads, attention.dim)
    attention.rotate(q.unsqueeze(0), positions)
    query_kv = attention.project_kv(hidden, positions)
    slots = torch.arange(WINDOW, device=hidden.device)
    history = start - 1 - (start - 1 - slots).remainder(WINDOW)
    metadata = SimpleNamespace(block_table_tensor=table)
    context = attention.cache.read(metadata, 0, history, 0, 0, HEAD_DIM).to(hidden.dtype)
    kv = torch.cat((context, query_kv), dim=0).unsqueeze(0)
    indices = torch.arange(WINDOW + DRAFT_ROWS, dtype=torch.int32, device=hidden.device)
    indices = indices.view(1, 1, -1).expand(1, DRAFT_ROWS, -1)
    attended = attention.attention_kernel(
        q.unsqueeze(0), kv, attention.attn_sink, indices, HEAD_DIM**-0.5, mode="SWA draft bidirectional"
    )[0]
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


class SingleDraftAttentionGraph:
    """Own one graph per attention layer; output projection and communication stay outside."""

    def __init__(self):
        self.entry = None

    def __call__(self, attention, hidden, positions, metadata):
        if not supports_single_attention_graph(attention, hidden, positions, metadata):
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
        table = metadata.block_table_tensor[:1]
        tensors = [attention.attn_sink, attention.frequencies]
        for module in (attention.wq_a, attention.wq_b, attention.wkv, attention.q_norm, attention.kv_norm):
            tensors.extend(module.parameters())
            tensors.extend(module.buffers())
        if any(tensor.device != hidden.device for tensor in tensors):
            return None
        signature = (
            hidden.device,
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
        entry = self.entry
        if entry is not None and entry.signature != signature:
            return None
        if entry is None:
            static_hidden, static_positions, static_table = hidden.clone(), positions.clone(), table.clone()
            start = torch.full((1,), metadata.requests[0].start_pos, dtype=torch.int64, device=hidden.device)
            # Warm lazy kernels and native attention metadata before recording once.
            single_attention_core(attention, static_hidden, static_positions, static_table, start)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = single_attention_core(attention, static_hidden, static_positions, static_table, start)
            entry = _AttentionGraphEntry(static_hidden, static_positions, static_table, start, graph, output, signature)
            self.entry = entry
        else:
            entry.hidden.copy_(hidden)
            entry.positions.copy_(positions)
            entry.table.copy_(table)
            entry.start.fill_(metadata.requests[0].start_pos)
        entry.graph.replay()
        return entry.output
