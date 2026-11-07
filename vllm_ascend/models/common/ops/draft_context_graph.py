# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the complete local context projection and request-owned cache write."""

from dataclasses import dataclass

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.deepseek_v41_context_store import store_context_rows

CONTEXT_ROWS = (6, 24)
HIDDEN_SIZE = 5120
KV_WIDTH = 512
ROPE_DIM = 64
TENSOR_PARALLEL_SIZE = 4
ND_FORMAT = 2


def supports_draft_context_graph(attention, states, positions, slots, field):
    return (
        store_context_rows is not None
        and states.device.type == "npu"
        and torch.is_inference_mode_enabled()
        and not attention.training
        and states.ndim == 2
        and states.shape[0] in CONTEXT_ROWS
        and states.shape[1] == HIDDEN_SIZE
        and states.dtype == field.dtype == torch.bfloat16
        and states.is_contiguous()
        and positions.shape == slots.shape == (states.shape[0],)
        and positions.dtype in (torch.int32, torch.int64)
        and slots.dtype in (torch.int32, torch.int64)
        and positions.is_contiguous()
        and slots.is_contiguous()
        and states.device == positions.device == slots.device == field.device
        and field.ndim == 2
        and field.shape[0] > 0
        and field.shape[1] == KV_WIDTH
        and field.stride(1) == 1
        and field.stride(0) >= KV_WIDTH
        and all(not tensor.is_neg() and not tensor.is_conj() for tensor in (states, positions, slots, field))
        and attention.dim == KV_WIDTH
        and attention.rope_dim == ROPE_DIM
        and attention.tp == TENSOR_PARALLEL_SIZE
        and not getattr(attention.wkv, "gather_output", False)
        and get_ascend_device_type() == AscendDeviceType.A3
    )


def _project_and_store(attention, states, positions, slots, field):
    from torch_npu import get_npu_format

    values = attention.project_kv(states, positions)
    if not (
        values.shape == (states.shape[0], KV_WIDTH)
        and values.dtype == field.dtype
        and values.device == field.device
        and values.stride(1) == 1
        and values.stride(0) >= KV_WIDTH
        and not values.is_neg()
        and not values.is_conj()
        and get_npu_format(values) == ND_FORMAT
    ):
        return False
    store_context_rows(field, slots, values)
    return True


@dataclass
class _ContextGraphEntry:
    states: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    graph: object
    signature: tuple


class DraftContextGraph:
    """Two entries per layer; all collectives and draft attention stay outside."""

    def __init__(self):
        self.entries = {}

    def __call__(self, attention, states, positions, slots):
        if slots is None or attention.dim != KV_WIDTH or states.device.type != "npu":
            return False
        field = attention.cache.field(0, 0, KV_WIDTH)
        if not supports_draft_context_graph(attention, states, positions, slots, field):
            return False
        if torch.npu.is_current_stream_capturing():
            return False
        from torch_npu import get_npu_format

        if any(get_npu_format(tensor) != ND_FORMAT for tensor in (states, positions, slots, field)):
            return False
        parameters = [attention.frequencies]
        for module in (attention.wkv, attention.kv_norm):
            parameters.extend(module.parameters())
            parameters.extend(module.buffers())
        if any(tensor.device != states.device for tensor in parameters):
            return False
        signature = (
            states.device,
            positions.dtype,
            slots.dtype,
            torch.npu.current_stream(states.device).npu_stream,
            field.data_ptr(),
            tuple(field.shape),
            field.stride(),
            tuple(
                (
                    tensor.data_ptr(),
                    tensor.dtype,
                    tuple(tensor.shape),
                    tensor.stride(),
                    tensor.is_neg(),
                    tensor.is_conj(),
                )
                for tensor in parameters
            ),
            bool(torch.npu.matmul.allow_hf32),
        )
        rows = states.shape[0]
        entry = self.entries.get(rows)
        if entry is not None and entry.signature != signature:
            return False
        if entry is None:
            static_states, static_positions, static_slots = states.clone(), positions.clone(), slots.clone()
            if not _project_and_store(attention, static_states, static_positions, static_slots, field):
                return False
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                if not _project_and_store(attention, static_states, static_positions, static_slots, field):
                    raise RuntimeError("Draft context projection geometry changed during capture")
            entry = _ContextGraphEntry(static_states, static_positions, static_slots, graph, signature)
            self.entries[rows] = entry
        else:
            entry.states.copy_(states)
            entry.positions.copy_(positions)
            entry.slots.copy_(slots)
        entry.graph.replay()
        return True
