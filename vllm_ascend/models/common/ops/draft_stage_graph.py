# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay complete draft-local regions while leaving all communication outside."""

import math
from dataclasses import dataclass

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

DRAFT_ROWS = (5, 20)
HC_WIDTH = 4
HIDDEN_SIZE = 5120
MIX_WIDTH = 24
TP_SIZE = 4
SINKHORN_ROUNDS = 20
HC_EPS = 1e-6
ND_FORMAT = 2


def supports_draft_stage_graph(stage, tensors, after_attention):
    """Qualify using host metadata; never inspect a device scalar or request state."""
    if len(tensors) != (5 if after_attention else 2):
        return False
    hidden = tensors[1] if after_attention else tensors[0]
    if (
        hidden.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or stage.training
        or hidden.ndim != 3
        or hidden.shape[0] not in DRAFT_ROWS
        or hidden.shape[1:] != (HC_WIDTH, HIDDEN_SIZE)
        or stage.config.hc_mult != HC_WIDTH
        or stage.config.hidden_size != HIDDEN_SIZE
        or stage.config.hc_sinkhorn_iters != SINKHORN_ROUNDS
        or stage.config.hc_eps != HC_EPS
        or not math.isfinite(stage.config.rms_norm_eps)
        or stage.config.rms_norm_eps <= 0
        or stage.attn.tp != TP_SIZE
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return False
    rows = hidden.shape[0]
    geometry = [((rows, HC_WIDTH, HIDDEN_SIZE), torch.bfloat16), ((rows, HC_WIDTH), torch.float32)]
    if after_attention:
        geometry = [((rows, HIDDEN_SIZE), torch.bfloat16)] + geometry
        geometry += [((rows, HC_WIDTH), torch.float32), ((rows, HC_WIDTH, HC_WIDTH), torch.float32)]
    return all(
        value.shape == shape
        and value.dtype == dtype
        and value.device == hidden.device
        and value.is_contiguous()
        and not value.is_neg()
        and not value.is_conj()
        for value, (shape, dtype) in zip(tensors, geometry)
    )


def prepare_draft_stage(stage, tensors, after_attention):
    """Use precisely the parent's residual, gate and normalization operations."""
    if after_attention:
        attended, hidden, previous_pre, post, comb = tensors
        hidden = stage.post_residual(attended, hidden, post, comb, None)
        part = "ffn"
    else:
        hidden, previous_pre = tensors
        part = "attn"
    states, next_pre, post, comb = stage.pre_and_mixes(hidden, previous_pre, part, None)
    normalized = getattr(stage, f"{part}_norm")(states)
    return normalized, next_pre, post, comb, hidden


@dataclass
class _StageGraphEntry:
    inputs: tuple
    graph: object
    outputs: tuple
    signature: tuple


class DraftStageGraph:
    """Own two bounded shape entries for one local region of one draft stage."""

    def __init__(self, *, after_attention=False):
        self.after_attention = after_attention
        self.entries = {}

    def __call__(self, stage, tensors):
        if not supports_draft_stage_graph(stage, tensors, self.after_attention):
            return None
        if torch.npu.is_current_stream_capturing():
            return None
        from torch_npu import get_npu_format

        part = "ffn" if self.after_attention else "attn"
        norm = getattr(stage, f"{part}_norm")
        parameters = tuple(getattr(stage, f"hc_{part}_{name}") for name in ("fn", "scale", "base"))
        shapes = ((MIX_WIDTH, HC_WIDTH * HIDDEN_SIZE), (3,), (MIX_WIDTH,))
        device = tensors[0].device
        if any(value.shape != shape or value.dtype != torch.float32 for value, shape in zip(parameters, shapes)):
            return None
        if norm.weight.shape != (HIDDEN_SIZE,) or norm.weight.dtype not in (torch.bfloat16, torch.float32):
            return None
        parameters += (norm.weight,)
        if any(
            value.device != device or not value.is_contiguous() or value.is_neg() or value.is_conj()
            for value in parameters
        ):
            return None
        if any(get_npu_format(value) != ND_FORMAT for value in (*tensors, *parameters)):
            return None
        signature = (
            device,
            torch.npu.current_stream(device).npu_stream,
            stage.config.rms_norm_eps,
            stage.config.hc_eps,
            stage.config.hc_sinkhorn_iters,
            norm.eps,
            tuple((value.data_ptr(), value.dtype, tuple(value.shape), value.stride()) for value in parameters),
            bool(torch.npu.matmul.allow_hf32),
        )
        rows = tensors[0].shape[0]
        entry = self.entries.get(rows)
        if entry is not None and entry.signature != signature:
            return None
        if entry is None:
            static_inputs = tuple(value.clone() for value in tensors)
            prepare_draft_stage(stage, static_inputs, self.after_attention)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = prepare_draft_stage(stage, static_inputs, self.after_attention)
            entry = _StageGraphEntry(static_inputs, graph, outputs, signature)
            self.entries[rows] = entry
        else:
            for destination, source in zip(entry.inputs, tensors):
                destination.copy_(source)
        entry.graph.replay()
        return entry.outputs
