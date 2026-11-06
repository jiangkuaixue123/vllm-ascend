# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded graph replay of the unchanged eager draft HC pre/mix chain."""

from dataclasses import dataclass

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41.core import hc_mixes, hc_pre

_DRAFT_ROWS = (5,)
_HC_WIDTH = 4
_HIDDEN_SIZE = 5120
_MIX_WIDTH = (_HC_WIDTH + 2) * _HC_WIDTH
_ND_FORMAT = 2


def supports_draft_hc_graph(x, previous_pre, fn, scale, base):
    """Inspect host tensor metadata only; unsupported inputs use the parent."""
    return not (
        x.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or x.ndim != 3
        or x.shape[0] not in _DRAFT_ROWS
        or x.shape[1:] != (_HC_WIDTH, _HIDDEN_SIZE)
        or x.dtype != torch.bfloat16
        or previous_pre.shape != x.shape[:2]
        or fn.shape != (_MIX_WIDTH, _HC_WIDTH * _HIDDEN_SIZE)
        or scale.shape != (3,)
        or base.shape != (_MIX_WIDTH,)
        or any(t.dtype != torch.float32 for t in (previous_pre, fn, scale, base))
        or any(
            t.device != x.device or not t.is_contiguous() or t.is_neg() or t.is_conj()
            for t in (x, previous_pre, fn, scale, base)
        )
        or get_ascend_device_type() != AscendDeviceType.A3
    )


def draft_hc_pre_and_mixes(x, previous_pre, fn, scale, base, norm_eps, hc_eps, sinkhorn_iters):
    next_pre, post, comb = hc_mixes(x, fn, scale, base, norm_eps, hc_eps, sinkhorn_iters)
    return hc_pre(x, previous_pre), next_pre, post, comb


@dataclass
class _HCGraphEntry:
    x: torch.Tensor
    previous_pre: torch.Tensor
    graph: object
    outputs: tuple
    signature: tuple


class DraftHCMixGraph:
    """Own one private graph for a five-row draft block and one HC sublayer.

    Results are consumed on the caller's stream before the same entry is replayed
    again. Attention, cache updates, sampling and collectives remain outside.
    """

    def __init__(self, norm_eps, hc_eps, sinkhorn_iters):
        self.parameters = (norm_eps, hc_eps, sinkhorn_iters)
        self.entries = {}

    def __call__(self, x, previous_pre, fn, scale, base):
        if not supports_draft_hc_graph(x, previous_pre, fn, scale, base):
            return None
        if torch.npu.is_current_stream_capturing():
            return None
        # Import the runtime only after the CPU/training/geometry fallbacks.
        from torch_npu import get_npu_format

        if any(get_npu_format(t) != _ND_FORMAT for t in (x, previous_pre, fn, scale, base)):
            return None
        signature = (
            x.device,
            torch.npu.current_stream(x.device).npu_stream,
            fn.data_ptr(),
            scale.data_ptr(),
            base.data_ptr(),
            bool(torch.npu.matmul.allow_hf32),
        )
        rows = x.shape[0]
        entry = self.entries.get(rows)
        if entry is not None and entry.signature != signature:
            return None
        if entry is None:
            static_x, static_pre = x.clone(), previous_pre.clone()
            # Resolve the parent's lazy Triton kernel before recording it once.
            draft_hc_pre_and_mixes(static_x, static_pre, fn, scale, base, *self.parameters)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = draft_hc_pre_and_mixes(static_x, static_pre, fn, scale, base, *self.parameters)
            entry = _HCGraphEntry(static_x, static_pre, graph, outputs, signature)
            self.entries[rows] = entry
        else:
            entry.x.copy_(x)
            entry.previous_pre.copy_(previous_pre)
        entry.graph.replay()
        return entry.outputs
