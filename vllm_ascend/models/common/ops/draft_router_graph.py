# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the original local draft expert selection without communication."""

import math
import os
from dataclasses import dataclass

import torch
from vllm.logger import logger

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41.core import route_sqrtsoftplus

DRAFT_ROWS = (2, 5)
DRAFT_EXPERTS = 128
DRAFT_TOP_K = 3
ND_FORMAT = 2


def supports_draft_router_graph(gate, logits, top_k, renormalize):
    return (
        logits.device.type == "npu"
        and torch.is_inference_mode_enabled()
        and not gate.training
        and logits.ndim == 2
        and logits.shape[0] in DRAFT_ROWS
        and logits.shape[1] == DRAFT_EXPERTS
        and top_k == gate.top_k == DRAFT_TOP_K
        and renormalize is True
        and gate.bias_vl is None
        and gate.bias.shape == (DRAFT_EXPERTS,)
        and gate.bias.dtype == logits.dtype == torch.float32
        and gate.bias.device == logits.device
        and all(t.is_contiguous() and not t.is_neg() and not t.is_conj() for t in (logits, gate.bias))
        and all(
            isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in (gate.temperature, gate.scaling_factor)
        )
        and get_ascend_device_type() == AscendDeviceType.A3
    )


@dataclass
class _RouterGraphEntry:
    logits: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    graph: object
    signature: tuple


class DraftRouterGraph:
    """Two row buckets per draft layer; a changed signature uses the parent."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.entries = {}

    def __call__(self, gate, logits, top_k, renormalize):
        if not supports_draft_router_graph(gate, logits, top_k, renormalize):
            return None
        if torch.npu.is_current_stream_capturing():
            return None
        from torch_npu import get_npu_format

        if any(get_npu_format(t) != ND_FORMAT for t in (logits, gate.bias)):
            return None
        signature = (
            logits.device,
            torch.npu.current_stream(logits.device).npu_stream,
            gate.bias.data_ptr(),
            gate.temperature,
            gate.scaling_factor,
        )
        rows = logits.shape[0]
        entry = self.entries.get(rows)
        if entry is not None and entry.signature != signature:
            return None
        if entry is None:
            static_logits = logits.clone()
            args = (static_logits, gate.bias, top_k, renormalize, gate.scaling_factor, gate.temperature)
            route_sqrtsoftplus(*args)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                weights, indices = route_sqrtsoftplus(*args)
            entry = _RouterGraphEntry(static_logits, weights, indices, graph, signature)
            self.entries[rows] = entry
        else:
            entry.logits.copy_(logits)
        # Every replay rewrites the complete route outputs, including IDs
        # that a downstream fused dispatcher may have invalidated in place.
        entry.graph.replay()
        logger.info_once(
            "DSPARK_ROUTER_GRAPH_RUNTIME pid=%d stage=%s rows=%d", os.getpid(), self.prefix, rows, scope="process"
        )
        return entry.weights, entry.indices
