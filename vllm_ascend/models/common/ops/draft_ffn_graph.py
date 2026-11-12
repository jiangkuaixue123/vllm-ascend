# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay draft routed and shared experts after the unchanged eager gate GEMM."""

from dataclasses import dataclass

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import MoECommType

HIDDEN_SIZE = 5120
DRAFT_EXPERTS = 128
TP_SIZE = 4
EP_SIZE = 16
MAX_PADDED_TOKENS = 40


@dataclass
class _ExpertGraphEntry:
    states: torch.Tensor
    logits: torch.Tensor
    output: torch.Tensor
    graph: object


@dataclass
class _ExpertGraphGroup:
    mask: torch.Tensor
    entries: dict
    signature: tuple


def draft_ffn_graph_capacity(moe, states):
    """Use a DP-uniform key; local row counts must not decide capture order."""
    if not is_forward_context_available() or states.device.type != "npu":
        return None
    context = get_forward_context()
    comm = moe.communication
    padded = getattr(context, "padded_num_tokens", 0)
    if not (
        getattr(context, "is_draft_model", False)
        and not getattr(context, "in_profile_run", False)
        and getattr(context, "moe_comm_type", None) == MoECommType.FUSED_MC2
        and isinstance(padded, int)
        and TP_SIZE <= padded <= MAX_PADDED_TOKENS
        and padded % TP_SIZE == 0
        and moe.use_fused_runner
        and comm.sequence_parallel
        and comm.replicated_shared
        and comm.tp_group.world_size == TP_SIZE
        and comm.ep_size == EP_SIZE
        and moe.n_routed_experts == DRAFT_EXPERTS
        and not moe.training
        and torch.is_inference_mode_enabled()
        and not torch.npu.is_current_stream_capturing()
    ):
        return None
    return padded


class DraftFFNGraph:
    """Capture every local width together before any rank starts replaying.

    Each DP can have a different number of draft requests, including zero. On
    first use of a common padding length, all EP ranks record the same ordered
    set of local widths. Subsequent calls may choose different widths while
    keeping the fused dispatch's padded communication shape identical.
    """

    def __init__(self):
        self.groups = {}

    def __call__(self, moe, states, logits):
        padded = draft_ffn_graph_capacity(moe, states)
        if padded is None:
            return None
        context = get_forward_context()
        mask = context.mc2_mask
        if not (
            states.ndim == 2
            and states.shape[1] == HIDDEN_SIZE
            and 0 <= states.shape[0] <= padded // TP_SIZE
            and states.dtype == torch.bfloat16
            and states.is_contiguous()
            and logits.shape == (states.shape[0], DRAFT_EXPERTS)
            and logits.dtype == torch.float32
            and logits.device == states.device
            and logits.is_contiguous()
            and mask.shape == (padded,)
            and mask.dtype == torch.bool
            and mask.device == states.device
            and mask.is_contiguous()
        ):
            raise RuntimeError("Draft FFN graph inputs violate the uniform TP4 layout")
        parameters = tuple(moe.parameters()) + tuple(moe.buffers())
        for module in moe.experts.modules():
            for name in ("v41_w2_scale_fp32", "w13_weight_scale_fp32"):
                value = getattr(module, name, None)
                if isinstance(value, torch.Tensor):
                    parameters += (value,)
        signature = (
            torch.npu.current_stream(states.device).npu_stream,
            tuple((p.data_ptr(), p.dtype, tuple(p.shape), p.stride()) for p in parameters),
            bool(torch.npu.matmul.allow_hf32),
            moe.gate.temperature,
            moe.gate.scaling_factor,
            moe.gate.top_k,
            moe.gate.renormalize,
        )
        group = self.groups.get(padded)
        if group is not None and group.signature != signature:
            # Never recapture on just one rank after a storage/stream change.
            return None
        if group is None:
            group = self._capture(moe, states, padded, mask, signature)
            self.groups[padded] = group
        entry = group.entries[states.shape[0]]
        entry.states.copy_(states)
        entry.logits.copy_(logits)
        group.mask.copy_(mask)
        entry.graph.replay()
        logger.info_once("DSPARK_FFN_GRAPH_RUNTIME padded=%d", padded, scope="process")
        return entry.output

    def _capture(self, moe, states, padded, mask, signature):
        context = get_forward_context()
        static_mask = mask.clone()
        entries = {}
        previous_mask = context.mc2_mask
        try:
            context.mc2_mask = static_mask
            # Including zero ensures idle replicas participate in precisely the
            # same capture collectives as replicas that own real requests.
            for rows in range(padded // TP_SIZE + 1):
                static_states = states.new_zeros((rows, HIDDEN_SIZE))
                static_logits = torch.zeros((rows, DRAFT_EXPERTS), dtype=torch.float32, device=states.device)
                moe._forward_experts(static_states, static_logits)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = moe._forward_experts(static_states, static_logits)
                entries[rows] = _ExpertGraphEntry(static_states, static_logits, output, graph)
        finally:
            context.mc2_mask = previous_mask
        return _ExpertGraphGroup(static_mask, entries, signature)
