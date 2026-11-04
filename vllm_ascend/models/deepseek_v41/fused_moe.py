# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend grouped expert dispatch for V4.1 text and DSpark blocks."""

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEFactory

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.quantization.quant_type import QuantType

from .fused_experts import DeepseekV41AllGatherComm, DeepseekV41FusedMC2Comm, DeepseekV41RoutedExperts

# Keep expanded routes well inside the existing HCCL row/return windows.
# Prefill continues to use the bounded grouped AllGather implementation.
MAX_FUSED_MC2_TOKENS_PER_RANK = 64


def select_v41_comm_type(enabled, requested_type, padded_tokens, tp_size, full_decode):
    if not enabled:
        return MoECommType.ALLGATHER
    if tp_size < 1 or padded_tokens < 1 or padded_tokens % tp_size:
        raise ValueError("V4.1 fused MC2 requires positive uniform TP-aligned token padding")
    if requested_type == MoECommType.FUSED_MC2 and padded_tokens // tp_size <= MAX_FUSED_MC2_TOKENS_PER_RANK:
        return MoECommType.FUSED_MC2
    if full_decode:
        raise ValueError("V4.1 FULL decode cannot fall back from requested fused MC2")
    return MoECommType.ALLGATHER


class DeepseekV41FusedMoERunner(AscendMoERunner):
    """Keep dispatch and quantization local to the target or draft expert layout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The target has 384 experts / top-6 and the draft 128 / top-3.
        # A single process-wide dispatcher cannot describe both layouts.
        self.v41_comm = DeepseekV41AllGatherComm(self.moe_config)
        self.v41_fused_mc2 = None
        if get_ascend_config().enable_fused_mc2 == 1:
            if self.quant_type != QuantType.W8A8 or self.moe_config.ep_size <= 1:
                raise ValueError("V4.1 fused MC2 requires W8A8 and expert parallelism")
            arguments = torch.ops._C_ascend.dispatch_ffn_combine.default._schema.arguments
            if not any(argument.name == "route_before_down" for argument in arguments):
                raise RuntimeError("Rebuild Ascend custom operators with route_before_down support")
            self.v41_fused_mc2 = DeepseekV41FusedMC2Comm(self.moe_config)

    def _encode_layer_name(self):
        # The registered layer list is shared with the independently executed
        # DSpark draft. On torch <2.11 a fresh positional cursor starts at
        # target layer zero even on a draft-only forward.
        name = super()._encode_layer_name()
        return self.layer_name if isinstance(name, str) else name

    def forward(self, *args, **kwargs):
        previous_type, previous_method = _EXTRA_CTX.moe_comm_type, _EXTRA_CTX.moe_comm_method
        full_decode = (
            is_forward_context_available() and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        )
        selected = select_v41_comm_type(
            self.v41_fused_mc2 is not None,
            previous_type,
            _EXTRA_CTX.padded_num_tokens,
            self.moe_config.tp_group.world_size,
            full_decode,
        )
        method = self.v41_fused_mc2 if selected == MoECommType.FUSED_MC2 else self.v41_comm
        if self.v41_fused_mc2 is not None and selected == MoECommType.ALLGATHER:
            logger.info_once("V4.1 fused MC2 prefill fallback: grouped W8A8 AllGather", scope="process")
        _EXTRA_CTX.moe_comm_type, _EXTRA_CTX.moe_comm_method = selected, method
        try:
            return super().forward(*args, **kwargs)
        finally:
            _EXTRA_CTX.moe_comm_type, _EXTRA_CTX.moe_comm_method = previous_type, previous_method


def build_fused_experts(config, quant_config, prefix, gate, sequence_parallel):
    return FusedMoEFactory(
        num_experts=config.n_routed_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        custom_routing_function=gate.select_experts,
        renormalize=gate.renormalize,
        # select_experts already includes the checkpoint's routed scale.
        routed_scaling_factor=1.0,
        quant_config=quant_config,
        prefix=prefix,
        is_sequence_parallel=sequence_parallel,
        ckpt_names=("w1", "w2", "w3"),
        activation="swigluoai_uninterleave" if config.swiglu_limit > 0 else "silu",
        swiglu_limit=config.swiglu_limit,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        router_logits_dtype=gate.weight.dtype,
        n_shared_experts=0,
        runner_cls=DeepseekV41FusedMoERunner,
        routed_experts_cls=DeepseekV41RoutedExperts,
    )
