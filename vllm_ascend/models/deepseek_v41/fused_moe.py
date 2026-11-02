# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend grouped expert dispatch for V4.1 text and DSpark blocks."""

from vllm.model_executor.layers.fused_moe import FusedMoEFactory

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

from .fused_experts import DeepseekV41AllGatherComm, DeepseekV41RoutedExperts


class DeepseekV41FusedMoERunner(AscendMoERunner):
    """Keep dispatch and quantization local to the target or draft expert layout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The target has 384 experts / top-6 and the draft 128 / top-3.
        # A single process-wide dispatcher cannot describe both layouts.
        self.v41_comm = DeepseekV41AllGatherComm(self.moe_config)

    def _encode_layer_name(self):
        # The registered layer list is shared with the independently executed
        # DSpark draft. On torch <2.11 a fresh positional cursor starts at
        # target layer zero even on a draft-only forward.
        name = super()._encode_layer_name()
        return self.layer_name if isinstance(name, str) else name

    def forward(self, *args, **kwargs):
        previous_type, previous_method = _EXTRA_CTX.moe_comm_type, _EXTRA_CTX.moe_comm_method
        _EXTRA_CTX.moe_comm_type, _EXTRA_CTX.moe_comm_method = MoECommType.ALLGATHER, self.v41_comm
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
