# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_moe import clipped_swiglu_route


def _apply_mlp(self, inputs):
    weights = inputs.weights
    if inputs.topk_scales is None:
        raise ValueError("V4.1 grouped experts require an FP32 factor per dispatched route")
    quantized = inputs.quant.quant_type == QuantType.W8A8
    mxfp = inputs.quant.quant_type == QuantType.W4A8MXFP
    if quantized and (inputs.hidden_states.dtype != torch.int8 or inputs.dynamic_scale is None):
        raise ValueError("V4.1 W8A8 grouped experts require dynamically quantized dispatch rows")
    gate_up = _grouped_projection(
        inputs.hidden_states,
        weights.w1,
        inputs.group_list,
        inputs.group_list_type,
        weights.w1_scale,
        inputs.dynamic_scale,
        mxfp=mxfp,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    factor = inputs.topk_scales.unsqueeze(-1)
    # Match the checkpoint's individual expert GEMMs: BF16 gate/up output,
    # FP32 nonlinear activation and route factor, then BF16 input to W2.
    activation = clipped_swiglu_route(gate, up, inputs.topk_scales, inputs.swiglu_limit)
    scale = None
    if quantized:
        activation, scale = torch_npu.npu_dynamic_quant(activation, dst_type=torch.int8)
    elif mxfp:
        activation, scale = torch_npu.npu_dynamic_mx_quant(activation, dst_type=torch.float8_e4m3fn, scale_alg=1)
    before_gmm2 = torch.npu.current_stream().record_event()
    output = _grouped_projection(
        activation, weights.w2, inputs.group_list, inputs.group_list_type, weights.w2_scale, scale, mxfp=mxfp
    )
    # GMM need only initialize rows covered by the local expert group list.
    # Clear padding before native unpermute, which can otherwise read NaNs.
    output = torch.where(factor != 0, output.float(), 0.0)
    logger.info_once(
        "V4.1 grouped expert down projection executed: experts=%d top_k=%d quant=%s",
        self.moe_config.num_experts,
        self.moe_config.experts_per_token,
        inputs.quant.quant_type.name,
    )
    return output, before_gmm2
