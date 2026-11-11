# SPDX-License-Identifier: Apache-2.0
"""Apply V4.1's FP32 routing coefficients before the expert W2 activation quantization."""

from dataclasses import replace

import torch
import torch_npu
from torch import nn

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoEAllGatherCombineMetadata,
    MoEAllToAllCombineMetadata,
    MoEMC2CombineMetadata,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_utils import async_all_to_all
from vllm_ascend.quantization.methods import AscendW4A8MXFPDSDynamicFusedMoEMethod


def _dispatched(values, reversed_rows):
    # Both init_routing_v2 and token_permute return original lane -> sorted row.
    # init_routing_v2 encodes non-local lanes with a signed destination.
    destination = reversed_rows.reshape(-1).abs().long()
    flat = values.reshape(-1)
    return torch.empty_like(flat).scatter_(0, destination, flat)


def _route_weights(dispatch, dispatcher):
    """Return one FP32 coefficient per local GMM row, and neutral combine weights."""
    metadata = dispatch.combine_metadata
    weights = metadata.topk_weights
    if isinstance(metadata, MoEMC2CombineMetadata):
        # With expert_scales supplied, A5 dispatch sends coefficients with their
        # tokens and returns them in expert order. Both scale inputs to the
        # reverse collective must be neutral after applying these at W2.
        expanded = metadata.expand_scales
        if expanded is None or expanded.numel() != dispatch.hidden_states.shape[0]:
            raise ValueError("A5 MC2 dispatch did not return one expert scale per expanded row")
        return expanded.reshape(-1), replace(
            metadata, topk_weights=torch.ones_like(weights), expand_scales=torch.ones_like(expanded)
        )
    if isinstance(metadata, MoEAllGatherCombineMetadata):
        # The stock dispatcher has already masked non-local lanes in weights.
        expanded = _dispatched(weights, metadata.expanded_row_idx)[: dispatch.hidden_states.shape[0]]
        return expanded, replace(metadata, topk_weights=weights.ne(0).to(torch.bfloat16))
    if isinstance(metadata, MoEAllToAllCombineMetadata):
        send = _dispatched(weights, metadata.reversed_local_input_permutation_mapping)
        _, received, work = async_all_to_all(send, metadata.output_splits, metadata.input_splits, dispatcher.ep_group)
        work.wait()
        reverse_local_experts = metadata.reversed_global_input_permutation_mapping
        if reverse_local_experts is not None:
            received = _dispatched(received, reverse_local_experts)
        return received, replace(metadata, topk_weights=torch.ones_like(weights, dtype=torch.bfloat16))
    raise TypeError(f"Unknown V4.1 MoE dispatch metadata {type(metadata).__name__}")


def _swiglu_origin(gate_up, route_weight, clamp):
    # Stock MX mode derives its shared exponent from exponent bits only. In
    # mode 3, output_origin gives the FP32 clipped/weighted SwiGLU rounded once
    # to BF16. DynamicMxQuantV3 then applies the checkpoint's ceiling rule.
    return torch.ops._C_ascend.npu_swiglu_group_quant(
        gate_up,
        topk_weight=route_weight,
        group_index=None,
        dst_type=torch.float8_e4m3fn,
        quant_mode=3,
        output_origin=True,
        clamp_value=clamp,
    )[2]


def _quantize(x):
    q, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=1)
    # HF floors the block maximum at 1e-4. Its smallest allowed E8M0
    # exponent is ceil(log2(1e-4/448)) = -22, biased byte 105. Re-encode
    # these groups from their BF16 inputs to avoid double-rounding FP8.
    exponent = scale.view(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 32)
    regular = q.view(torch.uint8).reshape(*exponent.shape, 32)
    floor = (x.float() * (2**22)).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).reshape_as(regular)
    encoded = torch.where((exponent < 105).unsqueeze(-1), floor, regular)
    return encoded.reshape_as(q).view(q.dtype), exponent.clamp_min(105).reshape_as(scale).view(scale.dtype)


class V41SharedActivation(nn.Module):
    """Original FP8 shared expert, including both HF per-32 activation scales."""

    def __init__(self, clamp):
        super().__init__()
        self.clamp = clamp

    @staticmethod
    def quantize_mx_input(x, *, dst_type):
        return _quantize(x)

    def forward(self, gate_up):
        return _swiglu_origin(gate_up, None, self.clamp)

    def quantize_mx(self, gate_up):
        return _quantize(self(gate_up))


def _project(activation, scales, weight, weight_scales, dispatch):
    return torch_npu.npu_grouped_matmul(
        x=[activation],
        weight=[weight],
        antiquant_scale=[weight_scales],
        per_token_scale=[scales],
        per_token_scale_dtype=torch.float8_e8m0fnu,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        group_list=dispatch.group_list,
        group_list_type=dispatch.group_list_type,
        group_type=0,
        split_item=2,
        output_dtype=torch.bfloat16,
    )[0]


class V41FP4Experts(AscendW4A8MXFPDSDynamicFusedMoEMethod):
    """Keep the platform's expert layout, dispatcher, grouped GEMMs and shared overlap."""

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        comm = _EXTRA_CTX.moe_comm_method
        packed = build_fused_experts_input(
            hidden_states=x,
            topk_weights=topk_weights.float(),
            topk_ids=topk_ids,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            quant_type=self.quant_type,
            dynamic_eplb=self.dynamic_eplb,
            expert_map=layer.ascend_expert_map,
            global_redundant_expert_num=layer.global_redundant_expert_num,
            mc2_mask=layer.ascend_mc2_mask,
            pertoken_scale=layer.ascend_pertoken_scale,
            activation=layer.activation,
            mxfp_act_quant_type=torch.float8_e4m3fn,
            mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
            mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
            mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            mxfp_use_bf16=True,
        )
        stream = torch.npu.current_stream()
        before_dispatch = stream.record_event()
        dispatch = comm.token_dispatcher.token_dispatch(build_token_dispatch_input(fused_experts_input=packed))
        weights, combine = _route_weights(dispatch, comm.token_dispatcher)
        activation, scales = dispatch.hidden_states, dispatch.dynamic_scale
        if scales is None:
            activation, scales = _quantize(activation)
        scales = scales.reshape(activation.shape[0], activation.shape[1] // 64, 2)
        gate_up = _project(activation, scales, layer.w13_weight, layer.w13_weight_scale, dispatch)
        activation, scales = _quantize(_swiglu_origin(gate_up, weights, comm.moe_config.swiglu_limit))
        before_gmm2 = stream.record_event()
        output = _project(activation, scales, layer.w2_weight, layer.w2_weight_scale, dispatch)
        before_combine = stream.record_event()
        routed_out = comm.token_dispatcher.token_combine(hidden_states=output, combine_metadata=combine)
        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch,
            before_gmm2_evt=before_gmm2,
            before_combine_evt=before_combine,
            group_list_type=dispatch.group_list_type,
            expert_tokens=dispatch.group_list,
        )
