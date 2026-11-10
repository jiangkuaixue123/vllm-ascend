# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve V4.1's per-route weight before quantizing the expert down projection."""

from dataclasses import replace

import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.models.deepseek_v41.core import clipped_swiglu
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.ops.register_custom_ops import ep_pad_and_reduce_scatter
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

# The native dispatcher allocates one expanded row per (gathered token, route),
# including remote routes. Bound its local workspace independently of DP and EP.
MAX_EAGER_ROUTED_TOKENS = 1024


def dispatched_scales(row_ids, route_weights, owned, num_rows):
    """Map original flat (token, top-k slot) to the expert-sorted row.

    init_routing_v2 indexes expanded_row_idx by the original pair, as used
    by native token_unpermute. Off-rank slots may alias a physical row. They
    must contribute zero even when the kernel uses negative or sentinel indices.
    """
    result = route_weights.new_zeros(num_rows)
    if num_rows:
        rows = row_ids.flatten().long().abs().clamp(max=num_rows - 1)
        result.scatter_add_(0, rows, torch.where(owned, route_weights, 0.0).flatten())
    return result


class DeepseekV41Dispatcher(TokenDispatcherWithAllGather):
    def token_dispatch(self, token_dispatch_input):
        output = super().token_dispatch(token_dispatch_input)
        meta = output.combine_metadata
        expert_map = token_dispatch_input.routing.expert_map
        ids = token_dispatch_input.topk_ids
        owned = torch.ones_like(ids, dtype=torch.bool) if expert_map is None else expert_map[ids.long()] >= 0
        scales = dispatched_scales(
            meta.expanded_row_idx, token_dispatch_input.topk_weights, owned, output.hidden_states.shape[0]
        )
        return replace(output, topk_scales=scales, combine_metadata=replace(meta, topk_weights=owned.float()))


def _grouped_projection(states, weights, groups, group_type, weight_scale=None, token_scale=None, *, mxfp=False):
    options = {}
    if weight_scale is not None:
        if mxfp:
            options = dict(
                # A5 W4A8 uses the antiquant scale slot for packed FP4.
                antiquant_scale=[weight_scale],
                per_token_scale=[token_scale.unflatten(-1, (-1, 2)) if token_scale.ndim == 2 else token_scale],
                per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                x_dtype=torch.float8_e4m3fn,
                weight_dtype=torch_npu.float4_e2m1fn_x2,
            )
        else:
            options = dict(scale=[weight_scale], per_token_scale=[token_scale.reshape(-1)])
    return torch_npu.npu_grouped_matmul(
        x=[states],
        weight=[weights],
        group_list=groups,
        group_list_type=group_type,
        group_type=0,
        split_item=2,
        output_dtype=torch.bfloat16,
        **options,
    )[0]


class DeepseekV41PrepareAndFinalize(PrepareAndFinalizeWithAllGather):
    def _prepare_with_ep_group(self, hidden_states, router_logits, quant_type=QuantType.NONE):
        # The runner's sizes describe EP communication slots. An eager V4.1
        # forward trims real tokens before sharding the residual across TP.
        local_shape = hidden_states.shape
        if router_logits.shape[0] != local_shape[0]:
            raise ValueError("V4.1 router and expert inputs must have the same number of token rows")
        metadata = get_forward_context().dp_metadata
        if metadata is not None:
            group = get_ep_group()
            sizes = metadata.get_chunk_sizes_across_dp_rank()
            if sizes is None or len(sizes) != group.world_size or local_shape[0] > sizes[group.rank_in_group]:
                raise ValueError("V4.1 local tokens exceed the EP gather layout")
            if not local_shape[0] and max(sizes) > 0:
                # Participate on an idle DP without invoking dynamic quant on
                # an empty tensor. Gather padding is discarded after scatter.
                hidden_states = hidden_states.new_zeros((1, *local_shape[1:]))
                router_logits = router_logits.new_zeros((1, *router_logits.shape[1:]))
        prepared = super()._prepare_with_ep_group(hidden_states, router_logits, quant_type)
        return replace(prepared, padded_hidden_states_shape=local_shape)

    def finalize(self, hidden_states, reduce_results, padded_hidden_states_shape=None):
        local = super().finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        if self._use_ep_sequence_parallel():
            if padded_hidden_states_shape is None or local.shape[0] < padded_hidden_states_shape[0]:
                raise ValueError("V4.1 EP scatter did not return the original local token shard")
            return local[: padded_hidden_states_shape[0]]
        return local

    def _finalize_with_ep_group(self, hidden_states):
        # Both the text target and its draft gather TP token shards across EP.
        # Their multimodal config wrapper does not change the output row owner.
        if self.moe_config.pcp_size != 1:
            raise ValueError("V4.1 text grouped experts require PCP1")
        return ep_pad_and_reduce_scatter(hidden_states, get_forward_context().dp_metadata)


class DeepseekV41AllGatherComm(AllGatherCommImpl):
    def _get_prepare_finalize(self):
        return DeepseekV41PrepareAndFinalize(self.moe_config)

    def _get_token_dispatcher(self):
        return DeepseekV41Dispatcher(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def fused_experts(self, fused_experts_input):
        inputs = fused_experts_input
        rows = inputs.hidden_states.shape[0]
        if rows <= MAX_EAGER_ROUTED_TOKENS or (
            is_forward_context_available() and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            return super().fused_experts(inputs)

        # prepare() has already gathered the rows and selected their routes.
        # Each chunk runs only local dispatch, grouped GEMMs and unpermute.
        # finalize() receives one complete output in the original gather order.
        routed_out = inputs.hidden_states.new_empty(inputs.hidden_states.shape, dtype=torch.float32)
        counts = inputs.topk_ids.new_zeros((self.moe_config.num_local_experts,), dtype=torch.int64)
        for start in range(0, rows, MAX_EAGER_ROUTED_TOKENS):
            part = slice(start, start + MAX_EAGER_ROUTED_TOKENS)
            routing = inputs.routing
            chunk = replace(
                inputs,
                hidden_states=inputs.hidden_states[part],
                topk_weights=inputs.topk_weights[part],
                topk_ids=inputs.topk_ids[part],
                routing=replace(
                    routing,
                    pertoken_scale=None if routing.pertoken_scale is None else routing.pertoken_scale[part],
                    mc2_mask=None if routing.mc2_mask is None else routing.mc2_mask[part],
                ),
            )
            result = super().fused_experts(chunk)
            if start == 0:
                before_dispatch = result.before_dispatch_evt
            routed_out[part].copy_(result.routed_out)
            # AllGather dispatch returns per-expert counts (group_list_type=1).
            counts.add_(result.expert_tokens)
        return replace(result, routed_out=routed_out, expert_tokens=counts, before_dispatch_evt=before_dispatch)

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
        activation = (clipped_swiglu(gate, up, inputs.swiglu_limit) * factor).to(torch.bfloat16)
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


class _RouteBeforeDown:
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if topk_weights.dtype != torch.float32:
            raise ValueError("V4.1 grouped experts require FP32 router weights")
        quant = layer.quant_type
        weights = {}
        if quant == QuantType.W8A8:
            # A8W8 grouped matmul with BF16 output requires BF16 channel
            # scales. The FP32 view injected for fused SwiGLU kernels is distinct.
            weights = dict(w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale)
        elif quant == QuantType.W4A8MXFP:
            weights = dict(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
                mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_use_bf16=True,
            )
        elif quant != QuantType.NONE:
            raise ValueError(f"Unsupported V4.1 grouped expert quantization: {quant}")
        return _EXTRA_CTX.moe_comm_method.fused_experts(
            build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=quant,
                dynamic_eplb=False,
                expert_map=layer.ascend_expert_map,
                pertoken_scale=layer.ascend_pertoken_scale,
                **weights,
            )
        )


class _V41W8A8Method(_RouteBeforeDown, AscendFusedMoEMethod):
    pass


class _V41BF16Method(_RouteBeforeDown, AscendUnquantizedFusedMoEMethod):
    pass


class DeepseekV41RoutedExperts(AscendRoutedExperts):
    def _get_quant_method(self, prefix, quant_config, moe_config):
        if quant_config is not None:
            method = super()._get_quant_method(prefix, quant_config, moe_config)
            if isinstance(method, AscendFusedMoEMethod):
                if method.quant_method.quant_type not in (QuantType.W8A8, QuantType.W4A8MXFP):
                    raise ValueError(f"Unsupported V4.1 grouped quantization for {prefix}")
                return _V41W8A8Method(method.quant_method, moe_config)
            if not isinstance(method, AscendUnquantizedFusedMoEMethod):
                raise ValueError(f"Unsupported V4.1 grouped expert method for {prefix}")
        return _V41BF16Method(moe_config)

    def _select_experts(self, hidden_states, router_logits, enable_force_load_balance, input_ids=None):
        # The trained gate already applies unbiased probabilities and route
        # scaling. Keep FP32 through the nonlinear activation before W2.
        if self.router is None:
            raise ValueError("Missing V4.1 expert router")
        return self.router._select_experts(
            hidden_states=hidden_states, router_logits=router_logits, input_ids=input_ids
        )
