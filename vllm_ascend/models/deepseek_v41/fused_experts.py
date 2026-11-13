# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve V4.1's per-route weight before quantizing the expert down projection."""

from dataclasses import dataclass, replace

import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.models.deepseek_v41.core import clipped_swiglu
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import build_mlp_compute_input
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoEAllGatherCombineMetadata,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    FusedExpertsResult,
    FusedMC2CommImpl,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.routed_experts import (
    AscendRoutedExperts,
    AscendUnquantizedFusedMoEMethod,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.ops.register_custom_ops import ep_pad_and_reduce_scatter
from vllm_ascend.ops.triton.deepseek_v41_prefill_experts import (
    prefill_combine_output,
    prefill_fusion_available,
    prefill_route_activation,
)
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

# The native dispatcher allocates one expanded row per (gathered token, route),
# including remote routes. Bound its local workspace independently of DP and EP.
MAX_EAGER_ROUTED_TOKENS = 1024
MAX_FUSED_PREFILL_ROUTED_TOKENS = 2048


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


@dataclass(frozen=True, slots=True)
class DeepseekV41PrefillCombineMetadata(MoEAllGatherCombineMetadata):
    route_scales: torch.Tensor


class DeepseekV41Dispatcher(TokenDispatcherWithAllGather):
    def token_dispatch(self, token_dispatch_input):
        output = super().token_dispatch(token_dispatch_input)
        meta = output.combine_metadata
        expert_map = token_dispatch_input.routing.expert_map
        ids = token_dispatch_input.topk_ids
        owned = torch.ones_like(ids, dtype=torch.bool) if expert_map is None else expert_map[ids.long()] >= 0
        scales = dispatched_scales(
            meta.expanded_row_idx,
            token_dispatch_input.topk_weights,
            owned,
            output.hidden_states.shape[0],
        )
        if prefill_fusion_available(output.hidden_states):
            meta = DeepseekV41PrefillCombineMetadata(
                topk_weights=owned.float(),
                expanded_row_idx=meta.expanded_row_idx,
                restore_shape=meta.restore_shape,
                route_scales=scales,
            )
        else:
            meta = replace(meta, topk_weights=owned.float())
        return replace(output, topk_scales=scales, combine_metadata=meta)

    def token_combine(self, hidden_states, combine_metadata, bias=None, *, out=None):
        if isinstance(combine_metadata, DeepseekV41PrefillCombineMetadata):
            output = prefill_combine_output(
                hidden_states,
                combine_metadata.route_scales,
                combine_metadata.expanded_row_idx,
                combine_metadata.topk_weights,
                out=out,
            )
            return output.view(combine_metadata.restore_shape)
        result = super().token_combine(hidden_states, combine_metadata, bias)
        if out is not None:
            out.copy_(result)
            return out
        return result


def _grouped_projection(states, weights, groups, group_type, weight_scale=None, token_scale=None):
    options = {}
    if weight_scale is not None:
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
        context = get_forward_context()
        metadata = context.dp_metadata
        if metadata is not None:
            group = get_ep_group()
            sizes = metadata.get_chunk_sizes_across_dp_rank()
            if sizes is None or len(sizes) != group.world_size or local_shape[0] > sizes[group.rank_in_group]:
                raise ValueError("V4.1 local tokens exceed the EP gather layout")
            actual_sizes = getattr(context, "ep_actual_local_sizes", None)
            counts = getattr(context, "num_actual_tokens_across_dp", None)
            if (
                actual_sizes is None
                and counts is not None
                and context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                and not context.is_draft_model
                and not context.in_profile_run
            ):
                tp_size = get_tensor_model_parallel_world_size()
                if counts.numel() * tp_size != group.world_size:
                    raise ValueError("V4.1 DP token counts do not cover the EP group")
                actual_sizes = [(n + tp_size - 1) // tp_size for n in counts.tolist() for _ in range(tp_size)]
                context.ep_actual_local_sizes = actual_sizes
            if actual_sizes is not None and local_shape[0] != actual_sizes[group.rank_in_group]:
                raise ValueError("V4.1 local tokens disagree with the actual EP gather layout")
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

    def _prefill_chunk_into(self, inputs, destination):
        """Keep chunk results in their final buffer without a second FP32 copy."""
        assert _EXTRA_CTX.moe_comm_method is not None, "Missing communication context"
        before_dispatch = torch.npu.current_stream().record_event()
        dispatched = self.token_dispatcher.token_dispatch(build_token_dispatch_input(fused_experts_input=inputs))
        mlp_input = build_mlp_compute_input(
            fused_experts_input=inputs,
            token_dispatch_output=dispatched,
            moe_config=self.moe_config,
        )
        values, before_gmm2 = self._apply_mlp(mlp_input)
        before_combine = torch.npu.current_stream().record_event()
        result = self.token_dispatcher.token_combine(values, dispatched.combine_metadata, out=destination)
        return FusedExpertsResult(
            routed_out=result,
            before_dispatch_evt=before_dispatch,
            before_gmm2_evt=before_gmm2,
            before_combine_evt=before_combine,
            group_list_type=dispatched.group_list_type,
            expert_tokens=dispatched.group_list,
        )

    def fused_experts(self, fused_experts_input):
        inputs = fused_experts_input
        rows = inputs.hidden_states.shape[0]
        # Larger batches rely on the fused chain's smaller live intermediates.
        chunk_size = (
            MAX_FUSED_PREFILL_ROUTED_TOKENS
            if prefill_fusion_available(inputs.hidden_states)
            else MAX_EAGER_ROUTED_TOKENS
        )
        if rows <= chunk_size or (
            is_forward_context_available() and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            return super().fused_experts(inputs)

        # prepare() has already gathered the rows and selected their routes.
        # Each chunk runs only local dispatch, grouped GEMMs and unpermute.
        # finalize() receives one complete output in the original gather order.
        routed_out = inputs.hidden_states.new_empty(inputs.hidden_states.shape, dtype=torch.float32)
        counts = inputs.topk_ids.new_zeros((self.moe_config.num_local_experts,), dtype=torch.int64)
        for start in range(0, rows, chunk_size):
            part = slice(start, start + chunk_size)
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
            if prefill_fusion_available(inputs.hidden_states):
                result = self._prefill_chunk_into(chunk, routed_out[part])
            else:
                result = super().fused_experts(chunk)
                routed_out[part].copy_(result.routed_out)
            if start == 0:
                before_dispatch = result.before_dispatch_evt
            # AllGather dispatch returns per-expert counts (group_list_type=1).
            counts.add_(result.expert_tokens)
        return replace(
            result,
            routed_out=routed_out,
            expert_tokens=counts,
            before_dispatch_evt=before_dispatch,
        )

    def _apply_mlp(self, inputs):
        weights = inputs.weights
        if inputs.topk_scales is None:
            raise ValueError("V4.1 grouped experts require an FP32 factor per dispatched route")
        quantized = inputs.quant.quant_type == QuantType.W8A8
        if quantized and (inputs.hidden_states.dtype != torch.int8 or inputs.dynamic_scale is None):
            raise ValueError("V4.1 W8A8 grouped experts require dynamically quantized dispatch rows")
        gate_up = _grouped_projection(
            inputs.hidden_states,
            weights.w1,
            inputs.group_list,
            inputs.group_list_type,
            weights.w1_scale,
            inputs.dynamic_scale,
        )
        factor = inputs.topk_scales.unsqueeze(-1)
        # Match the checkpoint's individual expert GEMMs: BF16 gate/up output,
        # FP32 nonlinear activation and route factor, then BF16 input to W2.
        fused_prefill = prefill_fusion_available(inputs.hidden_states)
        if fused_prefill:
            activation = prefill_route_activation(
                gate_up,
                inputs.topk_scales,
                inputs.swiglu_limit,
                groups=inputs.group_list if inputs.group_list_type == 1 else None,
            )
            del gate_up
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            activation = (clipped_swiglu(gate, up, inputs.swiglu_limit) * factor).to(torch.bfloat16)
        scale = None
        if quantized:
            activation, scale = torch_npu.npu_dynamic_quant(activation, dst_type=torch.int8)
        before_gmm2 = torch.npu.current_stream().record_event()
        output = _grouped_projection(
            activation,
            weights.w2,
            inputs.group_list,
            inputs.group_list_type,
            weights.w2_scale,
            scale,
        )
        # The prefill consumer masks padding while reading BF16 expert rows.
        if not fused_prefill:
            output = torch.where(factor != 0, output.float(), 0.0)
        logger.info_once(
            "V4.1 grouped expert down projection executed: experts=%d top_k=%d quant=%s",
            self.moe_config.num_experts,
            self.moe_config.experts_per_token,
            inputs.quant.quant_type.name,
        )
        return output, before_gmm2


class DeepseekV41FusedMC2Comm(FusedMC2CommImpl):
    """Grouped W8A8 with the route factor before W2 quantization.

    Each runner owns its communication/shape state: the target and DSpark
    draft have different local expert counts and top-k widths.
    """

    def fused_experts(self, fused_experts_input):
        inputs = fused_experts_input
        if inputs.quant.quant_type != QuantType.W8A8 or inputs.hidden_states.dtype != torch.bfloat16:
            raise ValueError("V4.1 fused MC2 requires grouped W8A8 with BF16 input")
        if inputs.topk_weights.dtype != torch.float32:
            raise ValueError("V4.1 fused MC2 requires FP32 router weights")
        weights = inputs.weights
        if any(weight.dtype != torch.int8 or weight.ndim != 3 for weight in (weights.w1, weights.w2)):
            raise ValueError("V4.1 fused MC2 requires stacked INT8 expert weights")
        if any(scale.dtype != torch.float32 for scale in (weights.w1_scale, weights.w2_scale)):
            raise ValueError("V4.1 fused MC2 requires FP32 channel scales")
        # Uniform MC2 padding makes this conservative bound identical across
        # EP ranks, including repeated expert IDs, and prevents the fused
        # kernel's capacity truncation from dropping any route.
        capacity = inputs.hidden_states.shape[0] * self.moe_config.ep_size * self.moe_config.experts_per_token
        if not 0 < capacity <= get_ascend_config().mega_moe_max_tokens:
            raise ValueError("V4.1 fused MC2 route capacity exceeds the configured workspace")
        out = torch.empty_like(inputs.hidden_states, dtype=torch.float32)
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=inputs.hidden_states,
            weight1=[weights.w1],
            weight2=[weights.w2],
            expert_idx=inputs.topk_ids.to(torch.int32),
            scale1=[weights.w1_scale],
            scale2=[weights.w2_scale],
            bias1=[],
            bias2=[],
            probs=inputs.topk_weights,
            group=self.token_dispatcher.moe_all_to_all_group_name,
            max_output_size=capacity,
            swiglu_limit=self.swiglu_limit,
            x_active_mask=inputs.routing.mc2_mask,
            out=out,
            expert_token_nums=self.expert_token_nums,
            route_before_down=True,
        )
        logger.info_once(
            "V4.1 fused MC2 W8A8 executed: experts=%d top_k=%d ep_rank=%d route_before_down=1",
            self.moe_config.num_experts,
            self.moe_config.experts_per_token,
            self.moe_config.ep_group.rank_in_group,
            scope="process",
        )
        return FusedExpertsResult(routed_out=out, expert_tokens=self.expert_token_nums)


class _RouteBeforeDown:
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if topk_weights.dtype != torch.float32:
            raise ValueError("V4.1 grouped experts require FP32 router weights")
        quant = layer.quant_type
        weights = {}
        if quant == QuantType.W8A8:
            # A8W8 grouped matmul with BF16 output requires BF16 channel
            # scales. The FP32 view injected for fused SwiGLU kernels is distinct.
            if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
                weights = dict(
                    w1_scale=layer.w13_weight_scale_fp32,
                    w2_scale=layer.v41_w2_scale_fp32,
                )
            else:
                weights = dict(w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale)
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
                mc2_mask=layer.ascend_mc2_mask,
                **weights,
            )
        )


class _V41W8A8Method(_RouteBeforeDown, AscendFusedMoEMethod):
    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        if get_ascend_config().enable_fused_mc2 == 1:
            # Match the existing BF16 channel values exactly; materialize the
            # FP32 view once, outside graph capture and the decode hot path.
            layer.v41_w2_scale_fp32 = layer.w2_weight_scale.float()


class _V41BF16Method(_RouteBeforeDown, AscendUnquantizedFusedMoEMethod):
    pass


class DeepseekV41RoutedExperts(AscendRoutedExperts):
    def _get_quant_method(self, prefix, quant_config, moe_config):
        if quant_config is not None:
            method = super()._get_quant_method(prefix, quant_config, moe_config)
            if isinstance(method, AscendFusedMoEMethod):
                if method.quant_method.quant_type != QuantType.W8A8:
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
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
