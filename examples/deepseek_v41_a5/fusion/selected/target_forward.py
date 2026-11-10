# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_hc import hc_pre_norm, hc_post
from vllm_ascend.ops.triton.deepseek_v41 import hc_gates_sinkhorn


def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, image_token_mask=None):
    # Statically expanded model blocks retain the measured execution order.
    if intermediate_tensors is not None or input_ids is None:
        raise ValueError("V4.1 requires raw token IDs and no pipeline tensors")
    metadata = self.layers[0].attn.cache.metadata()
    if getattr(self, "decode_enabled", False) and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
        return self._forward_decode(input_ids, metadata)
    actual_tokens = metadata.num_actual_tokens if metadata else input_ids.numel()
    runtime = DeepseekV41Runtime(metadata.requests if metadata else [RequestSpan(0, actual_tokens, actual_tokens)])
    ids = input_ids[:actual_tokens]
    image_mask = image_token_mask[:actual_tokens] if image_token_mask is not None else None
    hashes = self._hash_tokens(ids, runtime, metadata, image_mask)
    local_image = None
    local_text = None
    if image_mask is not None:
        mask = image_mask.to(ids.device, non_blocking=True)
        local_image = sp_shard(mask) if self.sequence_parallel else mask
        local_text = sp_shard(~mask) if self.sequence_parallel else ~mask
    if actual_tokens == 0:
        hidden_states = self.embed.weight.new_empty((0, self.config.hidden_size))
    else:
        hidden_states = self.embed(ids) if inputs_embeds is None else inputs_embeds[:actual_tokens]
    hidden_states, pre_mix = self._start_hc(hidden_states)
    auxiliary = []
    if self.layers[0].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(0)
        hidden_states = self.layers[0].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 0, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[0]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_1_hidden_states = block_hidden_states
            inline_1_fn = block_self.hc_attn_fn
            inline_1_scale = block_self.hc_attn_scale
            inline_1_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_1_states = inline_1_hidden_states.flatten(-2).float()
            inline_1_rstd = torch.rsqrt(inline_1_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_1_mixes = F.linear(inline_1_states, inline_1_fn.float()) * inline_1_rstd
            inline_2_mixes = inline_1_mixes
            inline_2_scale = inline_1_scale
            inline_2_base = inline_1_base
            inline_2_result = hc_gates_sinkhorn(
                inline_2_mixes, inline_2_scale, inline_2_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_2_mixes, inline_2_scale, inline_2_base
            inline_1_result = inline_2_result
            del inline_2_result
            del (
                inline_1_base,
                inline_1_fn,
                inline_1_hidden_states,
                inline_1_mixes,
                inline_1_rstd,
                inline_1_scale,
                inline_1_states,
            )
            block_attn_pre, block_post, block_comb = inline_1_result
            del inline_1_result
            inline_3_hidden_states = block_hidden_states
            inline_3_pre_mix = block_pre_mix
            inline_4_result = hc_pre_norm(inline_3_hidden_states, inline_3_pre_mix, block_self.attn_norm.weight, 1e-20)
            del inline_3_hidden_states, inline_3_pre_mix
            block_attended = block_self.attn(inline_4_result, block_runtime)
            del inline_4_result
            inline_5_sublayer_output = block_attended
            inline_5_residual = block_hidden_states
            inline_5_post = block_post
            inline_5_comb = block_comb
            inline_5_result = hc_post(inline_5_sublayer_output, inline_5_residual, inline_5_post, inline_5_comb)
            del inline_5_sublayer_output, inline_5_residual, inline_5_post, inline_5_comb
            block_hidden_states = inline_5_result
            del inline_5_result
            inline_6_hidden_states = block_hidden_states
            inline_6_fn = block_self.hc_ffn_fn
            inline_6_scale = block_self.hc_ffn_scale
            inline_6_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_6_states = inline_6_hidden_states.flatten(-2).float()
            inline_6_rstd = torch.rsqrt(inline_6_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_6_mixes = F.linear(inline_6_states, inline_6_fn.float()) * inline_6_rstd
            inline_7_mixes = inline_6_mixes
            inline_7_scale = inline_6_scale
            inline_7_base = inline_6_base
            inline_7_result = hc_gates_sinkhorn(
                inline_7_mixes, inline_7_scale, inline_7_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_7_mixes, inline_7_scale, inline_7_base
            inline_6_result = inline_7_result
            del inline_7_result
            del (
                inline_6_base,
                inline_6_fn,
                inline_6_hidden_states,
                inline_6_mixes,
                inline_6_rstd,
                inline_6_scale,
                inline_6_states,
            )
            block_ffn_pre, block_post, block_comb = inline_6_result
            del inline_6_result
            inline_8_hidden_states = block_hidden_states
            inline_8_pre_mix = block_attn_pre
            inline_9_result = hc_pre_norm(inline_8_hidden_states, inline_8_pre_mix, block_self.ffn_norm.weight, 1e-20)
            del inline_8_hidden_states, inline_8_pre_mix
            block_ffn_input = inline_9_result
            del inline_9_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_10_sublayer_output = block_output
            inline_10_residual = block_hidden_states
            inline_10_post = block_post
            inline_10_comb = block_comb
            inline_10_result = hc_post(inline_10_sublayer_output, inline_10_residual, inline_10_post, inline_10_comb)
            del inline_10_sublayer_output, inline_10_residual, inline_10_post, inline_10_comb
            hidden_states, pre_mix = (inline_10_result, block_ffn_pre)
            del inline_10_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[0]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_11_hidden_states = block_hidden_states
            inline_11_fn = block_self.hc_attn_fn
            inline_11_scale = block_self.hc_attn_scale
            inline_11_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_11_states = inline_11_hidden_states.flatten(-2).float()
            inline_11_rstd = torch.rsqrt(inline_11_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_11_mixes = F.linear(inline_11_states, inline_11_fn.float()) * inline_11_rstd
            inline_12_mixes = inline_11_mixes
            inline_12_scale = inline_11_scale
            inline_12_base = inline_11_base
            inline_12_result = hc_gates_sinkhorn(
                inline_12_mixes, inline_12_scale, inline_12_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_12_mixes, inline_12_scale, inline_12_base
            inline_11_result = inline_12_result
            del inline_12_result
            del (
                inline_11_base,
                inline_11_fn,
                inline_11_hidden_states,
                inline_11_mixes,
                inline_11_rstd,
                inline_11_scale,
                inline_11_states,
            )
            block_attn_pre, block_post, block_comb = inline_11_result
            del inline_11_result
            inline_13_hidden_states = block_hidden_states
            inline_13_pre_mix = block_pre_mix
            inline_14_result = hc_pre_norm(
                inline_13_hidden_states, inline_13_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_13_hidden_states, inline_13_pre_mix
            block_attended = block_self.attn(inline_14_result, block_runtime)
            del inline_14_result
            inline_15_sublayer_output = block_attended
            inline_15_residual = block_hidden_states
            inline_15_post = block_post
            inline_15_comb = block_comb
            inline_15_result = hc_post(inline_15_sublayer_output, inline_15_residual, inline_15_post, inline_15_comb)
            del inline_15_sublayer_output, inline_15_residual, inline_15_post, inline_15_comb
            block_hidden_states = inline_15_result
            del inline_15_result
            inline_16_hidden_states = block_hidden_states
            inline_16_fn = block_self.hc_ffn_fn
            inline_16_scale = block_self.hc_ffn_scale
            inline_16_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_16_states = inline_16_hidden_states.flatten(-2).float()
            inline_16_rstd = torch.rsqrt(inline_16_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_16_mixes = F.linear(inline_16_states, inline_16_fn.float()) * inline_16_rstd
            inline_17_mixes = inline_16_mixes
            inline_17_scale = inline_16_scale
            inline_17_base = inline_16_base
            inline_17_result = hc_gates_sinkhorn(
                inline_17_mixes, inline_17_scale, inline_17_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_17_mixes, inline_17_scale, inline_17_base
            inline_16_result = inline_17_result
            del inline_17_result
            del (
                inline_16_base,
                inline_16_fn,
                inline_16_hidden_states,
                inline_16_mixes,
                inline_16_rstd,
                inline_16_scale,
                inline_16_states,
            )
            block_ffn_pre, block_post, block_comb = inline_16_result
            del inline_16_result
            inline_18_hidden_states = block_hidden_states
            inline_18_pre_mix = block_attn_pre
            inline_19_result = hc_pre_norm(
                inline_18_hidden_states, inline_18_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_18_hidden_states, inline_18_pre_mix
            block_ffn_input = inline_19_result
            del inline_19_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_20_sublayer_output = block_output
            inline_20_residual = block_hidden_states
            inline_20_post = block_post
            inline_20_comb = block_comb
            inline_20_result = hc_post(inline_20_sublayer_output, inline_20_residual, inline_20_post, inline_20_comb)
            del inline_20_sublayer_output, inline_20_residual, inline_20_post, inline_20_comb
            hidden_states, pre_mix = (inline_20_result, block_ffn_pre)
            del inline_20_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[1].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(1)
        hidden_states = self.layers[1].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 1, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[1]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_21_hidden_states = block_hidden_states
            inline_21_fn = block_self.hc_attn_fn
            inline_21_scale = block_self.hc_attn_scale
            inline_21_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_21_states = inline_21_hidden_states.flatten(-2).float()
            inline_21_rstd = torch.rsqrt(inline_21_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_21_mixes = F.linear(inline_21_states, inline_21_fn.float()) * inline_21_rstd
            inline_22_mixes = inline_21_mixes
            inline_22_scale = inline_21_scale
            inline_22_base = inline_21_base
            inline_22_result = hc_gates_sinkhorn(
                inline_22_mixes, inline_22_scale, inline_22_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_22_mixes, inline_22_scale, inline_22_base
            inline_21_result = inline_22_result
            del inline_22_result
            del (
                inline_21_base,
                inline_21_fn,
                inline_21_hidden_states,
                inline_21_mixes,
                inline_21_rstd,
                inline_21_scale,
                inline_21_states,
            )
            block_attn_pre, block_post, block_comb = inline_21_result
            del inline_21_result
            inline_23_hidden_states = block_hidden_states
            inline_23_pre_mix = block_pre_mix
            inline_24_result = hc_pre_norm(
                inline_23_hidden_states, inline_23_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_23_hidden_states, inline_23_pre_mix
            block_attended = block_self.attn(inline_24_result, block_runtime)
            del inline_24_result
            inline_25_sublayer_output = block_attended
            inline_25_residual = block_hidden_states
            inline_25_post = block_post
            inline_25_comb = block_comb
            inline_25_result = hc_post(inline_25_sublayer_output, inline_25_residual, inline_25_post, inline_25_comb)
            del inline_25_sublayer_output, inline_25_residual, inline_25_post, inline_25_comb
            block_hidden_states = inline_25_result
            del inline_25_result
            inline_26_hidden_states = block_hidden_states
            inline_26_fn = block_self.hc_ffn_fn
            inline_26_scale = block_self.hc_ffn_scale
            inline_26_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_26_states = inline_26_hidden_states.flatten(-2).float()
            inline_26_rstd = torch.rsqrt(inline_26_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_26_mixes = F.linear(inline_26_states, inline_26_fn.float()) * inline_26_rstd
            inline_27_mixes = inline_26_mixes
            inline_27_scale = inline_26_scale
            inline_27_base = inline_26_base
            inline_27_result = hc_gates_sinkhorn(
                inline_27_mixes, inline_27_scale, inline_27_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_27_mixes, inline_27_scale, inline_27_base
            inline_26_result = inline_27_result
            del inline_27_result
            del (
                inline_26_base,
                inline_26_fn,
                inline_26_hidden_states,
                inline_26_mixes,
                inline_26_rstd,
                inline_26_scale,
                inline_26_states,
            )
            block_ffn_pre, block_post, block_comb = inline_26_result
            del inline_26_result
            inline_28_hidden_states = block_hidden_states
            inline_28_pre_mix = block_attn_pre
            inline_29_result = hc_pre_norm(
                inline_28_hidden_states, inline_28_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_28_hidden_states, inline_28_pre_mix
            block_ffn_input = inline_29_result
            del inline_29_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_30_sublayer_output = block_output
            inline_30_residual = block_hidden_states
            inline_30_post = block_post
            inline_30_comb = block_comb
            inline_30_result = hc_post(inline_30_sublayer_output, inline_30_residual, inline_30_post, inline_30_comb)
            del inline_30_sublayer_output, inline_30_residual, inline_30_post, inline_30_comb
            hidden_states, pre_mix = (inline_30_result, block_ffn_pre)
            del inline_30_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[1]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_31_hidden_states = block_hidden_states
            inline_31_fn = block_self.hc_attn_fn
            inline_31_scale = block_self.hc_attn_scale
            inline_31_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_31_states = inline_31_hidden_states.flatten(-2).float()
            inline_31_rstd = torch.rsqrt(inline_31_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_31_mixes = F.linear(inline_31_states, inline_31_fn.float()) * inline_31_rstd
            inline_32_mixes = inline_31_mixes
            inline_32_scale = inline_31_scale
            inline_32_base = inline_31_base
            inline_32_result = hc_gates_sinkhorn(
                inline_32_mixes, inline_32_scale, inline_32_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_32_mixes, inline_32_scale, inline_32_base
            inline_31_result = inline_32_result
            del inline_32_result
            del (
                inline_31_base,
                inline_31_fn,
                inline_31_hidden_states,
                inline_31_mixes,
                inline_31_rstd,
                inline_31_scale,
                inline_31_states,
            )
            block_attn_pre, block_post, block_comb = inline_31_result
            del inline_31_result
            inline_33_hidden_states = block_hidden_states
            inline_33_pre_mix = block_pre_mix
            inline_34_result = hc_pre_norm(
                inline_33_hidden_states, inline_33_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_33_hidden_states, inline_33_pre_mix
            block_attended = block_self.attn(inline_34_result, block_runtime)
            del inline_34_result
            inline_35_sublayer_output = block_attended
            inline_35_residual = block_hidden_states
            inline_35_post = block_post
            inline_35_comb = block_comb
            inline_35_result = hc_post(inline_35_sublayer_output, inline_35_residual, inline_35_post, inline_35_comb)
            del inline_35_sublayer_output, inline_35_residual, inline_35_post, inline_35_comb
            block_hidden_states = inline_35_result
            del inline_35_result
            inline_36_hidden_states = block_hidden_states
            inline_36_fn = block_self.hc_ffn_fn
            inline_36_scale = block_self.hc_ffn_scale
            inline_36_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_36_states = inline_36_hidden_states.flatten(-2).float()
            inline_36_rstd = torch.rsqrt(inline_36_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_36_mixes = F.linear(inline_36_states, inline_36_fn.float()) * inline_36_rstd
            inline_37_mixes = inline_36_mixes
            inline_37_scale = inline_36_scale
            inline_37_base = inline_36_base
            inline_37_result = hc_gates_sinkhorn(
                inline_37_mixes, inline_37_scale, inline_37_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_37_mixes, inline_37_scale, inline_37_base
            inline_36_result = inline_37_result
            del inline_37_result
            del (
                inline_36_base,
                inline_36_fn,
                inline_36_hidden_states,
                inline_36_mixes,
                inline_36_rstd,
                inline_36_scale,
                inline_36_states,
            )
            block_ffn_pre, block_post, block_comb = inline_36_result
            del inline_36_result
            inline_38_hidden_states = block_hidden_states
            inline_38_pre_mix = block_attn_pre
            inline_39_result = hc_pre_norm(
                inline_38_hidden_states, inline_38_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_38_hidden_states, inline_38_pre_mix
            block_ffn_input = inline_39_result
            del inline_39_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_40_sublayer_output = block_output
            inline_40_residual = block_hidden_states
            inline_40_post = block_post
            inline_40_comb = block_comb
            inline_40_result = hc_post(inline_40_sublayer_output, inline_40_residual, inline_40_post, inline_40_comb)
            del inline_40_sublayer_output, inline_40_residual, inline_40_post, inline_40_comb
            hidden_states, pre_mix = (inline_40_result, block_ffn_pre)
            del inline_40_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[2].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(2)
        hidden_states = self.layers[2].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 2, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[2]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_41_hidden_states = block_hidden_states
            inline_41_fn = block_self.hc_attn_fn
            inline_41_scale = block_self.hc_attn_scale
            inline_41_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_41_states = inline_41_hidden_states.flatten(-2).float()
            inline_41_rstd = torch.rsqrt(inline_41_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_41_mixes = F.linear(inline_41_states, inline_41_fn.float()) * inline_41_rstd
            inline_42_mixes = inline_41_mixes
            inline_42_scale = inline_41_scale
            inline_42_base = inline_41_base
            inline_42_result = hc_gates_sinkhorn(
                inline_42_mixes, inline_42_scale, inline_42_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_42_mixes, inline_42_scale, inline_42_base
            inline_41_result = inline_42_result
            del inline_42_result
            del (
                inline_41_base,
                inline_41_fn,
                inline_41_hidden_states,
                inline_41_mixes,
                inline_41_rstd,
                inline_41_scale,
                inline_41_states,
            )
            block_attn_pre, block_post, block_comb = inline_41_result
            del inline_41_result
            inline_43_hidden_states = block_hidden_states
            inline_43_pre_mix = block_pre_mix
            inline_44_result = hc_pre_norm(
                inline_43_hidden_states, inline_43_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_43_hidden_states, inline_43_pre_mix
            block_attended = block_self.attn(inline_44_result, block_runtime)
            del inline_44_result
            inline_45_sublayer_output = block_attended
            inline_45_residual = block_hidden_states
            inline_45_post = block_post
            inline_45_comb = block_comb
            inline_45_result = hc_post(inline_45_sublayer_output, inline_45_residual, inline_45_post, inline_45_comb)
            del inline_45_sublayer_output, inline_45_residual, inline_45_post, inline_45_comb
            block_hidden_states = inline_45_result
            del inline_45_result
            inline_46_hidden_states = block_hidden_states
            inline_46_fn = block_self.hc_ffn_fn
            inline_46_scale = block_self.hc_ffn_scale
            inline_46_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_46_states = inline_46_hidden_states.flatten(-2).float()
            inline_46_rstd = torch.rsqrt(inline_46_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_46_mixes = F.linear(inline_46_states, inline_46_fn.float()) * inline_46_rstd
            inline_47_mixes = inline_46_mixes
            inline_47_scale = inline_46_scale
            inline_47_base = inline_46_base
            inline_47_result = hc_gates_sinkhorn(
                inline_47_mixes, inline_47_scale, inline_47_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_47_mixes, inline_47_scale, inline_47_base
            inline_46_result = inline_47_result
            del inline_47_result
            del (
                inline_46_base,
                inline_46_fn,
                inline_46_hidden_states,
                inline_46_mixes,
                inline_46_rstd,
                inline_46_scale,
                inline_46_states,
            )
            block_ffn_pre, block_post, block_comb = inline_46_result
            del inline_46_result
            inline_48_hidden_states = block_hidden_states
            inline_48_pre_mix = block_attn_pre
            inline_49_result = hc_pre_norm(
                inline_48_hidden_states, inline_48_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_48_hidden_states, inline_48_pre_mix
            block_ffn_input = inline_49_result
            del inline_49_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_50_sublayer_output = block_output
            inline_50_residual = block_hidden_states
            inline_50_post = block_post
            inline_50_comb = block_comb
            inline_50_result = hc_post(inline_50_sublayer_output, inline_50_residual, inline_50_post, inline_50_comb)
            del inline_50_sublayer_output, inline_50_residual, inline_50_post, inline_50_comb
            hidden_states, pre_mix = (inline_50_result, block_ffn_pre)
            del inline_50_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[2]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_51_hidden_states = block_hidden_states
            inline_51_fn = block_self.hc_attn_fn
            inline_51_scale = block_self.hc_attn_scale
            inline_51_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_51_states = inline_51_hidden_states.flatten(-2).float()
            inline_51_rstd = torch.rsqrt(inline_51_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_51_mixes = F.linear(inline_51_states, inline_51_fn.float()) * inline_51_rstd
            inline_52_mixes = inline_51_mixes
            inline_52_scale = inline_51_scale
            inline_52_base = inline_51_base
            inline_52_result = hc_gates_sinkhorn(
                inline_52_mixes, inline_52_scale, inline_52_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_52_mixes, inline_52_scale, inline_52_base
            inline_51_result = inline_52_result
            del inline_52_result
            del (
                inline_51_base,
                inline_51_fn,
                inline_51_hidden_states,
                inline_51_mixes,
                inline_51_rstd,
                inline_51_scale,
                inline_51_states,
            )
            block_attn_pre, block_post, block_comb = inline_51_result
            del inline_51_result
            inline_53_hidden_states = block_hidden_states
            inline_53_pre_mix = block_pre_mix
            inline_54_result = hc_pre_norm(
                inline_53_hidden_states, inline_53_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_53_hidden_states, inline_53_pre_mix
            block_attended = block_self.attn(inline_54_result, block_runtime)
            del inline_54_result
            inline_55_sublayer_output = block_attended
            inline_55_residual = block_hidden_states
            inline_55_post = block_post
            inline_55_comb = block_comb
            inline_55_result = hc_post(inline_55_sublayer_output, inline_55_residual, inline_55_post, inline_55_comb)
            del inline_55_sublayer_output, inline_55_residual, inline_55_post, inline_55_comb
            block_hidden_states = inline_55_result
            del inline_55_result
            inline_56_hidden_states = block_hidden_states
            inline_56_fn = block_self.hc_ffn_fn
            inline_56_scale = block_self.hc_ffn_scale
            inline_56_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_56_states = inline_56_hidden_states.flatten(-2).float()
            inline_56_rstd = torch.rsqrt(inline_56_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_56_mixes = F.linear(inline_56_states, inline_56_fn.float()) * inline_56_rstd
            inline_57_mixes = inline_56_mixes
            inline_57_scale = inline_56_scale
            inline_57_base = inline_56_base
            inline_57_result = hc_gates_sinkhorn(
                inline_57_mixes, inline_57_scale, inline_57_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_57_mixes, inline_57_scale, inline_57_base
            inline_56_result = inline_57_result
            del inline_57_result
            del (
                inline_56_base,
                inline_56_fn,
                inline_56_hidden_states,
                inline_56_mixes,
                inline_56_rstd,
                inline_56_scale,
                inline_56_states,
            )
            block_ffn_pre, block_post, block_comb = inline_56_result
            del inline_56_result
            inline_58_hidden_states = block_hidden_states
            inline_58_pre_mix = block_attn_pre
            inline_59_result = hc_pre_norm(
                inline_58_hidden_states, inline_58_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_58_hidden_states, inline_58_pre_mix
            block_ffn_input = inline_59_result
            del inline_59_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_60_sublayer_output = block_output
            inline_60_residual = block_hidden_states
            inline_60_post = block_post
            inline_60_comb = block_comb
            inline_60_result = hc_post(inline_60_sublayer_output, inline_60_residual, inline_60_post, inline_60_comb)
            del inline_60_sublayer_output, inline_60_residual, inline_60_post, inline_60_comb
            hidden_states, pre_mix = (inline_60_result, block_ffn_pre)
            del inline_60_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[3].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(3)
        hidden_states = self.layers[3].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 3, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[3]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_61_hidden_states = block_hidden_states
            inline_61_fn = block_self.hc_attn_fn
            inline_61_scale = block_self.hc_attn_scale
            inline_61_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_61_states = inline_61_hidden_states.flatten(-2).float()
            inline_61_rstd = torch.rsqrt(inline_61_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_61_mixes = F.linear(inline_61_states, inline_61_fn.float()) * inline_61_rstd
            inline_62_mixes = inline_61_mixes
            inline_62_scale = inline_61_scale
            inline_62_base = inline_61_base
            inline_62_result = hc_gates_sinkhorn(
                inline_62_mixes, inline_62_scale, inline_62_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_62_mixes, inline_62_scale, inline_62_base
            inline_61_result = inline_62_result
            del inline_62_result
            del (
                inline_61_base,
                inline_61_fn,
                inline_61_hidden_states,
                inline_61_mixes,
                inline_61_rstd,
                inline_61_scale,
                inline_61_states,
            )
            block_attn_pre, block_post, block_comb = inline_61_result
            del inline_61_result
            inline_63_hidden_states = block_hidden_states
            inline_63_pre_mix = block_pre_mix
            inline_64_result = hc_pre_norm(
                inline_63_hidden_states, inline_63_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_63_hidden_states, inline_63_pre_mix
            block_attended = block_self.attn(inline_64_result, block_runtime)
            del inline_64_result
            inline_65_sublayer_output = block_attended
            inline_65_residual = block_hidden_states
            inline_65_post = block_post
            inline_65_comb = block_comb
            inline_65_result = hc_post(inline_65_sublayer_output, inline_65_residual, inline_65_post, inline_65_comb)
            del inline_65_sublayer_output, inline_65_residual, inline_65_post, inline_65_comb
            block_hidden_states = inline_65_result
            del inline_65_result
            inline_66_hidden_states = block_hidden_states
            inline_66_fn = block_self.hc_ffn_fn
            inline_66_scale = block_self.hc_ffn_scale
            inline_66_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_66_states = inline_66_hidden_states.flatten(-2).float()
            inline_66_rstd = torch.rsqrt(inline_66_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_66_mixes = F.linear(inline_66_states, inline_66_fn.float()) * inline_66_rstd
            inline_67_mixes = inline_66_mixes
            inline_67_scale = inline_66_scale
            inline_67_base = inline_66_base
            inline_67_result = hc_gates_sinkhorn(
                inline_67_mixes, inline_67_scale, inline_67_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_67_mixes, inline_67_scale, inline_67_base
            inline_66_result = inline_67_result
            del inline_67_result
            del (
                inline_66_base,
                inline_66_fn,
                inline_66_hidden_states,
                inline_66_mixes,
                inline_66_rstd,
                inline_66_scale,
                inline_66_states,
            )
            block_ffn_pre, block_post, block_comb = inline_66_result
            del inline_66_result
            inline_68_hidden_states = block_hidden_states
            inline_68_pre_mix = block_attn_pre
            inline_69_result = hc_pre_norm(
                inline_68_hidden_states, inline_68_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_68_hidden_states, inline_68_pre_mix
            block_ffn_input = inline_69_result
            del inline_69_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_70_sublayer_output = block_output
            inline_70_residual = block_hidden_states
            inline_70_post = block_post
            inline_70_comb = block_comb
            inline_70_result = hc_post(inline_70_sublayer_output, inline_70_residual, inline_70_post, inline_70_comb)
            del inline_70_sublayer_output, inline_70_residual, inline_70_post, inline_70_comb
            hidden_states, pre_mix = (inline_70_result, block_ffn_pre)
            del inline_70_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[3]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_71_hidden_states = block_hidden_states
            inline_71_fn = block_self.hc_attn_fn
            inline_71_scale = block_self.hc_attn_scale
            inline_71_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_71_states = inline_71_hidden_states.flatten(-2).float()
            inline_71_rstd = torch.rsqrt(inline_71_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_71_mixes = F.linear(inline_71_states, inline_71_fn.float()) * inline_71_rstd
            inline_72_mixes = inline_71_mixes
            inline_72_scale = inline_71_scale
            inline_72_base = inline_71_base
            inline_72_result = hc_gates_sinkhorn(
                inline_72_mixes, inline_72_scale, inline_72_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_72_mixes, inline_72_scale, inline_72_base
            inline_71_result = inline_72_result
            del inline_72_result
            del (
                inline_71_base,
                inline_71_fn,
                inline_71_hidden_states,
                inline_71_mixes,
                inline_71_rstd,
                inline_71_scale,
                inline_71_states,
            )
            block_attn_pre, block_post, block_comb = inline_71_result
            del inline_71_result
            inline_73_hidden_states = block_hidden_states
            inline_73_pre_mix = block_pre_mix
            inline_74_result = hc_pre_norm(
                inline_73_hidden_states, inline_73_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_73_hidden_states, inline_73_pre_mix
            block_attended = block_self.attn(inline_74_result, block_runtime)
            del inline_74_result
            inline_75_sublayer_output = block_attended
            inline_75_residual = block_hidden_states
            inline_75_post = block_post
            inline_75_comb = block_comb
            inline_75_result = hc_post(inline_75_sublayer_output, inline_75_residual, inline_75_post, inline_75_comb)
            del inline_75_sublayer_output, inline_75_residual, inline_75_post, inline_75_comb
            block_hidden_states = inline_75_result
            del inline_75_result
            inline_76_hidden_states = block_hidden_states
            inline_76_fn = block_self.hc_ffn_fn
            inline_76_scale = block_self.hc_ffn_scale
            inline_76_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_76_states = inline_76_hidden_states.flatten(-2).float()
            inline_76_rstd = torch.rsqrt(inline_76_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_76_mixes = F.linear(inline_76_states, inline_76_fn.float()) * inline_76_rstd
            inline_77_mixes = inline_76_mixes
            inline_77_scale = inline_76_scale
            inline_77_base = inline_76_base
            inline_77_result = hc_gates_sinkhorn(
                inline_77_mixes, inline_77_scale, inline_77_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_77_mixes, inline_77_scale, inline_77_base
            inline_76_result = inline_77_result
            del inline_77_result
            del (
                inline_76_base,
                inline_76_fn,
                inline_76_hidden_states,
                inline_76_mixes,
                inline_76_rstd,
                inline_76_scale,
                inline_76_states,
            )
            block_ffn_pre, block_post, block_comb = inline_76_result
            del inline_76_result
            inline_78_hidden_states = block_hidden_states
            inline_78_pre_mix = block_attn_pre
            inline_79_result = hc_pre_norm(
                inline_78_hidden_states, inline_78_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_78_hidden_states, inline_78_pre_mix
            block_ffn_input = inline_79_result
            del inline_79_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_80_sublayer_output = block_output
            inline_80_residual = block_hidden_states
            inline_80_post = block_post
            inline_80_comb = block_comb
            inline_80_result = hc_post(inline_80_sublayer_output, inline_80_residual, inline_80_post, inline_80_comb)
            del inline_80_sublayer_output, inline_80_residual, inline_80_post, inline_80_comb
            hidden_states, pre_mix = (inline_80_result, block_ffn_pre)
            del inline_80_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[4].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(4)
        hidden_states = self.layers[4].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 4, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[4]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_81_hidden_states = block_hidden_states
            inline_81_fn = block_self.hc_attn_fn
            inline_81_scale = block_self.hc_attn_scale
            inline_81_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_81_states = inline_81_hidden_states.flatten(-2).float()
            inline_81_rstd = torch.rsqrt(inline_81_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_81_mixes = F.linear(inline_81_states, inline_81_fn.float()) * inline_81_rstd
            inline_82_mixes = inline_81_mixes
            inline_82_scale = inline_81_scale
            inline_82_base = inline_81_base
            inline_82_result = hc_gates_sinkhorn(
                inline_82_mixes, inline_82_scale, inline_82_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_82_mixes, inline_82_scale, inline_82_base
            inline_81_result = inline_82_result
            del inline_82_result
            del (
                inline_81_base,
                inline_81_fn,
                inline_81_hidden_states,
                inline_81_mixes,
                inline_81_rstd,
                inline_81_scale,
                inline_81_states,
            )
            block_attn_pre, block_post, block_comb = inline_81_result
            del inline_81_result
            inline_83_hidden_states = block_hidden_states
            inline_83_pre_mix = block_pre_mix
            inline_84_result = hc_pre_norm(
                inline_83_hidden_states, inline_83_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_83_hidden_states, inline_83_pre_mix
            block_attended = block_self.attn(inline_84_result, block_runtime)
            del inline_84_result
            inline_85_sublayer_output = block_attended
            inline_85_residual = block_hidden_states
            inline_85_post = block_post
            inline_85_comb = block_comb
            inline_85_result = hc_post(inline_85_sublayer_output, inline_85_residual, inline_85_post, inline_85_comb)
            del inline_85_sublayer_output, inline_85_residual, inline_85_post, inline_85_comb
            block_hidden_states = inline_85_result
            del inline_85_result
            inline_86_hidden_states = block_hidden_states
            inline_86_fn = block_self.hc_ffn_fn
            inline_86_scale = block_self.hc_ffn_scale
            inline_86_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_86_states = inline_86_hidden_states.flatten(-2).float()
            inline_86_rstd = torch.rsqrt(inline_86_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_86_mixes = F.linear(inline_86_states, inline_86_fn.float()) * inline_86_rstd
            inline_87_mixes = inline_86_mixes
            inline_87_scale = inline_86_scale
            inline_87_base = inline_86_base
            inline_87_result = hc_gates_sinkhorn(
                inline_87_mixes, inline_87_scale, inline_87_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_87_mixes, inline_87_scale, inline_87_base
            inline_86_result = inline_87_result
            del inline_87_result
            del (
                inline_86_base,
                inline_86_fn,
                inline_86_hidden_states,
                inline_86_mixes,
                inline_86_rstd,
                inline_86_scale,
                inline_86_states,
            )
            block_ffn_pre, block_post, block_comb = inline_86_result
            del inline_86_result
            inline_88_hidden_states = block_hidden_states
            inline_88_pre_mix = block_attn_pre
            inline_89_result = hc_pre_norm(
                inline_88_hidden_states, inline_88_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_88_hidden_states, inline_88_pre_mix
            block_ffn_input = inline_89_result
            del inline_89_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_90_sublayer_output = block_output
            inline_90_residual = block_hidden_states
            inline_90_post = block_post
            inline_90_comb = block_comb
            inline_90_result = hc_post(inline_90_sublayer_output, inline_90_residual, inline_90_post, inline_90_comb)
            del inline_90_sublayer_output, inline_90_residual, inline_90_post, inline_90_comb
            hidden_states, pre_mix = (inline_90_result, block_ffn_pre)
            del inline_90_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[4]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_91_hidden_states = block_hidden_states
            inline_91_fn = block_self.hc_attn_fn
            inline_91_scale = block_self.hc_attn_scale
            inline_91_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_91_states = inline_91_hidden_states.flatten(-2).float()
            inline_91_rstd = torch.rsqrt(inline_91_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_91_mixes = F.linear(inline_91_states, inline_91_fn.float()) * inline_91_rstd
            inline_92_mixes = inline_91_mixes
            inline_92_scale = inline_91_scale
            inline_92_base = inline_91_base
            inline_92_result = hc_gates_sinkhorn(
                inline_92_mixes, inline_92_scale, inline_92_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_92_mixes, inline_92_scale, inline_92_base
            inline_91_result = inline_92_result
            del inline_92_result
            del (
                inline_91_base,
                inline_91_fn,
                inline_91_hidden_states,
                inline_91_mixes,
                inline_91_rstd,
                inline_91_scale,
                inline_91_states,
            )
            block_attn_pre, block_post, block_comb = inline_91_result
            del inline_91_result
            inline_93_hidden_states = block_hidden_states
            inline_93_pre_mix = block_pre_mix
            inline_94_result = hc_pre_norm(
                inline_93_hidden_states, inline_93_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_93_hidden_states, inline_93_pre_mix
            block_attended = block_self.attn(inline_94_result, block_runtime)
            del inline_94_result
            inline_95_sublayer_output = block_attended
            inline_95_residual = block_hidden_states
            inline_95_post = block_post
            inline_95_comb = block_comb
            inline_95_result = hc_post(inline_95_sublayer_output, inline_95_residual, inline_95_post, inline_95_comb)
            del inline_95_sublayer_output, inline_95_residual, inline_95_post, inline_95_comb
            block_hidden_states = inline_95_result
            del inline_95_result
            inline_96_hidden_states = block_hidden_states
            inline_96_fn = block_self.hc_ffn_fn
            inline_96_scale = block_self.hc_ffn_scale
            inline_96_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_96_states = inline_96_hidden_states.flatten(-2).float()
            inline_96_rstd = torch.rsqrt(inline_96_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_96_mixes = F.linear(inline_96_states, inline_96_fn.float()) * inline_96_rstd
            inline_97_mixes = inline_96_mixes
            inline_97_scale = inline_96_scale
            inline_97_base = inline_96_base
            inline_97_result = hc_gates_sinkhorn(
                inline_97_mixes, inline_97_scale, inline_97_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_97_mixes, inline_97_scale, inline_97_base
            inline_96_result = inline_97_result
            del inline_97_result
            del (
                inline_96_base,
                inline_96_fn,
                inline_96_hidden_states,
                inline_96_mixes,
                inline_96_rstd,
                inline_96_scale,
                inline_96_states,
            )
            block_ffn_pre, block_post, block_comb = inline_96_result
            del inline_96_result
            inline_98_hidden_states = block_hidden_states
            inline_98_pre_mix = block_attn_pre
            inline_99_result = hc_pre_norm(
                inline_98_hidden_states, inline_98_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_98_hidden_states, inline_98_pre_mix
            block_ffn_input = inline_99_result
            del inline_99_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_100_sublayer_output = block_output
            inline_100_residual = block_hidden_states
            inline_100_post = block_post
            inline_100_comb = block_comb
            inline_100_result = hc_post(
                inline_100_sublayer_output, inline_100_residual, inline_100_post, inline_100_comb
            )
            del inline_100_sublayer_output, inline_100_residual, inline_100_post, inline_100_comb
            hidden_states, pre_mix = (inline_100_result, block_ffn_pre)
            del inline_100_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[5].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(5)
        hidden_states = self.layers[5].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 5, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[5]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_101_hidden_states = block_hidden_states
            inline_101_fn = block_self.hc_attn_fn
            inline_101_scale = block_self.hc_attn_scale
            inline_101_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_101_states = inline_101_hidden_states.flatten(-2).float()
            inline_101_rstd = torch.rsqrt(inline_101_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_101_mixes = F.linear(inline_101_states, inline_101_fn.float()) * inline_101_rstd
            inline_102_mixes = inline_101_mixes
            inline_102_scale = inline_101_scale
            inline_102_base = inline_101_base
            inline_102_result = hc_gates_sinkhorn(
                inline_102_mixes, inline_102_scale, inline_102_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_102_mixes, inline_102_scale, inline_102_base
            inline_101_result = inline_102_result
            del inline_102_result
            del (
                inline_101_base,
                inline_101_fn,
                inline_101_hidden_states,
                inline_101_mixes,
                inline_101_rstd,
                inline_101_scale,
                inline_101_states,
            )
            block_attn_pre, block_post, block_comb = inline_101_result
            del inline_101_result
            inline_103_hidden_states = block_hidden_states
            inline_103_pre_mix = block_pre_mix
            inline_104_result = hc_pre_norm(
                inline_103_hidden_states, inline_103_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_103_hidden_states, inline_103_pre_mix
            block_attended = block_self.attn(inline_104_result, block_runtime)
            del inline_104_result
            inline_105_sublayer_output = block_attended
            inline_105_residual = block_hidden_states
            inline_105_post = block_post
            inline_105_comb = block_comb
            inline_105_result = hc_post(
                inline_105_sublayer_output, inline_105_residual, inline_105_post, inline_105_comb
            )
            del inline_105_sublayer_output, inline_105_residual, inline_105_post, inline_105_comb
            block_hidden_states = inline_105_result
            del inline_105_result
            inline_106_hidden_states = block_hidden_states
            inline_106_fn = block_self.hc_ffn_fn
            inline_106_scale = block_self.hc_ffn_scale
            inline_106_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_106_states = inline_106_hidden_states.flatten(-2).float()
            inline_106_rstd = torch.rsqrt(inline_106_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_106_mixes = F.linear(inline_106_states, inline_106_fn.float()) * inline_106_rstd
            inline_107_mixes = inline_106_mixes
            inline_107_scale = inline_106_scale
            inline_107_base = inline_106_base
            inline_107_result = hc_gates_sinkhorn(
                inline_107_mixes, inline_107_scale, inline_107_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_107_mixes, inline_107_scale, inline_107_base
            inline_106_result = inline_107_result
            del inline_107_result
            del (
                inline_106_base,
                inline_106_fn,
                inline_106_hidden_states,
                inline_106_mixes,
                inline_106_rstd,
                inline_106_scale,
                inline_106_states,
            )
            block_ffn_pre, block_post, block_comb = inline_106_result
            del inline_106_result
            inline_108_hidden_states = block_hidden_states
            inline_108_pre_mix = block_attn_pre
            inline_109_result = hc_pre_norm(
                inline_108_hidden_states, inline_108_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_108_hidden_states, inline_108_pre_mix
            block_ffn_input = inline_109_result
            del inline_109_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_110_sublayer_output = block_output
            inline_110_residual = block_hidden_states
            inline_110_post = block_post
            inline_110_comb = block_comb
            inline_110_result = hc_post(
                inline_110_sublayer_output, inline_110_residual, inline_110_post, inline_110_comb
            )
            del inline_110_sublayer_output, inline_110_residual, inline_110_post, inline_110_comb
            hidden_states, pre_mix = (inline_110_result, block_ffn_pre)
            del inline_110_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[5]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_111_hidden_states = block_hidden_states
            inline_111_fn = block_self.hc_attn_fn
            inline_111_scale = block_self.hc_attn_scale
            inline_111_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_111_states = inline_111_hidden_states.flatten(-2).float()
            inline_111_rstd = torch.rsqrt(inline_111_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_111_mixes = F.linear(inline_111_states, inline_111_fn.float()) * inline_111_rstd
            inline_112_mixes = inline_111_mixes
            inline_112_scale = inline_111_scale
            inline_112_base = inline_111_base
            inline_112_result = hc_gates_sinkhorn(
                inline_112_mixes, inline_112_scale, inline_112_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_112_mixes, inline_112_scale, inline_112_base
            inline_111_result = inline_112_result
            del inline_112_result
            del (
                inline_111_base,
                inline_111_fn,
                inline_111_hidden_states,
                inline_111_mixes,
                inline_111_rstd,
                inline_111_scale,
                inline_111_states,
            )
            block_attn_pre, block_post, block_comb = inline_111_result
            del inline_111_result
            inline_113_hidden_states = block_hidden_states
            inline_113_pre_mix = block_pre_mix
            inline_114_result = hc_pre_norm(
                inline_113_hidden_states, inline_113_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_113_hidden_states, inline_113_pre_mix
            block_attended = block_self.attn(inline_114_result, block_runtime)
            del inline_114_result
            inline_115_sublayer_output = block_attended
            inline_115_residual = block_hidden_states
            inline_115_post = block_post
            inline_115_comb = block_comb
            inline_115_result = hc_post(
                inline_115_sublayer_output, inline_115_residual, inline_115_post, inline_115_comb
            )
            del inline_115_sublayer_output, inline_115_residual, inline_115_post, inline_115_comb
            block_hidden_states = inline_115_result
            del inline_115_result
            inline_116_hidden_states = block_hidden_states
            inline_116_fn = block_self.hc_ffn_fn
            inline_116_scale = block_self.hc_ffn_scale
            inline_116_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_116_states = inline_116_hidden_states.flatten(-2).float()
            inline_116_rstd = torch.rsqrt(inline_116_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_116_mixes = F.linear(inline_116_states, inline_116_fn.float()) * inline_116_rstd
            inline_117_mixes = inline_116_mixes
            inline_117_scale = inline_116_scale
            inline_117_base = inline_116_base
            inline_117_result = hc_gates_sinkhorn(
                inline_117_mixes, inline_117_scale, inline_117_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_117_mixes, inline_117_scale, inline_117_base
            inline_116_result = inline_117_result
            del inline_117_result
            del (
                inline_116_base,
                inline_116_fn,
                inline_116_hidden_states,
                inline_116_mixes,
                inline_116_rstd,
                inline_116_scale,
                inline_116_states,
            )
            block_ffn_pre, block_post, block_comb = inline_116_result
            del inline_116_result
            inline_118_hidden_states = block_hidden_states
            inline_118_pre_mix = block_attn_pre
            inline_119_result = hc_pre_norm(
                inline_118_hidden_states, inline_118_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_118_hidden_states, inline_118_pre_mix
            block_ffn_input = inline_119_result
            del inline_119_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_120_sublayer_output = block_output
            inline_120_residual = block_hidden_states
            inline_120_post = block_post
            inline_120_comb = block_comb
            inline_120_result = hc_post(
                inline_120_sublayer_output, inline_120_residual, inline_120_post, inline_120_comb
            )
            del inline_120_sublayer_output, inline_120_residual, inline_120_post, inline_120_comb
            hidden_states, pre_mix = (inline_120_result, block_ffn_pre)
            del inline_120_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[6].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(6)
        hidden_states = self.layers[6].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 6, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[6]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_121_hidden_states = block_hidden_states
            inline_121_fn = block_self.hc_attn_fn
            inline_121_scale = block_self.hc_attn_scale
            inline_121_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_121_states = inline_121_hidden_states.flatten(-2).float()
            inline_121_rstd = torch.rsqrt(inline_121_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_121_mixes = F.linear(inline_121_states, inline_121_fn.float()) * inline_121_rstd
            inline_122_mixes = inline_121_mixes
            inline_122_scale = inline_121_scale
            inline_122_base = inline_121_base
            inline_122_result = hc_gates_sinkhorn(
                inline_122_mixes, inline_122_scale, inline_122_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_122_mixes, inline_122_scale, inline_122_base
            inline_121_result = inline_122_result
            del inline_122_result
            del (
                inline_121_base,
                inline_121_fn,
                inline_121_hidden_states,
                inline_121_mixes,
                inline_121_rstd,
                inline_121_scale,
                inline_121_states,
            )
            block_attn_pre, block_post, block_comb = inline_121_result
            del inline_121_result
            inline_123_hidden_states = block_hidden_states
            inline_123_pre_mix = block_pre_mix
            inline_124_result = hc_pre_norm(
                inline_123_hidden_states, inline_123_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_123_hidden_states, inline_123_pre_mix
            block_attended = block_self.attn(inline_124_result, block_runtime)
            del inline_124_result
            inline_125_sublayer_output = block_attended
            inline_125_residual = block_hidden_states
            inline_125_post = block_post
            inline_125_comb = block_comb
            inline_125_result = hc_post(
                inline_125_sublayer_output, inline_125_residual, inline_125_post, inline_125_comb
            )
            del inline_125_sublayer_output, inline_125_residual, inline_125_post, inline_125_comb
            block_hidden_states = inline_125_result
            del inline_125_result
            inline_126_hidden_states = block_hidden_states
            inline_126_fn = block_self.hc_ffn_fn
            inline_126_scale = block_self.hc_ffn_scale
            inline_126_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_126_states = inline_126_hidden_states.flatten(-2).float()
            inline_126_rstd = torch.rsqrt(inline_126_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_126_mixes = F.linear(inline_126_states, inline_126_fn.float()) * inline_126_rstd
            inline_127_mixes = inline_126_mixes
            inline_127_scale = inline_126_scale
            inline_127_base = inline_126_base
            inline_127_result = hc_gates_sinkhorn(
                inline_127_mixes, inline_127_scale, inline_127_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_127_mixes, inline_127_scale, inline_127_base
            inline_126_result = inline_127_result
            del inline_127_result
            del (
                inline_126_base,
                inline_126_fn,
                inline_126_hidden_states,
                inline_126_mixes,
                inline_126_rstd,
                inline_126_scale,
                inline_126_states,
            )
            block_ffn_pre, block_post, block_comb = inline_126_result
            del inline_126_result
            inline_128_hidden_states = block_hidden_states
            inline_128_pre_mix = block_attn_pre
            inline_129_result = hc_pre_norm(
                inline_128_hidden_states, inline_128_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_128_hidden_states, inline_128_pre_mix
            block_ffn_input = inline_129_result
            del inline_129_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_130_sublayer_output = block_output
            inline_130_residual = block_hidden_states
            inline_130_post = block_post
            inline_130_comb = block_comb
            inline_130_result = hc_post(
                inline_130_sublayer_output, inline_130_residual, inline_130_post, inline_130_comb
            )
            del inline_130_sublayer_output, inline_130_residual, inline_130_post, inline_130_comb
            hidden_states, pre_mix = (inline_130_result, block_ffn_pre)
            del inline_130_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[6]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_131_hidden_states = block_hidden_states
            inline_131_fn = block_self.hc_attn_fn
            inline_131_scale = block_self.hc_attn_scale
            inline_131_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_131_states = inline_131_hidden_states.flatten(-2).float()
            inline_131_rstd = torch.rsqrt(inline_131_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_131_mixes = F.linear(inline_131_states, inline_131_fn.float()) * inline_131_rstd
            inline_132_mixes = inline_131_mixes
            inline_132_scale = inline_131_scale
            inline_132_base = inline_131_base
            inline_132_result = hc_gates_sinkhorn(
                inline_132_mixes, inline_132_scale, inline_132_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_132_mixes, inline_132_scale, inline_132_base
            inline_131_result = inline_132_result
            del inline_132_result
            del (
                inline_131_base,
                inline_131_fn,
                inline_131_hidden_states,
                inline_131_mixes,
                inline_131_rstd,
                inline_131_scale,
                inline_131_states,
            )
            block_attn_pre, block_post, block_comb = inline_131_result
            del inline_131_result
            inline_133_hidden_states = block_hidden_states
            inline_133_pre_mix = block_pre_mix
            inline_134_result = hc_pre_norm(
                inline_133_hidden_states, inline_133_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_133_hidden_states, inline_133_pre_mix
            block_attended = block_self.attn(inline_134_result, block_runtime)
            del inline_134_result
            inline_135_sublayer_output = block_attended
            inline_135_residual = block_hidden_states
            inline_135_post = block_post
            inline_135_comb = block_comb
            inline_135_result = hc_post(
                inline_135_sublayer_output, inline_135_residual, inline_135_post, inline_135_comb
            )
            del inline_135_sublayer_output, inline_135_residual, inline_135_post, inline_135_comb
            block_hidden_states = inline_135_result
            del inline_135_result
            inline_136_hidden_states = block_hidden_states
            inline_136_fn = block_self.hc_ffn_fn
            inline_136_scale = block_self.hc_ffn_scale
            inline_136_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_136_states = inline_136_hidden_states.flatten(-2).float()
            inline_136_rstd = torch.rsqrt(inline_136_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_136_mixes = F.linear(inline_136_states, inline_136_fn.float()) * inline_136_rstd
            inline_137_mixes = inline_136_mixes
            inline_137_scale = inline_136_scale
            inline_137_base = inline_136_base
            inline_137_result = hc_gates_sinkhorn(
                inline_137_mixes, inline_137_scale, inline_137_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_137_mixes, inline_137_scale, inline_137_base
            inline_136_result = inline_137_result
            del inline_137_result
            del (
                inline_136_base,
                inline_136_fn,
                inline_136_hidden_states,
                inline_136_mixes,
                inline_136_rstd,
                inline_136_scale,
                inline_136_states,
            )
            block_ffn_pre, block_post, block_comb = inline_136_result
            del inline_136_result
            inline_138_hidden_states = block_hidden_states
            inline_138_pre_mix = block_attn_pre
            inline_139_result = hc_pre_norm(
                inline_138_hidden_states, inline_138_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_138_hidden_states, inline_138_pre_mix
            block_ffn_input = inline_139_result
            del inline_139_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_140_sublayer_output = block_output
            inline_140_residual = block_hidden_states
            inline_140_post = block_post
            inline_140_comb = block_comb
            inline_140_result = hc_post(
                inline_140_sublayer_output, inline_140_residual, inline_140_post, inline_140_comb
            )
            del inline_140_sublayer_output, inline_140_residual, inline_140_post, inline_140_comb
            hidden_states, pre_mix = (inline_140_result, block_ffn_pre)
            del inline_140_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[7].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(7)
        hidden_states = self.layers[7].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 7, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[7]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_141_hidden_states = block_hidden_states
            inline_141_fn = block_self.hc_attn_fn
            inline_141_scale = block_self.hc_attn_scale
            inline_141_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_141_states = inline_141_hidden_states.flatten(-2).float()
            inline_141_rstd = torch.rsqrt(inline_141_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_141_mixes = F.linear(inline_141_states, inline_141_fn.float()) * inline_141_rstd
            inline_142_mixes = inline_141_mixes
            inline_142_scale = inline_141_scale
            inline_142_base = inline_141_base
            inline_142_result = hc_gates_sinkhorn(
                inline_142_mixes, inline_142_scale, inline_142_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_142_mixes, inline_142_scale, inline_142_base
            inline_141_result = inline_142_result
            del inline_142_result
            del (
                inline_141_base,
                inline_141_fn,
                inline_141_hidden_states,
                inline_141_mixes,
                inline_141_rstd,
                inline_141_scale,
                inline_141_states,
            )
            block_attn_pre, block_post, block_comb = inline_141_result
            del inline_141_result
            inline_143_hidden_states = block_hidden_states
            inline_143_pre_mix = block_pre_mix
            inline_144_result = hc_pre_norm(
                inline_143_hidden_states, inline_143_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_143_hidden_states, inline_143_pre_mix
            block_attended = block_self.attn(inline_144_result, block_runtime)
            del inline_144_result
            inline_145_sublayer_output = block_attended
            inline_145_residual = block_hidden_states
            inline_145_post = block_post
            inline_145_comb = block_comb
            inline_145_result = hc_post(
                inline_145_sublayer_output, inline_145_residual, inline_145_post, inline_145_comb
            )
            del inline_145_sublayer_output, inline_145_residual, inline_145_post, inline_145_comb
            block_hidden_states = inline_145_result
            del inline_145_result
            inline_146_hidden_states = block_hidden_states
            inline_146_fn = block_self.hc_ffn_fn
            inline_146_scale = block_self.hc_ffn_scale
            inline_146_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_146_states = inline_146_hidden_states.flatten(-2).float()
            inline_146_rstd = torch.rsqrt(inline_146_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_146_mixes = F.linear(inline_146_states, inline_146_fn.float()) * inline_146_rstd
            inline_147_mixes = inline_146_mixes
            inline_147_scale = inline_146_scale
            inline_147_base = inline_146_base
            inline_147_result = hc_gates_sinkhorn(
                inline_147_mixes, inline_147_scale, inline_147_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_147_mixes, inline_147_scale, inline_147_base
            inline_146_result = inline_147_result
            del inline_147_result
            del (
                inline_146_base,
                inline_146_fn,
                inline_146_hidden_states,
                inline_146_mixes,
                inline_146_rstd,
                inline_146_scale,
                inline_146_states,
            )
            block_ffn_pre, block_post, block_comb = inline_146_result
            del inline_146_result
            inline_148_hidden_states = block_hidden_states
            inline_148_pre_mix = block_attn_pre
            inline_149_result = hc_pre_norm(
                inline_148_hidden_states, inline_148_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_148_hidden_states, inline_148_pre_mix
            block_ffn_input = inline_149_result
            del inline_149_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_150_sublayer_output = block_output
            inline_150_residual = block_hidden_states
            inline_150_post = block_post
            inline_150_comb = block_comb
            inline_150_result = hc_post(
                inline_150_sublayer_output, inline_150_residual, inline_150_post, inline_150_comb
            )
            del inline_150_sublayer_output, inline_150_residual, inline_150_post, inline_150_comb
            hidden_states, pre_mix = (inline_150_result, block_ffn_pre)
            del inline_150_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[7]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_151_hidden_states = block_hidden_states
            inline_151_fn = block_self.hc_attn_fn
            inline_151_scale = block_self.hc_attn_scale
            inline_151_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_151_states = inline_151_hidden_states.flatten(-2).float()
            inline_151_rstd = torch.rsqrt(inline_151_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_151_mixes = F.linear(inline_151_states, inline_151_fn.float()) * inline_151_rstd
            inline_152_mixes = inline_151_mixes
            inline_152_scale = inline_151_scale
            inline_152_base = inline_151_base
            inline_152_result = hc_gates_sinkhorn(
                inline_152_mixes, inline_152_scale, inline_152_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_152_mixes, inline_152_scale, inline_152_base
            inline_151_result = inline_152_result
            del inline_152_result
            del (
                inline_151_base,
                inline_151_fn,
                inline_151_hidden_states,
                inline_151_mixes,
                inline_151_rstd,
                inline_151_scale,
                inline_151_states,
            )
            block_attn_pre, block_post, block_comb = inline_151_result
            del inline_151_result
            inline_153_hidden_states = block_hidden_states
            inline_153_pre_mix = block_pre_mix
            inline_154_result = hc_pre_norm(
                inline_153_hidden_states, inline_153_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_153_hidden_states, inline_153_pre_mix
            block_attended = block_self.attn(inline_154_result, block_runtime)
            del inline_154_result
            inline_155_sublayer_output = block_attended
            inline_155_residual = block_hidden_states
            inline_155_post = block_post
            inline_155_comb = block_comb
            inline_155_result = hc_post(
                inline_155_sublayer_output, inline_155_residual, inline_155_post, inline_155_comb
            )
            del inline_155_sublayer_output, inline_155_residual, inline_155_post, inline_155_comb
            block_hidden_states = inline_155_result
            del inline_155_result
            inline_156_hidden_states = block_hidden_states
            inline_156_fn = block_self.hc_ffn_fn
            inline_156_scale = block_self.hc_ffn_scale
            inline_156_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_156_states = inline_156_hidden_states.flatten(-2).float()
            inline_156_rstd = torch.rsqrt(inline_156_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_156_mixes = F.linear(inline_156_states, inline_156_fn.float()) * inline_156_rstd
            inline_157_mixes = inline_156_mixes
            inline_157_scale = inline_156_scale
            inline_157_base = inline_156_base
            inline_157_result = hc_gates_sinkhorn(
                inline_157_mixes, inline_157_scale, inline_157_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_157_mixes, inline_157_scale, inline_157_base
            inline_156_result = inline_157_result
            del inline_157_result
            del (
                inline_156_base,
                inline_156_fn,
                inline_156_hidden_states,
                inline_156_mixes,
                inline_156_rstd,
                inline_156_scale,
                inline_156_states,
            )
            block_ffn_pre, block_post, block_comb = inline_156_result
            del inline_156_result
            inline_158_hidden_states = block_hidden_states
            inline_158_pre_mix = block_attn_pre
            inline_159_result = hc_pre_norm(
                inline_158_hidden_states, inline_158_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_158_hidden_states, inline_158_pre_mix
            block_ffn_input = inline_159_result
            del inline_159_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_160_sublayer_output = block_output
            inline_160_residual = block_hidden_states
            inline_160_post = block_post
            inline_160_comb = block_comb
            inline_160_result = hc_post(
                inline_160_sublayer_output, inline_160_residual, inline_160_post, inline_160_comb
            )
            del inline_160_sublayer_output, inline_160_residual, inline_160_post, inline_160_comb
            hidden_states, pre_mix = (inline_160_result, block_ffn_pre)
            del inline_160_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[8].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(8)
        hidden_states = self.layers[8].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 8, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[8]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_161_hidden_states = block_hidden_states
            inline_161_fn = block_self.hc_attn_fn
            inline_161_scale = block_self.hc_attn_scale
            inline_161_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_161_states = inline_161_hidden_states.flatten(-2).float()
            inline_161_rstd = torch.rsqrt(inline_161_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_161_mixes = F.linear(inline_161_states, inline_161_fn.float()) * inline_161_rstd
            inline_162_mixes = inline_161_mixes
            inline_162_scale = inline_161_scale
            inline_162_base = inline_161_base
            inline_162_result = hc_gates_sinkhorn(
                inline_162_mixes, inline_162_scale, inline_162_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_162_mixes, inline_162_scale, inline_162_base
            inline_161_result = inline_162_result
            del inline_162_result
            del (
                inline_161_base,
                inline_161_fn,
                inline_161_hidden_states,
                inline_161_mixes,
                inline_161_rstd,
                inline_161_scale,
                inline_161_states,
            )
            block_attn_pre, block_post, block_comb = inline_161_result
            del inline_161_result
            inline_163_hidden_states = block_hidden_states
            inline_163_pre_mix = block_pre_mix
            inline_164_result = hc_pre_norm(
                inline_163_hidden_states, inline_163_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_163_hidden_states, inline_163_pre_mix
            block_attended = block_self.attn(inline_164_result, block_runtime)
            del inline_164_result
            inline_165_sublayer_output = block_attended
            inline_165_residual = block_hidden_states
            inline_165_post = block_post
            inline_165_comb = block_comb
            inline_165_result = hc_post(
                inline_165_sublayer_output, inline_165_residual, inline_165_post, inline_165_comb
            )
            del inline_165_sublayer_output, inline_165_residual, inline_165_post, inline_165_comb
            block_hidden_states = inline_165_result
            del inline_165_result
            inline_166_hidden_states = block_hidden_states
            inline_166_fn = block_self.hc_ffn_fn
            inline_166_scale = block_self.hc_ffn_scale
            inline_166_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_166_states = inline_166_hidden_states.flatten(-2).float()
            inline_166_rstd = torch.rsqrt(inline_166_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_166_mixes = F.linear(inline_166_states, inline_166_fn.float()) * inline_166_rstd
            inline_167_mixes = inline_166_mixes
            inline_167_scale = inline_166_scale
            inline_167_base = inline_166_base
            inline_167_result = hc_gates_sinkhorn(
                inline_167_mixes, inline_167_scale, inline_167_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_167_mixes, inline_167_scale, inline_167_base
            inline_166_result = inline_167_result
            del inline_167_result
            del (
                inline_166_base,
                inline_166_fn,
                inline_166_hidden_states,
                inline_166_mixes,
                inline_166_rstd,
                inline_166_scale,
                inline_166_states,
            )
            block_ffn_pre, block_post, block_comb = inline_166_result
            del inline_166_result
            inline_168_hidden_states = block_hidden_states
            inline_168_pre_mix = block_attn_pre
            inline_169_result = hc_pre_norm(
                inline_168_hidden_states, inline_168_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_168_hidden_states, inline_168_pre_mix
            block_ffn_input = inline_169_result
            del inline_169_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_170_sublayer_output = block_output
            inline_170_residual = block_hidden_states
            inline_170_post = block_post
            inline_170_comb = block_comb
            inline_170_result = hc_post(
                inline_170_sublayer_output, inline_170_residual, inline_170_post, inline_170_comb
            )
            del inline_170_sublayer_output, inline_170_residual, inline_170_post, inline_170_comb
            hidden_states, pre_mix = (inline_170_result, block_ffn_pre)
            del inline_170_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[8]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_171_hidden_states = block_hidden_states
            inline_171_fn = block_self.hc_attn_fn
            inline_171_scale = block_self.hc_attn_scale
            inline_171_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_171_states = inline_171_hidden_states.flatten(-2).float()
            inline_171_rstd = torch.rsqrt(inline_171_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_171_mixes = F.linear(inline_171_states, inline_171_fn.float()) * inline_171_rstd
            inline_172_mixes = inline_171_mixes
            inline_172_scale = inline_171_scale
            inline_172_base = inline_171_base
            inline_172_result = hc_gates_sinkhorn(
                inline_172_mixes, inline_172_scale, inline_172_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_172_mixes, inline_172_scale, inline_172_base
            inline_171_result = inline_172_result
            del inline_172_result
            del (
                inline_171_base,
                inline_171_fn,
                inline_171_hidden_states,
                inline_171_mixes,
                inline_171_rstd,
                inline_171_scale,
                inline_171_states,
            )
            block_attn_pre, block_post, block_comb = inline_171_result
            del inline_171_result
            inline_173_hidden_states = block_hidden_states
            inline_173_pre_mix = block_pre_mix
            inline_174_result = hc_pre_norm(
                inline_173_hidden_states, inline_173_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_173_hidden_states, inline_173_pre_mix
            block_attended = block_self.attn(inline_174_result, block_runtime)
            del inline_174_result
            inline_175_sublayer_output = block_attended
            inline_175_residual = block_hidden_states
            inline_175_post = block_post
            inline_175_comb = block_comb
            inline_175_result = hc_post(
                inline_175_sublayer_output, inline_175_residual, inline_175_post, inline_175_comb
            )
            del inline_175_sublayer_output, inline_175_residual, inline_175_post, inline_175_comb
            block_hidden_states = inline_175_result
            del inline_175_result
            inline_176_hidden_states = block_hidden_states
            inline_176_fn = block_self.hc_ffn_fn
            inline_176_scale = block_self.hc_ffn_scale
            inline_176_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_176_states = inline_176_hidden_states.flatten(-2).float()
            inline_176_rstd = torch.rsqrt(inline_176_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_176_mixes = F.linear(inline_176_states, inline_176_fn.float()) * inline_176_rstd
            inline_177_mixes = inline_176_mixes
            inline_177_scale = inline_176_scale
            inline_177_base = inline_176_base
            inline_177_result = hc_gates_sinkhorn(
                inline_177_mixes, inline_177_scale, inline_177_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_177_mixes, inline_177_scale, inline_177_base
            inline_176_result = inline_177_result
            del inline_177_result
            del (
                inline_176_base,
                inline_176_fn,
                inline_176_hidden_states,
                inline_176_mixes,
                inline_176_rstd,
                inline_176_scale,
                inline_176_states,
            )
            block_ffn_pre, block_post, block_comb = inline_176_result
            del inline_176_result
            inline_178_hidden_states = block_hidden_states
            inline_178_pre_mix = block_attn_pre
            inline_179_result = hc_pre_norm(
                inline_178_hidden_states, inline_178_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_178_hidden_states, inline_178_pre_mix
            block_ffn_input = inline_179_result
            del inline_179_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_180_sublayer_output = block_output
            inline_180_residual = block_hidden_states
            inline_180_post = block_post
            inline_180_comb = block_comb
            inline_180_result = hc_post(
                inline_180_sublayer_output, inline_180_residual, inline_180_post, inline_180_comb
            )
            del inline_180_sublayer_output, inline_180_residual, inline_180_post, inline_180_comb
            hidden_states, pre_mix = (inline_180_result, block_ffn_pre)
            del inline_180_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[9].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(9)
        hidden_states = self.layers[9].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 9, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[9]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_181_hidden_states = block_hidden_states
            inline_181_fn = block_self.hc_attn_fn
            inline_181_scale = block_self.hc_attn_scale
            inline_181_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_181_states = inline_181_hidden_states.flatten(-2).float()
            inline_181_rstd = torch.rsqrt(inline_181_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_181_mixes = F.linear(inline_181_states, inline_181_fn.float()) * inline_181_rstd
            inline_182_mixes = inline_181_mixes
            inline_182_scale = inline_181_scale
            inline_182_base = inline_181_base
            inline_182_result = hc_gates_sinkhorn(
                inline_182_mixes, inline_182_scale, inline_182_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_182_mixes, inline_182_scale, inline_182_base
            inline_181_result = inline_182_result
            del inline_182_result
            del (
                inline_181_base,
                inline_181_fn,
                inline_181_hidden_states,
                inline_181_mixes,
                inline_181_rstd,
                inline_181_scale,
                inline_181_states,
            )
            block_attn_pre, block_post, block_comb = inline_181_result
            del inline_181_result
            inline_183_hidden_states = block_hidden_states
            inline_183_pre_mix = block_pre_mix
            inline_184_result = hc_pre_norm(
                inline_183_hidden_states, inline_183_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_183_hidden_states, inline_183_pre_mix
            block_attended = block_self.attn(inline_184_result, block_runtime)
            del inline_184_result
            inline_185_sublayer_output = block_attended
            inline_185_residual = block_hidden_states
            inline_185_post = block_post
            inline_185_comb = block_comb
            inline_185_result = hc_post(
                inline_185_sublayer_output, inline_185_residual, inline_185_post, inline_185_comb
            )
            del inline_185_sublayer_output, inline_185_residual, inline_185_post, inline_185_comb
            block_hidden_states = inline_185_result
            del inline_185_result
            inline_186_hidden_states = block_hidden_states
            inline_186_fn = block_self.hc_ffn_fn
            inline_186_scale = block_self.hc_ffn_scale
            inline_186_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_186_states = inline_186_hidden_states.flatten(-2).float()
            inline_186_rstd = torch.rsqrt(inline_186_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_186_mixes = F.linear(inline_186_states, inline_186_fn.float()) * inline_186_rstd
            inline_187_mixes = inline_186_mixes
            inline_187_scale = inline_186_scale
            inline_187_base = inline_186_base
            inline_187_result = hc_gates_sinkhorn(
                inline_187_mixes, inline_187_scale, inline_187_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_187_mixes, inline_187_scale, inline_187_base
            inline_186_result = inline_187_result
            del inline_187_result
            del (
                inline_186_base,
                inline_186_fn,
                inline_186_hidden_states,
                inline_186_mixes,
                inline_186_rstd,
                inline_186_scale,
                inline_186_states,
            )
            block_ffn_pre, block_post, block_comb = inline_186_result
            del inline_186_result
            inline_188_hidden_states = block_hidden_states
            inline_188_pre_mix = block_attn_pre
            inline_189_result = hc_pre_norm(
                inline_188_hidden_states, inline_188_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_188_hidden_states, inline_188_pre_mix
            block_ffn_input = inline_189_result
            del inline_189_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_190_sublayer_output = block_output
            inline_190_residual = block_hidden_states
            inline_190_post = block_post
            inline_190_comb = block_comb
            inline_190_result = hc_post(
                inline_190_sublayer_output, inline_190_residual, inline_190_post, inline_190_comb
            )
            del inline_190_sublayer_output, inline_190_residual, inline_190_post, inline_190_comb
            hidden_states, pre_mix = (inline_190_result, block_ffn_pre)
            del inline_190_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[9]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_191_hidden_states = block_hidden_states
            inline_191_fn = block_self.hc_attn_fn
            inline_191_scale = block_self.hc_attn_scale
            inline_191_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_191_states = inline_191_hidden_states.flatten(-2).float()
            inline_191_rstd = torch.rsqrt(inline_191_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_191_mixes = F.linear(inline_191_states, inline_191_fn.float()) * inline_191_rstd
            inline_192_mixes = inline_191_mixes
            inline_192_scale = inline_191_scale
            inline_192_base = inline_191_base
            inline_192_result = hc_gates_sinkhorn(
                inline_192_mixes, inline_192_scale, inline_192_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_192_mixes, inline_192_scale, inline_192_base
            inline_191_result = inline_192_result
            del inline_192_result
            del (
                inline_191_base,
                inline_191_fn,
                inline_191_hidden_states,
                inline_191_mixes,
                inline_191_rstd,
                inline_191_scale,
                inline_191_states,
            )
            block_attn_pre, block_post, block_comb = inline_191_result
            del inline_191_result
            inline_193_hidden_states = block_hidden_states
            inline_193_pre_mix = block_pre_mix
            inline_194_result = hc_pre_norm(
                inline_193_hidden_states, inline_193_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_193_hidden_states, inline_193_pre_mix
            block_attended = block_self.attn(inline_194_result, block_runtime)
            del inline_194_result
            inline_195_sublayer_output = block_attended
            inline_195_residual = block_hidden_states
            inline_195_post = block_post
            inline_195_comb = block_comb
            inline_195_result = hc_post(
                inline_195_sublayer_output, inline_195_residual, inline_195_post, inline_195_comb
            )
            del inline_195_sublayer_output, inline_195_residual, inline_195_post, inline_195_comb
            block_hidden_states = inline_195_result
            del inline_195_result
            inline_196_hidden_states = block_hidden_states
            inline_196_fn = block_self.hc_ffn_fn
            inline_196_scale = block_self.hc_ffn_scale
            inline_196_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_196_states = inline_196_hidden_states.flatten(-2).float()
            inline_196_rstd = torch.rsqrt(inline_196_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_196_mixes = F.linear(inline_196_states, inline_196_fn.float()) * inline_196_rstd
            inline_197_mixes = inline_196_mixes
            inline_197_scale = inline_196_scale
            inline_197_base = inline_196_base
            inline_197_result = hc_gates_sinkhorn(
                inline_197_mixes, inline_197_scale, inline_197_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_197_mixes, inline_197_scale, inline_197_base
            inline_196_result = inline_197_result
            del inline_197_result
            del (
                inline_196_base,
                inline_196_fn,
                inline_196_hidden_states,
                inline_196_mixes,
                inline_196_rstd,
                inline_196_scale,
                inline_196_states,
            )
            block_ffn_pre, block_post, block_comb = inline_196_result
            del inline_196_result
            inline_198_hidden_states = block_hidden_states
            inline_198_pre_mix = block_attn_pre
            inline_199_result = hc_pre_norm(
                inline_198_hidden_states, inline_198_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_198_hidden_states, inline_198_pre_mix
            block_ffn_input = inline_199_result
            del inline_199_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_200_sublayer_output = block_output
            inline_200_residual = block_hidden_states
            inline_200_post = block_post
            inline_200_comb = block_comb
            inline_200_result = hc_post(
                inline_200_sublayer_output, inline_200_residual, inline_200_post, inline_200_comb
            )
            del inline_200_sublayer_output, inline_200_residual, inline_200_post, inline_200_comb
            hidden_states, pre_mix = (inline_200_result, block_ffn_pre)
            del inline_200_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[10].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(10)
        hidden_states = self.layers[10].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 10, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[10]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_201_hidden_states = block_hidden_states
            inline_201_fn = block_self.hc_attn_fn
            inline_201_scale = block_self.hc_attn_scale
            inline_201_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_201_states = inline_201_hidden_states.flatten(-2).float()
            inline_201_rstd = torch.rsqrt(inline_201_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_201_mixes = F.linear(inline_201_states, inline_201_fn.float()) * inline_201_rstd
            inline_202_mixes = inline_201_mixes
            inline_202_scale = inline_201_scale
            inline_202_base = inline_201_base
            inline_202_result = hc_gates_sinkhorn(
                inline_202_mixes, inline_202_scale, inline_202_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_202_mixes, inline_202_scale, inline_202_base
            inline_201_result = inline_202_result
            del inline_202_result
            del (
                inline_201_base,
                inline_201_fn,
                inline_201_hidden_states,
                inline_201_mixes,
                inline_201_rstd,
                inline_201_scale,
                inline_201_states,
            )
            block_attn_pre, block_post, block_comb = inline_201_result
            del inline_201_result
            inline_203_hidden_states = block_hidden_states
            inline_203_pre_mix = block_pre_mix
            inline_204_result = hc_pre_norm(
                inline_203_hidden_states, inline_203_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_203_hidden_states, inline_203_pre_mix
            block_attended = block_self.attn(inline_204_result, block_runtime)
            del inline_204_result
            inline_205_sublayer_output = block_attended
            inline_205_residual = block_hidden_states
            inline_205_post = block_post
            inline_205_comb = block_comb
            inline_205_result = hc_post(
                inline_205_sublayer_output, inline_205_residual, inline_205_post, inline_205_comb
            )
            del inline_205_sublayer_output, inline_205_residual, inline_205_post, inline_205_comb
            block_hidden_states = inline_205_result
            del inline_205_result
            inline_206_hidden_states = block_hidden_states
            inline_206_fn = block_self.hc_ffn_fn
            inline_206_scale = block_self.hc_ffn_scale
            inline_206_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_206_states = inline_206_hidden_states.flatten(-2).float()
            inline_206_rstd = torch.rsqrt(inline_206_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_206_mixes = F.linear(inline_206_states, inline_206_fn.float()) * inline_206_rstd
            inline_207_mixes = inline_206_mixes
            inline_207_scale = inline_206_scale
            inline_207_base = inline_206_base
            inline_207_result = hc_gates_sinkhorn(
                inline_207_mixes, inline_207_scale, inline_207_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_207_mixes, inline_207_scale, inline_207_base
            inline_206_result = inline_207_result
            del inline_207_result
            del (
                inline_206_base,
                inline_206_fn,
                inline_206_hidden_states,
                inline_206_mixes,
                inline_206_rstd,
                inline_206_scale,
                inline_206_states,
            )
            block_ffn_pre, block_post, block_comb = inline_206_result
            del inline_206_result
            inline_208_hidden_states = block_hidden_states
            inline_208_pre_mix = block_attn_pre
            inline_209_result = hc_pre_norm(
                inline_208_hidden_states, inline_208_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_208_hidden_states, inline_208_pre_mix
            block_ffn_input = inline_209_result
            del inline_209_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_210_sublayer_output = block_output
            inline_210_residual = block_hidden_states
            inline_210_post = block_post
            inline_210_comb = block_comb
            inline_210_result = hc_post(
                inline_210_sublayer_output, inline_210_residual, inline_210_post, inline_210_comb
            )
            del inline_210_sublayer_output, inline_210_residual, inline_210_post, inline_210_comb
            hidden_states, pre_mix = (inline_210_result, block_ffn_pre)
            del inline_210_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[10]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_211_hidden_states = block_hidden_states
            inline_211_fn = block_self.hc_attn_fn
            inline_211_scale = block_self.hc_attn_scale
            inline_211_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_211_states = inline_211_hidden_states.flatten(-2).float()
            inline_211_rstd = torch.rsqrt(inline_211_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_211_mixes = F.linear(inline_211_states, inline_211_fn.float()) * inline_211_rstd
            inline_212_mixes = inline_211_mixes
            inline_212_scale = inline_211_scale
            inline_212_base = inline_211_base
            inline_212_result = hc_gates_sinkhorn(
                inline_212_mixes, inline_212_scale, inline_212_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_212_mixes, inline_212_scale, inline_212_base
            inline_211_result = inline_212_result
            del inline_212_result
            del (
                inline_211_base,
                inline_211_fn,
                inline_211_hidden_states,
                inline_211_mixes,
                inline_211_rstd,
                inline_211_scale,
                inline_211_states,
            )
            block_attn_pre, block_post, block_comb = inline_211_result
            del inline_211_result
            inline_213_hidden_states = block_hidden_states
            inline_213_pre_mix = block_pre_mix
            inline_214_result = hc_pre_norm(
                inline_213_hidden_states, inline_213_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_213_hidden_states, inline_213_pre_mix
            block_attended = block_self.attn(inline_214_result, block_runtime)
            del inline_214_result
            inline_215_sublayer_output = block_attended
            inline_215_residual = block_hidden_states
            inline_215_post = block_post
            inline_215_comb = block_comb
            inline_215_result = hc_post(
                inline_215_sublayer_output, inline_215_residual, inline_215_post, inline_215_comb
            )
            del inline_215_sublayer_output, inline_215_residual, inline_215_post, inline_215_comb
            block_hidden_states = inline_215_result
            del inline_215_result
            inline_216_hidden_states = block_hidden_states
            inline_216_fn = block_self.hc_ffn_fn
            inline_216_scale = block_self.hc_ffn_scale
            inline_216_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_216_states = inline_216_hidden_states.flatten(-2).float()
            inline_216_rstd = torch.rsqrt(inline_216_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_216_mixes = F.linear(inline_216_states, inline_216_fn.float()) * inline_216_rstd
            inline_217_mixes = inline_216_mixes
            inline_217_scale = inline_216_scale
            inline_217_base = inline_216_base
            inline_217_result = hc_gates_sinkhorn(
                inline_217_mixes, inline_217_scale, inline_217_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_217_mixes, inline_217_scale, inline_217_base
            inline_216_result = inline_217_result
            del inline_217_result
            del (
                inline_216_base,
                inline_216_fn,
                inline_216_hidden_states,
                inline_216_mixes,
                inline_216_rstd,
                inline_216_scale,
                inline_216_states,
            )
            block_ffn_pre, block_post, block_comb = inline_216_result
            del inline_216_result
            inline_218_hidden_states = block_hidden_states
            inline_218_pre_mix = block_attn_pre
            inline_219_result = hc_pre_norm(
                inline_218_hidden_states, inline_218_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_218_hidden_states, inline_218_pre_mix
            block_ffn_input = inline_219_result
            del inline_219_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_220_sublayer_output = block_output
            inline_220_residual = block_hidden_states
            inline_220_post = block_post
            inline_220_comb = block_comb
            inline_220_result = hc_post(
                inline_220_sublayer_output, inline_220_residual, inline_220_post, inline_220_comb
            )
            del inline_220_sublayer_output, inline_220_residual, inline_220_post, inline_220_comb
            hidden_states, pre_mix = (inline_220_result, block_ffn_pre)
            del inline_220_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[11].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(11)
        hidden_states = self.layers[11].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 11, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[11]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_221_hidden_states = block_hidden_states
            inline_221_fn = block_self.hc_attn_fn
            inline_221_scale = block_self.hc_attn_scale
            inline_221_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_221_states = inline_221_hidden_states.flatten(-2).float()
            inline_221_rstd = torch.rsqrt(inline_221_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_221_mixes = F.linear(inline_221_states, inline_221_fn.float()) * inline_221_rstd
            inline_222_mixes = inline_221_mixes
            inline_222_scale = inline_221_scale
            inline_222_base = inline_221_base
            inline_222_result = hc_gates_sinkhorn(
                inline_222_mixes, inline_222_scale, inline_222_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_222_mixes, inline_222_scale, inline_222_base
            inline_221_result = inline_222_result
            del inline_222_result
            del (
                inline_221_base,
                inline_221_fn,
                inline_221_hidden_states,
                inline_221_mixes,
                inline_221_rstd,
                inline_221_scale,
                inline_221_states,
            )
            block_attn_pre, block_post, block_comb = inline_221_result
            del inline_221_result
            inline_223_hidden_states = block_hidden_states
            inline_223_pre_mix = block_pre_mix
            inline_224_result = hc_pre_norm(
                inline_223_hidden_states, inline_223_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_223_hidden_states, inline_223_pre_mix
            block_attended = block_self.attn(inline_224_result, block_runtime)
            del inline_224_result
            inline_225_sublayer_output = block_attended
            inline_225_residual = block_hidden_states
            inline_225_post = block_post
            inline_225_comb = block_comb
            inline_225_result = hc_post(
                inline_225_sublayer_output, inline_225_residual, inline_225_post, inline_225_comb
            )
            del inline_225_sublayer_output, inline_225_residual, inline_225_post, inline_225_comb
            block_hidden_states = inline_225_result
            del inline_225_result
            inline_226_hidden_states = block_hidden_states
            inline_226_fn = block_self.hc_ffn_fn
            inline_226_scale = block_self.hc_ffn_scale
            inline_226_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_226_states = inline_226_hidden_states.flatten(-2).float()
            inline_226_rstd = torch.rsqrt(inline_226_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_226_mixes = F.linear(inline_226_states, inline_226_fn.float()) * inline_226_rstd
            inline_227_mixes = inline_226_mixes
            inline_227_scale = inline_226_scale
            inline_227_base = inline_226_base
            inline_227_result = hc_gates_sinkhorn(
                inline_227_mixes, inline_227_scale, inline_227_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_227_mixes, inline_227_scale, inline_227_base
            inline_226_result = inline_227_result
            del inline_227_result
            del (
                inline_226_base,
                inline_226_fn,
                inline_226_hidden_states,
                inline_226_mixes,
                inline_226_rstd,
                inline_226_scale,
                inline_226_states,
            )
            block_ffn_pre, block_post, block_comb = inline_226_result
            del inline_226_result
            inline_228_hidden_states = block_hidden_states
            inline_228_pre_mix = block_attn_pre
            inline_229_result = hc_pre_norm(
                inline_228_hidden_states, inline_228_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_228_hidden_states, inline_228_pre_mix
            block_ffn_input = inline_229_result
            del inline_229_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_230_sublayer_output = block_output
            inline_230_residual = block_hidden_states
            inline_230_post = block_post
            inline_230_comb = block_comb
            inline_230_result = hc_post(
                inline_230_sublayer_output, inline_230_residual, inline_230_post, inline_230_comb
            )
            del inline_230_sublayer_output, inline_230_residual, inline_230_post, inline_230_comb
            hidden_states, pre_mix = (inline_230_result, block_ffn_pre)
            del inline_230_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[11]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_231_hidden_states = block_hidden_states
            inline_231_fn = block_self.hc_attn_fn
            inline_231_scale = block_self.hc_attn_scale
            inline_231_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_231_states = inline_231_hidden_states.flatten(-2).float()
            inline_231_rstd = torch.rsqrt(inline_231_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_231_mixes = F.linear(inline_231_states, inline_231_fn.float()) * inline_231_rstd
            inline_232_mixes = inline_231_mixes
            inline_232_scale = inline_231_scale
            inline_232_base = inline_231_base
            inline_232_result = hc_gates_sinkhorn(
                inline_232_mixes, inline_232_scale, inline_232_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_232_mixes, inline_232_scale, inline_232_base
            inline_231_result = inline_232_result
            del inline_232_result
            del (
                inline_231_base,
                inline_231_fn,
                inline_231_hidden_states,
                inline_231_mixes,
                inline_231_rstd,
                inline_231_scale,
                inline_231_states,
            )
            block_attn_pre, block_post, block_comb = inline_231_result
            del inline_231_result
            inline_233_hidden_states = block_hidden_states
            inline_233_pre_mix = block_pre_mix
            inline_234_result = hc_pre_norm(
                inline_233_hidden_states, inline_233_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_233_hidden_states, inline_233_pre_mix
            block_attended = block_self.attn(inline_234_result, block_runtime)
            del inline_234_result
            inline_235_sublayer_output = block_attended
            inline_235_residual = block_hidden_states
            inline_235_post = block_post
            inline_235_comb = block_comb
            inline_235_result = hc_post(
                inline_235_sublayer_output, inline_235_residual, inline_235_post, inline_235_comb
            )
            del inline_235_sublayer_output, inline_235_residual, inline_235_post, inline_235_comb
            block_hidden_states = inline_235_result
            del inline_235_result
            inline_236_hidden_states = block_hidden_states
            inline_236_fn = block_self.hc_ffn_fn
            inline_236_scale = block_self.hc_ffn_scale
            inline_236_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_236_states = inline_236_hidden_states.flatten(-2).float()
            inline_236_rstd = torch.rsqrt(inline_236_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_236_mixes = F.linear(inline_236_states, inline_236_fn.float()) * inline_236_rstd
            inline_237_mixes = inline_236_mixes
            inline_237_scale = inline_236_scale
            inline_237_base = inline_236_base
            inline_237_result = hc_gates_sinkhorn(
                inline_237_mixes, inline_237_scale, inline_237_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_237_mixes, inline_237_scale, inline_237_base
            inline_236_result = inline_237_result
            del inline_237_result
            del (
                inline_236_base,
                inline_236_fn,
                inline_236_hidden_states,
                inline_236_mixes,
                inline_236_rstd,
                inline_236_scale,
                inline_236_states,
            )
            block_ffn_pre, block_post, block_comb = inline_236_result
            del inline_236_result
            inline_238_hidden_states = block_hidden_states
            inline_238_pre_mix = block_attn_pre
            inline_239_result = hc_pre_norm(
                inline_238_hidden_states, inline_238_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_238_hidden_states, inline_238_pre_mix
            block_ffn_input = inline_239_result
            del inline_239_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_240_sublayer_output = block_output
            inline_240_residual = block_hidden_states
            inline_240_post = block_post
            inline_240_comb = block_comb
            inline_240_result = hc_post(
                inline_240_sublayer_output, inline_240_residual, inline_240_post, inline_240_comb
            )
            del inline_240_sublayer_output, inline_240_residual, inline_240_post, inline_240_comb
            hidden_states, pre_mix = (inline_240_result, block_ffn_pre)
            del inline_240_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[12].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(12)
        hidden_states = self.layers[12].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 12, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[12]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_241_hidden_states = block_hidden_states
            inline_241_fn = block_self.hc_attn_fn
            inline_241_scale = block_self.hc_attn_scale
            inline_241_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_241_states = inline_241_hidden_states.flatten(-2).float()
            inline_241_rstd = torch.rsqrt(inline_241_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_241_mixes = F.linear(inline_241_states, inline_241_fn.float()) * inline_241_rstd
            inline_242_mixes = inline_241_mixes
            inline_242_scale = inline_241_scale
            inline_242_base = inline_241_base
            inline_242_result = hc_gates_sinkhorn(
                inline_242_mixes, inline_242_scale, inline_242_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_242_mixes, inline_242_scale, inline_242_base
            inline_241_result = inline_242_result
            del inline_242_result
            del (
                inline_241_base,
                inline_241_fn,
                inline_241_hidden_states,
                inline_241_mixes,
                inline_241_rstd,
                inline_241_scale,
                inline_241_states,
            )
            block_attn_pre, block_post, block_comb = inline_241_result
            del inline_241_result
            inline_243_hidden_states = block_hidden_states
            inline_243_pre_mix = block_pre_mix
            inline_244_result = hc_pre_norm(
                inline_243_hidden_states, inline_243_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_243_hidden_states, inline_243_pre_mix
            block_attended = block_self.attn(inline_244_result, block_runtime)
            del inline_244_result
            inline_245_sublayer_output = block_attended
            inline_245_residual = block_hidden_states
            inline_245_post = block_post
            inline_245_comb = block_comb
            inline_245_result = hc_post(
                inline_245_sublayer_output, inline_245_residual, inline_245_post, inline_245_comb
            )
            del inline_245_sublayer_output, inline_245_residual, inline_245_post, inline_245_comb
            block_hidden_states = inline_245_result
            del inline_245_result
            inline_246_hidden_states = block_hidden_states
            inline_246_fn = block_self.hc_ffn_fn
            inline_246_scale = block_self.hc_ffn_scale
            inline_246_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_246_states = inline_246_hidden_states.flatten(-2).float()
            inline_246_rstd = torch.rsqrt(inline_246_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_246_mixes = F.linear(inline_246_states, inline_246_fn.float()) * inline_246_rstd
            inline_247_mixes = inline_246_mixes
            inline_247_scale = inline_246_scale
            inline_247_base = inline_246_base
            inline_247_result = hc_gates_sinkhorn(
                inline_247_mixes, inline_247_scale, inline_247_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_247_mixes, inline_247_scale, inline_247_base
            inline_246_result = inline_247_result
            del inline_247_result
            del (
                inline_246_base,
                inline_246_fn,
                inline_246_hidden_states,
                inline_246_mixes,
                inline_246_rstd,
                inline_246_scale,
                inline_246_states,
            )
            block_ffn_pre, block_post, block_comb = inline_246_result
            del inline_246_result
            inline_248_hidden_states = block_hidden_states
            inline_248_pre_mix = block_attn_pre
            inline_249_result = hc_pre_norm(
                inline_248_hidden_states, inline_248_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_248_hidden_states, inline_248_pre_mix
            block_ffn_input = inline_249_result
            del inline_249_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_250_sublayer_output = block_output
            inline_250_residual = block_hidden_states
            inline_250_post = block_post
            inline_250_comb = block_comb
            inline_250_result = hc_post(
                inline_250_sublayer_output, inline_250_residual, inline_250_post, inline_250_comb
            )
            del inline_250_sublayer_output, inline_250_residual, inline_250_post, inline_250_comb
            hidden_states, pre_mix = (inline_250_result, block_ffn_pre)
            del inline_250_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[12]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_251_hidden_states = block_hidden_states
            inline_251_fn = block_self.hc_attn_fn
            inline_251_scale = block_self.hc_attn_scale
            inline_251_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_251_states = inline_251_hidden_states.flatten(-2).float()
            inline_251_rstd = torch.rsqrt(inline_251_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_251_mixes = F.linear(inline_251_states, inline_251_fn.float()) * inline_251_rstd
            inline_252_mixes = inline_251_mixes
            inline_252_scale = inline_251_scale
            inline_252_base = inline_251_base
            inline_252_result = hc_gates_sinkhorn(
                inline_252_mixes, inline_252_scale, inline_252_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_252_mixes, inline_252_scale, inline_252_base
            inline_251_result = inline_252_result
            del inline_252_result
            del (
                inline_251_base,
                inline_251_fn,
                inline_251_hidden_states,
                inline_251_mixes,
                inline_251_rstd,
                inline_251_scale,
                inline_251_states,
            )
            block_attn_pre, block_post, block_comb = inline_251_result
            del inline_251_result
            inline_253_hidden_states = block_hidden_states
            inline_253_pre_mix = block_pre_mix
            inline_254_result = hc_pre_norm(
                inline_253_hidden_states, inline_253_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_253_hidden_states, inline_253_pre_mix
            block_attended = block_self.attn(inline_254_result, block_runtime)
            del inline_254_result
            inline_255_sublayer_output = block_attended
            inline_255_residual = block_hidden_states
            inline_255_post = block_post
            inline_255_comb = block_comb
            inline_255_result = hc_post(
                inline_255_sublayer_output, inline_255_residual, inline_255_post, inline_255_comb
            )
            del inline_255_sublayer_output, inline_255_residual, inline_255_post, inline_255_comb
            block_hidden_states = inline_255_result
            del inline_255_result
            inline_256_hidden_states = block_hidden_states
            inline_256_fn = block_self.hc_ffn_fn
            inline_256_scale = block_self.hc_ffn_scale
            inline_256_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_256_states = inline_256_hidden_states.flatten(-2).float()
            inline_256_rstd = torch.rsqrt(inline_256_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_256_mixes = F.linear(inline_256_states, inline_256_fn.float()) * inline_256_rstd
            inline_257_mixes = inline_256_mixes
            inline_257_scale = inline_256_scale
            inline_257_base = inline_256_base
            inline_257_result = hc_gates_sinkhorn(
                inline_257_mixes, inline_257_scale, inline_257_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_257_mixes, inline_257_scale, inline_257_base
            inline_256_result = inline_257_result
            del inline_257_result
            del (
                inline_256_base,
                inline_256_fn,
                inline_256_hidden_states,
                inline_256_mixes,
                inline_256_rstd,
                inline_256_scale,
                inline_256_states,
            )
            block_ffn_pre, block_post, block_comb = inline_256_result
            del inline_256_result
            inline_258_hidden_states = block_hidden_states
            inline_258_pre_mix = block_attn_pre
            inline_259_result = hc_pre_norm(
                inline_258_hidden_states, inline_258_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_258_hidden_states, inline_258_pre_mix
            block_ffn_input = inline_259_result
            del inline_259_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_260_sublayer_output = block_output
            inline_260_residual = block_hidden_states
            inline_260_post = block_post
            inline_260_comb = block_comb
            inline_260_result = hc_post(
                inline_260_sublayer_output, inline_260_residual, inline_260_post, inline_260_comb
            )
            del inline_260_sublayer_output, inline_260_residual, inline_260_post, inline_260_comb
            hidden_states, pre_mix = (inline_260_result, block_ffn_pre)
            del inline_260_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[13].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(13)
        hidden_states = self.layers[13].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 13, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[13]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_261_hidden_states = block_hidden_states
            inline_261_fn = block_self.hc_attn_fn
            inline_261_scale = block_self.hc_attn_scale
            inline_261_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_261_states = inline_261_hidden_states.flatten(-2).float()
            inline_261_rstd = torch.rsqrt(inline_261_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_261_mixes = F.linear(inline_261_states, inline_261_fn.float()) * inline_261_rstd
            inline_262_mixes = inline_261_mixes
            inline_262_scale = inline_261_scale
            inline_262_base = inline_261_base
            inline_262_result = hc_gates_sinkhorn(
                inline_262_mixes, inline_262_scale, inline_262_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_262_mixes, inline_262_scale, inline_262_base
            inline_261_result = inline_262_result
            del inline_262_result
            del (
                inline_261_base,
                inline_261_fn,
                inline_261_hidden_states,
                inline_261_mixes,
                inline_261_rstd,
                inline_261_scale,
                inline_261_states,
            )
            block_attn_pre, block_post, block_comb = inline_261_result
            del inline_261_result
            inline_263_hidden_states = block_hidden_states
            inline_263_pre_mix = block_pre_mix
            inline_264_result = hc_pre_norm(
                inline_263_hidden_states, inline_263_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_263_hidden_states, inline_263_pre_mix
            block_attended = block_self.attn(inline_264_result, block_runtime)
            del inline_264_result
            inline_265_sublayer_output = block_attended
            inline_265_residual = block_hidden_states
            inline_265_post = block_post
            inline_265_comb = block_comb
            inline_265_result = hc_post(
                inline_265_sublayer_output, inline_265_residual, inline_265_post, inline_265_comb
            )
            del inline_265_sublayer_output, inline_265_residual, inline_265_post, inline_265_comb
            block_hidden_states = inline_265_result
            del inline_265_result
            inline_266_hidden_states = block_hidden_states
            inline_266_fn = block_self.hc_ffn_fn
            inline_266_scale = block_self.hc_ffn_scale
            inline_266_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_266_states = inline_266_hidden_states.flatten(-2).float()
            inline_266_rstd = torch.rsqrt(inline_266_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_266_mixes = F.linear(inline_266_states, inline_266_fn.float()) * inline_266_rstd
            inline_267_mixes = inline_266_mixes
            inline_267_scale = inline_266_scale
            inline_267_base = inline_266_base
            inline_267_result = hc_gates_sinkhorn(
                inline_267_mixes, inline_267_scale, inline_267_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_267_mixes, inline_267_scale, inline_267_base
            inline_266_result = inline_267_result
            del inline_267_result
            del (
                inline_266_base,
                inline_266_fn,
                inline_266_hidden_states,
                inline_266_mixes,
                inline_266_rstd,
                inline_266_scale,
                inline_266_states,
            )
            block_ffn_pre, block_post, block_comb = inline_266_result
            del inline_266_result
            inline_268_hidden_states = block_hidden_states
            inline_268_pre_mix = block_attn_pre
            inline_269_result = hc_pre_norm(
                inline_268_hidden_states, inline_268_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_268_hidden_states, inline_268_pre_mix
            block_ffn_input = inline_269_result
            del inline_269_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_270_sublayer_output = block_output
            inline_270_residual = block_hidden_states
            inline_270_post = block_post
            inline_270_comb = block_comb
            inline_270_result = hc_post(
                inline_270_sublayer_output, inline_270_residual, inline_270_post, inline_270_comb
            )
            del inline_270_sublayer_output, inline_270_residual, inline_270_post, inline_270_comb
            hidden_states, pre_mix = (inline_270_result, block_ffn_pre)
            del inline_270_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[13]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_271_hidden_states = block_hidden_states
            inline_271_fn = block_self.hc_attn_fn
            inline_271_scale = block_self.hc_attn_scale
            inline_271_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_271_states = inline_271_hidden_states.flatten(-2).float()
            inline_271_rstd = torch.rsqrt(inline_271_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_271_mixes = F.linear(inline_271_states, inline_271_fn.float()) * inline_271_rstd
            inline_272_mixes = inline_271_mixes
            inline_272_scale = inline_271_scale
            inline_272_base = inline_271_base
            inline_272_result = hc_gates_sinkhorn(
                inline_272_mixes, inline_272_scale, inline_272_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_272_mixes, inline_272_scale, inline_272_base
            inline_271_result = inline_272_result
            del inline_272_result
            del (
                inline_271_base,
                inline_271_fn,
                inline_271_hidden_states,
                inline_271_mixes,
                inline_271_rstd,
                inline_271_scale,
                inline_271_states,
            )
            block_attn_pre, block_post, block_comb = inline_271_result
            del inline_271_result
            inline_273_hidden_states = block_hidden_states
            inline_273_pre_mix = block_pre_mix
            inline_274_result = hc_pre_norm(
                inline_273_hidden_states, inline_273_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_273_hidden_states, inline_273_pre_mix
            block_attended = block_self.attn(inline_274_result, block_runtime)
            del inline_274_result
            inline_275_sublayer_output = block_attended
            inline_275_residual = block_hidden_states
            inline_275_post = block_post
            inline_275_comb = block_comb
            inline_275_result = hc_post(
                inline_275_sublayer_output, inline_275_residual, inline_275_post, inline_275_comb
            )
            del inline_275_sublayer_output, inline_275_residual, inline_275_post, inline_275_comb
            block_hidden_states = inline_275_result
            del inline_275_result
            inline_276_hidden_states = block_hidden_states
            inline_276_fn = block_self.hc_ffn_fn
            inline_276_scale = block_self.hc_ffn_scale
            inline_276_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_276_states = inline_276_hidden_states.flatten(-2).float()
            inline_276_rstd = torch.rsqrt(inline_276_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_276_mixes = F.linear(inline_276_states, inline_276_fn.float()) * inline_276_rstd
            inline_277_mixes = inline_276_mixes
            inline_277_scale = inline_276_scale
            inline_277_base = inline_276_base
            inline_277_result = hc_gates_sinkhorn(
                inline_277_mixes, inline_277_scale, inline_277_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_277_mixes, inline_277_scale, inline_277_base
            inline_276_result = inline_277_result
            del inline_277_result
            del (
                inline_276_base,
                inline_276_fn,
                inline_276_hidden_states,
                inline_276_mixes,
                inline_276_rstd,
                inline_276_scale,
                inline_276_states,
            )
            block_ffn_pre, block_post, block_comb = inline_276_result
            del inline_276_result
            inline_278_hidden_states = block_hidden_states
            inline_278_pre_mix = block_attn_pre
            inline_279_result = hc_pre_norm(
                inline_278_hidden_states, inline_278_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_278_hidden_states, inline_278_pre_mix
            block_ffn_input = inline_279_result
            del inline_279_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_280_sublayer_output = block_output
            inline_280_residual = block_hidden_states
            inline_280_post = block_post
            inline_280_comb = block_comb
            inline_280_result = hc_post(
                inline_280_sublayer_output, inline_280_residual, inline_280_post, inline_280_comb
            )
            del inline_280_sublayer_output, inline_280_residual, inline_280_post, inline_280_comb
            hidden_states, pre_mix = (inline_280_result, block_ffn_pre)
            del inline_280_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[14].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(14)
        hidden_states = self.layers[14].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 14, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[14]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_281_hidden_states = block_hidden_states
            inline_281_fn = block_self.hc_attn_fn
            inline_281_scale = block_self.hc_attn_scale
            inline_281_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_281_states = inline_281_hidden_states.flatten(-2).float()
            inline_281_rstd = torch.rsqrt(inline_281_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_281_mixes = F.linear(inline_281_states, inline_281_fn.float()) * inline_281_rstd
            inline_282_mixes = inline_281_mixes
            inline_282_scale = inline_281_scale
            inline_282_base = inline_281_base
            inline_282_result = hc_gates_sinkhorn(
                inline_282_mixes, inline_282_scale, inline_282_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_282_mixes, inline_282_scale, inline_282_base
            inline_281_result = inline_282_result
            del inline_282_result
            del (
                inline_281_base,
                inline_281_fn,
                inline_281_hidden_states,
                inline_281_mixes,
                inline_281_rstd,
                inline_281_scale,
                inline_281_states,
            )
            block_attn_pre, block_post, block_comb = inline_281_result
            del inline_281_result
            inline_283_hidden_states = block_hidden_states
            inline_283_pre_mix = block_pre_mix
            inline_284_result = hc_pre_norm(
                inline_283_hidden_states, inline_283_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_283_hidden_states, inline_283_pre_mix
            block_attended = block_self.attn(inline_284_result, block_runtime)
            del inline_284_result
            inline_285_sublayer_output = block_attended
            inline_285_residual = block_hidden_states
            inline_285_post = block_post
            inline_285_comb = block_comb
            inline_285_result = hc_post(
                inline_285_sublayer_output, inline_285_residual, inline_285_post, inline_285_comb
            )
            del inline_285_sublayer_output, inline_285_residual, inline_285_post, inline_285_comb
            block_hidden_states = inline_285_result
            del inline_285_result
            inline_286_hidden_states = block_hidden_states
            inline_286_fn = block_self.hc_ffn_fn
            inline_286_scale = block_self.hc_ffn_scale
            inline_286_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_286_states = inline_286_hidden_states.flatten(-2).float()
            inline_286_rstd = torch.rsqrt(inline_286_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_286_mixes = F.linear(inline_286_states, inline_286_fn.float()) * inline_286_rstd
            inline_287_mixes = inline_286_mixes
            inline_287_scale = inline_286_scale
            inline_287_base = inline_286_base
            inline_287_result = hc_gates_sinkhorn(
                inline_287_mixes, inline_287_scale, inline_287_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_287_mixes, inline_287_scale, inline_287_base
            inline_286_result = inline_287_result
            del inline_287_result
            del (
                inline_286_base,
                inline_286_fn,
                inline_286_hidden_states,
                inline_286_mixes,
                inline_286_rstd,
                inline_286_scale,
                inline_286_states,
            )
            block_ffn_pre, block_post, block_comb = inline_286_result
            del inline_286_result
            inline_288_hidden_states = block_hidden_states
            inline_288_pre_mix = block_attn_pre
            inline_289_result = hc_pre_norm(
                inline_288_hidden_states, inline_288_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_288_hidden_states, inline_288_pre_mix
            block_ffn_input = inline_289_result
            del inline_289_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_290_sublayer_output = block_output
            inline_290_residual = block_hidden_states
            inline_290_post = block_post
            inline_290_comb = block_comb
            inline_290_result = hc_post(
                inline_290_sublayer_output, inline_290_residual, inline_290_post, inline_290_comb
            )
            del inline_290_sublayer_output, inline_290_residual, inline_290_post, inline_290_comb
            hidden_states, pre_mix = (inline_290_result, block_ffn_pre)
            del inline_290_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[14]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_291_hidden_states = block_hidden_states
            inline_291_fn = block_self.hc_attn_fn
            inline_291_scale = block_self.hc_attn_scale
            inline_291_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_291_states = inline_291_hidden_states.flatten(-2).float()
            inline_291_rstd = torch.rsqrt(inline_291_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_291_mixes = F.linear(inline_291_states, inline_291_fn.float()) * inline_291_rstd
            inline_292_mixes = inline_291_mixes
            inline_292_scale = inline_291_scale
            inline_292_base = inline_291_base
            inline_292_result = hc_gates_sinkhorn(
                inline_292_mixes, inline_292_scale, inline_292_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_292_mixes, inline_292_scale, inline_292_base
            inline_291_result = inline_292_result
            del inline_292_result
            del (
                inline_291_base,
                inline_291_fn,
                inline_291_hidden_states,
                inline_291_mixes,
                inline_291_rstd,
                inline_291_scale,
                inline_291_states,
            )
            block_attn_pre, block_post, block_comb = inline_291_result
            del inline_291_result
            inline_293_hidden_states = block_hidden_states
            inline_293_pre_mix = block_pre_mix
            inline_294_result = hc_pre_norm(
                inline_293_hidden_states, inline_293_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_293_hidden_states, inline_293_pre_mix
            block_attended = block_self.attn(inline_294_result, block_runtime)
            del inline_294_result
            inline_295_sublayer_output = block_attended
            inline_295_residual = block_hidden_states
            inline_295_post = block_post
            inline_295_comb = block_comb
            inline_295_result = hc_post(
                inline_295_sublayer_output, inline_295_residual, inline_295_post, inline_295_comb
            )
            del inline_295_sublayer_output, inline_295_residual, inline_295_post, inline_295_comb
            block_hidden_states = inline_295_result
            del inline_295_result
            inline_296_hidden_states = block_hidden_states
            inline_296_fn = block_self.hc_ffn_fn
            inline_296_scale = block_self.hc_ffn_scale
            inline_296_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_296_states = inline_296_hidden_states.flatten(-2).float()
            inline_296_rstd = torch.rsqrt(inline_296_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_296_mixes = F.linear(inline_296_states, inline_296_fn.float()) * inline_296_rstd
            inline_297_mixes = inline_296_mixes
            inline_297_scale = inline_296_scale
            inline_297_base = inline_296_base
            inline_297_result = hc_gates_sinkhorn(
                inline_297_mixes, inline_297_scale, inline_297_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_297_mixes, inline_297_scale, inline_297_base
            inline_296_result = inline_297_result
            del inline_297_result
            del (
                inline_296_base,
                inline_296_fn,
                inline_296_hidden_states,
                inline_296_mixes,
                inline_296_rstd,
                inline_296_scale,
                inline_296_states,
            )
            block_ffn_pre, block_post, block_comb = inline_296_result
            del inline_296_result
            inline_298_hidden_states = block_hidden_states
            inline_298_pre_mix = block_attn_pre
            inline_299_result = hc_pre_norm(
                inline_298_hidden_states, inline_298_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_298_hidden_states, inline_298_pre_mix
            block_ffn_input = inline_299_result
            del inline_299_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_300_sublayer_output = block_output
            inline_300_residual = block_hidden_states
            inline_300_post = block_post
            inline_300_comb = block_comb
            inline_300_result = hc_post(
                inline_300_sublayer_output, inline_300_residual, inline_300_post, inline_300_comb
            )
            del inline_300_sublayer_output, inline_300_residual, inline_300_post, inline_300_comb
            hidden_states, pre_mix = (inline_300_result, block_ffn_pre)
            del inline_300_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[15].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(15)
        hidden_states = self.layers[15].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 15, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[15]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_301_hidden_states = block_hidden_states
            inline_301_fn = block_self.hc_attn_fn
            inline_301_scale = block_self.hc_attn_scale
            inline_301_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_301_states = inline_301_hidden_states.flatten(-2).float()
            inline_301_rstd = torch.rsqrt(inline_301_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_301_mixes = F.linear(inline_301_states, inline_301_fn.float()) * inline_301_rstd
            inline_302_mixes = inline_301_mixes
            inline_302_scale = inline_301_scale
            inline_302_base = inline_301_base
            inline_302_result = hc_gates_sinkhorn(
                inline_302_mixes, inline_302_scale, inline_302_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_302_mixes, inline_302_scale, inline_302_base
            inline_301_result = inline_302_result
            del inline_302_result
            del (
                inline_301_base,
                inline_301_fn,
                inline_301_hidden_states,
                inline_301_mixes,
                inline_301_rstd,
                inline_301_scale,
                inline_301_states,
            )
            block_attn_pre, block_post, block_comb = inline_301_result
            del inline_301_result
            inline_303_hidden_states = block_hidden_states
            inline_303_pre_mix = block_pre_mix
            inline_304_result = hc_pre_norm(
                inline_303_hidden_states, inline_303_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_303_hidden_states, inline_303_pre_mix
            block_attended = block_self.attn(inline_304_result, block_runtime)
            del inline_304_result
            inline_305_sublayer_output = block_attended
            inline_305_residual = block_hidden_states
            inline_305_post = block_post
            inline_305_comb = block_comb
            inline_305_result = hc_post(
                inline_305_sublayer_output, inline_305_residual, inline_305_post, inline_305_comb
            )
            del inline_305_sublayer_output, inline_305_residual, inline_305_post, inline_305_comb
            block_hidden_states = inline_305_result
            del inline_305_result
            inline_306_hidden_states = block_hidden_states
            inline_306_fn = block_self.hc_ffn_fn
            inline_306_scale = block_self.hc_ffn_scale
            inline_306_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_306_states = inline_306_hidden_states.flatten(-2).float()
            inline_306_rstd = torch.rsqrt(inline_306_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_306_mixes = F.linear(inline_306_states, inline_306_fn.float()) * inline_306_rstd
            inline_307_mixes = inline_306_mixes
            inline_307_scale = inline_306_scale
            inline_307_base = inline_306_base
            inline_307_result = hc_gates_sinkhorn(
                inline_307_mixes, inline_307_scale, inline_307_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_307_mixes, inline_307_scale, inline_307_base
            inline_306_result = inline_307_result
            del inline_307_result
            del (
                inline_306_base,
                inline_306_fn,
                inline_306_hidden_states,
                inline_306_mixes,
                inline_306_rstd,
                inline_306_scale,
                inline_306_states,
            )
            block_ffn_pre, block_post, block_comb = inline_306_result
            del inline_306_result
            inline_308_hidden_states = block_hidden_states
            inline_308_pre_mix = block_attn_pre
            inline_309_result = hc_pre_norm(
                inline_308_hidden_states, inline_308_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_308_hidden_states, inline_308_pre_mix
            block_ffn_input = inline_309_result
            del inline_309_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_310_sublayer_output = block_output
            inline_310_residual = block_hidden_states
            inline_310_post = block_post
            inline_310_comb = block_comb
            inline_310_result = hc_post(
                inline_310_sublayer_output, inline_310_residual, inline_310_post, inline_310_comb
            )
            del inline_310_sublayer_output, inline_310_residual, inline_310_post, inline_310_comb
            hidden_states, pre_mix = (inline_310_result, block_ffn_pre)
            del inline_310_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[15]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_311_hidden_states = block_hidden_states
            inline_311_fn = block_self.hc_attn_fn
            inline_311_scale = block_self.hc_attn_scale
            inline_311_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_311_states = inline_311_hidden_states.flatten(-2).float()
            inline_311_rstd = torch.rsqrt(inline_311_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_311_mixes = F.linear(inline_311_states, inline_311_fn.float()) * inline_311_rstd
            inline_312_mixes = inline_311_mixes
            inline_312_scale = inline_311_scale
            inline_312_base = inline_311_base
            inline_312_result = hc_gates_sinkhorn(
                inline_312_mixes, inline_312_scale, inline_312_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_312_mixes, inline_312_scale, inline_312_base
            inline_311_result = inline_312_result
            del inline_312_result
            del (
                inline_311_base,
                inline_311_fn,
                inline_311_hidden_states,
                inline_311_mixes,
                inline_311_rstd,
                inline_311_scale,
                inline_311_states,
            )
            block_attn_pre, block_post, block_comb = inline_311_result
            del inline_311_result
            inline_313_hidden_states = block_hidden_states
            inline_313_pre_mix = block_pre_mix
            inline_314_result = hc_pre_norm(
                inline_313_hidden_states, inline_313_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_313_hidden_states, inline_313_pre_mix
            block_attended = block_self.attn(inline_314_result, block_runtime)
            del inline_314_result
            inline_315_sublayer_output = block_attended
            inline_315_residual = block_hidden_states
            inline_315_post = block_post
            inline_315_comb = block_comb
            inline_315_result = hc_post(
                inline_315_sublayer_output, inline_315_residual, inline_315_post, inline_315_comb
            )
            del inline_315_sublayer_output, inline_315_residual, inline_315_post, inline_315_comb
            block_hidden_states = inline_315_result
            del inline_315_result
            inline_316_hidden_states = block_hidden_states
            inline_316_fn = block_self.hc_ffn_fn
            inline_316_scale = block_self.hc_ffn_scale
            inline_316_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_316_states = inline_316_hidden_states.flatten(-2).float()
            inline_316_rstd = torch.rsqrt(inline_316_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_316_mixes = F.linear(inline_316_states, inline_316_fn.float()) * inline_316_rstd
            inline_317_mixes = inline_316_mixes
            inline_317_scale = inline_316_scale
            inline_317_base = inline_316_base
            inline_317_result = hc_gates_sinkhorn(
                inline_317_mixes, inline_317_scale, inline_317_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_317_mixes, inline_317_scale, inline_317_base
            inline_316_result = inline_317_result
            del inline_317_result
            del (
                inline_316_base,
                inline_316_fn,
                inline_316_hidden_states,
                inline_316_mixes,
                inline_316_rstd,
                inline_316_scale,
                inline_316_states,
            )
            block_ffn_pre, block_post, block_comb = inline_316_result
            del inline_316_result
            inline_318_hidden_states = block_hidden_states
            inline_318_pre_mix = block_attn_pre
            inline_319_result = hc_pre_norm(
                inline_318_hidden_states, inline_318_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_318_hidden_states, inline_318_pre_mix
            block_ffn_input = inline_319_result
            del inline_319_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_320_sublayer_output = block_output
            inline_320_residual = block_hidden_states
            inline_320_post = block_post
            inline_320_comb = block_comb
            inline_320_result = hc_post(
                inline_320_sublayer_output, inline_320_residual, inline_320_post, inline_320_comb
            )
            del inline_320_sublayer_output, inline_320_residual, inline_320_post, inline_320_comb
            hidden_states, pre_mix = (inline_320_result, block_ffn_pre)
            del inline_320_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[16].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(16)
        hidden_states = self.layers[16].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 16, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[16]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_321_hidden_states = block_hidden_states
            inline_321_fn = block_self.hc_attn_fn
            inline_321_scale = block_self.hc_attn_scale
            inline_321_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_321_states = inline_321_hidden_states.flatten(-2).float()
            inline_321_rstd = torch.rsqrt(inline_321_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_321_mixes = F.linear(inline_321_states, inline_321_fn.float()) * inline_321_rstd
            inline_322_mixes = inline_321_mixes
            inline_322_scale = inline_321_scale
            inline_322_base = inline_321_base
            inline_322_result = hc_gates_sinkhorn(
                inline_322_mixes, inline_322_scale, inline_322_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_322_mixes, inline_322_scale, inline_322_base
            inline_321_result = inline_322_result
            del inline_322_result
            del (
                inline_321_base,
                inline_321_fn,
                inline_321_hidden_states,
                inline_321_mixes,
                inline_321_rstd,
                inline_321_scale,
                inline_321_states,
            )
            block_attn_pre, block_post, block_comb = inline_321_result
            del inline_321_result
            inline_323_hidden_states = block_hidden_states
            inline_323_pre_mix = block_pre_mix
            inline_324_result = hc_pre_norm(
                inline_323_hidden_states, inline_323_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_323_hidden_states, inline_323_pre_mix
            block_attended = block_self.attn(inline_324_result, block_runtime)
            del inline_324_result
            inline_325_sublayer_output = block_attended
            inline_325_residual = block_hidden_states
            inline_325_post = block_post
            inline_325_comb = block_comb
            inline_325_result = hc_post(
                inline_325_sublayer_output, inline_325_residual, inline_325_post, inline_325_comb
            )
            del inline_325_sublayer_output, inline_325_residual, inline_325_post, inline_325_comb
            block_hidden_states = inline_325_result
            del inline_325_result
            inline_326_hidden_states = block_hidden_states
            inline_326_fn = block_self.hc_ffn_fn
            inline_326_scale = block_self.hc_ffn_scale
            inline_326_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_326_states = inline_326_hidden_states.flatten(-2).float()
            inline_326_rstd = torch.rsqrt(inline_326_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_326_mixes = F.linear(inline_326_states, inline_326_fn.float()) * inline_326_rstd
            inline_327_mixes = inline_326_mixes
            inline_327_scale = inline_326_scale
            inline_327_base = inline_326_base
            inline_327_result = hc_gates_sinkhorn(
                inline_327_mixes, inline_327_scale, inline_327_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_327_mixes, inline_327_scale, inline_327_base
            inline_326_result = inline_327_result
            del inline_327_result
            del (
                inline_326_base,
                inline_326_fn,
                inline_326_hidden_states,
                inline_326_mixes,
                inline_326_rstd,
                inline_326_scale,
                inline_326_states,
            )
            block_ffn_pre, block_post, block_comb = inline_326_result
            del inline_326_result
            inline_328_hidden_states = block_hidden_states
            inline_328_pre_mix = block_attn_pre
            inline_329_result = hc_pre_norm(
                inline_328_hidden_states, inline_328_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_328_hidden_states, inline_328_pre_mix
            block_ffn_input = inline_329_result
            del inline_329_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_330_sublayer_output = block_output
            inline_330_residual = block_hidden_states
            inline_330_post = block_post
            inline_330_comb = block_comb
            inline_330_result = hc_post(
                inline_330_sublayer_output, inline_330_residual, inline_330_post, inline_330_comb
            )
            del inline_330_sublayer_output, inline_330_residual, inline_330_post, inline_330_comb
            hidden_states, pre_mix = (inline_330_result, block_ffn_pre)
            del inline_330_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[16]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_331_hidden_states = block_hidden_states
            inline_331_fn = block_self.hc_attn_fn
            inline_331_scale = block_self.hc_attn_scale
            inline_331_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_331_states = inline_331_hidden_states.flatten(-2).float()
            inline_331_rstd = torch.rsqrt(inline_331_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_331_mixes = F.linear(inline_331_states, inline_331_fn.float()) * inline_331_rstd
            inline_332_mixes = inline_331_mixes
            inline_332_scale = inline_331_scale
            inline_332_base = inline_331_base
            inline_332_result = hc_gates_sinkhorn(
                inline_332_mixes, inline_332_scale, inline_332_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_332_mixes, inline_332_scale, inline_332_base
            inline_331_result = inline_332_result
            del inline_332_result
            del (
                inline_331_base,
                inline_331_fn,
                inline_331_hidden_states,
                inline_331_mixes,
                inline_331_rstd,
                inline_331_scale,
                inline_331_states,
            )
            block_attn_pre, block_post, block_comb = inline_331_result
            del inline_331_result
            inline_333_hidden_states = block_hidden_states
            inline_333_pre_mix = block_pre_mix
            inline_334_result = hc_pre_norm(
                inline_333_hidden_states, inline_333_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_333_hidden_states, inline_333_pre_mix
            block_attended = block_self.attn(inline_334_result, block_runtime)
            del inline_334_result
            inline_335_sublayer_output = block_attended
            inline_335_residual = block_hidden_states
            inline_335_post = block_post
            inline_335_comb = block_comb
            inline_335_result = hc_post(
                inline_335_sublayer_output, inline_335_residual, inline_335_post, inline_335_comb
            )
            del inline_335_sublayer_output, inline_335_residual, inline_335_post, inline_335_comb
            block_hidden_states = inline_335_result
            del inline_335_result
            inline_336_hidden_states = block_hidden_states
            inline_336_fn = block_self.hc_ffn_fn
            inline_336_scale = block_self.hc_ffn_scale
            inline_336_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_336_states = inline_336_hidden_states.flatten(-2).float()
            inline_336_rstd = torch.rsqrt(inline_336_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_336_mixes = F.linear(inline_336_states, inline_336_fn.float()) * inline_336_rstd
            inline_337_mixes = inline_336_mixes
            inline_337_scale = inline_336_scale
            inline_337_base = inline_336_base
            inline_337_result = hc_gates_sinkhorn(
                inline_337_mixes, inline_337_scale, inline_337_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_337_mixes, inline_337_scale, inline_337_base
            inline_336_result = inline_337_result
            del inline_337_result
            del (
                inline_336_base,
                inline_336_fn,
                inline_336_hidden_states,
                inline_336_mixes,
                inline_336_rstd,
                inline_336_scale,
                inline_336_states,
            )
            block_ffn_pre, block_post, block_comb = inline_336_result
            del inline_336_result
            inline_338_hidden_states = block_hidden_states
            inline_338_pre_mix = block_attn_pre
            inline_339_result = hc_pre_norm(
                inline_338_hidden_states, inline_338_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_338_hidden_states, inline_338_pre_mix
            block_ffn_input = inline_339_result
            del inline_339_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_340_sublayer_output = block_output
            inline_340_residual = block_hidden_states
            inline_340_post = block_post
            inline_340_comb = block_comb
            inline_340_result = hc_post(
                inline_340_sublayer_output, inline_340_residual, inline_340_post, inline_340_comb
            )
            del inline_340_sublayer_output, inline_340_residual, inline_340_post, inline_340_comb
            hidden_states, pre_mix = (inline_340_result, block_ffn_pre)
            del inline_340_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[17].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(17)
        hidden_states = self.layers[17].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 17, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[17]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_341_hidden_states = block_hidden_states
            inline_341_fn = block_self.hc_attn_fn
            inline_341_scale = block_self.hc_attn_scale
            inline_341_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_341_states = inline_341_hidden_states.flatten(-2).float()
            inline_341_rstd = torch.rsqrt(inline_341_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_341_mixes = F.linear(inline_341_states, inline_341_fn.float()) * inline_341_rstd
            inline_342_mixes = inline_341_mixes
            inline_342_scale = inline_341_scale
            inline_342_base = inline_341_base
            inline_342_result = hc_gates_sinkhorn(
                inline_342_mixes, inline_342_scale, inline_342_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_342_mixes, inline_342_scale, inline_342_base
            inline_341_result = inline_342_result
            del inline_342_result
            del (
                inline_341_base,
                inline_341_fn,
                inline_341_hidden_states,
                inline_341_mixes,
                inline_341_rstd,
                inline_341_scale,
                inline_341_states,
            )
            block_attn_pre, block_post, block_comb = inline_341_result
            del inline_341_result
            inline_343_hidden_states = block_hidden_states
            inline_343_pre_mix = block_pre_mix
            inline_344_result = hc_pre_norm(
                inline_343_hidden_states, inline_343_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_343_hidden_states, inline_343_pre_mix
            block_attended = block_self.attn(inline_344_result, block_runtime)
            del inline_344_result
            inline_345_sublayer_output = block_attended
            inline_345_residual = block_hidden_states
            inline_345_post = block_post
            inline_345_comb = block_comb
            inline_345_result = hc_post(
                inline_345_sublayer_output, inline_345_residual, inline_345_post, inline_345_comb
            )
            del inline_345_sublayer_output, inline_345_residual, inline_345_post, inline_345_comb
            block_hidden_states = inline_345_result
            del inline_345_result
            inline_346_hidden_states = block_hidden_states
            inline_346_fn = block_self.hc_ffn_fn
            inline_346_scale = block_self.hc_ffn_scale
            inline_346_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_346_states = inline_346_hidden_states.flatten(-2).float()
            inline_346_rstd = torch.rsqrt(inline_346_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_346_mixes = F.linear(inline_346_states, inline_346_fn.float()) * inline_346_rstd
            inline_347_mixes = inline_346_mixes
            inline_347_scale = inline_346_scale
            inline_347_base = inline_346_base
            inline_347_result = hc_gates_sinkhorn(
                inline_347_mixes, inline_347_scale, inline_347_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_347_mixes, inline_347_scale, inline_347_base
            inline_346_result = inline_347_result
            del inline_347_result
            del (
                inline_346_base,
                inline_346_fn,
                inline_346_hidden_states,
                inline_346_mixes,
                inline_346_rstd,
                inline_346_scale,
                inline_346_states,
            )
            block_ffn_pre, block_post, block_comb = inline_346_result
            del inline_346_result
            inline_348_hidden_states = block_hidden_states
            inline_348_pre_mix = block_attn_pre
            inline_349_result = hc_pre_norm(
                inline_348_hidden_states, inline_348_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_348_hidden_states, inline_348_pre_mix
            block_ffn_input = inline_349_result
            del inline_349_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_350_sublayer_output = block_output
            inline_350_residual = block_hidden_states
            inline_350_post = block_post
            inline_350_comb = block_comb
            inline_350_result = hc_post(
                inline_350_sublayer_output, inline_350_residual, inline_350_post, inline_350_comb
            )
            del inline_350_sublayer_output, inline_350_residual, inline_350_post, inline_350_comb
            hidden_states, pre_mix = (inline_350_result, block_ffn_pre)
            del inline_350_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[17]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_351_hidden_states = block_hidden_states
            inline_351_fn = block_self.hc_attn_fn
            inline_351_scale = block_self.hc_attn_scale
            inline_351_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_351_states = inline_351_hidden_states.flatten(-2).float()
            inline_351_rstd = torch.rsqrt(inline_351_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_351_mixes = F.linear(inline_351_states, inline_351_fn.float()) * inline_351_rstd
            inline_352_mixes = inline_351_mixes
            inline_352_scale = inline_351_scale
            inline_352_base = inline_351_base
            inline_352_result = hc_gates_sinkhorn(
                inline_352_mixes, inline_352_scale, inline_352_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_352_mixes, inline_352_scale, inline_352_base
            inline_351_result = inline_352_result
            del inline_352_result
            del (
                inline_351_base,
                inline_351_fn,
                inline_351_hidden_states,
                inline_351_mixes,
                inline_351_rstd,
                inline_351_scale,
                inline_351_states,
            )
            block_attn_pre, block_post, block_comb = inline_351_result
            del inline_351_result
            inline_353_hidden_states = block_hidden_states
            inline_353_pre_mix = block_pre_mix
            inline_354_result = hc_pre_norm(
                inline_353_hidden_states, inline_353_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_353_hidden_states, inline_353_pre_mix
            block_attended = block_self.attn(inline_354_result, block_runtime)
            del inline_354_result
            inline_355_sublayer_output = block_attended
            inline_355_residual = block_hidden_states
            inline_355_post = block_post
            inline_355_comb = block_comb
            inline_355_result = hc_post(
                inline_355_sublayer_output, inline_355_residual, inline_355_post, inline_355_comb
            )
            del inline_355_sublayer_output, inline_355_residual, inline_355_post, inline_355_comb
            block_hidden_states = inline_355_result
            del inline_355_result
            inline_356_hidden_states = block_hidden_states
            inline_356_fn = block_self.hc_ffn_fn
            inline_356_scale = block_self.hc_ffn_scale
            inline_356_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_356_states = inline_356_hidden_states.flatten(-2).float()
            inline_356_rstd = torch.rsqrt(inline_356_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_356_mixes = F.linear(inline_356_states, inline_356_fn.float()) * inline_356_rstd
            inline_357_mixes = inline_356_mixes
            inline_357_scale = inline_356_scale
            inline_357_base = inline_356_base
            inline_357_result = hc_gates_sinkhorn(
                inline_357_mixes, inline_357_scale, inline_357_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_357_mixes, inline_357_scale, inline_357_base
            inline_356_result = inline_357_result
            del inline_357_result
            del (
                inline_356_base,
                inline_356_fn,
                inline_356_hidden_states,
                inline_356_mixes,
                inline_356_rstd,
                inline_356_scale,
                inline_356_states,
            )
            block_ffn_pre, block_post, block_comb = inline_356_result
            del inline_356_result
            inline_358_hidden_states = block_hidden_states
            inline_358_pre_mix = block_attn_pre
            inline_359_result = hc_pre_norm(
                inline_358_hidden_states, inline_358_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_358_hidden_states, inline_358_pre_mix
            block_ffn_input = inline_359_result
            del inline_359_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_360_sublayer_output = block_output
            inline_360_residual = block_hidden_states
            inline_360_post = block_post
            inline_360_comb = block_comb
            inline_360_result = hc_post(
                inline_360_sublayer_output, inline_360_residual, inline_360_post, inline_360_comb
            )
            del inline_360_sublayer_output, inline_360_residual, inline_360_post, inline_360_comb
            hidden_states, pre_mix = (inline_360_result, block_ffn_pre)
            del inline_360_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[18].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(18)
        hidden_states = self.layers[18].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 18, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[18]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_361_hidden_states = block_hidden_states
            inline_361_fn = block_self.hc_attn_fn
            inline_361_scale = block_self.hc_attn_scale
            inline_361_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_361_states = inline_361_hidden_states.flatten(-2).float()
            inline_361_rstd = torch.rsqrt(inline_361_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_361_mixes = F.linear(inline_361_states, inline_361_fn.float()) * inline_361_rstd
            inline_362_mixes = inline_361_mixes
            inline_362_scale = inline_361_scale
            inline_362_base = inline_361_base
            inline_362_result = hc_gates_sinkhorn(
                inline_362_mixes, inline_362_scale, inline_362_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_362_mixes, inline_362_scale, inline_362_base
            inline_361_result = inline_362_result
            del inline_362_result
            del (
                inline_361_base,
                inline_361_fn,
                inline_361_hidden_states,
                inline_361_mixes,
                inline_361_rstd,
                inline_361_scale,
                inline_361_states,
            )
            block_attn_pre, block_post, block_comb = inline_361_result
            del inline_361_result
            inline_363_hidden_states = block_hidden_states
            inline_363_pre_mix = block_pre_mix
            inline_364_result = hc_pre_norm(
                inline_363_hidden_states, inline_363_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_363_hidden_states, inline_363_pre_mix
            block_attended = block_self.attn(inline_364_result, block_runtime)
            del inline_364_result
            inline_365_sublayer_output = block_attended
            inline_365_residual = block_hidden_states
            inline_365_post = block_post
            inline_365_comb = block_comb
            inline_365_result = hc_post(
                inline_365_sublayer_output, inline_365_residual, inline_365_post, inline_365_comb
            )
            del inline_365_sublayer_output, inline_365_residual, inline_365_post, inline_365_comb
            block_hidden_states = inline_365_result
            del inline_365_result
            inline_366_hidden_states = block_hidden_states
            inline_366_fn = block_self.hc_ffn_fn
            inline_366_scale = block_self.hc_ffn_scale
            inline_366_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_366_states = inline_366_hidden_states.flatten(-2).float()
            inline_366_rstd = torch.rsqrt(inline_366_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_366_mixes = F.linear(inline_366_states, inline_366_fn.float()) * inline_366_rstd
            inline_367_mixes = inline_366_mixes
            inline_367_scale = inline_366_scale
            inline_367_base = inline_366_base
            inline_367_result = hc_gates_sinkhorn(
                inline_367_mixes, inline_367_scale, inline_367_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_367_mixes, inline_367_scale, inline_367_base
            inline_366_result = inline_367_result
            del inline_367_result
            del (
                inline_366_base,
                inline_366_fn,
                inline_366_hidden_states,
                inline_366_mixes,
                inline_366_rstd,
                inline_366_scale,
                inline_366_states,
            )
            block_ffn_pre, block_post, block_comb = inline_366_result
            del inline_366_result
            inline_368_hidden_states = block_hidden_states
            inline_368_pre_mix = block_attn_pre
            inline_369_result = hc_pre_norm(
                inline_368_hidden_states, inline_368_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_368_hidden_states, inline_368_pre_mix
            block_ffn_input = inline_369_result
            del inline_369_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_370_sublayer_output = block_output
            inline_370_residual = block_hidden_states
            inline_370_post = block_post
            inline_370_comb = block_comb
            inline_370_result = hc_post(
                inline_370_sublayer_output, inline_370_residual, inline_370_post, inline_370_comb
            )
            del inline_370_sublayer_output, inline_370_residual, inline_370_post, inline_370_comb
            hidden_states, pre_mix = (inline_370_result, block_ffn_pre)
            del inline_370_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[18]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_371_hidden_states = block_hidden_states
            inline_371_fn = block_self.hc_attn_fn
            inline_371_scale = block_self.hc_attn_scale
            inline_371_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_371_states = inline_371_hidden_states.flatten(-2).float()
            inline_371_rstd = torch.rsqrt(inline_371_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_371_mixes = F.linear(inline_371_states, inline_371_fn.float()) * inline_371_rstd
            inline_372_mixes = inline_371_mixes
            inline_372_scale = inline_371_scale
            inline_372_base = inline_371_base
            inline_372_result = hc_gates_sinkhorn(
                inline_372_mixes, inline_372_scale, inline_372_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_372_mixes, inline_372_scale, inline_372_base
            inline_371_result = inline_372_result
            del inline_372_result
            del (
                inline_371_base,
                inline_371_fn,
                inline_371_hidden_states,
                inline_371_mixes,
                inline_371_rstd,
                inline_371_scale,
                inline_371_states,
            )
            block_attn_pre, block_post, block_comb = inline_371_result
            del inline_371_result
            inline_373_hidden_states = block_hidden_states
            inline_373_pre_mix = block_pre_mix
            inline_374_result = hc_pre_norm(
                inline_373_hidden_states, inline_373_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_373_hidden_states, inline_373_pre_mix
            block_attended = block_self.attn(inline_374_result, block_runtime)
            del inline_374_result
            inline_375_sublayer_output = block_attended
            inline_375_residual = block_hidden_states
            inline_375_post = block_post
            inline_375_comb = block_comb
            inline_375_result = hc_post(
                inline_375_sublayer_output, inline_375_residual, inline_375_post, inline_375_comb
            )
            del inline_375_sublayer_output, inline_375_residual, inline_375_post, inline_375_comb
            block_hidden_states = inline_375_result
            del inline_375_result
            inline_376_hidden_states = block_hidden_states
            inline_376_fn = block_self.hc_ffn_fn
            inline_376_scale = block_self.hc_ffn_scale
            inline_376_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_376_states = inline_376_hidden_states.flatten(-2).float()
            inline_376_rstd = torch.rsqrt(inline_376_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_376_mixes = F.linear(inline_376_states, inline_376_fn.float()) * inline_376_rstd
            inline_377_mixes = inline_376_mixes
            inline_377_scale = inline_376_scale
            inline_377_base = inline_376_base
            inline_377_result = hc_gates_sinkhorn(
                inline_377_mixes, inline_377_scale, inline_377_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_377_mixes, inline_377_scale, inline_377_base
            inline_376_result = inline_377_result
            del inline_377_result
            del (
                inline_376_base,
                inline_376_fn,
                inline_376_hidden_states,
                inline_376_mixes,
                inline_376_rstd,
                inline_376_scale,
                inline_376_states,
            )
            block_ffn_pre, block_post, block_comb = inline_376_result
            del inline_376_result
            inline_378_hidden_states = block_hidden_states
            inline_378_pre_mix = block_attn_pre
            inline_379_result = hc_pre_norm(
                inline_378_hidden_states, inline_378_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_378_hidden_states, inline_378_pre_mix
            block_ffn_input = inline_379_result
            del inline_379_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_380_sublayer_output = block_output
            inline_380_residual = block_hidden_states
            inline_380_post = block_post
            inline_380_comb = block_comb
            inline_380_result = hc_post(
                inline_380_sublayer_output, inline_380_residual, inline_380_post, inline_380_comb
            )
            del inline_380_sublayer_output, inline_380_residual, inline_380_post, inline_380_comb
            hidden_states, pre_mix = (inline_380_result, block_ffn_pre)
            del inline_380_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[19].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(19)
        hidden_states = self.layers[19].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 19, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[19]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_381_hidden_states = block_hidden_states
            inline_381_fn = block_self.hc_attn_fn
            inline_381_scale = block_self.hc_attn_scale
            inline_381_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_381_states = inline_381_hidden_states.flatten(-2).float()
            inline_381_rstd = torch.rsqrt(inline_381_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_381_mixes = F.linear(inline_381_states, inline_381_fn.float()) * inline_381_rstd
            inline_382_mixes = inline_381_mixes
            inline_382_scale = inline_381_scale
            inline_382_base = inline_381_base
            inline_382_result = hc_gates_sinkhorn(
                inline_382_mixes, inline_382_scale, inline_382_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_382_mixes, inline_382_scale, inline_382_base
            inline_381_result = inline_382_result
            del inline_382_result
            del (
                inline_381_base,
                inline_381_fn,
                inline_381_hidden_states,
                inline_381_mixes,
                inline_381_rstd,
                inline_381_scale,
                inline_381_states,
            )
            block_attn_pre, block_post, block_comb = inline_381_result
            del inline_381_result
            inline_383_hidden_states = block_hidden_states
            inline_383_pre_mix = block_pre_mix
            inline_384_result = hc_pre_norm(
                inline_383_hidden_states, inline_383_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_383_hidden_states, inline_383_pre_mix
            block_attended = block_self.attn(inline_384_result, block_runtime)
            del inline_384_result
            inline_385_sublayer_output = block_attended
            inline_385_residual = block_hidden_states
            inline_385_post = block_post
            inline_385_comb = block_comb
            inline_385_result = hc_post(
                inline_385_sublayer_output, inline_385_residual, inline_385_post, inline_385_comb
            )
            del inline_385_sublayer_output, inline_385_residual, inline_385_post, inline_385_comb
            block_hidden_states = inline_385_result
            del inline_385_result
            inline_386_hidden_states = block_hidden_states
            inline_386_fn = block_self.hc_ffn_fn
            inline_386_scale = block_self.hc_ffn_scale
            inline_386_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_386_states = inline_386_hidden_states.flatten(-2).float()
            inline_386_rstd = torch.rsqrt(inline_386_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_386_mixes = F.linear(inline_386_states, inline_386_fn.float()) * inline_386_rstd
            inline_387_mixes = inline_386_mixes
            inline_387_scale = inline_386_scale
            inline_387_base = inline_386_base
            inline_387_result = hc_gates_sinkhorn(
                inline_387_mixes, inline_387_scale, inline_387_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_387_mixes, inline_387_scale, inline_387_base
            inline_386_result = inline_387_result
            del inline_387_result
            del (
                inline_386_base,
                inline_386_fn,
                inline_386_hidden_states,
                inline_386_mixes,
                inline_386_rstd,
                inline_386_scale,
                inline_386_states,
            )
            block_ffn_pre, block_post, block_comb = inline_386_result
            del inline_386_result
            inline_388_hidden_states = block_hidden_states
            inline_388_pre_mix = block_attn_pre
            inline_389_result = hc_pre_norm(
                inline_388_hidden_states, inline_388_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_388_hidden_states, inline_388_pre_mix
            block_ffn_input = inline_389_result
            del inline_389_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_390_sublayer_output = block_output
            inline_390_residual = block_hidden_states
            inline_390_post = block_post
            inline_390_comb = block_comb
            inline_390_result = hc_post(
                inline_390_sublayer_output, inline_390_residual, inline_390_post, inline_390_comb
            )
            del inline_390_sublayer_output, inline_390_residual, inline_390_post, inline_390_comb
            hidden_states, pre_mix = (inline_390_result, block_ffn_pre)
            del inline_390_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[19]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_391_hidden_states = block_hidden_states
            inline_391_fn = block_self.hc_attn_fn
            inline_391_scale = block_self.hc_attn_scale
            inline_391_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_391_states = inline_391_hidden_states.flatten(-2).float()
            inline_391_rstd = torch.rsqrt(inline_391_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_391_mixes = F.linear(inline_391_states, inline_391_fn.float()) * inline_391_rstd
            inline_392_mixes = inline_391_mixes
            inline_392_scale = inline_391_scale
            inline_392_base = inline_391_base
            inline_392_result = hc_gates_sinkhorn(
                inline_392_mixes, inline_392_scale, inline_392_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_392_mixes, inline_392_scale, inline_392_base
            inline_391_result = inline_392_result
            del inline_392_result
            del (
                inline_391_base,
                inline_391_fn,
                inline_391_hidden_states,
                inline_391_mixes,
                inline_391_rstd,
                inline_391_scale,
                inline_391_states,
            )
            block_attn_pre, block_post, block_comb = inline_391_result
            del inline_391_result
            inline_393_hidden_states = block_hidden_states
            inline_393_pre_mix = block_pre_mix
            inline_394_result = hc_pre_norm(
                inline_393_hidden_states, inline_393_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_393_hidden_states, inline_393_pre_mix
            block_attended = block_self.attn(inline_394_result, block_runtime)
            del inline_394_result
            inline_395_sublayer_output = block_attended
            inline_395_residual = block_hidden_states
            inline_395_post = block_post
            inline_395_comb = block_comb
            inline_395_result = hc_post(
                inline_395_sublayer_output, inline_395_residual, inline_395_post, inline_395_comb
            )
            del inline_395_sublayer_output, inline_395_residual, inline_395_post, inline_395_comb
            block_hidden_states = inline_395_result
            del inline_395_result
            inline_396_hidden_states = block_hidden_states
            inline_396_fn = block_self.hc_ffn_fn
            inline_396_scale = block_self.hc_ffn_scale
            inline_396_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_396_states = inline_396_hidden_states.flatten(-2).float()
            inline_396_rstd = torch.rsqrt(inline_396_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_396_mixes = F.linear(inline_396_states, inline_396_fn.float()) * inline_396_rstd
            inline_397_mixes = inline_396_mixes
            inline_397_scale = inline_396_scale
            inline_397_base = inline_396_base
            inline_397_result = hc_gates_sinkhorn(
                inline_397_mixes, inline_397_scale, inline_397_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_397_mixes, inline_397_scale, inline_397_base
            inline_396_result = inline_397_result
            del inline_397_result
            del (
                inline_396_base,
                inline_396_fn,
                inline_396_hidden_states,
                inline_396_mixes,
                inline_396_rstd,
                inline_396_scale,
                inline_396_states,
            )
            block_ffn_pre, block_post, block_comb = inline_396_result
            del inline_396_result
            inline_398_hidden_states = block_hidden_states
            inline_398_pre_mix = block_attn_pre
            inline_399_result = hc_pre_norm(
                inline_398_hidden_states, inline_398_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_398_hidden_states, inline_398_pre_mix
            block_ffn_input = inline_399_result
            del inline_399_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_400_sublayer_output = block_output
            inline_400_residual = block_hidden_states
            inline_400_post = block_post
            inline_400_comb = block_comb
            inline_400_result = hc_post(
                inline_400_sublayer_output, inline_400_residual, inline_400_post, inline_400_comb
            )
            del inline_400_sublayer_output, inline_400_residual, inline_400_post, inline_400_comb
            hidden_states, pre_mix = (inline_400_result, block_ffn_pre)
            del inline_400_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[20].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(20)
        hidden_states = self.layers[20].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 20, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[20]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_401_hidden_states = block_hidden_states
            inline_401_fn = block_self.hc_attn_fn
            inline_401_scale = block_self.hc_attn_scale
            inline_401_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_401_states = inline_401_hidden_states.flatten(-2).float()
            inline_401_rstd = torch.rsqrt(inline_401_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_401_mixes = F.linear(inline_401_states, inline_401_fn.float()) * inline_401_rstd
            inline_402_mixes = inline_401_mixes
            inline_402_scale = inline_401_scale
            inline_402_base = inline_401_base
            inline_402_result = hc_gates_sinkhorn(
                inline_402_mixes, inline_402_scale, inline_402_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_402_mixes, inline_402_scale, inline_402_base
            inline_401_result = inline_402_result
            del inline_402_result
            del (
                inline_401_base,
                inline_401_fn,
                inline_401_hidden_states,
                inline_401_mixes,
                inline_401_rstd,
                inline_401_scale,
                inline_401_states,
            )
            block_attn_pre, block_post, block_comb = inline_401_result
            del inline_401_result
            inline_403_hidden_states = block_hidden_states
            inline_403_pre_mix = block_pre_mix
            inline_404_result = hc_pre_norm(
                inline_403_hidden_states, inline_403_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_403_hidden_states, inline_403_pre_mix
            block_attended = block_self.attn(inline_404_result, block_runtime)
            del inline_404_result
            inline_405_sublayer_output = block_attended
            inline_405_residual = block_hidden_states
            inline_405_post = block_post
            inline_405_comb = block_comb
            inline_405_result = hc_post(
                inline_405_sublayer_output, inline_405_residual, inline_405_post, inline_405_comb
            )
            del inline_405_sublayer_output, inline_405_residual, inline_405_post, inline_405_comb
            block_hidden_states = inline_405_result
            del inline_405_result
            inline_406_hidden_states = block_hidden_states
            inline_406_fn = block_self.hc_ffn_fn
            inline_406_scale = block_self.hc_ffn_scale
            inline_406_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_406_states = inline_406_hidden_states.flatten(-2).float()
            inline_406_rstd = torch.rsqrt(inline_406_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_406_mixes = F.linear(inline_406_states, inline_406_fn.float()) * inline_406_rstd
            inline_407_mixes = inline_406_mixes
            inline_407_scale = inline_406_scale
            inline_407_base = inline_406_base
            inline_407_result = hc_gates_sinkhorn(
                inline_407_mixes, inline_407_scale, inline_407_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_407_mixes, inline_407_scale, inline_407_base
            inline_406_result = inline_407_result
            del inline_407_result
            del (
                inline_406_base,
                inline_406_fn,
                inline_406_hidden_states,
                inline_406_mixes,
                inline_406_rstd,
                inline_406_scale,
                inline_406_states,
            )
            block_ffn_pre, block_post, block_comb = inline_406_result
            del inline_406_result
            inline_408_hidden_states = block_hidden_states
            inline_408_pre_mix = block_attn_pre
            inline_409_result = hc_pre_norm(
                inline_408_hidden_states, inline_408_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_408_hidden_states, inline_408_pre_mix
            block_ffn_input = inline_409_result
            del inline_409_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_410_sublayer_output = block_output
            inline_410_residual = block_hidden_states
            inline_410_post = block_post
            inline_410_comb = block_comb
            inline_410_result = hc_post(
                inline_410_sublayer_output, inline_410_residual, inline_410_post, inline_410_comb
            )
            del inline_410_sublayer_output, inline_410_residual, inline_410_post, inline_410_comb
            hidden_states, pre_mix = (inline_410_result, block_ffn_pre)
            del inline_410_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[20]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_411_hidden_states = block_hidden_states
            inline_411_fn = block_self.hc_attn_fn
            inline_411_scale = block_self.hc_attn_scale
            inline_411_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_411_states = inline_411_hidden_states.flatten(-2).float()
            inline_411_rstd = torch.rsqrt(inline_411_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_411_mixes = F.linear(inline_411_states, inline_411_fn.float()) * inline_411_rstd
            inline_412_mixes = inline_411_mixes
            inline_412_scale = inline_411_scale
            inline_412_base = inline_411_base
            inline_412_result = hc_gates_sinkhorn(
                inline_412_mixes, inline_412_scale, inline_412_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_412_mixes, inline_412_scale, inline_412_base
            inline_411_result = inline_412_result
            del inline_412_result
            del (
                inline_411_base,
                inline_411_fn,
                inline_411_hidden_states,
                inline_411_mixes,
                inline_411_rstd,
                inline_411_scale,
                inline_411_states,
            )
            block_attn_pre, block_post, block_comb = inline_411_result
            del inline_411_result
            inline_413_hidden_states = block_hidden_states
            inline_413_pre_mix = block_pre_mix
            inline_414_result = hc_pre_norm(
                inline_413_hidden_states, inline_413_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_413_hidden_states, inline_413_pre_mix
            block_attended = block_self.attn(inline_414_result, block_runtime)
            del inline_414_result
            inline_415_sublayer_output = block_attended
            inline_415_residual = block_hidden_states
            inline_415_post = block_post
            inline_415_comb = block_comb
            inline_415_result = hc_post(
                inline_415_sublayer_output, inline_415_residual, inline_415_post, inline_415_comb
            )
            del inline_415_sublayer_output, inline_415_residual, inline_415_post, inline_415_comb
            block_hidden_states = inline_415_result
            del inline_415_result
            inline_416_hidden_states = block_hidden_states
            inline_416_fn = block_self.hc_ffn_fn
            inline_416_scale = block_self.hc_ffn_scale
            inline_416_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_416_states = inline_416_hidden_states.flatten(-2).float()
            inline_416_rstd = torch.rsqrt(inline_416_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_416_mixes = F.linear(inline_416_states, inline_416_fn.float()) * inline_416_rstd
            inline_417_mixes = inline_416_mixes
            inline_417_scale = inline_416_scale
            inline_417_base = inline_416_base
            inline_417_result = hc_gates_sinkhorn(
                inline_417_mixes, inline_417_scale, inline_417_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_417_mixes, inline_417_scale, inline_417_base
            inline_416_result = inline_417_result
            del inline_417_result
            del (
                inline_416_base,
                inline_416_fn,
                inline_416_hidden_states,
                inline_416_mixes,
                inline_416_rstd,
                inline_416_scale,
                inline_416_states,
            )
            block_ffn_pre, block_post, block_comb = inline_416_result
            del inline_416_result
            inline_418_hidden_states = block_hidden_states
            inline_418_pre_mix = block_attn_pre
            inline_419_result = hc_pre_norm(
                inline_418_hidden_states, inline_418_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_418_hidden_states, inline_418_pre_mix
            block_ffn_input = inline_419_result
            del inline_419_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_420_sublayer_output = block_output
            inline_420_residual = block_hidden_states
            inline_420_post = block_post
            inline_420_comb = block_comb
            inline_420_result = hc_post(
                inline_420_sublayer_output, inline_420_residual, inline_420_post, inline_420_comb
            )
            del inline_420_sublayer_output, inline_420_residual, inline_420_post, inline_420_comb
            hidden_states, pre_mix = (inline_420_result, block_ffn_pre)
            del inline_420_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[21].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(21)
        hidden_states = self.layers[21].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 21, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[21]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_421_hidden_states = block_hidden_states
            inline_421_fn = block_self.hc_attn_fn
            inline_421_scale = block_self.hc_attn_scale
            inline_421_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_421_states = inline_421_hidden_states.flatten(-2).float()
            inline_421_rstd = torch.rsqrt(inline_421_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_421_mixes = F.linear(inline_421_states, inline_421_fn.float()) * inline_421_rstd
            inline_422_mixes = inline_421_mixes
            inline_422_scale = inline_421_scale
            inline_422_base = inline_421_base
            inline_422_result = hc_gates_sinkhorn(
                inline_422_mixes, inline_422_scale, inline_422_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_422_mixes, inline_422_scale, inline_422_base
            inline_421_result = inline_422_result
            del inline_422_result
            del (
                inline_421_base,
                inline_421_fn,
                inline_421_hidden_states,
                inline_421_mixes,
                inline_421_rstd,
                inline_421_scale,
                inline_421_states,
            )
            block_attn_pre, block_post, block_comb = inline_421_result
            del inline_421_result
            inline_423_hidden_states = block_hidden_states
            inline_423_pre_mix = block_pre_mix
            inline_424_result = hc_pre_norm(
                inline_423_hidden_states, inline_423_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_423_hidden_states, inline_423_pre_mix
            block_attended = block_self.attn(inline_424_result, block_runtime)
            del inline_424_result
            inline_425_sublayer_output = block_attended
            inline_425_residual = block_hidden_states
            inline_425_post = block_post
            inline_425_comb = block_comb
            inline_425_result = hc_post(
                inline_425_sublayer_output, inline_425_residual, inline_425_post, inline_425_comb
            )
            del inline_425_sublayer_output, inline_425_residual, inline_425_post, inline_425_comb
            block_hidden_states = inline_425_result
            del inline_425_result
            inline_426_hidden_states = block_hidden_states
            inline_426_fn = block_self.hc_ffn_fn
            inline_426_scale = block_self.hc_ffn_scale
            inline_426_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_426_states = inline_426_hidden_states.flatten(-2).float()
            inline_426_rstd = torch.rsqrt(inline_426_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_426_mixes = F.linear(inline_426_states, inline_426_fn.float()) * inline_426_rstd
            inline_427_mixes = inline_426_mixes
            inline_427_scale = inline_426_scale
            inline_427_base = inline_426_base
            inline_427_result = hc_gates_sinkhorn(
                inline_427_mixes, inline_427_scale, inline_427_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_427_mixes, inline_427_scale, inline_427_base
            inline_426_result = inline_427_result
            del inline_427_result
            del (
                inline_426_base,
                inline_426_fn,
                inline_426_hidden_states,
                inline_426_mixes,
                inline_426_rstd,
                inline_426_scale,
                inline_426_states,
            )
            block_ffn_pre, block_post, block_comb = inline_426_result
            del inline_426_result
            inline_428_hidden_states = block_hidden_states
            inline_428_pre_mix = block_attn_pre
            inline_429_result = hc_pre_norm(
                inline_428_hidden_states, inline_428_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_428_hidden_states, inline_428_pre_mix
            block_ffn_input = inline_429_result
            del inline_429_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_430_sublayer_output = block_output
            inline_430_residual = block_hidden_states
            inline_430_post = block_post
            inline_430_comb = block_comb
            inline_430_result = hc_post(
                inline_430_sublayer_output, inline_430_residual, inline_430_post, inline_430_comb
            )
            del inline_430_sublayer_output, inline_430_residual, inline_430_post, inline_430_comb
            hidden_states, pre_mix = (inline_430_result, block_ffn_pre)
            del inline_430_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[21]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_431_hidden_states = block_hidden_states
            inline_431_fn = block_self.hc_attn_fn
            inline_431_scale = block_self.hc_attn_scale
            inline_431_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_431_states = inline_431_hidden_states.flatten(-2).float()
            inline_431_rstd = torch.rsqrt(inline_431_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_431_mixes = F.linear(inline_431_states, inline_431_fn.float()) * inline_431_rstd
            inline_432_mixes = inline_431_mixes
            inline_432_scale = inline_431_scale
            inline_432_base = inline_431_base
            inline_432_result = hc_gates_sinkhorn(
                inline_432_mixes, inline_432_scale, inline_432_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_432_mixes, inline_432_scale, inline_432_base
            inline_431_result = inline_432_result
            del inline_432_result
            del (
                inline_431_base,
                inline_431_fn,
                inline_431_hidden_states,
                inline_431_mixes,
                inline_431_rstd,
                inline_431_scale,
                inline_431_states,
            )
            block_attn_pre, block_post, block_comb = inline_431_result
            del inline_431_result
            inline_433_hidden_states = block_hidden_states
            inline_433_pre_mix = block_pre_mix
            inline_434_result = hc_pre_norm(
                inline_433_hidden_states, inline_433_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_433_hidden_states, inline_433_pre_mix
            block_attended = block_self.attn(inline_434_result, block_runtime)
            del inline_434_result
            inline_435_sublayer_output = block_attended
            inline_435_residual = block_hidden_states
            inline_435_post = block_post
            inline_435_comb = block_comb
            inline_435_result = hc_post(
                inline_435_sublayer_output, inline_435_residual, inline_435_post, inline_435_comb
            )
            del inline_435_sublayer_output, inline_435_residual, inline_435_post, inline_435_comb
            block_hidden_states = inline_435_result
            del inline_435_result
            inline_436_hidden_states = block_hidden_states
            inline_436_fn = block_self.hc_ffn_fn
            inline_436_scale = block_self.hc_ffn_scale
            inline_436_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_436_states = inline_436_hidden_states.flatten(-2).float()
            inline_436_rstd = torch.rsqrt(inline_436_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_436_mixes = F.linear(inline_436_states, inline_436_fn.float()) * inline_436_rstd
            inline_437_mixes = inline_436_mixes
            inline_437_scale = inline_436_scale
            inline_437_base = inline_436_base
            inline_437_result = hc_gates_sinkhorn(
                inline_437_mixes, inline_437_scale, inline_437_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_437_mixes, inline_437_scale, inline_437_base
            inline_436_result = inline_437_result
            del inline_437_result
            del (
                inline_436_base,
                inline_436_fn,
                inline_436_hidden_states,
                inline_436_mixes,
                inline_436_rstd,
                inline_436_scale,
                inline_436_states,
            )
            block_ffn_pre, block_post, block_comb = inline_436_result
            del inline_436_result
            inline_438_hidden_states = block_hidden_states
            inline_438_pre_mix = block_attn_pre
            inline_439_result = hc_pre_norm(
                inline_438_hidden_states, inline_438_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_438_hidden_states, inline_438_pre_mix
            block_ffn_input = inline_439_result
            del inline_439_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_440_sublayer_output = block_output
            inline_440_residual = block_hidden_states
            inline_440_post = block_post
            inline_440_comb = block_comb
            inline_440_result = hc_post(
                inline_440_sublayer_output, inline_440_residual, inline_440_post, inline_440_comb
            )
            del inline_440_sublayer_output, inline_440_residual, inline_440_post, inline_440_comb
            hidden_states, pre_mix = (inline_440_result, block_ffn_pre)
            del inline_440_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[22].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(22)
        hidden_states = self.layers[22].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 22, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[22]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_441_hidden_states = block_hidden_states
            inline_441_fn = block_self.hc_attn_fn
            inline_441_scale = block_self.hc_attn_scale
            inline_441_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_441_states = inline_441_hidden_states.flatten(-2).float()
            inline_441_rstd = torch.rsqrt(inline_441_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_441_mixes = F.linear(inline_441_states, inline_441_fn.float()) * inline_441_rstd
            inline_442_mixes = inline_441_mixes
            inline_442_scale = inline_441_scale
            inline_442_base = inline_441_base
            inline_442_result = hc_gates_sinkhorn(
                inline_442_mixes, inline_442_scale, inline_442_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_442_mixes, inline_442_scale, inline_442_base
            inline_441_result = inline_442_result
            del inline_442_result
            del (
                inline_441_base,
                inline_441_fn,
                inline_441_hidden_states,
                inline_441_mixes,
                inline_441_rstd,
                inline_441_scale,
                inline_441_states,
            )
            block_attn_pre, block_post, block_comb = inline_441_result
            del inline_441_result
            inline_443_hidden_states = block_hidden_states
            inline_443_pre_mix = block_pre_mix
            inline_444_result = hc_pre_norm(
                inline_443_hidden_states, inline_443_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_443_hidden_states, inline_443_pre_mix
            block_attended = block_self.attn(inline_444_result, block_runtime)
            del inline_444_result
            inline_445_sublayer_output = block_attended
            inline_445_residual = block_hidden_states
            inline_445_post = block_post
            inline_445_comb = block_comb
            inline_445_result = hc_post(
                inline_445_sublayer_output, inline_445_residual, inline_445_post, inline_445_comb
            )
            del inline_445_sublayer_output, inline_445_residual, inline_445_post, inline_445_comb
            block_hidden_states = inline_445_result
            del inline_445_result
            inline_446_hidden_states = block_hidden_states
            inline_446_fn = block_self.hc_ffn_fn
            inline_446_scale = block_self.hc_ffn_scale
            inline_446_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_446_states = inline_446_hidden_states.flatten(-2).float()
            inline_446_rstd = torch.rsqrt(inline_446_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_446_mixes = F.linear(inline_446_states, inline_446_fn.float()) * inline_446_rstd
            inline_447_mixes = inline_446_mixes
            inline_447_scale = inline_446_scale
            inline_447_base = inline_446_base
            inline_447_result = hc_gates_sinkhorn(
                inline_447_mixes, inline_447_scale, inline_447_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_447_mixes, inline_447_scale, inline_447_base
            inline_446_result = inline_447_result
            del inline_447_result
            del (
                inline_446_base,
                inline_446_fn,
                inline_446_hidden_states,
                inline_446_mixes,
                inline_446_rstd,
                inline_446_scale,
                inline_446_states,
            )
            block_ffn_pre, block_post, block_comb = inline_446_result
            del inline_446_result
            inline_448_hidden_states = block_hidden_states
            inline_448_pre_mix = block_attn_pre
            inline_449_result = hc_pre_norm(
                inline_448_hidden_states, inline_448_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_448_hidden_states, inline_448_pre_mix
            block_ffn_input = inline_449_result
            del inline_449_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_450_sublayer_output = block_output
            inline_450_residual = block_hidden_states
            inline_450_post = block_post
            inline_450_comb = block_comb
            inline_450_result = hc_post(
                inline_450_sublayer_output, inline_450_residual, inline_450_post, inline_450_comb
            )
            del inline_450_sublayer_output, inline_450_residual, inline_450_post, inline_450_comb
            hidden_states, pre_mix = (inline_450_result, block_ffn_pre)
            del inline_450_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[22]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_451_hidden_states = block_hidden_states
            inline_451_fn = block_self.hc_attn_fn
            inline_451_scale = block_self.hc_attn_scale
            inline_451_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_451_states = inline_451_hidden_states.flatten(-2).float()
            inline_451_rstd = torch.rsqrt(inline_451_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_451_mixes = F.linear(inline_451_states, inline_451_fn.float()) * inline_451_rstd
            inline_452_mixes = inline_451_mixes
            inline_452_scale = inline_451_scale
            inline_452_base = inline_451_base
            inline_452_result = hc_gates_sinkhorn(
                inline_452_mixes, inline_452_scale, inline_452_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_452_mixes, inline_452_scale, inline_452_base
            inline_451_result = inline_452_result
            del inline_452_result
            del (
                inline_451_base,
                inline_451_fn,
                inline_451_hidden_states,
                inline_451_mixes,
                inline_451_rstd,
                inline_451_scale,
                inline_451_states,
            )
            block_attn_pre, block_post, block_comb = inline_451_result
            del inline_451_result
            inline_453_hidden_states = block_hidden_states
            inline_453_pre_mix = block_pre_mix
            inline_454_result = hc_pre_norm(
                inline_453_hidden_states, inline_453_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_453_hidden_states, inline_453_pre_mix
            block_attended = block_self.attn(inline_454_result, block_runtime)
            del inline_454_result
            inline_455_sublayer_output = block_attended
            inline_455_residual = block_hidden_states
            inline_455_post = block_post
            inline_455_comb = block_comb
            inline_455_result = hc_post(
                inline_455_sublayer_output, inline_455_residual, inline_455_post, inline_455_comb
            )
            del inline_455_sublayer_output, inline_455_residual, inline_455_post, inline_455_comb
            block_hidden_states = inline_455_result
            del inline_455_result
            inline_456_hidden_states = block_hidden_states
            inline_456_fn = block_self.hc_ffn_fn
            inline_456_scale = block_self.hc_ffn_scale
            inline_456_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_456_states = inline_456_hidden_states.flatten(-2).float()
            inline_456_rstd = torch.rsqrt(inline_456_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_456_mixes = F.linear(inline_456_states, inline_456_fn.float()) * inline_456_rstd
            inline_457_mixes = inline_456_mixes
            inline_457_scale = inline_456_scale
            inline_457_base = inline_456_base
            inline_457_result = hc_gates_sinkhorn(
                inline_457_mixes, inline_457_scale, inline_457_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_457_mixes, inline_457_scale, inline_457_base
            inline_456_result = inline_457_result
            del inline_457_result
            del (
                inline_456_base,
                inline_456_fn,
                inline_456_hidden_states,
                inline_456_mixes,
                inline_456_rstd,
                inline_456_scale,
                inline_456_states,
            )
            block_ffn_pre, block_post, block_comb = inline_456_result
            del inline_456_result
            inline_458_hidden_states = block_hidden_states
            inline_458_pre_mix = block_attn_pre
            inline_459_result = hc_pre_norm(
                inline_458_hidden_states, inline_458_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_458_hidden_states, inline_458_pre_mix
            block_ffn_input = inline_459_result
            del inline_459_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_460_sublayer_output = block_output
            inline_460_residual = block_hidden_states
            inline_460_post = block_post
            inline_460_comb = block_comb
            inline_460_result = hc_post(
                inline_460_sublayer_output, inline_460_residual, inline_460_post, inline_460_comb
            )
            del inline_460_sublayer_output, inline_460_residual, inline_460_post, inline_460_comb
            hidden_states, pre_mix = (inline_460_result, block_ffn_pre)
            del inline_460_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[23].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(23)
        hidden_states = self.layers[23].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 23, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[23]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_461_hidden_states = block_hidden_states
            inline_461_fn = block_self.hc_attn_fn
            inline_461_scale = block_self.hc_attn_scale
            inline_461_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_461_states = inline_461_hidden_states.flatten(-2).float()
            inline_461_rstd = torch.rsqrt(inline_461_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_461_mixes = F.linear(inline_461_states, inline_461_fn.float()) * inline_461_rstd
            inline_462_mixes = inline_461_mixes
            inline_462_scale = inline_461_scale
            inline_462_base = inline_461_base
            inline_462_result = hc_gates_sinkhorn(
                inline_462_mixes, inline_462_scale, inline_462_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_462_mixes, inline_462_scale, inline_462_base
            inline_461_result = inline_462_result
            del inline_462_result
            del (
                inline_461_base,
                inline_461_fn,
                inline_461_hidden_states,
                inline_461_mixes,
                inline_461_rstd,
                inline_461_scale,
                inline_461_states,
            )
            block_attn_pre, block_post, block_comb = inline_461_result
            del inline_461_result
            inline_463_hidden_states = block_hidden_states
            inline_463_pre_mix = block_pre_mix
            inline_464_result = hc_pre_norm(
                inline_463_hidden_states, inline_463_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_463_hidden_states, inline_463_pre_mix
            block_attended = block_self.attn(inline_464_result, block_runtime)
            del inline_464_result
            inline_465_sublayer_output = block_attended
            inline_465_residual = block_hidden_states
            inline_465_post = block_post
            inline_465_comb = block_comb
            inline_465_result = hc_post(
                inline_465_sublayer_output, inline_465_residual, inline_465_post, inline_465_comb
            )
            del inline_465_sublayer_output, inline_465_residual, inline_465_post, inline_465_comb
            block_hidden_states = inline_465_result
            del inline_465_result
            inline_466_hidden_states = block_hidden_states
            inline_466_fn = block_self.hc_ffn_fn
            inline_466_scale = block_self.hc_ffn_scale
            inline_466_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_466_states = inline_466_hidden_states.flatten(-2).float()
            inline_466_rstd = torch.rsqrt(inline_466_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_466_mixes = F.linear(inline_466_states, inline_466_fn.float()) * inline_466_rstd
            inline_467_mixes = inline_466_mixes
            inline_467_scale = inline_466_scale
            inline_467_base = inline_466_base
            inline_467_result = hc_gates_sinkhorn(
                inline_467_mixes, inline_467_scale, inline_467_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_467_mixes, inline_467_scale, inline_467_base
            inline_466_result = inline_467_result
            del inline_467_result
            del (
                inline_466_base,
                inline_466_fn,
                inline_466_hidden_states,
                inline_466_mixes,
                inline_466_rstd,
                inline_466_scale,
                inline_466_states,
            )
            block_ffn_pre, block_post, block_comb = inline_466_result
            del inline_466_result
            inline_468_hidden_states = block_hidden_states
            inline_468_pre_mix = block_attn_pre
            inline_469_result = hc_pre_norm(
                inline_468_hidden_states, inline_468_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_468_hidden_states, inline_468_pre_mix
            block_ffn_input = inline_469_result
            del inline_469_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_470_sublayer_output = block_output
            inline_470_residual = block_hidden_states
            inline_470_post = block_post
            inline_470_comb = block_comb
            inline_470_result = hc_post(
                inline_470_sublayer_output, inline_470_residual, inline_470_post, inline_470_comb
            )
            del inline_470_sublayer_output, inline_470_residual, inline_470_post, inline_470_comb
            hidden_states, pre_mix = (inline_470_result, block_ffn_pre)
            del inline_470_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[23]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_471_hidden_states = block_hidden_states
            inline_471_fn = block_self.hc_attn_fn
            inline_471_scale = block_self.hc_attn_scale
            inline_471_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_471_states = inline_471_hidden_states.flatten(-2).float()
            inline_471_rstd = torch.rsqrt(inline_471_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_471_mixes = F.linear(inline_471_states, inline_471_fn.float()) * inline_471_rstd
            inline_472_mixes = inline_471_mixes
            inline_472_scale = inline_471_scale
            inline_472_base = inline_471_base
            inline_472_result = hc_gates_sinkhorn(
                inline_472_mixes, inline_472_scale, inline_472_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_472_mixes, inline_472_scale, inline_472_base
            inline_471_result = inline_472_result
            del inline_472_result
            del (
                inline_471_base,
                inline_471_fn,
                inline_471_hidden_states,
                inline_471_mixes,
                inline_471_rstd,
                inline_471_scale,
                inline_471_states,
            )
            block_attn_pre, block_post, block_comb = inline_471_result
            del inline_471_result
            inline_473_hidden_states = block_hidden_states
            inline_473_pre_mix = block_pre_mix
            inline_474_result = hc_pre_norm(
                inline_473_hidden_states, inline_473_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_473_hidden_states, inline_473_pre_mix
            block_attended = block_self.attn(inline_474_result, block_runtime)
            del inline_474_result
            inline_475_sublayer_output = block_attended
            inline_475_residual = block_hidden_states
            inline_475_post = block_post
            inline_475_comb = block_comb
            inline_475_result = hc_post(
                inline_475_sublayer_output, inline_475_residual, inline_475_post, inline_475_comb
            )
            del inline_475_sublayer_output, inline_475_residual, inline_475_post, inline_475_comb
            block_hidden_states = inline_475_result
            del inline_475_result
            inline_476_hidden_states = block_hidden_states
            inline_476_fn = block_self.hc_ffn_fn
            inline_476_scale = block_self.hc_ffn_scale
            inline_476_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_476_states = inline_476_hidden_states.flatten(-2).float()
            inline_476_rstd = torch.rsqrt(inline_476_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_476_mixes = F.linear(inline_476_states, inline_476_fn.float()) * inline_476_rstd
            inline_477_mixes = inline_476_mixes
            inline_477_scale = inline_476_scale
            inline_477_base = inline_476_base
            inline_477_result = hc_gates_sinkhorn(
                inline_477_mixes, inline_477_scale, inline_477_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_477_mixes, inline_477_scale, inline_477_base
            inline_476_result = inline_477_result
            del inline_477_result
            del (
                inline_476_base,
                inline_476_fn,
                inline_476_hidden_states,
                inline_476_mixes,
                inline_476_rstd,
                inline_476_scale,
                inline_476_states,
            )
            block_ffn_pre, block_post, block_comb = inline_476_result
            del inline_476_result
            inline_478_hidden_states = block_hidden_states
            inline_478_pre_mix = block_attn_pre
            inline_479_result = hc_pre_norm(
                inline_478_hidden_states, inline_478_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_478_hidden_states, inline_478_pre_mix
            block_ffn_input = inline_479_result
            del inline_479_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_480_sublayer_output = block_output
            inline_480_residual = block_hidden_states
            inline_480_post = block_post
            inline_480_comb = block_comb
            inline_480_result = hc_post(
                inline_480_sublayer_output, inline_480_residual, inline_480_post, inline_480_comb
            )
            del inline_480_sublayer_output, inline_480_residual, inline_480_post, inline_480_comb
            hidden_states, pre_mix = (inline_480_result, block_ffn_pre)
            del inline_480_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[24].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(24)
        hidden_states = self.layers[24].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 24, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[24]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_481_hidden_states = block_hidden_states
            inline_481_fn = block_self.hc_attn_fn
            inline_481_scale = block_self.hc_attn_scale
            inline_481_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_481_states = inline_481_hidden_states.flatten(-2).float()
            inline_481_rstd = torch.rsqrt(inline_481_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_481_mixes = F.linear(inline_481_states, inline_481_fn.float()) * inline_481_rstd
            inline_482_mixes = inline_481_mixes
            inline_482_scale = inline_481_scale
            inline_482_base = inline_481_base
            inline_482_result = hc_gates_sinkhorn(
                inline_482_mixes, inline_482_scale, inline_482_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_482_mixes, inline_482_scale, inline_482_base
            inline_481_result = inline_482_result
            del inline_482_result
            del (
                inline_481_base,
                inline_481_fn,
                inline_481_hidden_states,
                inline_481_mixes,
                inline_481_rstd,
                inline_481_scale,
                inline_481_states,
            )
            block_attn_pre, block_post, block_comb = inline_481_result
            del inline_481_result
            inline_483_hidden_states = block_hidden_states
            inline_483_pre_mix = block_pre_mix
            inline_484_result = hc_pre_norm(
                inline_483_hidden_states, inline_483_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_483_hidden_states, inline_483_pre_mix
            block_attended = block_self.attn(inline_484_result, block_runtime)
            del inline_484_result
            inline_485_sublayer_output = block_attended
            inline_485_residual = block_hidden_states
            inline_485_post = block_post
            inline_485_comb = block_comb
            inline_485_result = hc_post(
                inline_485_sublayer_output, inline_485_residual, inline_485_post, inline_485_comb
            )
            del inline_485_sublayer_output, inline_485_residual, inline_485_post, inline_485_comb
            block_hidden_states = inline_485_result
            del inline_485_result
            inline_486_hidden_states = block_hidden_states
            inline_486_fn = block_self.hc_ffn_fn
            inline_486_scale = block_self.hc_ffn_scale
            inline_486_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_486_states = inline_486_hidden_states.flatten(-2).float()
            inline_486_rstd = torch.rsqrt(inline_486_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_486_mixes = F.linear(inline_486_states, inline_486_fn.float()) * inline_486_rstd
            inline_487_mixes = inline_486_mixes
            inline_487_scale = inline_486_scale
            inline_487_base = inline_486_base
            inline_487_result = hc_gates_sinkhorn(
                inline_487_mixes, inline_487_scale, inline_487_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_487_mixes, inline_487_scale, inline_487_base
            inline_486_result = inline_487_result
            del inline_487_result
            del (
                inline_486_base,
                inline_486_fn,
                inline_486_hidden_states,
                inline_486_mixes,
                inline_486_rstd,
                inline_486_scale,
                inline_486_states,
            )
            block_ffn_pre, block_post, block_comb = inline_486_result
            del inline_486_result
            inline_488_hidden_states = block_hidden_states
            inline_488_pre_mix = block_attn_pre
            inline_489_result = hc_pre_norm(
                inline_488_hidden_states, inline_488_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_488_hidden_states, inline_488_pre_mix
            block_ffn_input = inline_489_result
            del inline_489_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_490_sublayer_output = block_output
            inline_490_residual = block_hidden_states
            inline_490_post = block_post
            inline_490_comb = block_comb
            inline_490_result = hc_post(
                inline_490_sublayer_output, inline_490_residual, inline_490_post, inline_490_comb
            )
            del inline_490_sublayer_output, inline_490_residual, inline_490_post, inline_490_comb
            hidden_states, pre_mix = (inline_490_result, block_ffn_pre)
            del inline_490_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[24]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_491_hidden_states = block_hidden_states
            inline_491_fn = block_self.hc_attn_fn
            inline_491_scale = block_self.hc_attn_scale
            inline_491_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_491_states = inline_491_hidden_states.flatten(-2).float()
            inline_491_rstd = torch.rsqrt(inline_491_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_491_mixes = F.linear(inline_491_states, inline_491_fn.float()) * inline_491_rstd
            inline_492_mixes = inline_491_mixes
            inline_492_scale = inline_491_scale
            inline_492_base = inline_491_base
            inline_492_result = hc_gates_sinkhorn(
                inline_492_mixes, inline_492_scale, inline_492_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_492_mixes, inline_492_scale, inline_492_base
            inline_491_result = inline_492_result
            del inline_492_result
            del (
                inline_491_base,
                inline_491_fn,
                inline_491_hidden_states,
                inline_491_mixes,
                inline_491_rstd,
                inline_491_scale,
                inline_491_states,
            )
            block_attn_pre, block_post, block_comb = inline_491_result
            del inline_491_result
            inline_493_hidden_states = block_hidden_states
            inline_493_pre_mix = block_pre_mix
            inline_494_result = hc_pre_norm(
                inline_493_hidden_states, inline_493_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_493_hidden_states, inline_493_pre_mix
            block_attended = block_self.attn(inline_494_result, block_runtime)
            del inline_494_result
            inline_495_sublayer_output = block_attended
            inline_495_residual = block_hidden_states
            inline_495_post = block_post
            inline_495_comb = block_comb
            inline_495_result = hc_post(
                inline_495_sublayer_output, inline_495_residual, inline_495_post, inline_495_comb
            )
            del inline_495_sublayer_output, inline_495_residual, inline_495_post, inline_495_comb
            block_hidden_states = inline_495_result
            del inline_495_result
            inline_496_hidden_states = block_hidden_states
            inline_496_fn = block_self.hc_ffn_fn
            inline_496_scale = block_self.hc_ffn_scale
            inline_496_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_496_states = inline_496_hidden_states.flatten(-2).float()
            inline_496_rstd = torch.rsqrt(inline_496_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_496_mixes = F.linear(inline_496_states, inline_496_fn.float()) * inline_496_rstd
            inline_497_mixes = inline_496_mixes
            inline_497_scale = inline_496_scale
            inline_497_base = inline_496_base
            inline_497_result = hc_gates_sinkhorn(
                inline_497_mixes, inline_497_scale, inline_497_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_497_mixes, inline_497_scale, inline_497_base
            inline_496_result = inline_497_result
            del inline_497_result
            del (
                inline_496_base,
                inline_496_fn,
                inline_496_hidden_states,
                inline_496_mixes,
                inline_496_rstd,
                inline_496_scale,
                inline_496_states,
            )
            block_ffn_pre, block_post, block_comb = inline_496_result
            del inline_496_result
            inline_498_hidden_states = block_hidden_states
            inline_498_pre_mix = block_attn_pre
            inline_499_result = hc_pre_norm(
                inline_498_hidden_states, inline_498_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_498_hidden_states, inline_498_pre_mix
            block_ffn_input = inline_499_result
            del inline_499_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_500_sublayer_output = block_output
            inline_500_residual = block_hidden_states
            inline_500_post = block_post
            inline_500_comb = block_comb
            inline_500_result = hc_post(
                inline_500_sublayer_output, inline_500_residual, inline_500_post, inline_500_comb
            )
            del inline_500_sublayer_output, inline_500_residual, inline_500_post, inline_500_comb
            hidden_states, pre_mix = (inline_500_result, block_ffn_pre)
            del inline_500_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[25].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(25)
        hidden_states = self.layers[25].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 25, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[25]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_501_hidden_states = block_hidden_states
            inline_501_fn = block_self.hc_attn_fn
            inline_501_scale = block_self.hc_attn_scale
            inline_501_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_501_states = inline_501_hidden_states.flatten(-2).float()
            inline_501_rstd = torch.rsqrt(inline_501_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_501_mixes = F.linear(inline_501_states, inline_501_fn.float()) * inline_501_rstd
            inline_502_mixes = inline_501_mixes
            inline_502_scale = inline_501_scale
            inline_502_base = inline_501_base
            inline_502_result = hc_gates_sinkhorn(
                inline_502_mixes, inline_502_scale, inline_502_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_502_mixes, inline_502_scale, inline_502_base
            inline_501_result = inline_502_result
            del inline_502_result
            del (
                inline_501_base,
                inline_501_fn,
                inline_501_hidden_states,
                inline_501_mixes,
                inline_501_rstd,
                inline_501_scale,
                inline_501_states,
            )
            block_attn_pre, block_post, block_comb = inline_501_result
            del inline_501_result
            inline_503_hidden_states = block_hidden_states
            inline_503_pre_mix = block_pre_mix
            inline_504_result = hc_pre_norm(
                inline_503_hidden_states, inline_503_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_503_hidden_states, inline_503_pre_mix
            block_attended = block_self.attn(inline_504_result, block_runtime)
            del inline_504_result
            inline_505_sublayer_output = block_attended
            inline_505_residual = block_hidden_states
            inline_505_post = block_post
            inline_505_comb = block_comb
            inline_505_result = hc_post(
                inline_505_sublayer_output, inline_505_residual, inline_505_post, inline_505_comb
            )
            del inline_505_sublayer_output, inline_505_residual, inline_505_post, inline_505_comb
            block_hidden_states = inline_505_result
            del inline_505_result
            inline_506_hidden_states = block_hidden_states
            inline_506_fn = block_self.hc_ffn_fn
            inline_506_scale = block_self.hc_ffn_scale
            inline_506_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_506_states = inline_506_hidden_states.flatten(-2).float()
            inline_506_rstd = torch.rsqrt(inline_506_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_506_mixes = F.linear(inline_506_states, inline_506_fn.float()) * inline_506_rstd
            inline_507_mixes = inline_506_mixes
            inline_507_scale = inline_506_scale
            inline_507_base = inline_506_base
            inline_507_result = hc_gates_sinkhorn(
                inline_507_mixes, inline_507_scale, inline_507_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_507_mixes, inline_507_scale, inline_507_base
            inline_506_result = inline_507_result
            del inline_507_result
            del (
                inline_506_base,
                inline_506_fn,
                inline_506_hidden_states,
                inline_506_mixes,
                inline_506_rstd,
                inline_506_scale,
                inline_506_states,
            )
            block_ffn_pre, block_post, block_comb = inline_506_result
            del inline_506_result
            inline_508_hidden_states = block_hidden_states
            inline_508_pre_mix = block_attn_pre
            inline_509_result = hc_pre_norm(
                inline_508_hidden_states, inline_508_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_508_hidden_states, inline_508_pre_mix
            block_ffn_input = inline_509_result
            del inline_509_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_510_sublayer_output = block_output
            inline_510_residual = block_hidden_states
            inline_510_post = block_post
            inline_510_comb = block_comb
            inline_510_result = hc_post(
                inline_510_sublayer_output, inline_510_residual, inline_510_post, inline_510_comb
            )
            del inline_510_sublayer_output, inline_510_residual, inline_510_post, inline_510_comb
            hidden_states, pre_mix = (inline_510_result, block_ffn_pre)
            del inline_510_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[25]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_511_hidden_states = block_hidden_states
            inline_511_fn = block_self.hc_attn_fn
            inline_511_scale = block_self.hc_attn_scale
            inline_511_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_511_states = inline_511_hidden_states.flatten(-2).float()
            inline_511_rstd = torch.rsqrt(inline_511_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_511_mixes = F.linear(inline_511_states, inline_511_fn.float()) * inline_511_rstd
            inline_512_mixes = inline_511_mixes
            inline_512_scale = inline_511_scale
            inline_512_base = inline_511_base
            inline_512_result = hc_gates_sinkhorn(
                inline_512_mixes, inline_512_scale, inline_512_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_512_mixes, inline_512_scale, inline_512_base
            inline_511_result = inline_512_result
            del inline_512_result
            del (
                inline_511_base,
                inline_511_fn,
                inline_511_hidden_states,
                inline_511_mixes,
                inline_511_rstd,
                inline_511_scale,
                inline_511_states,
            )
            block_attn_pre, block_post, block_comb = inline_511_result
            del inline_511_result
            inline_513_hidden_states = block_hidden_states
            inline_513_pre_mix = block_pre_mix
            inline_514_result = hc_pre_norm(
                inline_513_hidden_states, inline_513_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_513_hidden_states, inline_513_pre_mix
            block_attended = block_self.attn(inline_514_result, block_runtime)
            del inline_514_result
            inline_515_sublayer_output = block_attended
            inline_515_residual = block_hidden_states
            inline_515_post = block_post
            inline_515_comb = block_comb
            inline_515_result = hc_post(
                inline_515_sublayer_output, inline_515_residual, inline_515_post, inline_515_comb
            )
            del inline_515_sublayer_output, inline_515_residual, inline_515_post, inline_515_comb
            block_hidden_states = inline_515_result
            del inline_515_result
            inline_516_hidden_states = block_hidden_states
            inline_516_fn = block_self.hc_ffn_fn
            inline_516_scale = block_self.hc_ffn_scale
            inline_516_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_516_states = inline_516_hidden_states.flatten(-2).float()
            inline_516_rstd = torch.rsqrt(inline_516_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_516_mixes = F.linear(inline_516_states, inline_516_fn.float()) * inline_516_rstd
            inline_517_mixes = inline_516_mixes
            inline_517_scale = inline_516_scale
            inline_517_base = inline_516_base
            inline_517_result = hc_gates_sinkhorn(
                inline_517_mixes, inline_517_scale, inline_517_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_517_mixes, inline_517_scale, inline_517_base
            inline_516_result = inline_517_result
            del inline_517_result
            del (
                inline_516_base,
                inline_516_fn,
                inline_516_hidden_states,
                inline_516_mixes,
                inline_516_rstd,
                inline_516_scale,
                inline_516_states,
            )
            block_ffn_pre, block_post, block_comb = inline_516_result
            del inline_516_result
            inline_518_hidden_states = block_hidden_states
            inline_518_pre_mix = block_attn_pre
            inline_519_result = hc_pre_norm(
                inline_518_hidden_states, inline_518_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_518_hidden_states, inline_518_pre_mix
            block_ffn_input = inline_519_result
            del inline_519_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_520_sublayer_output = block_output
            inline_520_residual = block_hidden_states
            inline_520_post = block_post
            inline_520_comb = block_comb
            inline_520_result = hc_post(
                inline_520_sublayer_output, inline_520_residual, inline_520_post, inline_520_comb
            )
            del inline_520_sublayer_output, inline_520_residual, inline_520_post, inline_520_comb
            hidden_states, pre_mix = (inline_520_result, block_ffn_pre)
            del inline_520_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[26].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(26)
        hidden_states = self.layers[26].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 26, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[26]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_521_hidden_states = block_hidden_states
            inline_521_fn = block_self.hc_attn_fn
            inline_521_scale = block_self.hc_attn_scale
            inline_521_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_521_states = inline_521_hidden_states.flatten(-2).float()
            inline_521_rstd = torch.rsqrt(inline_521_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_521_mixes = F.linear(inline_521_states, inline_521_fn.float()) * inline_521_rstd
            inline_522_mixes = inline_521_mixes
            inline_522_scale = inline_521_scale
            inline_522_base = inline_521_base
            inline_522_result = hc_gates_sinkhorn(
                inline_522_mixes, inline_522_scale, inline_522_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_522_mixes, inline_522_scale, inline_522_base
            inline_521_result = inline_522_result
            del inline_522_result
            del (
                inline_521_base,
                inline_521_fn,
                inline_521_hidden_states,
                inline_521_mixes,
                inline_521_rstd,
                inline_521_scale,
                inline_521_states,
            )
            block_attn_pre, block_post, block_comb = inline_521_result
            del inline_521_result
            inline_523_hidden_states = block_hidden_states
            inline_523_pre_mix = block_pre_mix
            inline_524_result = hc_pre_norm(
                inline_523_hidden_states, inline_523_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_523_hidden_states, inline_523_pre_mix
            block_attended = block_self.attn(inline_524_result, block_runtime)
            del inline_524_result
            inline_525_sublayer_output = block_attended
            inline_525_residual = block_hidden_states
            inline_525_post = block_post
            inline_525_comb = block_comb
            inline_525_result = hc_post(
                inline_525_sublayer_output, inline_525_residual, inline_525_post, inline_525_comb
            )
            del inline_525_sublayer_output, inline_525_residual, inline_525_post, inline_525_comb
            block_hidden_states = inline_525_result
            del inline_525_result
            inline_526_hidden_states = block_hidden_states
            inline_526_fn = block_self.hc_ffn_fn
            inline_526_scale = block_self.hc_ffn_scale
            inline_526_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_526_states = inline_526_hidden_states.flatten(-2).float()
            inline_526_rstd = torch.rsqrt(inline_526_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_526_mixes = F.linear(inline_526_states, inline_526_fn.float()) * inline_526_rstd
            inline_527_mixes = inline_526_mixes
            inline_527_scale = inline_526_scale
            inline_527_base = inline_526_base
            inline_527_result = hc_gates_sinkhorn(
                inline_527_mixes, inline_527_scale, inline_527_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_527_mixes, inline_527_scale, inline_527_base
            inline_526_result = inline_527_result
            del inline_527_result
            del (
                inline_526_base,
                inline_526_fn,
                inline_526_hidden_states,
                inline_526_mixes,
                inline_526_rstd,
                inline_526_scale,
                inline_526_states,
            )
            block_ffn_pre, block_post, block_comb = inline_526_result
            del inline_526_result
            inline_528_hidden_states = block_hidden_states
            inline_528_pre_mix = block_attn_pre
            inline_529_result = hc_pre_norm(
                inline_528_hidden_states, inline_528_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_528_hidden_states, inline_528_pre_mix
            block_ffn_input = inline_529_result
            del inline_529_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_530_sublayer_output = block_output
            inline_530_residual = block_hidden_states
            inline_530_post = block_post
            inline_530_comb = block_comb
            inline_530_result = hc_post(
                inline_530_sublayer_output, inline_530_residual, inline_530_post, inline_530_comb
            )
            del inline_530_sublayer_output, inline_530_residual, inline_530_post, inline_530_comb
            hidden_states, pre_mix = (inline_530_result, block_ffn_pre)
            del inline_530_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[26]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_531_hidden_states = block_hidden_states
            inline_531_fn = block_self.hc_attn_fn
            inline_531_scale = block_self.hc_attn_scale
            inline_531_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_531_states = inline_531_hidden_states.flatten(-2).float()
            inline_531_rstd = torch.rsqrt(inline_531_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_531_mixes = F.linear(inline_531_states, inline_531_fn.float()) * inline_531_rstd
            inline_532_mixes = inline_531_mixes
            inline_532_scale = inline_531_scale
            inline_532_base = inline_531_base
            inline_532_result = hc_gates_sinkhorn(
                inline_532_mixes, inline_532_scale, inline_532_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_532_mixes, inline_532_scale, inline_532_base
            inline_531_result = inline_532_result
            del inline_532_result
            del (
                inline_531_base,
                inline_531_fn,
                inline_531_hidden_states,
                inline_531_mixes,
                inline_531_rstd,
                inline_531_scale,
                inline_531_states,
            )
            block_attn_pre, block_post, block_comb = inline_531_result
            del inline_531_result
            inline_533_hidden_states = block_hidden_states
            inline_533_pre_mix = block_pre_mix
            inline_534_result = hc_pre_norm(
                inline_533_hidden_states, inline_533_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_533_hidden_states, inline_533_pre_mix
            block_attended = block_self.attn(inline_534_result, block_runtime)
            del inline_534_result
            inline_535_sublayer_output = block_attended
            inline_535_residual = block_hidden_states
            inline_535_post = block_post
            inline_535_comb = block_comb
            inline_535_result = hc_post(
                inline_535_sublayer_output, inline_535_residual, inline_535_post, inline_535_comb
            )
            del inline_535_sublayer_output, inline_535_residual, inline_535_post, inline_535_comb
            block_hidden_states = inline_535_result
            del inline_535_result
            inline_536_hidden_states = block_hidden_states
            inline_536_fn = block_self.hc_ffn_fn
            inline_536_scale = block_self.hc_ffn_scale
            inline_536_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_536_states = inline_536_hidden_states.flatten(-2).float()
            inline_536_rstd = torch.rsqrt(inline_536_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_536_mixes = F.linear(inline_536_states, inline_536_fn.float()) * inline_536_rstd
            inline_537_mixes = inline_536_mixes
            inline_537_scale = inline_536_scale
            inline_537_base = inline_536_base
            inline_537_result = hc_gates_sinkhorn(
                inline_537_mixes, inline_537_scale, inline_537_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_537_mixes, inline_537_scale, inline_537_base
            inline_536_result = inline_537_result
            del inline_537_result
            del (
                inline_536_base,
                inline_536_fn,
                inline_536_hidden_states,
                inline_536_mixes,
                inline_536_rstd,
                inline_536_scale,
                inline_536_states,
            )
            block_ffn_pre, block_post, block_comb = inline_536_result
            del inline_536_result
            inline_538_hidden_states = block_hidden_states
            inline_538_pre_mix = block_attn_pre
            inline_539_result = hc_pre_norm(
                inline_538_hidden_states, inline_538_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_538_hidden_states, inline_538_pre_mix
            block_ffn_input = inline_539_result
            del inline_539_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_540_sublayer_output = block_output
            inline_540_residual = block_hidden_states
            inline_540_post = block_post
            inline_540_comb = block_comb
            inline_540_result = hc_post(
                inline_540_sublayer_output, inline_540_residual, inline_540_post, inline_540_comb
            )
            del inline_540_sublayer_output, inline_540_residual, inline_540_post, inline_540_comb
            hidden_states, pre_mix = (inline_540_result, block_ffn_pre)
            del inline_540_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[27].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(27)
        hidden_states = self.layers[27].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 27, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[27]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_541_hidden_states = block_hidden_states
            inline_541_fn = block_self.hc_attn_fn
            inline_541_scale = block_self.hc_attn_scale
            inline_541_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_541_states = inline_541_hidden_states.flatten(-2).float()
            inline_541_rstd = torch.rsqrt(inline_541_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_541_mixes = F.linear(inline_541_states, inline_541_fn.float()) * inline_541_rstd
            inline_542_mixes = inline_541_mixes
            inline_542_scale = inline_541_scale
            inline_542_base = inline_541_base
            inline_542_result = hc_gates_sinkhorn(
                inline_542_mixes, inline_542_scale, inline_542_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_542_mixes, inline_542_scale, inline_542_base
            inline_541_result = inline_542_result
            del inline_542_result
            del (
                inline_541_base,
                inline_541_fn,
                inline_541_hidden_states,
                inline_541_mixes,
                inline_541_rstd,
                inline_541_scale,
                inline_541_states,
            )
            block_attn_pre, block_post, block_comb = inline_541_result
            del inline_541_result
            inline_543_hidden_states = block_hidden_states
            inline_543_pre_mix = block_pre_mix
            inline_544_result = hc_pre_norm(
                inline_543_hidden_states, inline_543_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_543_hidden_states, inline_543_pre_mix
            block_attended = block_self.attn(inline_544_result, block_runtime)
            del inline_544_result
            inline_545_sublayer_output = block_attended
            inline_545_residual = block_hidden_states
            inline_545_post = block_post
            inline_545_comb = block_comb
            inline_545_result = hc_post(
                inline_545_sublayer_output, inline_545_residual, inline_545_post, inline_545_comb
            )
            del inline_545_sublayer_output, inline_545_residual, inline_545_post, inline_545_comb
            block_hidden_states = inline_545_result
            del inline_545_result
            inline_546_hidden_states = block_hidden_states
            inline_546_fn = block_self.hc_ffn_fn
            inline_546_scale = block_self.hc_ffn_scale
            inline_546_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_546_states = inline_546_hidden_states.flatten(-2).float()
            inline_546_rstd = torch.rsqrt(inline_546_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_546_mixes = F.linear(inline_546_states, inline_546_fn.float()) * inline_546_rstd
            inline_547_mixes = inline_546_mixes
            inline_547_scale = inline_546_scale
            inline_547_base = inline_546_base
            inline_547_result = hc_gates_sinkhorn(
                inline_547_mixes, inline_547_scale, inline_547_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_547_mixes, inline_547_scale, inline_547_base
            inline_546_result = inline_547_result
            del inline_547_result
            del (
                inline_546_base,
                inline_546_fn,
                inline_546_hidden_states,
                inline_546_mixes,
                inline_546_rstd,
                inline_546_scale,
                inline_546_states,
            )
            block_ffn_pre, block_post, block_comb = inline_546_result
            del inline_546_result
            inline_548_hidden_states = block_hidden_states
            inline_548_pre_mix = block_attn_pre
            inline_549_result = hc_pre_norm(
                inline_548_hidden_states, inline_548_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_548_hidden_states, inline_548_pre_mix
            block_ffn_input = inline_549_result
            del inline_549_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_550_sublayer_output = block_output
            inline_550_residual = block_hidden_states
            inline_550_post = block_post
            inline_550_comb = block_comb
            inline_550_result = hc_post(
                inline_550_sublayer_output, inline_550_residual, inline_550_post, inline_550_comb
            )
            del inline_550_sublayer_output, inline_550_residual, inline_550_post, inline_550_comb
            hidden_states, pre_mix = (inline_550_result, block_ffn_pre)
            del inline_550_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[27]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_551_hidden_states = block_hidden_states
            inline_551_fn = block_self.hc_attn_fn
            inline_551_scale = block_self.hc_attn_scale
            inline_551_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_551_states = inline_551_hidden_states.flatten(-2).float()
            inline_551_rstd = torch.rsqrt(inline_551_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_551_mixes = F.linear(inline_551_states, inline_551_fn.float()) * inline_551_rstd
            inline_552_mixes = inline_551_mixes
            inline_552_scale = inline_551_scale
            inline_552_base = inline_551_base
            inline_552_result = hc_gates_sinkhorn(
                inline_552_mixes, inline_552_scale, inline_552_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_552_mixes, inline_552_scale, inline_552_base
            inline_551_result = inline_552_result
            del inline_552_result
            del (
                inline_551_base,
                inline_551_fn,
                inline_551_hidden_states,
                inline_551_mixes,
                inline_551_rstd,
                inline_551_scale,
                inline_551_states,
            )
            block_attn_pre, block_post, block_comb = inline_551_result
            del inline_551_result
            inline_553_hidden_states = block_hidden_states
            inline_553_pre_mix = block_pre_mix
            inline_554_result = hc_pre_norm(
                inline_553_hidden_states, inline_553_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_553_hidden_states, inline_553_pre_mix
            block_attended = block_self.attn(inline_554_result, block_runtime)
            del inline_554_result
            inline_555_sublayer_output = block_attended
            inline_555_residual = block_hidden_states
            inline_555_post = block_post
            inline_555_comb = block_comb
            inline_555_result = hc_post(
                inline_555_sublayer_output, inline_555_residual, inline_555_post, inline_555_comb
            )
            del inline_555_sublayer_output, inline_555_residual, inline_555_post, inline_555_comb
            block_hidden_states = inline_555_result
            del inline_555_result
            inline_556_hidden_states = block_hidden_states
            inline_556_fn = block_self.hc_ffn_fn
            inline_556_scale = block_self.hc_ffn_scale
            inline_556_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_556_states = inline_556_hidden_states.flatten(-2).float()
            inline_556_rstd = torch.rsqrt(inline_556_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_556_mixes = F.linear(inline_556_states, inline_556_fn.float()) * inline_556_rstd
            inline_557_mixes = inline_556_mixes
            inline_557_scale = inline_556_scale
            inline_557_base = inline_556_base
            inline_557_result = hc_gates_sinkhorn(
                inline_557_mixes, inline_557_scale, inline_557_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_557_mixes, inline_557_scale, inline_557_base
            inline_556_result = inline_557_result
            del inline_557_result
            del (
                inline_556_base,
                inline_556_fn,
                inline_556_hidden_states,
                inline_556_mixes,
                inline_556_rstd,
                inline_556_scale,
                inline_556_states,
            )
            block_ffn_pre, block_post, block_comb = inline_556_result
            del inline_556_result
            inline_558_hidden_states = block_hidden_states
            inline_558_pre_mix = block_attn_pre
            inline_559_result = hc_pre_norm(
                inline_558_hidden_states, inline_558_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_558_hidden_states, inline_558_pre_mix
            block_ffn_input = inline_559_result
            del inline_559_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_560_sublayer_output = block_output
            inline_560_residual = block_hidden_states
            inline_560_post = block_post
            inline_560_comb = block_comb
            inline_560_result = hc_post(
                inline_560_sublayer_output, inline_560_residual, inline_560_post, inline_560_comb
            )
            del inline_560_sublayer_output, inline_560_residual, inline_560_post, inline_560_comb
            hidden_states, pre_mix = (inline_560_result, block_ffn_pre)
            del inline_560_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[28].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(28)
        hidden_states = self.layers[28].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 28, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[28]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_561_hidden_states = block_hidden_states
            inline_561_fn = block_self.hc_attn_fn
            inline_561_scale = block_self.hc_attn_scale
            inline_561_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_561_states = inline_561_hidden_states.flatten(-2).float()
            inline_561_rstd = torch.rsqrt(inline_561_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_561_mixes = F.linear(inline_561_states, inline_561_fn.float()) * inline_561_rstd
            inline_562_mixes = inline_561_mixes
            inline_562_scale = inline_561_scale
            inline_562_base = inline_561_base
            inline_562_result = hc_gates_sinkhorn(
                inline_562_mixes, inline_562_scale, inline_562_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_562_mixes, inline_562_scale, inline_562_base
            inline_561_result = inline_562_result
            del inline_562_result
            del (
                inline_561_base,
                inline_561_fn,
                inline_561_hidden_states,
                inline_561_mixes,
                inline_561_rstd,
                inline_561_scale,
                inline_561_states,
            )
            block_attn_pre, block_post, block_comb = inline_561_result
            del inline_561_result
            inline_563_hidden_states = block_hidden_states
            inline_563_pre_mix = block_pre_mix
            inline_564_result = hc_pre_norm(
                inline_563_hidden_states, inline_563_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_563_hidden_states, inline_563_pre_mix
            block_attended = block_self.attn(inline_564_result, block_runtime)
            del inline_564_result
            inline_565_sublayer_output = block_attended
            inline_565_residual = block_hidden_states
            inline_565_post = block_post
            inline_565_comb = block_comb
            inline_565_result = hc_post(
                inline_565_sublayer_output, inline_565_residual, inline_565_post, inline_565_comb
            )
            del inline_565_sublayer_output, inline_565_residual, inline_565_post, inline_565_comb
            block_hidden_states = inline_565_result
            del inline_565_result
            inline_566_hidden_states = block_hidden_states
            inline_566_fn = block_self.hc_ffn_fn
            inline_566_scale = block_self.hc_ffn_scale
            inline_566_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_566_states = inline_566_hidden_states.flatten(-2).float()
            inline_566_rstd = torch.rsqrt(inline_566_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_566_mixes = F.linear(inline_566_states, inline_566_fn.float()) * inline_566_rstd
            inline_567_mixes = inline_566_mixes
            inline_567_scale = inline_566_scale
            inline_567_base = inline_566_base
            inline_567_result = hc_gates_sinkhorn(
                inline_567_mixes, inline_567_scale, inline_567_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_567_mixes, inline_567_scale, inline_567_base
            inline_566_result = inline_567_result
            del inline_567_result
            del (
                inline_566_base,
                inline_566_fn,
                inline_566_hidden_states,
                inline_566_mixes,
                inline_566_rstd,
                inline_566_scale,
                inline_566_states,
            )
            block_ffn_pre, block_post, block_comb = inline_566_result
            del inline_566_result
            inline_568_hidden_states = block_hidden_states
            inline_568_pre_mix = block_attn_pre
            inline_569_result = hc_pre_norm(
                inline_568_hidden_states, inline_568_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_568_hidden_states, inline_568_pre_mix
            block_ffn_input = inline_569_result
            del inline_569_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_570_sublayer_output = block_output
            inline_570_residual = block_hidden_states
            inline_570_post = block_post
            inline_570_comb = block_comb
            inline_570_result = hc_post(
                inline_570_sublayer_output, inline_570_residual, inline_570_post, inline_570_comb
            )
            del inline_570_sublayer_output, inline_570_residual, inline_570_post, inline_570_comb
            hidden_states, pre_mix = (inline_570_result, block_ffn_pre)
            del inline_570_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[28]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_571_hidden_states = block_hidden_states
            inline_571_fn = block_self.hc_attn_fn
            inline_571_scale = block_self.hc_attn_scale
            inline_571_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_571_states = inline_571_hidden_states.flatten(-2).float()
            inline_571_rstd = torch.rsqrt(inline_571_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_571_mixes = F.linear(inline_571_states, inline_571_fn.float()) * inline_571_rstd
            inline_572_mixes = inline_571_mixes
            inline_572_scale = inline_571_scale
            inline_572_base = inline_571_base
            inline_572_result = hc_gates_sinkhorn(
                inline_572_mixes, inline_572_scale, inline_572_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_572_mixes, inline_572_scale, inline_572_base
            inline_571_result = inline_572_result
            del inline_572_result
            del (
                inline_571_base,
                inline_571_fn,
                inline_571_hidden_states,
                inline_571_mixes,
                inline_571_rstd,
                inline_571_scale,
                inline_571_states,
            )
            block_attn_pre, block_post, block_comb = inline_571_result
            del inline_571_result
            inline_573_hidden_states = block_hidden_states
            inline_573_pre_mix = block_pre_mix
            inline_574_result = hc_pre_norm(
                inline_573_hidden_states, inline_573_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_573_hidden_states, inline_573_pre_mix
            block_attended = block_self.attn(inline_574_result, block_runtime)
            del inline_574_result
            inline_575_sublayer_output = block_attended
            inline_575_residual = block_hidden_states
            inline_575_post = block_post
            inline_575_comb = block_comb
            inline_575_result = hc_post(
                inline_575_sublayer_output, inline_575_residual, inline_575_post, inline_575_comb
            )
            del inline_575_sublayer_output, inline_575_residual, inline_575_post, inline_575_comb
            block_hidden_states = inline_575_result
            del inline_575_result
            inline_576_hidden_states = block_hidden_states
            inline_576_fn = block_self.hc_ffn_fn
            inline_576_scale = block_self.hc_ffn_scale
            inline_576_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_576_states = inline_576_hidden_states.flatten(-2).float()
            inline_576_rstd = torch.rsqrt(inline_576_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_576_mixes = F.linear(inline_576_states, inline_576_fn.float()) * inline_576_rstd
            inline_577_mixes = inline_576_mixes
            inline_577_scale = inline_576_scale
            inline_577_base = inline_576_base
            inline_577_result = hc_gates_sinkhorn(
                inline_577_mixes, inline_577_scale, inline_577_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_577_mixes, inline_577_scale, inline_577_base
            inline_576_result = inline_577_result
            del inline_577_result
            del (
                inline_576_base,
                inline_576_fn,
                inline_576_hidden_states,
                inline_576_mixes,
                inline_576_rstd,
                inline_576_scale,
                inline_576_states,
            )
            block_ffn_pre, block_post, block_comb = inline_576_result
            del inline_576_result
            inline_578_hidden_states = block_hidden_states
            inline_578_pre_mix = block_attn_pre
            inline_579_result = hc_pre_norm(
                inline_578_hidden_states, inline_578_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_578_hidden_states, inline_578_pre_mix
            block_ffn_input = inline_579_result
            del inline_579_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_580_sublayer_output = block_output
            inline_580_residual = block_hidden_states
            inline_580_post = block_post
            inline_580_comb = block_comb
            inline_580_result = hc_post(
                inline_580_sublayer_output, inline_580_residual, inline_580_post, inline_580_comb
            )
            del inline_580_sublayer_output, inline_580_residual, inline_580_post, inline_580_comb
            hidden_states, pre_mix = (inline_580_result, block_ffn_pre)
            del inline_580_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[29].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(29)
        hidden_states = self.layers[29].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 29, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[29]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_581_hidden_states = block_hidden_states
            inline_581_fn = block_self.hc_attn_fn
            inline_581_scale = block_self.hc_attn_scale
            inline_581_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_581_states = inline_581_hidden_states.flatten(-2).float()
            inline_581_rstd = torch.rsqrt(inline_581_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_581_mixes = F.linear(inline_581_states, inline_581_fn.float()) * inline_581_rstd
            inline_582_mixes = inline_581_mixes
            inline_582_scale = inline_581_scale
            inline_582_base = inline_581_base
            inline_582_result = hc_gates_sinkhorn(
                inline_582_mixes, inline_582_scale, inline_582_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_582_mixes, inline_582_scale, inline_582_base
            inline_581_result = inline_582_result
            del inline_582_result
            del (
                inline_581_base,
                inline_581_fn,
                inline_581_hidden_states,
                inline_581_mixes,
                inline_581_rstd,
                inline_581_scale,
                inline_581_states,
            )
            block_attn_pre, block_post, block_comb = inline_581_result
            del inline_581_result
            inline_583_hidden_states = block_hidden_states
            inline_583_pre_mix = block_pre_mix
            inline_584_result = hc_pre_norm(
                inline_583_hidden_states, inline_583_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_583_hidden_states, inline_583_pre_mix
            block_attended = block_self.attn(inline_584_result, block_runtime)
            del inline_584_result
            inline_585_sublayer_output = block_attended
            inline_585_residual = block_hidden_states
            inline_585_post = block_post
            inline_585_comb = block_comb
            inline_585_result = hc_post(
                inline_585_sublayer_output, inline_585_residual, inline_585_post, inline_585_comb
            )
            del inline_585_sublayer_output, inline_585_residual, inline_585_post, inline_585_comb
            block_hidden_states = inline_585_result
            del inline_585_result
            inline_586_hidden_states = block_hidden_states
            inline_586_fn = block_self.hc_ffn_fn
            inline_586_scale = block_self.hc_ffn_scale
            inline_586_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_586_states = inline_586_hidden_states.flatten(-2).float()
            inline_586_rstd = torch.rsqrt(inline_586_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_586_mixes = F.linear(inline_586_states, inline_586_fn.float()) * inline_586_rstd
            inline_587_mixes = inline_586_mixes
            inline_587_scale = inline_586_scale
            inline_587_base = inline_586_base
            inline_587_result = hc_gates_sinkhorn(
                inline_587_mixes, inline_587_scale, inline_587_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_587_mixes, inline_587_scale, inline_587_base
            inline_586_result = inline_587_result
            del inline_587_result
            del (
                inline_586_base,
                inline_586_fn,
                inline_586_hidden_states,
                inline_586_mixes,
                inline_586_rstd,
                inline_586_scale,
                inline_586_states,
            )
            block_ffn_pre, block_post, block_comb = inline_586_result
            del inline_586_result
            inline_588_hidden_states = block_hidden_states
            inline_588_pre_mix = block_attn_pre
            inline_589_result = hc_pre_norm(
                inline_588_hidden_states, inline_588_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_588_hidden_states, inline_588_pre_mix
            block_ffn_input = inline_589_result
            del inline_589_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_590_sublayer_output = block_output
            inline_590_residual = block_hidden_states
            inline_590_post = block_post
            inline_590_comb = block_comb
            inline_590_result = hc_post(
                inline_590_sublayer_output, inline_590_residual, inline_590_post, inline_590_comb
            )
            del inline_590_sublayer_output, inline_590_residual, inline_590_post, inline_590_comb
            hidden_states, pre_mix = (inline_590_result, block_ffn_pre)
            del inline_590_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[29]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_591_hidden_states = block_hidden_states
            inline_591_fn = block_self.hc_attn_fn
            inline_591_scale = block_self.hc_attn_scale
            inline_591_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_591_states = inline_591_hidden_states.flatten(-2).float()
            inline_591_rstd = torch.rsqrt(inline_591_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_591_mixes = F.linear(inline_591_states, inline_591_fn.float()) * inline_591_rstd
            inline_592_mixes = inline_591_mixes
            inline_592_scale = inline_591_scale
            inline_592_base = inline_591_base
            inline_592_result = hc_gates_sinkhorn(
                inline_592_mixes, inline_592_scale, inline_592_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_592_mixes, inline_592_scale, inline_592_base
            inline_591_result = inline_592_result
            del inline_592_result
            del (
                inline_591_base,
                inline_591_fn,
                inline_591_hidden_states,
                inline_591_mixes,
                inline_591_rstd,
                inline_591_scale,
                inline_591_states,
            )
            block_attn_pre, block_post, block_comb = inline_591_result
            del inline_591_result
            inline_593_hidden_states = block_hidden_states
            inline_593_pre_mix = block_pre_mix
            inline_594_result = hc_pre_norm(
                inline_593_hidden_states, inline_593_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_593_hidden_states, inline_593_pre_mix
            block_attended = block_self.attn(inline_594_result, block_runtime)
            del inline_594_result
            inline_595_sublayer_output = block_attended
            inline_595_residual = block_hidden_states
            inline_595_post = block_post
            inline_595_comb = block_comb
            inline_595_result = hc_post(
                inline_595_sublayer_output, inline_595_residual, inline_595_post, inline_595_comb
            )
            del inline_595_sublayer_output, inline_595_residual, inline_595_post, inline_595_comb
            block_hidden_states = inline_595_result
            del inline_595_result
            inline_596_hidden_states = block_hidden_states
            inline_596_fn = block_self.hc_ffn_fn
            inline_596_scale = block_self.hc_ffn_scale
            inline_596_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_596_states = inline_596_hidden_states.flatten(-2).float()
            inline_596_rstd = torch.rsqrt(inline_596_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_596_mixes = F.linear(inline_596_states, inline_596_fn.float()) * inline_596_rstd
            inline_597_mixes = inline_596_mixes
            inline_597_scale = inline_596_scale
            inline_597_base = inline_596_base
            inline_597_result = hc_gates_sinkhorn(
                inline_597_mixes, inline_597_scale, inline_597_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_597_mixes, inline_597_scale, inline_597_base
            inline_596_result = inline_597_result
            del inline_597_result
            del (
                inline_596_base,
                inline_596_fn,
                inline_596_hidden_states,
                inline_596_mixes,
                inline_596_rstd,
                inline_596_scale,
                inline_596_states,
            )
            block_ffn_pre, block_post, block_comb = inline_596_result
            del inline_596_result
            inline_598_hidden_states = block_hidden_states
            inline_598_pre_mix = block_attn_pre
            inline_599_result = hc_pre_norm(
                inline_598_hidden_states, inline_598_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_598_hidden_states, inline_598_pre_mix
            block_ffn_input = inline_599_result
            del inline_599_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_600_sublayer_output = block_output
            inline_600_residual = block_hidden_states
            inline_600_post = block_post
            inline_600_comb = block_comb
            inline_600_result = hc_post(
                inline_600_sublayer_output, inline_600_residual, inline_600_post, inline_600_comb
            )
            del inline_600_sublayer_output, inline_600_residual, inline_600_post, inline_600_comb
            hidden_states, pre_mix = (inline_600_result, block_ffn_pre)
            del inline_600_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[30].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(30)
        hidden_states = self.layers[30].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 30, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[30]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_601_hidden_states = block_hidden_states
            inline_601_fn = block_self.hc_attn_fn
            inline_601_scale = block_self.hc_attn_scale
            inline_601_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_601_states = inline_601_hidden_states.flatten(-2).float()
            inline_601_rstd = torch.rsqrt(inline_601_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_601_mixes = F.linear(inline_601_states, inline_601_fn.float()) * inline_601_rstd
            inline_602_mixes = inline_601_mixes
            inline_602_scale = inline_601_scale
            inline_602_base = inline_601_base
            inline_602_result = hc_gates_sinkhorn(
                inline_602_mixes, inline_602_scale, inline_602_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_602_mixes, inline_602_scale, inline_602_base
            inline_601_result = inline_602_result
            del inline_602_result
            del (
                inline_601_base,
                inline_601_fn,
                inline_601_hidden_states,
                inline_601_mixes,
                inline_601_rstd,
                inline_601_scale,
                inline_601_states,
            )
            block_attn_pre, block_post, block_comb = inline_601_result
            del inline_601_result
            inline_603_hidden_states = block_hidden_states
            inline_603_pre_mix = block_pre_mix
            inline_604_result = hc_pre_norm(
                inline_603_hidden_states, inline_603_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_603_hidden_states, inline_603_pre_mix
            block_attended = block_self.attn(inline_604_result, block_runtime)
            del inline_604_result
            inline_605_sublayer_output = block_attended
            inline_605_residual = block_hidden_states
            inline_605_post = block_post
            inline_605_comb = block_comb
            inline_605_result = hc_post(
                inline_605_sublayer_output, inline_605_residual, inline_605_post, inline_605_comb
            )
            del inline_605_sublayer_output, inline_605_residual, inline_605_post, inline_605_comb
            block_hidden_states = inline_605_result
            del inline_605_result
            inline_606_hidden_states = block_hidden_states
            inline_606_fn = block_self.hc_ffn_fn
            inline_606_scale = block_self.hc_ffn_scale
            inline_606_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_606_states = inline_606_hidden_states.flatten(-2).float()
            inline_606_rstd = torch.rsqrt(inline_606_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_606_mixes = F.linear(inline_606_states, inline_606_fn.float()) * inline_606_rstd
            inline_607_mixes = inline_606_mixes
            inline_607_scale = inline_606_scale
            inline_607_base = inline_606_base
            inline_607_result = hc_gates_sinkhorn(
                inline_607_mixes, inline_607_scale, inline_607_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_607_mixes, inline_607_scale, inline_607_base
            inline_606_result = inline_607_result
            del inline_607_result
            del (
                inline_606_base,
                inline_606_fn,
                inline_606_hidden_states,
                inline_606_mixes,
                inline_606_rstd,
                inline_606_scale,
                inline_606_states,
            )
            block_ffn_pre, block_post, block_comb = inline_606_result
            del inline_606_result
            inline_608_hidden_states = block_hidden_states
            inline_608_pre_mix = block_attn_pre
            inline_609_result = hc_pre_norm(
                inline_608_hidden_states, inline_608_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_608_hidden_states, inline_608_pre_mix
            block_ffn_input = inline_609_result
            del inline_609_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_610_sublayer_output = block_output
            inline_610_residual = block_hidden_states
            inline_610_post = block_post
            inline_610_comb = block_comb
            inline_610_result = hc_post(
                inline_610_sublayer_output, inline_610_residual, inline_610_post, inline_610_comb
            )
            del inline_610_sublayer_output, inline_610_residual, inline_610_post, inline_610_comb
            hidden_states, pre_mix = (inline_610_result, block_ffn_pre)
            del inline_610_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[30]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_611_hidden_states = block_hidden_states
            inline_611_fn = block_self.hc_attn_fn
            inline_611_scale = block_self.hc_attn_scale
            inline_611_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_611_states = inline_611_hidden_states.flatten(-2).float()
            inline_611_rstd = torch.rsqrt(inline_611_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_611_mixes = F.linear(inline_611_states, inline_611_fn.float()) * inline_611_rstd
            inline_612_mixes = inline_611_mixes
            inline_612_scale = inline_611_scale
            inline_612_base = inline_611_base
            inline_612_result = hc_gates_sinkhorn(
                inline_612_mixes, inline_612_scale, inline_612_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_612_mixes, inline_612_scale, inline_612_base
            inline_611_result = inline_612_result
            del inline_612_result
            del (
                inline_611_base,
                inline_611_fn,
                inline_611_hidden_states,
                inline_611_mixes,
                inline_611_rstd,
                inline_611_scale,
                inline_611_states,
            )
            block_attn_pre, block_post, block_comb = inline_611_result
            del inline_611_result
            inline_613_hidden_states = block_hidden_states
            inline_613_pre_mix = block_pre_mix
            inline_614_result = hc_pre_norm(
                inline_613_hidden_states, inline_613_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_613_hidden_states, inline_613_pre_mix
            block_attended = block_self.attn(inline_614_result, block_runtime)
            del inline_614_result
            inline_615_sublayer_output = block_attended
            inline_615_residual = block_hidden_states
            inline_615_post = block_post
            inline_615_comb = block_comb
            inline_615_result = hc_post(
                inline_615_sublayer_output, inline_615_residual, inline_615_post, inline_615_comb
            )
            del inline_615_sublayer_output, inline_615_residual, inline_615_post, inline_615_comb
            block_hidden_states = inline_615_result
            del inline_615_result
            inline_616_hidden_states = block_hidden_states
            inline_616_fn = block_self.hc_ffn_fn
            inline_616_scale = block_self.hc_ffn_scale
            inline_616_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_616_states = inline_616_hidden_states.flatten(-2).float()
            inline_616_rstd = torch.rsqrt(inline_616_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_616_mixes = F.linear(inline_616_states, inline_616_fn.float()) * inline_616_rstd
            inline_617_mixes = inline_616_mixes
            inline_617_scale = inline_616_scale
            inline_617_base = inline_616_base
            inline_617_result = hc_gates_sinkhorn(
                inline_617_mixes, inline_617_scale, inline_617_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_617_mixes, inline_617_scale, inline_617_base
            inline_616_result = inline_617_result
            del inline_617_result
            del (
                inline_616_base,
                inline_616_fn,
                inline_616_hidden_states,
                inline_616_mixes,
                inline_616_rstd,
                inline_616_scale,
                inline_616_states,
            )
            block_ffn_pre, block_post, block_comb = inline_616_result
            del inline_616_result
            inline_618_hidden_states = block_hidden_states
            inline_618_pre_mix = block_attn_pre
            inline_619_result = hc_pre_norm(
                inline_618_hidden_states, inline_618_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_618_hidden_states, inline_618_pre_mix
            block_ffn_input = inline_619_result
            del inline_619_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_620_sublayer_output = block_output
            inline_620_residual = block_hidden_states
            inline_620_post = block_post
            inline_620_comb = block_comb
            inline_620_result = hc_post(
                inline_620_sublayer_output, inline_620_residual, inline_620_post, inline_620_comb
            )
            del inline_620_sublayer_output, inline_620_residual, inline_620_post, inline_620_comb
            hidden_states, pre_mix = (inline_620_result, block_ffn_pre)
            del inline_620_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[31].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(31)
        hidden_states = self.layers[31].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 31, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[31]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_621_hidden_states = block_hidden_states
            inline_621_fn = block_self.hc_attn_fn
            inline_621_scale = block_self.hc_attn_scale
            inline_621_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_621_states = inline_621_hidden_states.flatten(-2).float()
            inline_621_rstd = torch.rsqrt(inline_621_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_621_mixes = F.linear(inline_621_states, inline_621_fn.float()) * inline_621_rstd
            inline_622_mixes = inline_621_mixes
            inline_622_scale = inline_621_scale
            inline_622_base = inline_621_base
            inline_622_result = hc_gates_sinkhorn(
                inline_622_mixes, inline_622_scale, inline_622_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_622_mixes, inline_622_scale, inline_622_base
            inline_621_result = inline_622_result
            del inline_622_result
            del (
                inline_621_base,
                inline_621_fn,
                inline_621_hidden_states,
                inline_621_mixes,
                inline_621_rstd,
                inline_621_scale,
                inline_621_states,
            )
            block_attn_pre, block_post, block_comb = inline_621_result
            del inline_621_result
            inline_623_hidden_states = block_hidden_states
            inline_623_pre_mix = block_pre_mix
            inline_624_result = hc_pre_norm(
                inline_623_hidden_states, inline_623_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_623_hidden_states, inline_623_pre_mix
            block_attended = block_self.attn(inline_624_result, block_runtime)
            del inline_624_result
            inline_625_sublayer_output = block_attended
            inline_625_residual = block_hidden_states
            inline_625_post = block_post
            inline_625_comb = block_comb
            inline_625_result = hc_post(
                inline_625_sublayer_output, inline_625_residual, inline_625_post, inline_625_comb
            )
            del inline_625_sublayer_output, inline_625_residual, inline_625_post, inline_625_comb
            block_hidden_states = inline_625_result
            del inline_625_result
            inline_626_hidden_states = block_hidden_states
            inline_626_fn = block_self.hc_ffn_fn
            inline_626_scale = block_self.hc_ffn_scale
            inline_626_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_626_states = inline_626_hidden_states.flatten(-2).float()
            inline_626_rstd = torch.rsqrt(inline_626_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_626_mixes = F.linear(inline_626_states, inline_626_fn.float()) * inline_626_rstd
            inline_627_mixes = inline_626_mixes
            inline_627_scale = inline_626_scale
            inline_627_base = inline_626_base
            inline_627_result = hc_gates_sinkhorn(
                inline_627_mixes, inline_627_scale, inline_627_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_627_mixes, inline_627_scale, inline_627_base
            inline_626_result = inline_627_result
            del inline_627_result
            del (
                inline_626_base,
                inline_626_fn,
                inline_626_hidden_states,
                inline_626_mixes,
                inline_626_rstd,
                inline_626_scale,
                inline_626_states,
            )
            block_ffn_pre, block_post, block_comb = inline_626_result
            del inline_626_result
            inline_628_hidden_states = block_hidden_states
            inline_628_pre_mix = block_attn_pre
            inline_629_result = hc_pre_norm(
                inline_628_hidden_states, inline_628_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_628_hidden_states, inline_628_pre_mix
            block_ffn_input = inline_629_result
            del inline_629_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_630_sublayer_output = block_output
            inline_630_residual = block_hidden_states
            inline_630_post = block_post
            inline_630_comb = block_comb
            inline_630_result = hc_post(
                inline_630_sublayer_output, inline_630_residual, inline_630_post, inline_630_comb
            )
            del inline_630_sublayer_output, inline_630_residual, inline_630_post, inline_630_comb
            hidden_states, pre_mix = (inline_630_result, block_ffn_pre)
            del inline_630_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[31]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_631_hidden_states = block_hidden_states
            inline_631_fn = block_self.hc_attn_fn
            inline_631_scale = block_self.hc_attn_scale
            inline_631_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_631_states = inline_631_hidden_states.flatten(-2).float()
            inline_631_rstd = torch.rsqrt(inline_631_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_631_mixes = F.linear(inline_631_states, inline_631_fn.float()) * inline_631_rstd
            inline_632_mixes = inline_631_mixes
            inline_632_scale = inline_631_scale
            inline_632_base = inline_631_base
            inline_632_result = hc_gates_sinkhorn(
                inline_632_mixes, inline_632_scale, inline_632_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_632_mixes, inline_632_scale, inline_632_base
            inline_631_result = inline_632_result
            del inline_632_result
            del (
                inline_631_base,
                inline_631_fn,
                inline_631_hidden_states,
                inline_631_mixes,
                inline_631_rstd,
                inline_631_scale,
                inline_631_states,
            )
            block_attn_pre, block_post, block_comb = inline_631_result
            del inline_631_result
            inline_633_hidden_states = block_hidden_states
            inline_633_pre_mix = block_pre_mix
            inline_634_result = hc_pre_norm(
                inline_633_hidden_states, inline_633_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_633_hidden_states, inline_633_pre_mix
            block_attended = block_self.attn(inline_634_result, block_runtime)
            del inline_634_result
            inline_635_sublayer_output = block_attended
            inline_635_residual = block_hidden_states
            inline_635_post = block_post
            inline_635_comb = block_comb
            inline_635_result = hc_post(
                inline_635_sublayer_output, inline_635_residual, inline_635_post, inline_635_comb
            )
            del inline_635_sublayer_output, inline_635_residual, inline_635_post, inline_635_comb
            block_hidden_states = inline_635_result
            del inline_635_result
            inline_636_hidden_states = block_hidden_states
            inline_636_fn = block_self.hc_ffn_fn
            inline_636_scale = block_self.hc_ffn_scale
            inline_636_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_636_states = inline_636_hidden_states.flatten(-2).float()
            inline_636_rstd = torch.rsqrt(inline_636_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_636_mixes = F.linear(inline_636_states, inline_636_fn.float()) * inline_636_rstd
            inline_637_mixes = inline_636_mixes
            inline_637_scale = inline_636_scale
            inline_637_base = inline_636_base
            inline_637_result = hc_gates_sinkhorn(
                inline_637_mixes, inline_637_scale, inline_637_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_637_mixes, inline_637_scale, inline_637_base
            inline_636_result = inline_637_result
            del inline_637_result
            del (
                inline_636_base,
                inline_636_fn,
                inline_636_hidden_states,
                inline_636_mixes,
                inline_636_rstd,
                inline_636_scale,
                inline_636_states,
            )
            block_ffn_pre, block_post, block_comb = inline_636_result
            del inline_636_result
            inline_638_hidden_states = block_hidden_states
            inline_638_pre_mix = block_attn_pre
            inline_639_result = hc_pre_norm(
                inline_638_hidden_states, inline_638_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_638_hidden_states, inline_638_pre_mix
            block_ffn_input = inline_639_result
            del inline_639_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_640_sublayer_output = block_output
            inline_640_residual = block_hidden_states
            inline_640_post = block_post
            inline_640_comb = block_comb
            inline_640_result = hc_post(
                inline_640_sublayer_output, inline_640_residual, inline_640_post, inline_640_comb
            )
            del inline_640_sublayer_output, inline_640_residual, inline_640_post, inline_640_comb
            hidden_states, pre_mix = (inline_640_result, block_ffn_pre)
            del inline_640_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[32].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(32)
        hidden_states = self.layers[32].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 32, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[32]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_641_hidden_states = block_hidden_states
            inline_641_fn = block_self.hc_attn_fn
            inline_641_scale = block_self.hc_attn_scale
            inline_641_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_641_states = inline_641_hidden_states.flatten(-2).float()
            inline_641_rstd = torch.rsqrt(inline_641_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_641_mixes = F.linear(inline_641_states, inline_641_fn.float()) * inline_641_rstd
            inline_642_mixes = inline_641_mixes
            inline_642_scale = inline_641_scale
            inline_642_base = inline_641_base
            inline_642_result = hc_gates_sinkhorn(
                inline_642_mixes, inline_642_scale, inline_642_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_642_mixes, inline_642_scale, inline_642_base
            inline_641_result = inline_642_result
            del inline_642_result
            del (
                inline_641_base,
                inline_641_fn,
                inline_641_hidden_states,
                inline_641_mixes,
                inline_641_rstd,
                inline_641_scale,
                inline_641_states,
            )
            block_attn_pre, block_post, block_comb = inline_641_result
            del inline_641_result
            inline_643_hidden_states = block_hidden_states
            inline_643_pre_mix = block_pre_mix
            inline_644_result = hc_pre_norm(
                inline_643_hidden_states, inline_643_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_643_hidden_states, inline_643_pre_mix
            block_attended = block_self.attn(inline_644_result, block_runtime)
            del inline_644_result
            inline_645_sublayer_output = block_attended
            inline_645_residual = block_hidden_states
            inline_645_post = block_post
            inline_645_comb = block_comb
            inline_645_result = hc_post(
                inline_645_sublayer_output, inline_645_residual, inline_645_post, inline_645_comb
            )
            del inline_645_sublayer_output, inline_645_residual, inline_645_post, inline_645_comb
            block_hidden_states = inline_645_result
            del inline_645_result
            inline_646_hidden_states = block_hidden_states
            inline_646_fn = block_self.hc_ffn_fn
            inline_646_scale = block_self.hc_ffn_scale
            inline_646_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_646_states = inline_646_hidden_states.flatten(-2).float()
            inline_646_rstd = torch.rsqrt(inline_646_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_646_mixes = F.linear(inline_646_states, inline_646_fn.float()) * inline_646_rstd
            inline_647_mixes = inline_646_mixes
            inline_647_scale = inline_646_scale
            inline_647_base = inline_646_base
            inline_647_result = hc_gates_sinkhorn(
                inline_647_mixes, inline_647_scale, inline_647_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_647_mixes, inline_647_scale, inline_647_base
            inline_646_result = inline_647_result
            del inline_647_result
            del (
                inline_646_base,
                inline_646_fn,
                inline_646_hidden_states,
                inline_646_mixes,
                inline_646_rstd,
                inline_646_scale,
                inline_646_states,
            )
            block_ffn_pre, block_post, block_comb = inline_646_result
            del inline_646_result
            inline_648_hidden_states = block_hidden_states
            inline_648_pre_mix = block_attn_pre
            inline_649_result = hc_pre_norm(
                inline_648_hidden_states, inline_648_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_648_hidden_states, inline_648_pre_mix
            block_ffn_input = inline_649_result
            del inline_649_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_650_sublayer_output = block_output
            inline_650_residual = block_hidden_states
            inline_650_post = block_post
            inline_650_comb = block_comb
            inline_650_result = hc_post(
                inline_650_sublayer_output, inline_650_residual, inline_650_post, inline_650_comb
            )
            del inline_650_sublayer_output, inline_650_residual, inline_650_post, inline_650_comb
            hidden_states, pre_mix = (inline_650_result, block_ffn_pre)
            del inline_650_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[32]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_651_hidden_states = block_hidden_states
            inline_651_fn = block_self.hc_attn_fn
            inline_651_scale = block_self.hc_attn_scale
            inline_651_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_651_states = inline_651_hidden_states.flatten(-2).float()
            inline_651_rstd = torch.rsqrt(inline_651_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_651_mixes = F.linear(inline_651_states, inline_651_fn.float()) * inline_651_rstd
            inline_652_mixes = inline_651_mixes
            inline_652_scale = inline_651_scale
            inline_652_base = inline_651_base
            inline_652_result = hc_gates_sinkhorn(
                inline_652_mixes, inline_652_scale, inline_652_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_652_mixes, inline_652_scale, inline_652_base
            inline_651_result = inline_652_result
            del inline_652_result
            del (
                inline_651_base,
                inline_651_fn,
                inline_651_hidden_states,
                inline_651_mixes,
                inline_651_rstd,
                inline_651_scale,
                inline_651_states,
            )
            block_attn_pre, block_post, block_comb = inline_651_result
            del inline_651_result
            inline_653_hidden_states = block_hidden_states
            inline_653_pre_mix = block_pre_mix
            inline_654_result = hc_pre_norm(
                inline_653_hidden_states, inline_653_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_653_hidden_states, inline_653_pre_mix
            block_attended = block_self.attn(inline_654_result, block_runtime)
            del inline_654_result
            inline_655_sublayer_output = block_attended
            inline_655_residual = block_hidden_states
            inline_655_post = block_post
            inline_655_comb = block_comb
            inline_655_result = hc_post(
                inline_655_sublayer_output, inline_655_residual, inline_655_post, inline_655_comb
            )
            del inline_655_sublayer_output, inline_655_residual, inline_655_post, inline_655_comb
            block_hidden_states = inline_655_result
            del inline_655_result
            inline_656_hidden_states = block_hidden_states
            inline_656_fn = block_self.hc_ffn_fn
            inline_656_scale = block_self.hc_ffn_scale
            inline_656_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_656_states = inline_656_hidden_states.flatten(-2).float()
            inline_656_rstd = torch.rsqrt(inline_656_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_656_mixes = F.linear(inline_656_states, inline_656_fn.float()) * inline_656_rstd
            inline_657_mixes = inline_656_mixes
            inline_657_scale = inline_656_scale
            inline_657_base = inline_656_base
            inline_657_result = hc_gates_sinkhorn(
                inline_657_mixes, inline_657_scale, inline_657_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_657_mixes, inline_657_scale, inline_657_base
            inline_656_result = inline_657_result
            del inline_657_result
            del (
                inline_656_base,
                inline_656_fn,
                inline_656_hidden_states,
                inline_656_mixes,
                inline_656_rstd,
                inline_656_scale,
                inline_656_states,
            )
            block_ffn_pre, block_post, block_comb = inline_656_result
            del inline_656_result
            inline_658_hidden_states = block_hidden_states
            inline_658_pre_mix = block_attn_pre
            inline_659_result = hc_pre_norm(
                inline_658_hidden_states, inline_658_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_658_hidden_states, inline_658_pre_mix
            block_ffn_input = inline_659_result
            del inline_659_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_660_sublayer_output = block_output
            inline_660_residual = block_hidden_states
            inline_660_post = block_post
            inline_660_comb = block_comb
            inline_660_result = hc_post(
                inline_660_sublayer_output, inline_660_residual, inline_660_post, inline_660_comb
            )
            del inline_660_sublayer_output, inline_660_residual, inline_660_post, inline_660_comb
            hidden_states, pre_mix = (inline_660_result, block_ffn_pre)
            del inline_660_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[33].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(33)
        hidden_states = self.layers[33].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 33, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[33]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_661_hidden_states = block_hidden_states
            inline_661_fn = block_self.hc_attn_fn
            inline_661_scale = block_self.hc_attn_scale
            inline_661_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_661_states = inline_661_hidden_states.flatten(-2).float()
            inline_661_rstd = torch.rsqrt(inline_661_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_661_mixes = F.linear(inline_661_states, inline_661_fn.float()) * inline_661_rstd
            inline_662_mixes = inline_661_mixes
            inline_662_scale = inline_661_scale
            inline_662_base = inline_661_base
            inline_662_result = hc_gates_sinkhorn(
                inline_662_mixes, inline_662_scale, inline_662_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_662_mixes, inline_662_scale, inline_662_base
            inline_661_result = inline_662_result
            del inline_662_result
            del (
                inline_661_base,
                inline_661_fn,
                inline_661_hidden_states,
                inline_661_mixes,
                inline_661_rstd,
                inline_661_scale,
                inline_661_states,
            )
            block_attn_pre, block_post, block_comb = inline_661_result
            del inline_661_result
            inline_663_hidden_states = block_hidden_states
            inline_663_pre_mix = block_pre_mix
            inline_664_result = hc_pre_norm(
                inline_663_hidden_states, inline_663_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_663_hidden_states, inline_663_pre_mix
            block_attended = block_self.attn(inline_664_result, block_runtime)
            del inline_664_result
            inline_665_sublayer_output = block_attended
            inline_665_residual = block_hidden_states
            inline_665_post = block_post
            inline_665_comb = block_comb
            inline_665_result = hc_post(
                inline_665_sublayer_output, inline_665_residual, inline_665_post, inline_665_comb
            )
            del inline_665_sublayer_output, inline_665_residual, inline_665_post, inline_665_comb
            block_hidden_states = inline_665_result
            del inline_665_result
            inline_666_hidden_states = block_hidden_states
            inline_666_fn = block_self.hc_ffn_fn
            inline_666_scale = block_self.hc_ffn_scale
            inline_666_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_666_states = inline_666_hidden_states.flatten(-2).float()
            inline_666_rstd = torch.rsqrt(inline_666_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_666_mixes = F.linear(inline_666_states, inline_666_fn.float()) * inline_666_rstd
            inline_667_mixes = inline_666_mixes
            inline_667_scale = inline_666_scale
            inline_667_base = inline_666_base
            inline_667_result = hc_gates_sinkhorn(
                inline_667_mixes, inline_667_scale, inline_667_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_667_mixes, inline_667_scale, inline_667_base
            inline_666_result = inline_667_result
            del inline_667_result
            del (
                inline_666_base,
                inline_666_fn,
                inline_666_hidden_states,
                inline_666_mixes,
                inline_666_rstd,
                inline_666_scale,
                inline_666_states,
            )
            block_ffn_pre, block_post, block_comb = inline_666_result
            del inline_666_result
            inline_668_hidden_states = block_hidden_states
            inline_668_pre_mix = block_attn_pre
            inline_669_result = hc_pre_norm(
                inline_668_hidden_states, inline_668_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_668_hidden_states, inline_668_pre_mix
            block_ffn_input = inline_669_result
            del inline_669_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_670_sublayer_output = block_output
            inline_670_residual = block_hidden_states
            inline_670_post = block_post
            inline_670_comb = block_comb
            inline_670_result = hc_post(
                inline_670_sublayer_output, inline_670_residual, inline_670_post, inline_670_comb
            )
            del inline_670_sublayer_output, inline_670_residual, inline_670_post, inline_670_comb
            hidden_states, pre_mix = (inline_670_result, block_ffn_pre)
            del inline_670_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[33]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_671_hidden_states = block_hidden_states
            inline_671_fn = block_self.hc_attn_fn
            inline_671_scale = block_self.hc_attn_scale
            inline_671_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_671_states = inline_671_hidden_states.flatten(-2).float()
            inline_671_rstd = torch.rsqrt(inline_671_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_671_mixes = F.linear(inline_671_states, inline_671_fn.float()) * inline_671_rstd
            inline_672_mixes = inline_671_mixes
            inline_672_scale = inline_671_scale
            inline_672_base = inline_671_base
            inline_672_result = hc_gates_sinkhorn(
                inline_672_mixes, inline_672_scale, inline_672_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_672_mixes, inline_672_scale, inline_672_base
            inline_671_result = inline_672_result
            del inline_672_result
            del (
                inline_671_base,
                inline_671_fn,
                inline_671_hidden_states,
                inline_671_mixes,
                inline_671_rstd,
                inline_671_scale,
                inline_671_states,
            )
            block_attn_pre, block_post, block_comb = inline_671_result
            del inline_671_result
            inline_673_hidden_states = block_hidden_states
            inline_673_pre_mix = block_pre_mix
            inline_674_result = hc_pre_norm(
                inline_673_hidden_states, inline_673_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_673_hidden_states, inline_673_pre_mix
            block_attended = block_self.attn(inline_674_result, block_runtime)
            del inline_674_result
            inline_675_sublayer_output = block_attended
            inline_675_residual = block_hidden_states
            inline_675_post = block_post
            inline_675_comb = block_comb
            inline_675_result = hc_post(
                inline_675_sublayer_output, inline_675_residual, inline_675_post, inline_675_comb
            )
            del inline_675_sublayer_output, inline_675_residual, inline_675_post, inline_675_comb
            block_hidden_states = inline_675_result
            del inline_675_result
            inline_676_hidden_states = block_hidden_states
            inline_676_fn = block_self.hc_ffn_fn
            inline_676_scale = block_self.hc_ffn_scale
            inline_676_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_676_states = inline_676_hidden_states.flatten(-2).float()
            inline_676_rstd = torch.rsqrt(inline_676_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_676_mixes = F.linear(inline_676_states, inline_676_fn.float()) * inline_676_rstd
            inline_677_mixes = inline_676_mixes
            inline_677_scale = inline_676_scale
            inline_677_base = inline_676_base
            inline_677_result = hc_gates_sinkhorn(
                inline_677_mixes, inline_677_scale, inline_677_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_677_mixes, inline_677_scale, inline_677_base
            inline_676_result = inline_677_result
            del inline_677_result
            del (
                inline_676_base,
                inline_676_fn,
                inline_676_hidden_states,
                inline_676_mixes,
                inline_676_rstd,
                inline_676_scale,
                inline_676_states,
            )
            block_ffn_pre, block_post, block_comb = inline_676_result
            del inline_676_result
            inline_678_hidden_states = block_hidden_states
            inline_678_pre_mix = block_attn_pre
            inline_679_result = hc_pre_norm(
                inline_678_hidden_states, inline_678_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_678_hidden_states, inline_678_pre_mix
            block_ffn_input = inline_679_result
            del inline_679_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_680_sublayer_output = block_output
            inline_680_residual = block_hidden_states
            inline_680_post = block_post
            inline_680_comb = block_comb
            inline_680_result = hc_post(
                inline_680_sublayer_output, inline_680_residual, inline_680_post, inline_680_comb
            )
            del inline_680_sublayer_output, inline_680_residual, inline_680_post, inline_680_comb
            hidden_states, pre_mix = (inline_680_result, block_ffn_pre)
            del inline_680_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[34].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(34)
        hidden_states = self.layers[34].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 34, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[34]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_681_hidden_states = block_hidden_states
            inline_681_fn = block_self.hc_attn_fn
            inline_681_scale = block_self.hc_attn_scale
            inline_681_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_681_states = inline_681_hidden_states.flatten(-2).float()
            inline_681_rstd = torch.rsqrt(inline_681_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_681_mixes = F.linear(inline_681_states, inline_681_fn.float()) * inline_681_rstd
            inline_682_mixes = inline_681_mixes
            inline_682_scale = inline_681_scale
            inline_682_base = inline_681_base
            inline_682_result = hc_gates_sinkhorn(
                inline_682_mixes, inline_682_scale, inline_682_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_682_mixes, inline_682_scale, inline_682_base
            inline_681_result = inline_682_result
            del inline_682_result
            del (
                inline_681_base,
                inline_681_fn,
                inline_681_hidden_states,
                inline_681_mixes,
                inline_681_rstd,
                inline_681_scale,
                inline_681_states,
            )
            block_attn_pre, block_post, block_comb = inline_681_result
            del inline_681_result
            inline_683_hidden_states = block_hidden_states
            inline_683_pre_mix = block_pre_mix
            inline_684_result = hc_pre_norm(
                inline_683_hidden_states, inline_683_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_683_hidden_states, inline_683_pre_mix
            block_attended = block_self.attn(inline_684_result, block_runtime)
            del inline_684_result
            inline_685_sublayer_output = block_attended
            inline_685_residual = block_hidden_states
            inline_685_post = block_post
            inline_685_comb = block_comb
            inline_685_result = hc_post(
                inline_685_sublayer_output, inline_685_residual, inline_685_post, inline_685_comb
            )
            del inline_685_sublayer_output, inline_685_residual, inline_685_post, inline_685_comb
            block_hidden_states = inline_685_result
            del inline_685_result
            inline_686_hidden_states = block_hidden_states
            inline_686_fn = block_self.hc_ffn_fn
            inline_686_scale = block_self.hc_ffn_scale
            inline_686_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_686_states = inline_686_hidden_states.flatten(-2).float()
            inline_686_rstd = torch.rsqrt(inline_686_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_686_mixes = F.linear(inline_686_states, inline_686_fn.float()) * inline_686_rstd
            inline_687_mixes = inline_686_mixes
            inline_687_scale = inline_686_scale
            inline_687_base = inline_686_base
            inline_687_result = hc_gates_sinkhorn(
                inline_687_mixes, inline_687_scale, inline_687_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_687_mixes, inline_687_scale, inline_687_base
            inline_686_result = inline_687_result
            del inline_687_result
            del (
                inline_686_base,
                inline_686_fn,
                inline_686_hidden_states,
                inline_686_mixes,
                inline_686_rstd,
                inline_686_scale,
                inline_686_states,
            )
            block_ffn_pre, block_post, block_comb = inline_686_result
            del inline_686_result
            inline_688_hidden_states = block_hidden_states
            inline_688_pre_mix = block_attn_pre
            inline_689_result = hc_pre_norm(
                inline_688_hidden_states, inline_688_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_688_hidden_states, inline_688_pre_mix
            block_ffn_input = inline_689_result
            del inline_689_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_690_sublayer_output = block_output
            inline_690_residual = block_hidden_states
            inline_690_post = block_post
            inline_690_comb = block_comb
            inline_690_result = hc_post(
                inline_690_sublayer_output, inline_690_residual, inline_690_post, inline_690_comb
            )
            del inline_690_sublayer_output, inline_690_residual, inline_690_post, inline_690_comb
            hidden_states, pre_mix = (inline_690_result, block_ffn_pre)
            del inline_690_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[34]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_691_hidden_states = block_hidden_states
            inline_691_fn = block_self.hc_attn_fn
            inline_691_scale = block_self.hc_attn_scale
            inline_691_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_691_states = inline_691_hidden_states.flatten(-2).float()
            inline_691_rstd = torch.rsqrt(inline_691_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_691_mixes = F.linear(inline_691_states, inline_691_fn.float()) * inline_691_rstd
            inline_692_mixes = inline_691_mixes
            inline_692_scale = inline_691_scale
            inline_692_base = inline_691_base
            inline_692_result = hc_gates_sinkhorn(
                inline_692_mixes, inline_692_scale, inline_692_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_692_mixes, inline_692_scale, inline_692_base
            inline_691_result = inline_692_result
            del inline_692_result
            del (
                inline_691_base,
                inline_691_fn,
                inline_691_hidden_states,
                inline_691_mixes,
                inline_691_rstd,
                inline_691_scale,
                inline_691_states,
            )
            block_attn_pre, block_post, block_comb = inline_691_result
            del inline_691_result
            inline_693_hidden_states = block_hidden_states
            inline_693_pre_mix = block_pre_mix
            inline_694_result = hc_pre_norm(
                inline_693_hidden_states, inline_693_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_693_hidden_states, inline_693_pre_mix
            block_attended = block_self.attn(inline_694_result, block_runtime)
            del inline_694_result
            inline_695_sublayer_output = block_attended
            inline_695_residual = block_hidden_states
            inline_695_post = block_post
            inline_695_comb = block_comb
            inline_695_result = hc_post(
                inline_695_sublayer_output, inline_695_residual, inline_695_post, inline_695_comb
            )
            del inline_695_sublayer_output, inline_695_residual, inline_695_post, inline_695_comb
            block_hidden_states = inline_695_result
            del inline_695_result
            inline_696_hidden_states = block_hidden_states
            inline_696_fn = block_self.hc_ffn_fn
            inline_696_scale = block_self.hc_ffn_scale
            inline_696_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_696_states = inline_696_hidden_states.flatten(-2).float()
            inline_696_rstd = torch.rsqrt(inline_696_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_696_mixes = F.linear(inline_696_states, inline_696_fn.float()) * inline_696_rstd
            inline_697_mixes = inline_696_mixes
            inline_697_scale = inline_696_scale
            inline_697_base = inline_696_base
            inline_697_result = hc_gates_sinkhorn(
                inline_697_mixes, inline_697_scale, inline_697_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_697_mixes, inline_697_scale, inline_697_base
            inline_696_result = inline_697_result
            del inline_697_result
            del (
                inline_696_base,
                inline_696_fn,
                inline_696_hidden_states,
                inline_696_mixes,
                inline_696_rstd,
                inline_696_scale,
                inline_696_states,
            )
            block_ffn_pre, block_post, block_comb = inline_696_result
            del inline_696_result
            inline_698_hidden_states = block_hidden_states
            inline_698_pre_mix = block_attn_pre
            inline_699_result = hc_pre_norm(
                inline_698_hidden_states, inline_698_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_698_hidden_states, inline_698_pre_mix
            block_ffn_input = inline_699_result
            del inline_699_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_700_sublayer_output = block_output
            inline_700_residual = block_hidden_states
            inline_700_post = block_post
            inline_700_comb = block_comb
            inline_700_result = hc_post(
                inline_700_sublayer_output, inline_700_residual, inline_700_post, inline_700_comb
            )
            del inline_700_sublayer_output, inline_700_residual, inline_700_post, inline_700_comb
            hidden_states, pre_mix = (inline_700_result, block_ffn_pre)
            del inline_700_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[35].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(35)
        hidden_states = self.layers[35].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 35, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[35]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_701_hidden_states = block_hidden_states
            inline_701_fn = block_self.hc_attn_fn
            inline_701_scale = block_self.hc_attn_scale
            inline_701_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_701_states = inline_701_hidden_states.flatten(-2).float()
            inline_701_rstd = torch.rsqrt(inline_701_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_701_mixes = F.linear(inline_701_states, inline_701_fn.float()) * inline_701_rstd
            inline_702_mixes = inline_701_mixes
            inline_702_scale = inline_701_scale
            inline_702_base = inline_701_base
            inline_702_result = hc_gates_sinkhorn(
                inline_702_mixes, inline_702_scale, inline_702_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_702_mixes, inline_702_scale, inline_702_base
            inline_701_result = inline_702_result
            del inline_702_result
            del (
                inline_701_base,
                inline_701_fn,
                inline_701_hidden_states,
                inline_701_mixes,
                inline_701_rstd,
                inline_701_scale,
                inline_701_states,
            )
            block_attn_pre, block_post, block_comb = inline_701_result
            del inline_701_result
            inline_703_hidden_states = block_hidden_states
            inline_703_pre_mix = block_pre_mix
            inline_704_result = hc_pre_norm(
                inline_703_hidden_states, inline_703_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_703_hidden_states, inline_703_pre_mix
            block_attended = block_self.attn(inline_704_result, block_runtime)
            del inline_704_result
            inline_705_sublayer_output = block_attended
            inline_705_residual = block_hidden_states
            inline_705_post = block_post
            inline_705_comb = block_comb
            inline_705_result = hc_post(
                inline_705_sublayer_output, inline_705_residual, inline_705_post, inline_705_comb
            )
            del inline_705_sublayer_output, inline_705_residual, inline_705_post, inline_705_comb
            block_hidden_states = inline_705_result
            del inline_705_result
            inline_706_hidden_states = block_hidden_states
            inline_706_fn = block_self.hc_ffn_fn
            inline_706_scale = block_self.hc_ffn_scale
            inline_706_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_706_states = inline_706_hidden_states.flatten(-2).float()
            inline_706_rstd = torch.rsqrt(inline_706_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_706_mixes = F.linear(inline_706_states, inline_706_fn.float()) * inline_706_rstd
            inline_707_mixes = inline_706_mixes
            inline_707_scale = inline_706_scale
            inline_707_base = inline_706_base
            inline_707_result = hc_gates_sinkhorn(
                inline_707_mixes, inline_707_scale, inline_707_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_707_mixes, inline_707_scale, inline_707_base
            inline_706_result = inline_707_result
            del inline_707_result
            del (
                inline_706_base,
                inline_706_fn,
                inline_706_hidden_states,
                inline_706_mixes,
                inline_706_rstd,
                inline_706_scale,
                inline_706_states,
            )
            block_ffn_pre, block_post, block_comb = inline_706_result
            del inline_706_result
            inline_708_hidden_states = block_hidden_states
            inline_708_pre_mix = block_attn_pre
            inline_709_result = hc_pre_norm(
                inline_708_hidden_states, inline_708_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_708_hidden_states, inline_708_pre_mix
            block_ffn_input = inline_709_result
            del inline_709_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_710_sublayer_output = block_output
            inline_710_residual = block_hidden_states
            inline_710_post = block_post
            inline_710_comb = block_comb
            inline_710_result = hc_post(
                inline_710_sublayer_output, inline_710_residual, inline_710_post, inline_710_comb
            )
            del inline_710_sublayer_output, inline_710_residual, inline_710_post, inline_710_comb
            hidden_states, pre_mix = (inline_710_result, block_ffn_pre)
            del inline_710_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[35]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_711_hidden_states = block_hidden_states
            inline_711_fn = block_self.hc_attn_fn
            inline_711_scale = block_self.hc_attn_scale
            inline_711_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_711_states = inline_711_hidden_states.flatten(-2).float()
            inline_711_rstd = torch.rsqrt(inline_711_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_711_mixes = F.linear(inline_711_states, inline_711_fn.float()) * inline_711_rstd
            inline_712_mixes = inline_711_mixes
            inline_712_scale = inline_711_scale
            inline_712_base = inline_711_base
            inline_712_result = hc_gates_sinkhorn(
                inline_712_mixes, inline_712_scale, inline_712_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_712_mixes, inline_712_scale, inline_712_base
            inline_711_result = inline_712_result
            del inline_712_result
            del (
                inline_711_base,
                inline_711_fn,
                inline_711_hidden_states,
                inline_711_mixes,
                inline_711_rstd,
                inline_711_scale,
                inline_711_states,
            )
            block_attn_pre, block_post, block_comb = inline_711_result
            del inline_711_result
            inline_713_hidden_states = block_hidden_states
            inline_713_pre_mix = block_pre_mix
            inline_714_result = hc_pre_norm(
                inline_713_hidden_states, inline_713_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_713_hidden_states, inline_713_pre_mix
            block_attended = block_self.attn(inline_714_result, block_runtime)
            del inline_714_result
            inline_715_sublayer_output = block_attended
            inline_715_residual = block_hidden_states
            inline_715_post = block_post
            inline_715_comb = block_comb
            inline_715_result = hc_post(
                inline_715_sublayer_output, inline_715_residual, inline_715_post, inline_715_comb
            )
            del inline_715_sublayer_output, inline_715_residual, inline_715_post, inline_715_comb
            block_hidden_states = inline_715_result
            del inline_715_result
            inline_716_hidden_states = block_hidden_states
            inline_716_fn = block_self.hc_ffn_fn
            inline_716_scale = block_self.hc_ffn_scale
            inline_716_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_716_states = inline_716_hidden_states.flatten(-2).float()
            inline_716_rstd = torch.rsqrt(inline_716_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_716_mixes = F.linear(inline_716_states, inline_716_fn.float()) * inline_716_rstd
            inline_717_mixes = inline_716_mixes
            inline_717_scale = inline_716_scale
            inline_717_base = inline_716_base
            inline_717_result = hc_gates_sinkhorn(
                inline_717_mixes, inline_717_scale, inline_717_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_717_mixes, inline_717_scale, inline_717_base
            inline_716_result = inline_717_result
            del inline_717_result
            del (
                inline_716_base,
                inline_716_fn,
                inline_716_hidden_states,
                inline_716_mixes,
                inline_716_rstd,
                inline_716_scale,
                inline_716_states,
            )
            block_ffn_pre, block_post, block_comb = inline_716_result
            del inline_716_result
            inline_718_hidden_states = block_hidden_states
            inline_718_pre_mix = block_attn_pre
            inline_719_result = hc_pre_norm(
                inline_718_hidden_states, inline_718_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_718_hidden_states, inline_718_pre_mix
            block_ffn_input = inline_719_result
            del inline_719_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_720_sublayer_output = block_output
            inline_720_residual = block_hidden_states
            inline_720_post = block_post
            inline_720_comb = block_comb
            inline_720_result = hc_post(
                inline_720_sublayer_output, inline_720_residual, inline_720_post, inline_720_comb
            )
            del inline_720_sublayer_output, inline_720_residual, inline_720_post, inline_720_comb
            hidden_states, pre_mix = (inline_720_result, block_ffn_pre)
            del inline_720_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[36].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(36)
        hidden_states = self.layers[36].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 36, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[36]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_721_hidden_states = block_hidden_states
            inline_721_fn = block_self.hc_attn_fn
            inline_721_scale = block_self.hc_attn_scale
            inline_721_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_721_states = inline_721_hidden_states.flatten(-2).float()
            inline_721_rstd = torch.rsqrt(inline_721_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_721_mixes = F.linear(inline_721_states, inline_721_fn.float()) * inline_721_rstd
            inline_722_mixes = inline_721_mixes
            inline_722_scale = inline_721_scale
            inline_722_base = inline_721_base
            inline_722_result = hc_gates_sinkhorn(
                inline_722_mixes, inline_722_scale, inline_722_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_722_mixes, inline_722_scale, inline_722_base
            inline_721_result = inline_722_result
            del inline_722_result
            del (
                inline_721_base,
                inline_721_fn,
                inline_721_hidden_states,
                inline_721_mixes,
                inline_721_rstd,
                inline_721_scale,
                inline_721_states,
            )
            block_attn_pre, block_post, block_comb = inline_721_result
            del inline_721_result
            inline_723_hidden_states = block_hidden_states
            inline_723_pre_mix = block_pre_mix
            inline_724_result = hc_pre_norm(
                inline_723_hidden_states, inline_723_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_723_hidden_states, inline_723_pre_mix
            block_attended = block_self.attn(inline_724_result, block_runtime)
            del inline_724_result
            inline_725_sublayer_output = block_attended
            inline_725_residual = block_hidden_states
            inline_725_post = block_post
            inline_725_comb = block_comb
            inline_725_result = hc_post(
                inline_725_sublayer_output, inline_725_residual, inline_725_post, inline_725_comb
            )
            del inline_725_sublayer_output, inline_725_residual, inline_725_post, inline_725_comb
            block_hidden_states = inline_725_result
            del inline_725_result
            inline_726_hidden_states = block_hidden_states
            inline_726_fn = block_self.hc_ffn_fn
            inline_726_scale = block_self.hc_ffn_scale
            inline_726_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_726_states = inline_726_hidden_states.flatten(-2).float()
            inline_726_rstd = torch.rsqrt(inline_726_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_726_mixes = F.linear(inline_726_states, inline_726_fn.float()) * inline_726_rstd
            inline_727_mixes = inline_726_mixes
            inline_727_scale = inline_726_scale
            inline_727_base = inline_726_base
            inline_727_result = hc_gates_sinkhorn(
                inline_727_mixes, inline_727_scale, inline_727_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_727_mixes, inline_727_scale, inline_727_base
            inline_726_result = inline_727_result
            del inline_727_result
            del (
                inline_726_base,
                inline_726_fn,
                inline_726_hidden_states,
                inline_726_mixes,
                inline_726_rstd,
                inline_726_scale,
                inline_726_states,
            )
            block_ffn_pre, block_post, block_comb = inline_726_result
            del inline_726_result
            inline_728_hidden_states = block_hidden_states
            inline_728_pre_mix = block_attn_pre
            inline_729_result = hc_pre_norm(
                inline_728_hidden_states, inline_728_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_728_hidden_states, inline_728_pre_mix
            block_ffn_input = inline_729_result
            del inline_729_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_730_sublayer_output = block_output
            inline_730_residual = block_hidden_states
            inline_730_post = block_post
            inline_730_comb = block_comb
            inline_730_result = hc_post(
                inline_730_sublayer_output, inline_730_residual, inline_730_post, inline_730_comb
            )
            del inline_730_sublayer_output, inline_730_residual, inline_730_post, inline_730_comb
            hidden_states, pre_mix = (inline_730_result, block_ffn_pre)
            del inline_730_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[36]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_731_hidden_states = block_hidden_states
            inline_731_fn = block_self.hc_attn_fn
            inline_731_scale = block_self.hc_attn_scale
            inline_731_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_731_states = inline_731_hidden_states.flatten(-2).float()
            inline_731_rstd = torch.rsqrt(inline_731_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_731_mixes = F.linear(inline_731_states, inline_731_fn.float()) * inline_731_rstd
            inline_732_mixes = inline_731_mixes
            inline_732_scale = inline_731_scale
            inline_732_base = inline_731_base
            inline_732_result = hc_gates_sinkhorn(
                inline_732_mixes, inline_732_scale, inline_732_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_732_mixes, inline_732_scale, inline_732_base
            inline_731_result = inline_732_result
            del inline_732_result
            del (
                inline_731_base,
                inline_731_fn,
                inline_731_hidden_states,
                inline_731_mixes,
                inline_731_rstd,
                inline_731_scale,
                inline_731_states,
            )
            block_attn_pre, block_post, block_comb = inline_731_result
            del inline_731_result
            inline_733_hidden_states = block_hidden_states
            inline_733_pre_mix = block_pre_mix
            inline_734_result = hc_pre_norm(
                inline_733_hidden_states, inline_733_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_733_hidden_states, inline_733_pre_mix
            block_attended = block_self.attn(inline_734_result, block_runtime)
            del inline_734_result
            inline_735_sublayer_output = block_attended
            inline_735_residual = block_hidden_states
            inline_735_post = block_post
            inline_735_comb = block_comb
            inline_735_result = hc_post(
                inline_735_sublayer_output, inline_735_residual, inline_735_post, inline_735_comb
            )
            del inline_735_sublayer_output, inline_735_residual, inline_735_post, inline_735_comb
            block_hidden_states = inline_735_result
            del inline_735_result
            inline_736_hidden_states = block_hidden_states
            inline_736_fn = block_self.hc_ffn_fn
            inline_736_scale = block_self.hc_ffn_scale
            inline_736_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_736_states = inline_736_hidden_states.flatten(-2).float()
            inline_736_rstd = torch.rsqrt(inline_736_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_736_mixes = F.linear(inline_736_states, inline_736_fn.float()) * inline_736_rstd
            inline_737_mixes = inline_736_mixes
            inline_737_scale = inline_736_scale
            inline_737_base = inline_736_base
            inline_737_result = hc_gates_sinkhorn(
                inline_737_mixes, inline_737_scale, inline_737_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_737_mixes, inline_737_scale, inline_737_base
            inline_736_result = inline_737_result
            del inline_737_result
            del (
                inline_736_base,
                inline_736_fn,
                inline_736_hidden_states,
                inline_736_mixes,
                inline_736_rstd,
                inline_736_scale,
                inline_736_states,
            )
            block_ffn_pre, block_post, block_comb = inline_736_result
            del inline_736_result
            inline_738_hidden_states = block_hidden_states
            inline_738_pre_mix = block_attn_pre
            inline_739_result = hc_pre_norm(
                inline_738_hidden_states, inline_738_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_738_hidden_states, inline_738_pre_mix
            block_ffn_input = inline_739_result
            del inline_739_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_740_sublayer_output = block_output
            inline_740_residual = block_hidden_states
            inline_740_post = block_post
            inline_740_comb = block_comb
            inline_740_result = hc_post(
                inline_740_sublayer_output, inline_740_residual, inline_740_post, inline_740_comb
            )
            del inline_740_sublayer_output, inline_740_residual, inline_740_post, inline_740_comb
            hidden_states, pre_mix = (inline_740_result, block_ffn_pre)
            del inline_740_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[37].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(37)
        hidden_states = self.layers[37].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 37, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[37]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_741_hidden_states = block_hidden_states
            inline_741_fn = block_self.hc_attn_fn
            inline_741_scale = block_self.hc_attn_scale
            inline_741_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_741_states = inline_741_hidden_states.flatten(-2).float()
            inline_741_rstd = torch.rsqrt(inline_741_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_741_mixes = F.linear(inline_741_states, inline_741_fn.float()) * inline_741_rstd
            inline_742_mixes = inline_741_mixes
            inline_742_scale = inline_741_scale
            inline_742_base = inline_741_base
            inline_742_result = hc_gates_sinkhorn(
                inline_742_mixes, inline_742_scale, inline_742_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_742_mixes, inline_742_scale, inline_742_base
            inline_741_result = inline_742_result
            del inline_742_result
            del (
                inline_741_base,
                inline_741_fn,
                inline_741_hidden_states,
                inline_741_mixes,
                inline_741_rstd,
                inline_741_scale,
                inline_741_states,
            )
            block_attn_pre, block_post, block_comb = inline_741_result
            del inline_741_result
            inline_743_hidden_states = block_hidden_states
            inline_743_pre_mix = block_pre_mix
            inline_744_result = hc_pre_norm(
                inline_743_hidden_states, inline_743_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_743_hidden_states, inline_743_pre_mix
            block_attended = block_self.attn(inline_744_result, block_runtime)
            del inline_744_result
            inline_745_sublayer_output = block_attended
            inline_745_residual = block_hidden_states
            inline_745_post = block_post
            inline_745_comb = block_comb
            inline_745_result = hc_post(
                inline_745_sublayer_output, inline_745_residual, inline_745_post, inline_745_comb
            )
            del inline_745_sublayer_output, inline_745_residual, inline_745_post, inline_745_comb
            block_hidden_states = inline_745_result
            del inline_745_result
            inline_746_hidden_states = block_hidden_states
            inline_746_fn = block_self.hc_ffn_fn
            inline_746_scale = block_self.hc_ffn_scale
            inline_746_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_746_states = inline_746_hidden_states.flatten(-2).float()
            inline_746_rstd = torch.rsqrt(inline_746_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_746_mixes = F.linear(inline_746_states, inline_746_fn.float()) * inline_746_rstd
            inline_747_mixes = inline_746_mixes
            inline_747_scale = inline_746_scale
            inline_747_base = inline_746_base
            inline_747_result = hc_gates_sinkhorn(
                inline_747_mixes, inline_747_scale, inline_747_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_747_mixes, inline_747_scale, inline_747_base
            inline_746_result = inline_747_result
            del inline_747_result
            del (
                inline_746_base,
                inline_746_fn,
                inline_746_hidden_states,
                inline_746_mixes,
                inline_746_rstd,
                inline_746_scale,
                inline_746_states,
            )
            block_ffn_pre, block_post, block_comb = inline_746_result
            del inline_746_result
            inline_748_hidden_states = block_hidden_states
            inline_748_pre_mix = block_attn_pre
            inline_749_result = hc_pre_norm(
                inline_748_hidden_states, inline_748_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_748_hidden_states, inline_748_pre_mix
            block_ffn_input = inline_749_result
            del inline_749_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_750_sublayer_output = block_output
            inline_750_residual = block_hidden_states
            inline_750_post = block_post
            inline_750_comb = block_comb
            inline_750_result = hc_post(
                inline_750_sublayer_output, inline_750_residual, inline_750_post, inline_750_comb
            )
            del inline_750_sublayer_output, inline_750_residual, inline_750_post, inline_750_comb
            hidden_states, pre_mix = (inline_750_result, block_ffn_pre)
            del inline_750_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[37]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_751_hidden_states = block_hidden_states
            inline_751_fn = block_self.hc_attn_fn
            inline_751_scale = block_self.hc_attn_scale
            inline_751_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_751_states = inline_751_hidden_states.flatten(-2).float()
            inline_751_rstd = torch.rsqrt(inline_751_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_751_mixes = F.linear(inline_751_states, inline_751_fn.float()) * inline_751_rstd
            inline_752_mixes = inline_751_mixes
            inline_752_scale = inline_751_scale
            inline_752_base = inline_751_base
            inline_752_result = hc_gates_sinkhorn(
                inline_752_mixes, inline_752_scale, inline_752_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_752_mixes, inline_752_scale, inline_752_base
            inline_751_result = inline_752_result
            del inline_752_result
            del (
                inline_751_base,
                inline_751_fn,
                inline_751_hidden_states,
                inline_751_mixes,
                inline_751_rstd,
                inline_751_scale,
                inline_751_states,
            )
            block_attn_pre, block_post, block_comb = inline_751_result
            del inline_751_result
            inline_753_hidden_states = block_hidden_states
            inline_753_pre_mix = block_pre_mix
            inline_754_result = hc_pre_norm(
                inline_753_hidden_states, inline_753_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_753_hidden_states, inline_753_pre_mix
            block_attended = block_self.attn(inline_754_result, block_runtime)
            del inline_754_result
            inline_755_sublayer_output = block_attended
            inline_755_residual = block_hidden_states
            inline_755_post = block_post
            inline_755_comb = block_comb
            inline_755_result = hc_post(
                inline_755_sublayer_output, inline_755_residual, inline_755_post, inline_755_comb
            )
            del inline_755_sublayer_output, inline_755_residual, inline_755_post, inline_755_comb
            block_hidden_states = inline_755_result
            del inline_755_result
            inline_756_hidden_states = block_hidden_states
            inline_756_fn = block_self.hc_ffn_fn
            inline_756_scale = block_self.hc_ffn_scale
            inline_756_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_756_states = inline_756_hidden_states.flatten(-2).float()
            inline_756_rstd = torch.rsqrt(inline_756_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_756_mixes = F.linear(inline_756_states, inline_756_fn.float()) * inline_756_rstd
            inline_757_mixes = inline_756_mixes
            inline_757_scale = inline_756_scale
            inline_757_base = inline_756_base
            inline_757_result = hc_gates_sinkhorn(
                inline_757_mixes, inline_757_scale, inline_757_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_757_mixes, inline_757_scale, inline_757_base
            inline_756_result = inline_757_result
            del inline_757_result
            del (
                inline_756_base,
                inline_756_fn,
                inline_756_hidden_states,
                inline_756_mixes,
                inline_756_rstd,
                inline_756_scale,
                inline_756_states,
            )
            block_ffn_pre, block_post, block_comb = inline_756_result
            del inline_756_result
            inline_758_hidden_states = block_hidden_states
            inline_758_pre_mix = block_attn_pre
            inline_759_result = hc_pre_norm(
                inline_758_hidden_states, inline_758_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_758_hidden_states, inline_758_pre_mix
            block_ffn_input = inline_759_result
            del inline_759_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_760_sublayer_output = block_output
            inline_760_residual = block_hidden_states
            inline_760_post = block_post
            inline_760_comb = block_comb
            inline_760_result = hc_post(
                inline_760_sublayer_output, inline_760_residual, inline_760_post, inline_760_comb
            )
            del inline_760_sublayer_output, inline_760_residual, inline_760_post, inline_760_comb
            hidden_states, pre_mix = (inline_760_result, block_ffn_pre)
            del inline_760_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[38].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(38)
        hidden_states = self.layers[38].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 38, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[38]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_761_hidden_states = block_hidden_states
            inline_761_fn = block_self.hc_attn_fn
            inline_761_scale = block_self.hc_attn_scale
            inline_761_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_761_states = inline_761_hidden_states.flatten(-2).float()
            inline_761_rstd = torch.rsqrt(inline_761_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_761_mixes = F.linear(inline_761_states, inline_761_fn.float()) * inline_761_rstd
            inline_762_mixes = inline_761_mixes
            inline_762_scale = inline_761_scale
            inline_762_base = inline_761_base
            inline_762_result = hc_gates_sinkhorn(
                inline_762_mixes, inline_762_scale, inline_762_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_762_mixes, inline_762_scale, inline_762_base
            inline_761_result = inline_762_result
            del inline_762_result
            del (
                inline_761_base,
                inline_761_fn,
                inline_761_hidden_states,
                inline_761_mixes,
                inline_761_rstd,
                inline_761_scale,
                inline_761_states,
            )
            block_attn_pre, block_post, block_comb = inline_761_result
            del inline_761_result
            inline_763_hidden_states = block_hidden_states
            inline_763_pre_mix = block_pre_mix
            inline_764_result = hc_pre_norm(
                inline_763_hidden_states, inline_763_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_763_hidden_states, inline_763_pre_mix
            block_attended = block_self.attn(inline_764_result, block_runtime)
            del inline_764_result
            inline_765_sublayer_output = block_attended
            inline_765_residual = block_hidden_states
            inline_765_post = block_post
            inline_765_comb = block_comb
            inline_765_result = hc_post(
                inline_765_sublayer_output, inline_765_residual, inline_765_post, inline_765_comb
            )
            del inline_765_sublayer_output, inline_765_residual, inline_765_post, inline_765_comb
            block_hidden_states = inline_765_result
            del inline_765_result
            inline_766_hidden_states = block_hidden_states
            inline_766_fn = block_self.hc_ffn_fn
            inline_766_scale = block_self.hc_ffn_scale
            inline_766_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_766_states = inline_766_hidden_states.flatten(-2).float()
            inline_766_rstd = torch.rsqrt(inline_766_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_766_mixes = F.linear(inline_766_states, inline_766_fn.float()) * inline_766_rstd
            inline_767_mixes = inline_766_mixes
            inline_767_scale = inline_766_scale
            inline_767_base = inline_766_base
            inline_767_result = hc_gates_sinkhorn(
                inline_767_mixes, inline_767_scale, inline_767_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_767_mixes, inline_767_scale, inline_767_base
            inline_766_result = inline_767_result
            del inline_767_result
            del (
                inline_766_base,
                inline_766_fn,
                inline_766_hidden_states,
                inline_766_mixes,
                inline_766_rstd,
                inline_766_scale,
                inline_766_states,
            )
            block_ffn_pre, block_post, block_comb = inline_766_result
            del inline_766_result
            inline_768_hidden_states = block_hidden_states
            inline_768_pre_mix = block_attn_pre
            inline_769_result = hc_pre_norm(
                inline_768_hidden_states, inline_768_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_768_hidden_states, inline_768_pre_mix
            block_ffn_input = inline_769_result
            del inline_769_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_770_sublayer_output = block_output
            inline_770_residual = block_hidden_states
            inline_770_post = block_post
            inline_770_comb = block_comb
            inline_770_result = hc_post(
                inline_770_sublayer_output, inline_770_residual, inline_770_post, inline_770_comb
            )
            del inline_770_sublayer_output, inline_770_residual, inline_770_post, inline_770_comb
            hidden_states, pre_mix = (inline_770_result, block_ffn_pre)
            del inline_770_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[38]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_771_hidden_states = block_hidden_states
            inline_771_fn = block_self.hc_attn_fn
            inline_771_scale = block_self.hc_attn_scale
            inline_771_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_771_states = inline_771_hidden_states.flatten(-2).float()
            inline_771_rstd = torch.rsqrt(inline_771_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_771_mixes = F.linear(inline_771_states, inline_771_fn.float()) * inline_771_rstd
            inline_772_mixes = inline_771_mixes
            inline_772_scale = inline_771_scale
            inline_772_base = inline_771_base
            inline_772_result = hc_gates_sinkhorn(
                inline_772_mixes, inline_772_scale, inline_772_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_772_mixes, inline_772_scale, inline_772_base
            inline_771_result = inline_772_result
            del inline_772_result
            del (
                inline_771_base,
                inline_771_fn,
                inline_771_hidden_states,
                inline_771_mixes,
                inline_771_rstd,
                inline_771_scale,
                inline_771_states,
            )
            block_attn_pre, block_post, block_comb = inline_771_result
            del inline_771_result
            inline_773_hidden_states = block_hidden_states
            inline_773_pre_mix = block_pre_mix
            inline_774_result = hc_pre_norm(
                inline_773_hidden_states, inline_773_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_773_hidden_states, inline_773_pre_mix
            block_attended = block_self.attn(inline_774_result, block_runtime)
            del inline_774_result
            inline_775_sublayer_output = block_attended
            inline_775_residual = block_hidden_states
            inline_775_post = block_post
            inline_775_comb = block_comb
            inline_775_result = hc_post(
                inline_775_sublayer_output, inline_775_residual, inline_775_post, inline_775_comb
            )
            del inline_775_sublayer_output, inline_775_residual, inline_775_post, inline_775_comb
            block_hidden_states = inline_775_result
            del inline_775_result
            inline_776_hidden_states = block_hidden_states
            inline_776_fn = block_self.hc_ffn_fn
            inline_776_scale = block_self.hc_ffn_scale
            inline_776_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_776_states = inline_776_hidden_states.flatten(-2).float()
            inline_776_rstd = torch.rsqrt(inline_776_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_776_mixes = F.linear(inline_776_states, inline_776_fn.float()) * inline_776_rstd
            inline_777_mixes = inline_776_mixes
            inline_777_scale = inline_776_scale
            inline_777_base = inline_776_base
            inline_777_result = hc_gates_sinkhorn(
                inline_777_mixes, inline_777_scale, inline_777_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_777_mixes, inline_777_scale, inline_777_base
            inline_776_result = inline_777_result
            del inline_777_result
            del (
                inline_776_base,
                inline_776_fn,
                inline_776_hidden_states,
                inline_776_mixes,
                inline_776_rstd,
                inline_776_scale,
                inline_776_states,
            )
            block_ffn_pre, block_post, block_comb = inline_776_result
            del inline_776_result
            inline_778_hidden_states = block_hidden_states
            inline_778_pre_mix = block_attn_pre
            inline_779_result = hc_pre_norm(
                inline_778_hidden_states, inline_778_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_778_hidden_states, inline_778_pre_mix
            block_ffn_input = inline_779_result
            del inline_779_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_780_sublayer_output = block_output
            inline_780_residual = block_hidden_states
            inline_780_post = block_post
            inline_780_comb = block_comb
            inline_780_result = hc_post(
                inline_780_sublayer_output, inline_780_residual, inline_780_post, inline_780_comb
            )
            del inline_780_sublayer_output, inline_780_residual, inline_780_post, inline_780_comb
            hidden_states, pre_mix = (inline_780_result, block_ffn_pre)
            del inline_780_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[39].engram is not None and actual_tokens:
        index = self.engram_layer_ids.index(39)
        hidden_states = self.layers[39].engram(
            hidden_states, hashes[:, index], token_mask=local_text, sequence_parallel=self.sequence_parallel
        )
    self._capture_aux(auxiliary, 39, hidden_states, actual_tokens)
    if local_image is None:
        block_self = self.layers[39]
        block_decode_mask = None
        block_image_mask = None
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_781_hidden_states = block_hidden_states
            inline_781_fn = block_self.hc_attn_fn
            inline_781_scale = block_self.hc_attn_scale
            inline_781_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_781_states = inline_781_hidden_states.flatten(-2).float()
            inline_781_rstd = torch.rsqrt(inline_781_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_781_mixes = F.linear(inline_781_states, inline_781_fn.float()) * inline_781_rstd
            inline_782_mixes = inline_781_mixes
            inline_782_scale = inline_781_scale
            inline_782_base = inline_781_base
            inline_782_result = hc_gates_sinkhorn(
                inline_782_mixes, inline_782_scale, inline_782_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_782_mixes, inline_782_scale, inline_782_base
            inline_781_result = inline_782_result
            del inline_782_result
            del (
                inline_781_base,
                inline_781_fn,
                inline_781_hidden_states,
                inline_781_mixes,
                inline_781_rstd,
                inline_781_scale,
                inline_781_states,
            )
            block_attn_pre, block_post, block_comb = inline_781_result
            del inline_781_result
            inline_783_hidden_states = block_hidden_states
            inline_783_pre_mix = block_pre_mix
            inline_784_result = hc_pre_norm(
                inline_783_hidden_states, inline_783_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_783_hidden_states, inline_783_pre_mix
            block_attended = block_self.attn(inline_784_result, block_runtime)
            del inline_784_result
            inline_785_sublayer_output = block_attended
            inline_785_residual = block_hidden_states
            inline_785_post = block_post
            inline_785_comb = block_comb
            inline_785_result = hc_post(
                inline_785_sublayer_output, inline_785_residual, inline_785_post, inline_785_comb
            )
            del inline_785_sublayer_output, inline_785_residual, inline_785_post, inline_785_comb
            block_hidden_states = inline_785_result
            del inline_785_result
            inline_786_hidden_states = block_hidden_states
            inline_786_fn = block_self.hc_ffn_fn
            inline_786_scale = block_self.hc_ffn_scale
            inline_786_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_786_states = inline_786_hidden_states.flatten(-2).float()
            inline_786_rstd = torch.rsqrt(inline_786_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_786_mixes = F.linear(inline_786_states, inline_786_fn.float()) * inline_786_rstd
            inline_787_mixes = inline_786_mixes
            inline_787_scale = inline_786_scale
            inline_787_base = inline_786_base
            inline_787_result = hc_gates_sinkhorn(
                inline_787_mixes, inline_787_scale, inline_787_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_787_mixes, inline_787_scale, inline_787_base
            inline_786_result = inline_787_result
            del inline_787_result
            del (
                inline_786_base,
                inline_786_fn,
                inline_786_hidden_states,
                inline_786_mixes,
                inline_786_rstd,
                inline_786_scale,
                inline_786_states,
            )
            block_ffn_pre, block_post, block_comb = inline_786_result
            del inline_786_result
            inline_788_hidden_states = block_hidden_states
            inline_788_pre_mix = block_attn_pre
            inline_789_result = hc_pre_norm(
                inline_788_hidden_states, inline_788_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_788_hidden_states, inline_788_pre_mix
            block_ffn_input = inline_789_result
            del inline_789_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_790_sublayer_output = block_output
            inline_790_residual = block_hidden_states
            inline_790_post = block_post
            inline_790_comb = block_comb
            inline_790_result = hc_post(
                inline_790_sublayer_output, inline_790_residual, inline_790_post, inline_790_comb
            )
            del inline_790_sublayer_output, inline_790_residual, inline_790_post, inline_790_comb
            hidden_states, pre_mix = (inline_790_result, block_ffn_pre)
            del inline_790_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    else:
        block_self = self.layers[39]
        block_decode_mask = None
        block_image_mask = local_image
        block_hidden_states = hidden_states
        block_pre_mix = pre_mix
        block_runtime = runtime
        if block_hidden_states.shape[0] == 0:
            block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
            hidden_states, pre_mix = (block_hidden_states, block_pre_mix)
        else:
            inline_791_hidden_states = block_hidden_states
            inline_791_fn = block_self.hc_attn_fn
            inline_791_scale = block_self.hc_attn_scale
            inline_791_base = block_self.hc_attn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_791_states = inline_791_hidden_states.flatten(-2).float()
            inline_791_rstd = torch.rsqrt(inline_791_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_791_mixes = F.linear(inline_791_states, inline_791_fn.float()) * inline_791_rstd
            inline_792_mixes = inline_791_mixes
            inline_792_scale = inline_791_scale
            inline_792_base = inline_791_base
            inline_792_result = hc_gates_sinkhorn(
                inline_792_mixes, inline_792_scale, inline_792_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_792_mixes, inline_792_scale, inline_792_base
            inline_791_result = inline_792_result
            del inline_792_result
            del (
                inline_791_base,
                inline_791_fn,
                inline_791_hidden_states,
                inline_791_mixes,
                inline_791_rstd,
                inline_791_scale,
                inline_791_states,
            )
            block_attn_pre, block_post, block_comb = inline_791_result
            del inline_791_result
            inline_793_hidden_states = block_hidden_states
            inline_793_pre_mix = block_pre_mix
            inline_794_result = hc_pre_norm(
                inline_793_hidden_states, inline_793_pre_mix, block_self.attn_norm.weight, 1e-20
            )
            del inline_793_hidden_states, inline_793_pre_mix
            block_attended = block_self.attn(inline_794_result, block_runtime)
            del inline_794_result
            inline_795_sublayer_output = block_attended
            inline_795_residual = block_hidden_states
            inline_795_post = block_post
            inline_795_comb = block_comb
            inline_795_result = hc_post(
                inline_795_sublayer_output, inline_795_residual, inline_795_post, inline_795_comb
            )
            del inline_795_sublayer_output, inline_795_residual, inline_795_post, inline_795_comb
            block_hidden_states = inline_795_result
            del inline_795_result
            inline_796_hidden_states = block_hidden_states
            inline_796_fn = block_self.hc_ffn_fn
            inline_796_scale = block_self.hc_ffn_scale
            inline_796_base = block_self.hc_ffn_base
            "Normalize jointly over all HC copies before projecting their coefficients."
            inline_796_states = inline_796_hidden_states.flatten(-2).float()
            inline_796_rstd = torch.rsqrt(inline_796_states.square().mean(-1, keepdim=True) + 1e-20)
            inline_796_mixes = F.linear(inline_796_states, inline_796_fn.float()) * inline_796_rstd
            inline_797_mixes = inline_796_mixes
            inline_797_scale = inline_796_scale
            inline_797_base = inline_796_base
            inline_797_result = hc_gates_sinkhorn(
                inline_797_mixes, inline_797_scale, inline_797_base, sinkhorn_iters=20, eps=1e-06
            )
            del inline_797_mixes, inline_797_scale, inline_797_base
            inline_796_result = inline_797_result
            del inline_797_result
            del (
                inline_796_base,
                inline_796_fn,
                inline_796_hidden_states,
                inline_796_mixes,
                inline_796_rstd,
                inline_796_scale,
                inline_796_states,
            )
            block_ffn_pre, block_post, block_comb = inline_796_result
            del inline_796_result
            inline_798_hidden_states = block_hidden_states
            inline_798_pre_mix = block_attn_pre
            inline_799_result = hc_pre_norm(
                inline_798_hidden_states, inline_798_pre_mix, block_self.ffn_norm.weight, 1e-20
            )
            del inline_798_hidden_states, inline_798_pre_mix
            block_ffn_input = inline_799_result
            del inline_799_result
            if getattr(block_runtime, "decode", None) is not None:
                block_output = block_self.ffn.forward_decode(
                    block_ffn_input, block_runtime.decode.active if block_decode_mask is None else block_decode_mask
                )
            elif block_image_mask is not None:
                block_output = block_self.ffn(block_ffn_input, image_mask=block_image_mask)
            else:
                block_output = block_self.ffn(block_ffn_input)
            inline_800_sublayer_output = block_output
            inline_800_residual = block_hidden_states
            inline_800_post = block_post
            inline_800_comb = block_comb
            inline_800_result = hc_post(
                inline_800_sublayer_output, inline_800_residual, inline_800_post, inline_800_comb
            )
            del inline_800_sublayer_output, inline_800_residual, inline_800_post, inline_800_comb
            hidden_states, pre_mix = (inline_800_result, block_ffn_pre)
            del inline_800_result
            del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
        del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    output = self._finish_hc(hidden_states, pre_mix, actual_tokens)
    if actual_tokens < input_ids.numel():
        padding = output.new_zeros(input_ids.numel() - actual_tokens, self.config.hidden_size)
        output = torch.cat((output, padding))
        auxiliary = [torch.cat((value, padding)) for value in auxiliary]
    return (output, auxiliary) if self.aux_hidden_state_layers else output
