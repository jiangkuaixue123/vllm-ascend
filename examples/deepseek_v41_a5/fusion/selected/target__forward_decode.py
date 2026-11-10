# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_hc import hc_pre_norm, hc_post
from vllm_ascend.ops.triton.deepseek_v41 import hc_gates_sinkhorn


def _forward_decode(self, input_ids, metadata):
    # Statically expanded model blocks retain the measured execution order.
    runtime = DeepseekV41Runtime.for_decode(metadata)
    local_active = sp_shard(runtime.decode.active) if self.sequence_parallel else runtime.decode.active
    hidden_states, pre_mix = self._start_hc(self.embed(input_ids))
    auxiliary = []
    if self.layers[0].engram is not None:
        hidden_states = self.layers[0].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 0],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 0, hidden_states, input_ids.shape[0])
    block_self = self.layers[0]
    block_decode_mask = local_active
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
        inline_2_result = hc_gates_sinkhorn(inline_2_mixes, inline_2_scale, inline_2_base, sinkhorn_iters=20, eps=1e-06)
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
        inline_7_result = hc_gates_sinkhorn(inline_7_mixes, inline_7_scale, inline_7_base, sinkhorn_iters=20, eps=1e-06)
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
    if self.layers[1].engram is not None:
        hidden_states = self.layers[1].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 1],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 1, hidden_states, input_ids.shape[0])
    block_self = self.layers[1]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_14_result = hc_pre_norm(inline_13_hidden_states, inline_13_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_19_result = hc_pre_norm(inline_18_hidden_states, inline_18_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
    if self.layers[2].engram is not None:
        hidden_states = self.layers[2].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 2],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 2, hidden_states, input_ids.shape[0])
    block_self = self.layers[2]
    block_decode_mask = local_active
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
        inline_24_result = hc_pre_norm(inline_23_hidden_states, inline_23_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_29_result = hc_pre_norm(inline_28_hidden_states, inline_28_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
    if self.layers[3].engram is not None:
        hidden_states = self.layers[3].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 3],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 3, hidden_states, input_ids.shape[0])
    block_self = self.layers[3]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_34_result = hc_pre_norm(inline_33_hidden_states, inline_33_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_39_result = hc_pre_norm(inline_38_hidden_states, inline_38_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
    if self.layers[4].engram is not None:
        hidden_states = self.layers[4].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 4],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 4, hidden_states, input_ids.shape[0])
    block_self = self.layers[4]
    block_decode_mask = local_active
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
        inline_44_result = hc_pre_norm(inline_43_hidden_states, inline_43_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_49_result = hc_pre_norm(inline_48_hidden_states, inline_48_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
    if self.layers[5].engram is not None:
        hidden_states = self.layers[5].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 5],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 5, hidden_states, input_ids.shape[0])
    block_self = self.layers[5]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_54_result = hc_pre_norm(inline_53_hidden_states, inline_53_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_59_result = hc_pre_norm(inline_58_hidden_states, inline_58_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
    if self.layers[6].engram is not None:
        hidden_states = self.layers[6].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 6],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 6, hidden_states, input_ids.shape[0])
    block_self = self.layers[6]
    block_decode_mask = local_active
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
        inline_64_result = hc_pre_norm(inline_63_hidden_states, inline_63_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_69_result = hc_pre_norm(inline_68_hidden_states, inline_68_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
    if self.layers[7].engram is not None:
        hidden_states = self.layers[7].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 7],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 7, hidden_states, input_ids.shape[0])
    block_self = self.layers[7]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_74_result = hc_pre_norm(inline_73_hidden_states, inline_73_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_79_result = hc_pre_norm(inline_78_hidden_states, inline_78_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
    if self.layers[8].engram is not None:
        hidden_states = self.layers[8].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 8],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 8, hidden_states, input_ids.shape[0])
    block_self = self.layers[8]
    block_decode_mask = local_active
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
        inline_84_result = hc_pre_norm(inline_83_hidden_states, inline_83_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_89_result = hc_pre_norm(inline_88_hidden_states, inline_88_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
    if self.layers[9].engram is not None:
        hidden_states = self.layers[9].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 9],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 9, hidden_states, input_ids.shape[0])
    block_self = self.layers[9]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_94_result = hc_pre_norm(inline_93_hidden_states, inline_93_pre_mix, block_self.attn_norm.weight, 1e-20)
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
        inline_99_result = hc_pre_norm(inline_98_hidden_states, inline_98_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_100_result = hc_post(inline_100_sublayer_output, inline_100_residual, inline_100_post, inline_100_comb)
        del inline_100_sublayer_output, inline_100_residual, inline_100_post, inline_100_comb
        hidden_states, pre_mix = (inline_100_result, block_ffn_pre)
        del inline_100_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[10].engram is not None:
        hidden_states = self.layers[10].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 10],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 10, hidden_states, input_ids.shape[0])
    block_self = self.layers[10]
    block_decode_mask = local_active
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
        inline_105_result = hc_post(inline_105_sublayer_output, inline_105_residual, inline_105_post, inline_105_comb)
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
        inline_109_result = hc_pre_norm(inline_108_hidden_states, inline_108_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_110_result = hc_post(inline_110_sublayer_output, inline_110_residual, inline_110_post, inline_110_comb)
        del inline_110_sublayer_output, inline_110_residual, inline_110_post, inline_110_comb
        hidden_states, pre_mix = (inline_110_result, block_ffn_pre)
        del inline_110_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[11].engram is not None:
        hidden_states = self.layers[11].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 11],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 11, hidden_states, input_ids.shape[0])
    block_self = self.layers[11]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_115_result = hc_post(inline_115_sublayer_output, inline_115_residual, inline_115_post, inline_115_comb)
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
        inline_119_result = hc_pre_norm(inline_118_hidden_states, inline_118_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_120_result = hc_post(inline_120_sublayer_output, inline_120_residual, inline_120_post, inline_120_comb)
        del inline_120_sublayer_output, inline_120_residual, inline_120_post, inline_120_comb
        hidden_states, pre_mix = (inline_120_result, block_ffn_pre)
        del inline_120_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[12].engram is not None:
        hidden_states = self.layers[12].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 12],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 12, hidden_states, input_ids.shape[0])
    block_self = self.layers[12]
    block_decode_mask = local_active
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
        inline_125_result = hc_post(inline_125_sublayer_output, inline_125_residual, inline_125_post, inline_125_comb)
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
        inline_129_result = hc_pre_norm(inline_128_hidden_states, inline_128_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_130_result = hc_post(inline_130_sublayer_output, inline_130_residual, inline_130_post, inline_130_comb)
        del inline_130_sublayer_output, inline_130_residual, inline_130_post, inline_130_comb
        hidden_states, pre_mix = (inline_130_result, block_ffn_pre)
        del inline_130_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[13].engram is not None:
        hidden_states = self.layers[13].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 13],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 13, hidden_states, input_ids.shape[0])
    block_self = self.layers[13]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_135_result = hc_post(inline_135_sublayer_output, inline_135_residual, inline_135_post, inline_135_comb)
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
        inline_139_result = hc_pre_norm(inline_138_hidden_states, inline_138_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_140_result = hc_post(inline_140_sublayer_output, inline_140_residual, inline_140_post, inline_140_comb)
        del inline_140_sublayer_output, inline_140_residual, inline_140_post, inline_140_comb
        hidden_states, pre_mix = (inline_140_result, block_ffn_pre)
        del inline_140_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[14].engram is not None:
        hidden_states = self.layers[14].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 14],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 14, hidden_states, input_ids.shape[0])
    block_self = self.layers[14]
    block_decode_mask = local_active
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
        inline_145_result = hc_post(inline_145_sublayer_output, inline_145_residual, inline_145_post, inline_145_comb)
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
        inline_149_result = hc_pre_norm(inline_148_hidden_states, inline_148_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_150_result = hc_post(inline_150_sublayer_output, inline_150_residual, inline_150_post, inline_150_comb)
        del inline_150_sublayer_output, inline_150_residual, inline_150_post, inline_150_comb
        hidden_states, pre_mix = (inline_150_result, block_ffn_pre)
        del inline_150_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[15].engram is not None:
        hidden_states = self.layers[15].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 15],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 15, hidden_states, input_ids.shape[0])
    block_self = self.layers[15]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_155_result = hc_post(inline_155_sublayer_output, inline_155_residual, inline_155_post, inline_155_comb)
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
        inline_159_result = hc_pre_norm(inline_158_hidden_states, inline_158_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_160_result = hc_post(inline_160_sublayer_output, inline_160_residual, inline_160_post, inline_160_comb)
        del inline_160_sublayer_output, inline_160_residual, inline_160_post, inline_160_comb
        hidden_states, pre_mix = (inline_160_result, block_ffn_pre)
        del inline_160_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[16].engram is not None:
        hidden_states = self.layers[16].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 16],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 16, hidden_states, input_ids.shape[0])
    block_self = self.layers[16]
    block_decode_mask = local_active
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
        inline_165_result = hc_post(inline_165_sublayer_output, inline_165_residual, inline_165_post, inline_165_comb)
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
        inline_169_result = hc_pre_norm(inline_168_hidden_states, inline_168_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_170_result = hc_post(inline_170_sublayer_output, inline_170_residual, inline_170_post, inline_170_comb)
        del inline_170_sublayer_output, inline_170_residual, inline_170_post, inline_170_comb
        hidden_states, pre_mix = (inline_170_result, block_ffn_pre)
        del inline_170_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[17].engram is not None:
        hidden_states = self.layers[17].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 17],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 17, hidden_states, input_ids.shape[0])
    block_self = self.layers[17]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_175_result = hc_post(inline_175_sublayer_output, inline_175_residual, inline_175_post, inline_175_comb)
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
        inline_179_result = hc_pre_norm(inline_178_hidden_states, inline_178_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_180_result = hc_post(inline_180_sublayer_output, inline_180_residual, inline_180_post, inline_180_comb)
        del inline_180_sublayer_output, inline_180_residual, inline_180_post, inline_180_comb
        hidden_states, pre_mix = (inline_180_result, block_ffn_pre)
        del inline_180_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[18].engram is not None:
        hidden_states = self.layers[18].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 18],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 18, hidden_states, input_ids.shape[0])
    block_self = self.layers[18]
    block_decode_mask = local_active
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
        inline_185_result = hc_post(inline_185_sublayer_output, inline_185_residual, inline_185_post, inline_185_comb)
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
        inline_189_result = hc_pre_norm(inline_188_hidden_states, inline_188_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_190_result = hc_post(inline_190_sublayer_output, inline_190_residual, inline_190_post, inline_190_comb)
        del inline_190_sublayer_output, inline_190_residual, inline_190_post, inline_190_comb
        hidden_states, pre_mix = (inline_190_result, block_ffn_pre)
        del inline_190_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[19].engram is not None:
        hidden_states = self.layers[19].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 19],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 19, hidden_states, input_ids.shape[0])
    block_self = self.layers[19]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_195_result = hc_post(inline_195_sublayer_output, inline_195_residual, inline_195_post, inline_195_comb)
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
        inline_199_result = hc_pre_norm(inline_198_hidden_states, inline_198_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_200_result = hc_post(inline_200_sublayer_output, inline_200_residual, inline_200_post, inline_200_comb)
        del inline_200_sublayer_output, inline_200_residual, inline_200_post, inline_200_comb
        hidden_states, pre_mix = (inline_200_result, block_ffn_pre)
        del inline_200_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[20].engram is not None:
        hidden_states = self.layers[20].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 20],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 20, hidden_states, input_ids.shape[0])
    block_self = self.layers[20]
    block_decode_mask = local_active
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
        inline_205_result = hc_post(inline_205_sublayer_output, inline_205_residual, inline_205_post, inline_205_comb)
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
        inline_209_result = hc_pre_norm(inline_208_hidden_states, inline_208_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_210_result = hc_post(inline_210_sublayer_output, inline_210_residual, inline_210_post, inline_210_comb)
        del inline_210_sublayer_output, inline_210_residual, inline_210_post, inline_210_comb
        hidden_states, pre_mix = (inline_210_result, block_ffn_pre)
        del inline_210_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[21].engram is not None:
        hidden_states = self.layers[21].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 21],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 21, hidden_states, input_ids.shape[0])
    block_self = self.layers[21]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_215_result = hc_post(inline_215_sublayer_output, inline_215_residual, inline_215_post, inline_215_comb)
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
        inline_219_result = hc_pre_norm(inline_218_hidden_states, inline_218_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_220_result = hc_post(inline_220_sublayer_output, inline_220_residual, inline_220_post, inline_220_comb)
        del inline_220_sublayer_output, inline_220_residual, inline_220_post, inline_220_comb
        hidden_states, pre_mix = (inline_220_result, block_ffn_pre)
        del inline_220_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[22].engram is not None:
        hidden_states = self.layers[22].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 22],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 22, hidden_states, input_ids.shape[0])
    block_self = self.layers[22]
    block_decode_mask = local_active
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
        inline_225_result = hc_post(inline_225_sublayer_output, inline_225_residual, inline_225_post, inline_225_comb)
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
        inline_229_result = hc_pre_norm(inline_228_hidden_states, inline_228_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_230_result = hc_post(inline_230_sublayer_output, inline_230_residual, inline_230_post, inline_230_comb)
        del inline_230_sublayer_output, inline_230_residual, inline_230_post, inline_230_comb
        hidden_states, pre_mix = (inline_230_result, block_ffn_pre)
        del inline_230_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[23].engram is not None:
        hidden_states = self.layers[23].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 23],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 23, hidden_states, input_ids.shape[0])
    block_self = self.layers[23]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_235_result = hc_post(inline_235_sublayer_output, inline_235_residual, inline_235_post, inline_235_comb)
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
        inline_239_result = hc_pre_norm(inline_238_hidden_states, inline_238_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_240_result = hc_post(inline_240_sublayer_output, inline_240_residual, inline_240_post, inline_240_comb)
        del inline_240_sublayer_output, inline_240_residual, inline_240_post, inline_240_comb
        hidden_states, pre_mix = (inline_240_result, block_ffn_pre)
        del inline_240_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[24].engram is not None:
        hidden_states = self.layers[24].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 24],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 24, hidden_states, input_ids.shape[0])
    block_self = self.layers[24]
    block_decode_mask = local_active
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
        inline_245_result = hc_post(inline_245_sublayer_output, inline_245_residual, inline_245_post, inline_245_comb)
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
        inline_249_result = hc_pre_norm(inline_248_hidden_states, inline_248_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_250_result = hc_post(inline_250_sublayer_output, inline_250_residual, inline_250_post, inline_250_comb)
        del inline_250_sublayer_output, inline_250_residual, inline_250_post, inline_250_comb
        hidden_states, pre_mix = (inline_250_result, block_ffn_pre)
        del inline_250_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[25].engram is not None:
        hidden_states = self.layers[25].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 25],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 25, hidden_states, input_ids.shape[0])
    block_self = self.layers[25]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_255_result = hc_post(inline_255_sublayer_output, inline_255_residual, inline_255_post, inline_255_comb)
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
        inline_259_result = hc_pre_norm(inline_258_hidden_states, inline_258_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_260_result = hc_post(inline_260_sublayer_output, inline_260_residual, inline_260_post, inline_260_comb)
        del inline_260_sublayer_output, inline_260_residual, inline_260_post, inline_260_comb
        hidden_states, pre_mix = (inline_260_result, block_ffn_pre)
        del inline_260_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[26].engram is not None:
        hidden_states = self.layers[26].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 26],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 26, hidden_states, input_ids.shape[0])
    block_self = self.layers[26]
    block_decode_mask = local_active
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
        inline_265_result = hc_post(inline_265_sublayer_output, inline_265_residual, inline_265_post, inline_265_comb)
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
        inline_269_result = hc_pre_norm(inline_268_hidden_states, inline_268_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_270_result = hc_post(inline_270_sublayer_output, inline_270_residual, inline_270_post, inline_270_comb)
        del inline_270_sublayer_output, inline_270_residual, inline_270_post, inline_270_comb
        hidden_states, pre_mix = (inline_270_result, block_ffn_pre)
        del inline_270_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[27].engram is not None:
        hidden_states = self.layers[27].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 27],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 27, hidden_states, input_ids.shape[0])
    block_self = self.layers[27]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_275_result = hc_post(inline_275_sublayer_output, inline_275_residual, inline_275_post, inline_275_comb)
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
        inline_279_result = hc_pre_norm(inline_278_hidden_states, inline_278_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_280_result = hc_post(inline_280_sublayer_output, inline_280_residual, inline_280_post, inline_280_comb)
        del inline_280_sublayer_output, inline_280_residual, inline_280_post, inline_280_comb
        hidden_states, pre_mix = (inline_280_result, block_ffn_pre)
        del inline_280_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[28].engram is not None:
        hidden_states = self.layers[28].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 28],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 28, hidden_states, input_ids.shape[0])
    block_self = self.layers[28]
    block_decode_mask = local_active
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
        inline_285_result = hc_post(inline_285_sublayer_output, inline_285_residual, inline_285_post, inline_285_comb)
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
        inline_289_result = hc_pre_norm(inline_288_hidden_states, inline_288_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_290_result = hc_post(inline_290_sublayer_output, inline_290_residual, inline_290_post, inline_290_comb)
        del inline_290_sublayer_output, inline_290_residual, inline_290_post, inline_290_comb
        hidden_states, pre_mix = (inline_290_result, block_ffn_pre)
        del inline_290_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[29].engram is not None:
        hidden_states = self.layers[29].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 29],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 29, hidden_states, input_ids.shape[0])
    block_self = self.layers[29]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_295_result = hc_post(inline_295_sublayer_output, inline_295_residual, inline_295_post, inline_295_comb)
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
        inline_299_result = hc_pre_norm(inline_298_hidden_states, inline_298_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_300_result = hc_post(inline_300_sublayer_output, inline_300_residual, inline_300_post, inline_300_comb)
        del inline_300_sublayer_output, inline_300_residual, inline_300_post, inline_300_comb
        hidden_states, pre_mix = (inline_300_result, block_ffn_pre)
        del inline_300_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[30].engram is not None:
        hidden_states = self.layers[30].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 30],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 30, hidden_states, input_ids.shape[0])
    block_self = self.layers[30]
    block_decode_mask = local_active
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
        inline_305_result = hc_post(inline_305_sublayer_output, inline_305_residual, inline_305_post, inline_305_comb)
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
        inline_309_result = hc_pre_norm(inline_308_hidden_states, inline_308_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_310_result = hc_post(inline_310_sublayer_output, inline_310_residual, inline_310_post, inline_310_comb)
        del inline_310_sublayer_output, inline_310_residual, inline_310_post, inline_310_comb
        hidden_states, pre_mix = (inline_310_result, block_ffn_pre)
        del inline_310_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[31].engram is not None:
        hidden_states = self.layers[31].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 31],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 31, hidden_states, input_ids.shape[0])
    block_self = self.layers[31]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_315_result = hc_post(inline_315_sublayer_output, inline_315_residual, inline_315_post, inline_315_comb)
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
        inline_319_result = hc_pre_norm(inline_318_hidden_states, inline_318_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_320_result = hc_post(inline_320_sublayer_output, inline_320_residual, inline_320_post, inline_320_comb)
        del inline_320_sublayer_output, inline_320_residual, inline_320_post, inline_320_comb
        hidden_states, pre_mix = (inline_320_result, block_ffn_pre)
        del inline_320_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[32].engram is not None:
        hidden_states = self.layers[32].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 32],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 32, hidden_states, input_ids.shape[0])
    block_self = self.layers[32]
    block_decode_mask = local_active
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
        inline_325_result = hc_post(inline_325_sublayer_output, inline_325_residual, inline_325_post, inline_325_comb)
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
        inline_329_result = hc_pre_norm(inline_328_hidden_states, inline_328_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_330_result = hc_post(inline_330_sublayer_output, inline_330_residual, inline_330_post, inline_330_comb)
        del inline_330_sublayer_output, inline_330_residual, inline_330_post, inline_330_comb
        hidden_states, pre_mix = (inline_330_result, block_ffn_pre)
        del inline_330_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[33].engram is not None:
        hidden_states = self.layers[33].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 33],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 33, hidden_states, input_ids.shape[0])
    block_self = self.layers[33]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_335_result = hc_post(inline_335_sublayer_output, inline_335_residual, inline_335_post, inline_335_comb)
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
        inline_339_result = hc_pre_norm(inline_338_hidden_states, inline_338_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_340_result = hc_post(inline_340_sublayer_output, inline_340_residual, inline_340_post, inline_340_comb)
        del inline_340_sublayer_output, inline_340_residual, inline_340_post, inline_340_comb
        hidden_states, pre_mix = (inline_340_result, block_ffn_pre)
        del inline_340_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[34].engram is not None:
        hidden_states = self.layers[34].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 34],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 34, hidden_states, input_ids.shape[0])
    block_self = self.layers[34]
    block_decode_mask = local_active
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
        inline_345_result = hc_post(inline_345_sublayer_output, inline_345_residual, inline_345_post, inline_345_comb)
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
        inline_349_result = hc_pre_norm(inline_348_hidden_states, inline_348_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_350_result = hc_post(inline_350_sublayer_output, inline_350_residual, inline_350_post, inline_350_comb)
        del inline_350_sublayer_output, inline_350_residual, inline_350_post, inline_350_comb
        hidden_states, pre_mix = (inline_350_result, block_ffn_pre)
        del inline_350_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[35].engram is not None:
        hidden_states = self.layers[35].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 35],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 35, hidden_states, input_ids.shape[0])
    block_self = self.layers[35]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_355_result = hc_post(inline_355_sublayer_output, inline_355_residual, inline_355_post, inline_355_comb)
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
        inline_359_result = hc_pre_norm(inline_358_hidden_states, inline_358_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_360_result = hc_post(inline_360_sublayer_output, inline_360_residual, inline_360_post, inline_360_comb)
        del inline_360_sublayer_output, inline_360_residual, inline_360_post, inline_360_comb
        hidden_states, pre_mix = (inline_360_result, block_ffn_pre)
        del inline_360_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[36].engram is not None:
        hidden_states = self.layers[36].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 36],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 36, hidden_states, input_ids.shape[0])
    block_self = self.layers[36]
    block_decode_mask = local_active
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
        inline_365_result = hc_post(inline_365_sublayer_output, inline_365_residual, inline_365_post, inline_365_comb)
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
        inline_369_result = hc_pre_norm(inline_368_hidden_states, inline_368_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_370_result = hc_post(inline_370_sublayer_output, inline_370_residual, inline_370_post, inline_370_comb)
        del inline_370_sublayer_output, inline_370_residual, inline_370_post, inline_370_comb
        hidden_states, pre_mix = (inline_370_result, block_ffn_pre)
        del inline_370_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[37].engram is not None:
        hidden_states = self.layers[37].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 37],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 37, hidden_states, input_ids.shape[0])
    block_self = self.layers[37]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_375_result = hc_post(inline_375_sublayer_output, inline_375_residual, inline_375_post, inline_375_comb)
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
        inline_379_result = hc_pre_norm(inline_378_hidden_states, inline_378_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_380_result = hc_post(inline_380_sublayer_output, inline_380_residual, inline_380_post, inline_380_comb)
        del inline_380_sublayer_output, inline_380_residual, inline_380_post, inline_380_comb
        hidden_states, pre_mix = (inline_380_result, block_ffn_pre)
        del inline_380_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[38].engram is not None:
        hidden_states = self.layers[38].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 38],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 38, hidden_states, input_ids.shape[0])
    block_self = self.layers[38]
    block_decode_mask = local_active
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
        inline_385_result = hc_post(inline_385_sublayer_output, inline_385_residual, inline_385_post, inline_385_comb)
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
        inline_389_result = hc_pre_norm(inline_388_hidden_states, inline_388_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_390_result = hc_post(inline_390_sublayer_output, inline_390_residual, inline_390_post, inline_390_comb)
        del inline_390_sublayer_output, inline_390_residual, inline_390_post, inline_390_comb
        hidden_states, pre_mix = (inline_390_result, block_ffn_pre)
        del inline_390_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    if self.layers[39].engram is not None:
        hidden_states = self.layers[39].engram.forward_decode(
            hidden_states,
            self._decode_engram_values[input_ids.shape[0], 39],
            local_active,
            sequence_parallel=self.sequence_parallel,
        )
    self._capture_aux(auxiliary, 39, hidden_states, input_ids.shape[0])
    block_self = self.layers[39]
    block_decode_mask = local_active
    block_image_mask = None
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
        inline_395_result = hc_post(inline_395_sublayer_output, inline_395_residual, inline_395_post, inline_395_comb)
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
        inline_399_result = hc_pre_norm(inline_398_hidden_states, inline_398_pre_mix, block_self.ffn_norm.weight, 1e-20)
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
        inline_400_result = hc_post(inline_400_sublayer_output, inline_400_residual, inline_400_post, inline_400_comb)
        del inline_400_sublayer_output, inline_400_residual, inline_400_post, inline_400_comb
        hidden_states, pre_mix = (inline_400_result, block_ffn_pre)
        del inline_400_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    output = self._finish_hc(hidden_states, pre_mix, input_ids.shape[0])
    active = runtime.decode.active[:, None]
    output = torch.where(active, output, 0)
    if self.aux_hidden_state_layers:
        return (output, [torch.where(active, value, 0) for value in auxiliary])
    return output
