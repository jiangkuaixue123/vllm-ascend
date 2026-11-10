# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_hc import hc_pre_norm, hc_post
from vllm_ascend.ops.triton.deepseek_v41 import hc_gates_sinkhorn


def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
    # Statically expanded model blocks retain the measured execution order.
    hidden = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
    hidden = hidden.unsqueeze(-2).repeat(1, self.config.hc_mult, 1)
    pre = make_identity_pre_mix(hidden)
    block_self = self.mtp[0]
    block_decode_mask = None
    block_image_mask = None
    block_hidden_states = hidden
    block_pre_mix = pre
    block_runtime = positions
    if block_hidden_states.shape[0] == 0:
        block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
        hidden, pre = (block_hidden_states, block_pre_mix)
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
        hidden, pre = (inline_10_result, block_ffn_pre)
        del inline_10_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    block_self = self.mtp[1]
    block_decode_mask = None
    block_image_mask = None
    block_hidden_states = hidden
    block_pre_mix = pre
    block_runtime = positions
    if block_hidden_states.shape[0] == 0:
        block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
        hidden, pre = (block_hidden_states, block_pre_mix)
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
        hidden, pre = (inline_20_result, block_ffn_pre)
        del inline_20_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    block_self = self.mtp[2]
    block_decode_mask = None
    block_image_mask = None
    block_hidden_states = hidden
    block_pre_mix = pre
    block_runtime = positions
    if block_hidden_states.shape[0] == 0:
        block_self.ffn(block_hidden_states.new_empty((0, block_self.config.hidden_size)))
        hidden, pre = (block_hidden_states, block_pre_mix)
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
        hidden, pre = (inline_30_result, block_ffn_pre)
        del inline_30_result
        del block_attended, block_attn_pre, block_comb, block_ffn_input, block_ffn_pre, block_output, block_post
    del block_decode_mask, block_hidden_states, block_image_mask, block_pre_mix, block_runtime, block_self
    return hc_pre_norm(hidden, pre)
