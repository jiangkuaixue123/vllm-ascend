# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import sys

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.core import (
    DeepseekV41Expert,
    DeepseekV41HCMix,
    DeepseekV41ResidualSublayer,
    clipped_swiglu,
    hc_post,
    hc_pre,
    hc_split_sinkhorn,
    make_identity_pre_mix,
    rms_norm,
    route_sqrtsoftplus,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_scales_in_fp32_before_restoring_dtype(dtype):
    states = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=dtype)
    weight = torch.tensor([2.0, 0.5], dtype=dtype)
    actual = rms_norm(states, weight)
    expected = torch.tensor([[6 / math.sqrt(12.5), 2 / math.sqrt(12.5)], [0, 0]])
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected.to(dtype))


def test_hc_gates_and_sinkhorn_match_hand_calculated_symmetric_case():
    log_three = math.log(3)
    base = torch.tensor([log_three, -log_three, log_three, -log_three, log_three, 0, 0, log_three])
    pre, post, comb = hc_split_sinkhorn(torch.zeros(1, 8), torch.ones(3), base, hc_mult=2, eps=0)
    torch.testing.assert_close(pre, torch.tensor([[0.75, 0.25]]))
    torch.testing.assert_close(post, torch.tensor([[1.5, 0.5]]))
    torch.testing.assert_close(comb, torch.tensor([[[0.75, 0.25], [0.25, 0.75]]]))


def test_hc_projection_normalizes_jointly_over_all_residual_copies():
    fn = torch.zeros(8, 4)
    fn[0, 0] = 1
    projection = DeepseekV41HCMix(
        nn.Parameter(fn),
        nn.Parameter(torch.tensor([2.0, 1.0, 1.0])),
        nn.Parameter(torch.zeros(8)),
        norm_eps=0,
        hc_eps=0,
    )
    pre, _, _ = projection(torch.tensor([[[3.0, 4.0], [0.0, 0.0]]]))
    # Joint RMS is sqrt((9 + 16) / 4) = 2.5, not sqrt(12.5).
    expected = torch.tensor([[1 / (1 + math.exp(-2.4)), 0.5]])
    torch.testing.assert_close(pre, expected)


def test_four_stream_hc_on_cpu_preserves_reference_and_autograd(monkeypatch):
    # The CPU primitives also work when the optional Triton implementation cannot be imported.
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.hc_sinkhorn", None)
    gen = torch.Generator().manual_seed(42)
    mixes = torch.randn(3, 24, generator=gen, requires_grad=True)
    scale = torch.tensor([0.5, 1.5, -1.0])
    base = torch.randn(24, generator=gen)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, 4)
    (pre[:, 0] + post[:, 1] + comb[:, 1, 2]).sum().backward()
    assert torch.isfinite(mixes.grad).all()
    assert (mixes.grad[:, 8:].abs().sum(-1) > 0).all()
    with torch.inference_mode():
        inferred = hc_split_sinkhorn(mixes, scale, base, 4)
    for actual, expected in zip(inferred, (pre, post, comb)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hc_post_uses_comb_input_to_output_direction():
    output = torch.tensor([[10.0, 20.0]])
    residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    post = torch.tensor([[2.0, 3.0]])
    comb = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    expected = torch.tensor([[[21.0, 41.4], [31.4, 62.0]]])
    torch.testing.assert_close(hc_post(output, residual, post, comb), expected)


class FixedMix(nn.Module):
    def __init__(self, pre):
        super().__init__()
        self.register_buffer("pre", torch.tensor([pre]))
        self.register_buffer("post", torch.ones(1, 2))
        self.register_buffer("comb", torch.eye(2).unsqueeze(0))

    def forward(self, states):
        return self.pre, self.post, self.comb


class RecordInput(nn.Module):
    def __init__(self, round_output=False):
        super().__init__()
        self.seen = None
        self.round_output = round_output

    def forward(self, states):
        self.seen = states.detach().clone()
        return states.round() if self.round_output else states


def test_attention_and_ffn_use_preceding_sublayer_pre_mix():
    attention_op, ffn_op = RecordInput(), RecordInput()
    attention = DeepseekV41ResidualSublayer(FixedMix([0.0, 1.0]), nn.Identity(), attention_op)
    ffn = DeepseekV41ResidualSublayer(FixedMix([0.25, 0.75]), nn.Identity(), ffn_op)
    states = torch.tensor([[[2.0], [5.0]]])
    initial_pre = make_identity_pre_mix(states)
    states, next_pre = attention(states, initial_pre)
    states, final_pre = ffn(states, next_pre)
    # Attention reads 2, so residual becomes [4, 7]; FFN uses attention's
    # [0, 1] gate to read 7. Using either sublayer's own gate differs.
    torch.testing.assert_close(attention_op.seen, torch.tensor([[2.0]]))
    torch.testing.assert_close(ffn_op.seen, torch.tensor([[7.0]]))
    torch.testing.assert_close(states, torch.tensor([[[11.0], [14.0]]]))
    torch.testing.assert_close(hc_pre(states, final_pre), torch.tensor([[13.25]]))
    assert not torch.allclose(states, torch.tensor([[[16.25], [19.25]]]))


@pytest.mark.parametrize("outer_shape", [(3,), (2, 3)])
def test_hc_supports_packed_and_batched_tokens_with_fp32_coefficients(outer_shape):
    states = torch.ones(*outer_shape, 4, 2, dtype=torch.bfloat16)
    initial_pre = make_identity_pre_mix(states)
    assert initial_pre.dtype == torch.float32
    torch.testing.assert_close(hc_pre(states, initial_pre), states[..., 0, :])
    pre, post, comb = hc_split_sinkhorn(torch.zeros(*outer_shape, 24), torch.ones(3), torch.zeros(24), hc_mult=4)
    assert pre.shape == (*outer_shape, 4)
    assert comb.shape == (*outer_shape, 4, 4)
    assert pre.dtype == post.dtype == comb.dtype == torch.float32
    torch.testing.assert_close(pre, torch.full_like(pre, 0.5 + 1e-6))
    torch.testing.assert_close(comb.sum(-1), torch.ones_like(pre), atol=2e-6, rtol=0)


def test_correction_bias_changes_selection_but_not_routing_weights():
    # Inverse softplus gives unbiased sqrtsoftplus scores [1, 2, 3].
    logits = torch.tensor([[math.log(math.expm1(s * s)) for s in (1, 2, 3)]])
    weights, ids = route_sqrtsoftplus(logits, torch.tensor([10.0, 0.0, 0.0]), 2)
    torch.testing.assert_close(ids, torch.tensor([[0, 2]]))
    torch.testing.assert_close(weights, torch.tensor([[0.375, 1.125]]))


@pytest.mark.parametrize("top_k,renormalize", [(1, True), (2, False)])
def test_router_does_not_normalize_top_one_or_disabled_normalization(top_k, renormalize):
    logits = torch.tensor([[math.log(math.expm1(s * s)) for s in (1, 2, 3)]])
    weights, ids = route_sqrtsoftplus(logits, torch.zeros(3), top_k, renormalize=renormalize)
    torch.testing.assert_close(ids, torch.tensor([[2, 1]])[:, :top_k])
    torch.testing.assert_close(weights, torch.tensor([[4.5, 3.0]])[:, :top_k])


def test_zero_router_scores_remain_finite_with_normalization_epsilon():
    weights, _ = route_sqrtsoftplus(torch.full((1, 3), -torch.inf), torch.zeros(3), 2)
    torch.testing.assert_close(weights, torch.zeros(1, 2))


def test_swiglu_clamps_up_both_sides_but_preserves_negative_gate():
    actual = clipped_swiglu(torch.tensor([-20.0, 20.0]), torch.tensor([-20.0, 20.0]))
    expected = torch.tensor([200 / (1 + math.exp(20)), 100 / (1 + math.exp(-10))])
    torch.testing.assert_close(actual, expected)


def test_expert_applies_routing_weight_before_quantized_down_projection():
    down = RecordInput(round_output=True)
    expert = DeepseekV41Expert(nn.Identity(), nn.Identity(), down)
    output = expert(torch.ones(1, 1), torch.tensor([[0.5]]))
    torch.testing.assert_close(down.seen, torch.tensor([[0.5 / (1 + math.exp(-1))]]))
    # Rounding inside the down projection distinguishes pre-down weighting
    # (round(0.3655) = 0) from post-down weighting (round(0.7311) * 0.5).
    torch.testing.assert_close(output, torch.zeros(1, 1))
    assert not torch.equal(output, torch.full((1, 1), 0.5))


@pytest.mark.parametrize("top_k,temperature", [(0, 1), (4, 1), (1, 0)])
def test_invalid_routing_parameters_fail_before_computation(top_k, temperature):
    with pytest.raises(ValueError):
        route_sqrtsoftplus(torch.zeros(1, 3), torch.zeros(3), top_k, gate_temperature=temperature)


def test_sinkhorn_requires_at_least_one_iteration():
    with pytest.raises(ValueError, match="iterations"):
        hc_split_sinkhorn(torch.zeros(1, 8), torch.ones(3), torch.zeros(8), 2, 0)
