# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device-independent text primitives for DeepSeek V4.1.

Equations follow inference/model.py and inference/kernel.py in the official
deepseek-ai/DeepSeek-V4.1-Flash revision
314113f8a7dadb388215396ba5fbee2c610a04d6. Distributed execution and cache ownership
belong to the integrating model, not these primitives.
"""

import torch
import torch.nn.functional as F
from torch import nn

ROUTER_NORMALIZATION_EPS = 1e-20
_FUSED_HC_WIDTH = 4
_FUSED_HC_ROUNDS = 20
_FUSED_HC_EPS = 1e-6


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    states = hidden_states.float()
    normalized = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + eps)
    return (weight * normalized).to(hidden_states.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 pre/post gates and the input-to-output residual mixing matrix."""
    if hc_mult < 1 or sinkhorn_iters < 1 or eps < 0:
        raise ValueError("HC width/iterations must be positive and eps nonnegative")
    mix_width = (hc_mult + 2) * hc_mult
    if mixes.shape[-1] != mix_width or base.shape != (mix_width,):
        raise ValueError("HC mixes/base dimensions do not match hc_mult")
    if scale.shape != (3,):
        raise ValueError("HC scale must have three coefficients")

    mixes, scale, base = mixes.float(), scale.float(), base.float()
    if mixes.device.type == "npu":
        from vllm_ascend.models.common.ops.hc_gate_fused import try_hc_gate_fused

        fused = try_hc_gate_fused(mixes, scale, base, hc_mult, sinkhorn_iters, eps)
        if fused is not None:
            return fused
    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult])
    comb = (mixes[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]).unflatten(-1, (hc_mult, hc_mult))
    if (
        comb.device.type == "npu"
        and hc_mult == _FUSED_HC_WIDTH
        and sinkhorn_iters == _FUSED_HC_ROUNDS
        and eps == _FUSED_HC_EPS
        and torch.is_inference_mode_enabled()
        and comb.is_contiguous()
    ):
        # Defer Triton driver discovery until an NPU worker needs it. CPU and autograd use the reference below.
        from vllm_ascend.ops.triton.hc_sinkhorn import hc_sinkhorn4

        if hc_sinkhorn4 is not None:
            return pre, post, hc_sinkhorn4(comb)

    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(
    hidden_states: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float = 1e-20,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize jointly over all HC copies before projecting their coefficients."""
    states = hidden_states.flatten(-2).float()
    rstd = torch.rsqrt(states.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(states, fn.float()) * rstd
    return hc_split_sinkhorn(mixes, scale, base, hidden_states.shape[-2], sinkhorn_iters, hc_eps)


def make_identity_pre_mix(hidden_states: torch.Tensor) -> torch.Tensor:
    """Select the first HC copy at the start of the backbone."""
    pre_mix = hidden_states.new_zeros(hidden_states.shape[:-1], dtype=torch.float32)
    pre_mix[..., 0] = 1
    return pre_mix


def hc_pre(hidden_states: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    mixed = (pre_mix.unsqueeze(-1) * hidden_states.float()).sum(-2)
    return mixed.to(hidden_states.dtype)


def hc_post(
    sublayer_output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    mixed = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)
    output = post.unsqueeze(-1) * sublayer_output.unsqueeze(-2) + mixed
    return output.to(sublayer_output.dtype)


class DeepseekV41HCMix(nn.Module):
    """HC coefficient projection using caller-owned, checkpoint-loaded parameters."""

    def __init__(
        self,
        fn: nn.Parameter,
        scale: nn.Parameter,
        base: nn.Parameter,
        norm_eps: float = 1e-20,
        hc_eps: float = 1e-6,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.fn = fn
        self.scale = scale
        self.base = base
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, hidden_states: torch.Tensor):
        return hc_mixes(
            hidden_states,
            self.fn,
            self.scale,
            self.base,
            self.norm_eps,
            self.hc_eps,
            self.sinkhorn_iters,
        )


class DeepseekV41ResidualSublayer(nn.Module):
    """Consume the preceding sublayer's pre gate, returning this one's next gate.

    Compose attention followed by FFN using each call's returned pre_mix. Both
    mix and sublayer modules must return tensors directly (no linear bias tuple).
    The mix module returns the three tensors pre, post, comb.
    """

    def __init__(self, mix: nn.Module, norm: nn.Module, sublayer: nn.Module):
        super().__init__()
        self.mix = mix
        self.norm = norm
        self.sublayer = sublayer

    def forward(
        self, hidden_states: torch.Tensor, pre_mix: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_pre, post, comb = self.mix(hidden_states)
        sublayer_input = self.norm(hc_pre(hidden_states, pre_mix))
        output = self.sublayer(sublayer_input, *args, **kwargs)
        return hc_post(output, hidden_states, post, comb), next_pre


def route_sqrtsoftplus(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.5,
    gate_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts with bias; weight them using the unbiased FP32 scores."""
    if not 1 <= top_k <= router_logits.shape[-1]:
        raise ValueError("top_k must be between one and the number of experts")
    if gate_temperature <= 0:
        raise ValueError("gate_temperature must be positive")
    scores = F.softplus(router_logits.float() / gate_temperature).sqrt()
    indices = (scores + correction_bias.float()).topk(top_k, dim=-1).indices
    weights = scores.gather(-1, indices)
    if renormalize and top_k > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + ROUTER_NORMALIZATION_EPS)
    return weights * routed_scaling_factor, indices


def clipped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    """Return FP32 activations; only the up branch has a lower clamp bound."""
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class DeepseekV41Expert(nn.Module):
    """An expert with routing weights applied before the down projection.

    Projections are supplied by the model so its loader and quantization method
    retain ownership. Each projection must return a tensor, not a bias tuple.
    """

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        swiglu_limit: float = 10.0,
    ):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.swiglu_limit = swiglu_limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activations = clipped_swiglu(
            self.gate_proj(hidden_states),
            self.up_proj(hidden_states),
            self.swiglu_limit,
        )
        if routing_weights is not None:
            activations = routing_weights * activations
        return self.down_proj(activations.to(hidden_states.dtype))
