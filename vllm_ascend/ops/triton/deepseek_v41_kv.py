# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV RMSNorm, adjacent-pair RoPE and FP8 round-trip in one vector kernel."""

import torch
from vllm.triton_utils import tl, triton

from .deepseek_v41 import _round_binary_float, _round_scale_up_kernel


@triton.jit(do_not_specialize=["ROWS"])
def _kv_norm_rope_qdq_kernel(
    x, weight, cosine, sine, output, ROWS, DIM: tl.constexpr, ROPE: tl.constexpr, EPS: tl.constexpr
):
    channels = tl.arange(0, DIM)
    weights = tl.load(weight + channels).to(tl.float32)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        values = tl.load(x + row * DIM + channels).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / DIM
        normalized = weights * (values * tl.rsqrt(variance + EPS))
        normalized = normalized.to(x.dtype.element_ty).to(tl.float32)
        paired = tl.gather(normalized, channels ^ 1, axis=0)
        rotary = channels >= DIM - ROPE
        phase = (channels - (DIM - ROPE)) // 2
        c = tl.load(cosine + row * (ROPE // 2) + phase, rotary, other=1).to(tl.float32)
        s = tl.load(sine + row * (ROPE // 2) + phase, rotary, other=0).to(tl.float32)
        even = normalized * c - paired * s
        odd = paired * s + normalized * c
        rotated = tl.where((channels & 1) == 0, even, odd)
        values = tl.where(rotary, rotated, normalized).to(x.dtype.element_ty).to(tl.float32)
        groups = tl.reshape(values, (DIM // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 1e-4)
        amax = tl.where(tl.sum((groups != groups).to(tl.int32), axis=1) != 0, float("nan"), amax)
        scale = _round_scale_up_kernel(amax, 1.0 / 448.0)
        quantized = tl.div_rn(groups, scale[:, None])
        quantized = tl.where(quantized < -448.0, -448.0, quantized)
        quantized = tl.where(quantized > 448.0, 448.0, quantized)
        dequantized = _round_binary_float(quantized, 3, -6) * scale[:, None]
        tl.store(output + row * DIM + channels, tl.reshape(dequantized, (DIM,)))


def kv_norm_rope_qdq(x, weight, cosine, sine, eps=1e-20):
    if x.ndim != 2 or x.shape[1] != 512 or weight.shape != (512,):
        raise ValueError("KV fusion requires 512-dimensional rows and norm weights")
    if x.dtype != torch.bfloat16 or cosine.shape != (x.shape[0], 32) or sine.shape != cosine.shape:
        raise ValueError("KV fusion requires BF16 inputs and a 64-dimensional RoPE")
    if not all(t.is_contiguous() for t in (x, weight, cosine, sine)):
        raise ValueError("KV fusion requires contiguous input tensors")
    result = torch.empty_like(x)
    if x.shape[0]:
        _kv_norm_rope_qdq_kernel[(min(x.shape[0], 64),)](
            x, weight, cosine, sine, result, ROWS=x.shape[0], DIM=512, ROPE=64, EPS=eps, enable_fp_fusion=False
        )
    return result
