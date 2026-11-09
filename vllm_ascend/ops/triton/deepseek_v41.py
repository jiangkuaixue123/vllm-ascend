# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit V4.1 vector fusions used by versioned flattened forward bundles."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _hc_gates_sinkhorn_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    lane = tl.arange(0, 4)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    comb_scale = tl.load(scale_ptr + 2)
    pre_logit = tl.load(mixes_ptr + token * 24 + lane) * pre_scale + tl.load(base_ptr + lane)
    post_logit = tl.load(mixes_ptr + token * 24 + 4 + lane) * post_scale + tl.load(base_ptr + 4 + lane)
    pre = tl.div_rn(1.0, 1.0 + tl.exp(-pre_logit)) + EPS
    post = 2.0 * tl.div_rn(1.0, 1.0 + tl.exp(-post_logit))
    matrix_offset = lane[:, None] * 4 + lane[None, :]
    matrix = tl.load(mixes_ptr + token * 24 + 8 + matrix_offset) * comb_scale
    matrix = matrix + tl.load(base_ptr + 8 + matrix_offset)
    exponential = tl.exp(matrix - tl.max(matrix, axis=1)[:, None])
    matrix = tl.div_rn(exponential, tl.sum(exponential, axis=1)[:, None]) + EPS
    matrix = tl.div_rn(matrix, tl.sum(matrix, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        matrix = tl.div_rn(matrix, tl.sum(matrix, axis=1)[:, None] + EPS)
        matrix = tl.div_rn(matrix, tl.sum(matrix, axis=0)[None, :] + EPS)
    tl.store(pre_ptr + token * 4 + lane, pre)
    tl.store(post_ptr + token * 4 + lane, post)
    tl.store(comb_ptr + token * 16 + matrix_offset, matrix)


def hc_gates_sinkhorn(mixes, scale, base, *, sinkhorn_iters=20, eps=1e-6):
    """Return FP32 next-pre, post and comb; never consume the preceding pre."""
    if mixes.ndim != 2 or mixes.shape[1] != 24 or scale.shape != (3,) or base.shape != (24,):
        raise ValueError("V4.1 HC fusion requires mixes[T,24], scale[3], base[24]")
    if sinkhorn_iters < 1 or eps < 0:
        raise ValueError("Sinkhorn iterations must be positive and eps nonnegative")
    if any(value.dtype != torch.float32 or not value.is_contiguous() for value in (mixes, scale, base)):
        raise ValueError("V4.1 HC fusion requires contiguous FP32 inputs")
    if any(value.device != mixes.device for value in (scale, base)):
        raise ValueError("HC inputs must share a device")
    rows = mixes.shape[0]
    pre = torch.empty((rows, 4), dtype=torch.float32, device=mixes.device)
    post = torch.empty_like(pre)
    comb = torch.empty((rows, 4, 4), dtype=torch.float32, device=mixes.device)
    if rows:
        _hc_gates_sinkhorn_kernel[(rows,)](
            mixes, scale, base, pre, post, comb, ITERS=sinkhorn_iters, EPS=eps, enable_fp_fusion=False
        )
    return pre, post, comb


@triton.jit
def _round_binary_float(value, MANTISSA: tl.constexpr, MIN_EXPONENT: tl.constexpr):
    magnitude = tl.abs(value)
    bits = magnitude.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127
    quantum_bits = (tl.maximum(exponent, MIN_EXPONENT) - MANTISSA + 127) << 23
    quantum = quantum_bits.to(tl.float32, bitcast=True)
    scaled = tl.div_rn(magnitude, quantum)
    lower = tl.floor(scaled)
    fraction = scaled - lower
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    rounded = (lower + increment.to(tl.float32)) * quantum
    sign = value.to(tl.int32, bitcast=True) & (-2147483648)
    rounded_bits = (rounded.to(tl.int32, bitcast=True) & 2147483647) | sign
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _round_scale_up_kernel(amax, RECIPROCAL: tl.constexpr):
    scaled = amax * RECIPROCAL
    bits = scaled.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 8388607) != 0).to(tl.int32)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _quant_dequant_kernel(
    input_ptr,
    output_ptr,
    groups,
    BLOCK: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    FP8: tl.constexpr,
    E4M3_SCALE: tl.constexpr,
):
    columns = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), tl.cdiv(groups, GROUP_TILE), tl.num_programs(0)):
        rows = tile * GROUP_TILE + tl.arange(0, GROUP_TILE)
        offsets = rows[:, None] * BLOCK + columns[None, :]
        values = tl.load(input_ptr + offsets, mask=rows[:, None] < groups, other=0).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=1)
        has_nan = tl.sum((values != values).to(tl.int32), axis=1) != 0
        amax = tl.where(has_nan, float("nan"), amax)
        if FP8:
            amax = tl.where(amax < 1e-4, 1e-4, amax)
            scale = _round_scale_up_kernel(amax, 1.0 / 448.0)
            maximum: tl.constexpr = 448.0
        elif E4M3_SCALE:
            amax = tl.where(amax < 6.0 * 2.0**-9, 6.0 * 2.0**-9, amax)
            scale = _round_binary_float(tl.div_rn(amax, 6.0), 3, -6)
            scale = tl.where(tl.abs(scale) > 448.0, float("nan"), scale)
            maximum: tl.constexpr = 6.0
        else:
            amax = tl.where(amax < 6.0 * 2.0**-126, 6.0 * 2.0**-126, amax)
            scale = _round_scale_up_kernel(amax, 1.0 / 6.0)
            maximum: tl.constexpr = 6.0
        normalized = tl.div_rn(values, scale[:, None])
        clipped = tl.where(normalized < -maximum, -maximum, normalized)
        clipped = tl.where(clipped > maximum, maximum, clipped)
        if FP8:
            rounded = _round_binary_float(clipped, 3, -6)
            rounded = tl.where(tl.abs(rounded) > 448.0, float("nan"), rounded)
        else:
            rounded = _round_binary_float(clipped, 1, 0)
        result = rounded * scale[:, None]
        tl.store(output_ptr + offsets, result, mask=rows[:, None] < groups)


def _quant_dequant(x, block_size, *, fp8, scale_format, inplace):
    if x.ndim == 0 or block_size not in (16, 32) or x.shape[-1] % block_size:
        raise ValueError("QDQ fusion requires a last dimension divisible by block size 16 or 32")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("QDQ input must use BF16, FP16 or FP32")
    if scale_format not in ("e8m0", "e4m3"):
        raise ValueError("Scale format must be e8m0 or e4m3")
    source = x.contiguous()
    output = torch.empty_like(source)
    groups = source.numel() // block_size
    if groups:
        # Cap at the 64 vector cores of the measured A5; large tensors loop
        # over tiles within each program instead of launching thousands of CTAs.
        _quant_dequant_kernel[(min(triton.cdiv(groups, 16), 64),)](
            source,
            output,
            groups,
            BLOCK=block_size,
            GROUP_TILE=16,
            FP8=fp8,
            E4M3_SCALE=scale_format == "e4m3",
            enable_fp_fusion=False,
        )
    if inplace:
        x.copy_(output)
        return x
    return output


def fp8_quant_dequant(x, block_size=32, *, inplace=False):
    return _quant_dequant(x, block_size, fp8=True, scale_format="e8m0", inplace=inplace)


def fp4_quant_dequant(x, block_size=32, *, scale_format="e8m0", inplace=False):
    return _quant_dequant(x, block_size, fp8=False, scale_format=scale_format, inplace=inplace)
