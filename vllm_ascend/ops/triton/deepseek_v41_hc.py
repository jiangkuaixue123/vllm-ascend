# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HC residual mixing and normalization with explicit BF16 rounding boundaries."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _hc_post_kernel(output, residual, post, comb, result, ROWS: tl.constexpr, DIM: tl.constexpr, TILE: tl.constexpr):
    dim_tiles: tl.constexpr = triton.cdiv(DIM, TILE)
    channels = tl.arange(0, TILE)
    streams = tl.arange(0, 4)
    for tile in range(tl.program_id(0), ROWS * 4 * dim_tiles, tl.num_programs(0)):
        row = tile // (4 * dim_tiles)
        out_stream = tile // dim_tiles % 4
        dim = tile % dim_tiles * TILE + channels
        values = tl.load(residual + (row * 4 + streams[:, None]) * DIM + dim[None, :], dim[None, :] < DIM, other=0).to(
            tl.float32
        )
        coefficients = tl.load(comb + row * 16 + streams * 4 + out_stream)
        mixed = tl.sum(coefficients[:, None] * values, axis=0)
        sublayer = tl.load(output + row * DIM + dim, dim < DIM, other=0).to(tl.float32)
        gate = tl.load(post + row * 4 + out_stream)
        value = gate * sublayer + mixed
        tl.store(result + (row * 4 + out_stream) * DIM + dim, value, dim < DIM)


def hc_post(sublayer_output, residual, post, comb):
    rows, dim = sublayer_output.shape
    if residual.shape != (rows, 4, dim) or post.shape != (rows, 4) or comb.shape != (rows, 4, 4):
        raise ValueError("HC post fusion requires four streams and matching token rows")
    if not all(x.is_contiguous() for x in (sublayer_output, residual, post, comb)):
        raise ValueError("HC post fusion requires contiguous inputs")
    result = torch.empty((rows, 4, dim), device=sublayer_output.device, dtype=sublayer_output.dtype)
    if rows:
        _hc_post_kernel[(min(rows * 4 * triton.cdiv(dim, 512), 64),)](
            sublayer_output, residual, post, comb, result, ROWS=rows, DIM=dim, TILE=512, enable_fp_fusion=False
        )
    return result


@triton.jit
def _hc_pre_norm_kernel(
    hidden,
    pre,
    weight,
    output,
    ROWS: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    TILE: tl.constexpr,
):
    channels = tl.arange(0, TILE)
    streams = tl.arange(0, 4)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        values = tl.load(
            hidden + (row * 4 + streams[:, None]) * DIM + channels[None, :], channels[None, :] < DIM, other=0
        ).to(tl.float32)
        coefficients = tl.load(pre + row * 4 + streams)
        mixed = tl.sum(values * coefficients[:, None], axis=0)
        # hc_pre returns the input dtype before the following RMSNorm casts to FP32.
        mixed = mixed.to(hidden.dtype.element_ty).to(tl.float32)
        if NORM:
            variance = tl.sum(mixed * mixed, axis=0) / DIM
            normalized = mixed * tl.rsqrt(variance + EPS)
            weights = tl.load(weight + channels, channels < DIM, other=0).to(tl.float32)
            mixed = weights * normalized
        tl.store(output + row * DIM + channels, mixed, channels < DIM)


def hc_pre_norm(hidden, pre, weight=None, eps=1e-20):
    rows, streams, dim = hidden.shape
    if streams != 4 or pre.shape != (rows, 4) or (weight is not None and weight.shape != (dim,)):
        raise ValueError("HC pre fusion requires four streams and a matching norm weight")
    if not all(x.is_contiguous() for x in (hidden, pre)):
        raise ValueError("HC pre fusion requires contiguous inputs")
    output = torch.empty((rows, dim), dtype=hidden.dtype, device=hidden.device)
    if rows:
        _hc_pre_norm_kernel[(min(rows, 64),)](
            hidden,
            pre,
            weight if weight is not None else pre,
            output,
            ROWS=rows,
            DIM=dim,
            EPS=eps,
            NORM=weight is not None,
            TILE=triton.next_power_of_2(dim),
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return output
