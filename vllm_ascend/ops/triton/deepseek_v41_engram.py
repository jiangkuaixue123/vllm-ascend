# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unrotated Engram dual RMS gate and residual epilogue after native projection."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["ROWS"])
def _engram_scores(x, kv, qw, kw, dots, ROWS, DIM: tl.constexpr, EPS: tl.constexpr, TILE: tl.constexpr):
    channels = tl.arange(0, TILE)
    for block in range(tl.program_id(0), ROWS * 4, tl.num_programs(0)):
        row, stream = block // 4, block % 4
        query = tl.load(x + block * DIM + channels, channels < DIM, other=0).to(tl.float32)
        key = tl.load(kv + (row * 5 + stream) * DIM + channels, channels < DIM, other=0).to(tl.float32)
        q_weight = tl.load(qw + stream * DIM + channels, channels < DIM, other=0).to(tl.float32)
        k_weight = tl.load(kw + stream * DIM + channels, channels < DIM, other=0).to(tl.float32)
        weight = q_weight * k_weight
        rstd = tl.rsqrt(tl.sum(query * query, axis=0) / DIM + EPS)
        rstd = rstd * tl.rsqrt(tl.sum(key * key, axis=0) / DIM + EPS)
        dot = tl.sum((query * weight) * key, axis=0) * rstd * (DIM**-0.5)
        tl.store(dots + block, dot)


@triton.jit(do_not_specialize=["ROWS"])
def _engram_residual(x, kv, dots, mask, output, ROWS, DIM: tl.constexpr, HAS_MASK: tl.constexpr, TILE: tl.constexpr):
    channels = tl.arange(0, TILE)
    streams = tl.arange(0, 4)
    tiles: tl.constexpr = triton.cdiv(DIM, TILE)
    for block in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        row = block // tiles
        dim = block % tiles * TILE + channels
        dot = tl.load(dots + row * 4 + streams)
        magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
        bits = (magnitude.to(tl.int32, bitcast=True) & 2147483647) | (dot.to(tl.int32, bitcast=True) & -2147483648)
        gate = tl.div_rn(1.0, 1.0 + tl.exp(-bits.to(tl.float32, bitcast=True)))
        if HAS_MASK:
            gate = tl.where(tl.load(mask + row) != 0, gate, 0.0)
        query = tl.load(x + (row * 4 + streams[:, None]) * DIM + dim[None, :], dim[None, :] < DIM, other=0).to(
            tl.float32
        )
        value = tl.load(kv + (row * 5 + 4) * DIM + dim, dim < DIM, other=0).to(tl.float32)
        result = query + gate[:, None] * value[None, :]
        tl.store(output + (row * 4 + streams[:, None]) * DIM + dim[None, :], result, dim[None, :] < DIM)


def engram_gate_residual(x, projected_kv, q_weight, k_weight, token_mask=None, eps=1e-20):
    rows, streams, dim = x.shape
    if streams != 4 or projected_kv.shape != (rows, 5 * dim):
        raise ValueError("Engram fusion requires four HC streams and projected key/value rows")
    if q_weight.shape != (4, dim) or k_weight.shape != q_weight.shape:
        raise ValueError("Engram gate weights disagree with the HC geometry")
    if token_mask is not None and token_mask.shape != (rows,):
        raise ValueError("Engram mask must contain one entry per token")
    if not all(v.is_contiguous() for v in (x, projected_kv, q_weight, k_weight)):
        raise ValueError("Engram fusion requires contiguous tensors")
    result = torch.empty_like(x)
    if rows:
        dots = torch.empty((rows, 4), dtype=torch.float32, device=x.device)
        _engram_scores[(min(rows * 4, 64),)](
            x,
            projected_kv,
            q_weight,
            k_weight,
            dots,
            ROWS=rows,
            DIM=dim,
            EPS=eps,
            TILE=triton.next_power_of_2(dim),
            enable_fp_fusion=False,
            multibuffer=False,
        )
        _engram_residual[(min(rows * triton.cdiv(dim, 512), 64),)](
            x,
            projected_kv,
            dots,
            token_mask if token_mask is not None else x,
            result,
            ROWS=rows,
            DIM=dim,
            HAS_MASK=token_mask is not None,
            TILE=512,
            enable_fp_fusion=False,
        )
    return result
