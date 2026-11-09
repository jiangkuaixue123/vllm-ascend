# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiGLU and route-factor fusion at the native quantized GMM boundary."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["ROWS"])
def _activation_kernel(
    gate,
    up,
    factor,
    output,
    ROWS,
    DIM: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    U0: tl.constexpr,
    U1: tl.constexpr,
    HAS_FACTOR: tl.constexpr,
    F0: tl.constexpr,
    LIMIT: tl.constexpr,
    TILE: tl.constexpr,
):
    columns = tl.arange(0, TILE)
    tiles: tl.constexpr = triton.cdiv(DIM, TILE)
    for block in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        row = block // tiles
        dim = block % tiles * TILE + columns
        g = tl.load(gate + row * G0 + dim * G1, dim < DIM, other=0).to(tl.float32)
        u = tl.load(up + row * U0 + dim * U1, dim < DIM, other=0).to(tl.float32)
        if LIMIT > 0:
            g = tl.where(g > LIMIT, LIMIT, g)
            u = tl.where(u < -LIMIT, -LIMIT, u)
            u = tl.where(u > LIMIT, LIMIT, u)
        value = tl.div_rn(g, 1.0 + tl.exp(-g)) * u
        if HAS_FACTOR:
            value = value * tl.load(factor + row * F0).to(tl.float32)
        tl.store(output + row * DIM + dim, value, dim < DIM)


def clipped_swiglu_route(gate, up, factor=None, limit=10.0, dtype=torch.bfloat16):
    if gate.ndim != 2 or up.shape != gate.shape:
        raise ValueError("SwiGLU fusion requires matching two-dimensional gate/up tensors")
    rows, dim = gate.shape
    if factor is not None and factor.shape not in ((rows,), (rows, 1)):
        raise ValueError("SwiGLU fusion requires one factor per row")
    output = torch.empty((rows, dim), device=gate.device, dtype=dtype)
    if rows:
        _activation_kernel[(min(rows * triton.cdiv(dim, 1024), 64),)](
            gate,
            up,
            factor if factor is not None else gate,
            output,
            ROWS=rows,
            DIM=dim,
            G0=gate.stride(0),
            G1=gate.stride(1),
            U0=up.stride(0),
            U1=up.stride(1),
            HAS_FACTOR=factor is not None,
            F0=factor.stride(0) if factor is not None else 0,
            LIMIT=limit,
            TILE=1024,
            enable_fp_fusion=False,
        )
    return output
