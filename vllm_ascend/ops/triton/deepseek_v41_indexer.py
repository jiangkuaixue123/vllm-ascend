# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer GEMM epilogue, preserving the product's original dtype."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _indexer_reduce_kernel(
    values,
    weights,
    output,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    KEYS: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    H: tl.constexpr,
    TILE: tl.constexpr,
):
    heads = tl.arange(0, H)
    keys = tl.arange(0, TILE)
    tiles: tl.constexpr = triton.cdiv(KEYS, TILE)
    for tile in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        row = tile // tiles
        column = tile % tiles * TILE + keys
        value = tl.load(
            values + row * V0 + heads[:, None] * V1 + column[None, :] * V2,
            (heads[:, None] < HEADS) & (column[None, :] < KEYS),
            other=0,
        ).to(tl.float32)
        weight = tl.load(weights + row * W0 + heads * W1, heads < HEADS, other=0).to(tl.float32)
        value = tl.where(value < 0, 0, value) * weight[:, None]
        value = value.to(output.dtype.element_ty).to(tl.float32)
        result = tl.sum(value, axis=0)
        tl.store(output + row * KEYS + column, result, column < KEYS)


def indexer_reduce(per_head, weights):
    rows, heads, keys = per_head.shape
    if weights.shape != (rows, heads):
        raise ValueError("Indexer weights must match token and head dimensions")
    dtype = torch.promote_types(per_head.dtype, weights.dtype)
    output = torch.empty((rows, keys), dtype=dtype, device=per_head.device)
    if rows and keys:
        _indexer_reduce_kernel[(min(rows * triton.cdiv(keys, 128), 64),)](
            per_head,
            weights,
            output,
            ROWS=rows,
            HEADS=heads,
            KEYS=keys,
            V0=per_head.stride(0),
            V1=per_head.stride(1),
            V2=per_head.stride(2),
            W0=weights.stride(0),
            W1=weights.stride(1),
            H=triton.next_power_of_2(heads),
            TILE=128,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return output
