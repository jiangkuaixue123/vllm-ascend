# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse only RMSNorm normalization work; keep the native mean/add/rsqrt."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

_BLOCK_SIZE = 1024
_NORMALIZE_ROWS = 4
_NORMALIZE_CHANNELS = 256

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _square_kernel(states, squares, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
        channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offset = tl.program_id(0) * HIDDEN + channel
        value = tl.load(states + offset, channel < HIDDEN, 0).to(tl.float32)
        tl.store(squares + offset, value * value, channel < HIDDEN)

    @triton.jit
    def _normalize_kernel(
        states,
        weight,
        means,
        output,
        ROWS: tl.constexpr,
        HIDDEN: tl.constexpr,
        EPS: tl.constexpr,
        TILE_ROWS: tl.constexpr,
        TILE_CHANNELS: tl.constexpr,
    ):
        rows = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
        channels = tl.program_id(1) * TILE_CHANNELS + tl.arange(0, TILE_CHANNELS)
        offsets = rows[:, None] * HIDDEN + channels[None, :]
        valid = (rows[:, None] < ROWS) & (channels[None, :] < HIDDEN)
        value = tl.load(states + offsets, valid, 0).to(tl.float32)
        # A vector of row scales keeps the NPU vector rsqrt implementation.
        # Scalar rsqrt can differ at a BF16 midpoint even with FP fusion off.
        mean = tl.load(means + rows, rows < ROWS, 1.0)
        scales = tl.rsqrt(mean + EPS)
        normalized = value * scales[:, None]
        coefficient = tl.load(weight + channels, channels < HIDDEN, 0).to(tl.float32)
        result = coefficient[None, :] * normalized
        tl.store(output + offsets, result.to(tl.bfloat16), valid)

    def square_fp32(hidden_states):
        squares = torch.empty_like(hidden_states, dtype=torch.float32)
        rows, hidden = hidden_states.shape
        _square_kernel[(rows, triton.cdiv(hidden, _BLOCK_SIZE))](
            hidden_states, squares, HIDDEN=hidden, BLOCK=_BLOCK_SIZE, enable_fp_fusion=False
        )
        return squares

    def rms_rsqrt_fused(hidden_states, weight, eps):
        squares = square_fp32(hidden_states)
        # Preserve the original FP32 reduction shape and native reduction order.
        means = squares.mean(-1, keepdim=True)
        output = torch.empty_like(hidden_states)
        rows, hidden = hidden_states.shape
        _normalize_kernel[(triton.cdiv(rows, _NORMALIZE_ROWS), triton.cdiv(hidden, _NORMALIZE_CHANNELS))](
            hidden_states,
            weight,
            means,
            output,
            ROWS=rows,
            HIDDEN=hidden,
            EPS=eps,
            TILE_ROWS=_NORMALIZE_ROWS,
            TILE_CHANNELS=_NORMALIZE_CHANNELS,
            enable_fp_fusion=False,
        )
        return output

else:
    rms_rsqrt_fused = None
