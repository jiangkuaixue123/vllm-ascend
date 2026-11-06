# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse only RMSNorm normalization work; keep the native mean/add/rsqrt."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

_BLOCK_SIZE = 1024

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _square_kernel(states, squares, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
        channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offset = tl.program_id(0) * HIDDEN + channel
        value = tl.load(states + offset, channel < HIDDEN, 0).to(tl.float32)
        tl.store(squares + offset, value * value, channel < HIDDEN)

    @triton.jit
    def _normalize_kernel(states, weight, means, output, HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offset = row * HIDDEN + channel
        value = tl.load(states + offset, channel < HIDDEN, 0).to(tl.float32)
        scale = tl.rsqrt(tl.load(means + row) + EPS)
        normalized = value * scale
        coefficient = tl.load(weight + channel, channel < HIDDEN, 0).to(tl.float32)
        result = coefficient * normalized
        tl.store(output + offset, result.to(tl.bfloat16), channel < HIDDEN)

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
        _normalize_kernel[(rows, triton.cdiv(hidden, _BLOCK_SIZE))](
            hidden_states, weight, means, output, HIDDEN=hidden, EPS=eps, BLOCK=_BLOCK_SIZE, enable_fp_fusion=False
        )
        return output

else:
    rms_rsqrt_fused = None
