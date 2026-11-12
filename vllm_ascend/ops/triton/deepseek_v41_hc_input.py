# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute the delayed HC input with the native left-to-right FP32 sum."""

import torch
from vllm.triton_utils import tl, triton

_HIDDEN = 5120
_WIDTH = 4
_BLOCK = 1024


@triton.jit
def _hc_input_kernel(x, previous, output, HIDDEN: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = channels < HIDDEN
    base = row * WIDTH * HIDDEN + channels
    x0 = tl.load(x + base, valid, 0).to(tl.float32)
    x1 = tl.load(x + base + HIDDEN, valid, 0).to(tl.float32)
    x2 = tl.load(x + base + 2 * HIDDEN, valid, 0).to(tl.float32)
    x3 = tl.load(x + base + 3 * HIDDEN, valid, 0).to(tl.float32)
    w0 = tl.load(previous + row * WIDTH)
    w1 = tl.load(previous + row * WIDTH + 1)
    w2 = tl.load(previous + row * WIDTH + 2)
    w3 = tl.load(previous + row * WIDTH + 3)
    p0, p1, p2, p3 = x0 * w0, x1 * w1, x2 * w2, x3 * w3
    value = ((p0 + p1) + p2) + p3
    # Native ReduceSumARA starts from p0, so four negative-zero products
    # remain negative zero. Preserve this even if vector addition loses it.
    s0 = tl.cast(x0, tl.int32, bitcast=True) ^ tl.cast(w0, tl.int32, bitcast=True)
    s1 = tl.cast(x1, tl.int32, bitcast=True) ^ tl.cast(w1, tl.int32, bitcast=True)
    s2 = tl.cast(x2, tl.int32, bitcast=True) ^ tl.cast(w2, tl.int32, bitcast=True)
    s3 = tl.cast(x3, tl.int32, bitcast=True) ^ tl.cast(w3, tl.int32, bitcast=True)
    zero = tl.cast(s0 & s1 & s2 & s3 & -0x8000_0000, tl.float32, bitcast=True)
    value = tl.where((p0 == 0) & (p1 == 0) & (p2 == 0) & (p3 == 0), zero, value)
    tl.store(output + row * HIDDEN + channels, value.to(tl.bfloat16), valid)


def hc_input4(x, previous):
    output = torch.empty((x.shape[0], _HIDDEN), device=x.device, dtype=x.dtype)
    _hc_input_kernel[(x.shape[0], triton.cdiv(_HIDDEN, _BLOCK))](
        x, previous, output, HIDDEN=_HIDDEN, WIDTH=_WIDTH, BLOCK=_BLOCK, enable_fp_fusion=False
    )
    return output
