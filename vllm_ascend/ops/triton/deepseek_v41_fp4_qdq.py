# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 index Q/K quantize-dequantize with E2M1 values and E8M0 scales."""

import torch
from vllm.triton_utils import tl, triton

_GROUP_SIZE = 32
_GROUPS_PER_PROGRAM = 16


@triton.jit
def _fp4_e8m0(src):
    words = tl.cast(tl.cast(src, tl.int16, bitcast=True), tl.int32)
    maximum = tl.max(words & 0x7FFF, axis=1)
    # For BF16 amax, ceil(log2(amax * float32(1/6))) changes at fraction > 64.
    # The reference floor 6 * 2**-126 yields scale 2**-126.
    exponent = tl.maximum(-126, ((maximum + 63) >> 7) - 129)
    inf_group = maximum == 0x7F80
    nan_group = maximum > 0x7F80
    scale_word = tl.where(nan_group, -0x8000_0000, tl.where(inf_group, 0x7F80_0000, (127 + exponent) << 23))
    inverse_word = tl.where(nan_group, -0x0080_0000, tl.where(inf_group, 0, (127 - exponent) << 23))
    scale = tl.cast(scale_word, tl.float32, bitcast=True)
    inverse = tl.cast(inverse_word, tl.float32, bitcast=True)
    scaled = src.to(tl.float32) * inverse[:, None]
    magnitude = tl.abs(scaled)
    # The exact E2M1 grid with ties-to-even, including saturation at six.
    rounded = tl.where(magnitude <= 0.25, 0.0, 0.5)
    rounded = tl.where(magnitude >= 0.75, 1.0, rounded)
    rounded = tl.where(magnitude > 1.25, 1.5, rounded)
    rounded = tl.where(magnitude >= 1.75, 2.0, rounded)
    rounded = tl.where(magnitude > 2.5, 3.0, rounded)
    rounded = tl.where(magnitude >= 3.5, 4.0, rounded)
    rounded = tl.where(magnitude > 5.0, 6.0, rounded)
    rounded = tl.where(magnitude != magnitude, magnitude, rounded)
    sign = tl.cast(scaled, tl.int32, bitcast=True) & -0x8000_0000
    signed = tl.cast((tl.cast(rounded, tl.int32, bitcast=True) & 0x7FFF_FFFF) | sign, tl.float32, bitcast=True)
    return (signed * scale[:, None]).to(tl.bfloat16)


@triton.jit(do_not_specialize=["groups"])
def _kernel(states, output, groups, BLOCK: tl.constexpr):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    valid = group[:, None] < groups
    values = tl.load(states + offset, valid, 0)
    tl.store(output + offset, _fp4_e8m0(values), valid)


def fp4_e8m0_qdq(states, inplace=False):
    groups = states.numel() // _GROUP_SIZE
    output = states if inplace else torch.empty_like(states)
    _kernel[(triton.cdiv(groups, _GROUPS_PER_PROGRAM),)](
        states, output, groups=groups, BLOCK=_GROUPS_PER_PROGRAM, enable_fp_fusion=False
    )
    if inplace:
        torch.autograd.graph.increment_version(states)
    return output
