# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed-KV FP4 fusion retaining the native FP32 tensor division."""

import torch
from vllm.triton_utils import tl, triton

_GROUP_SIZE = 16
_GROUPS_PER_PROGRAM = 16


@triton.jit(do_not_specialize=["groups"])
def _prepare_kernel(states, fp32_states, scales, groups, BLOCK: tl.constexpr):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = group[:, None] * 16 + tl.arange(0, 16)[None, :]
    valid = group < groups
    source = tl.load(states + offset, valid[:, None], 0)
    words = tl.cast(tl.cast(source, tl.int16, bitcast=True), tl.int32) & 0x7FFF
    maximum_word = tl.max(words, axis=1)
    maximum = tl.cast(maximum_word << 16, tl.float32, bitcast=True)
    scaled = tl.maximum(maximum, 0.01171875) * (1.0 / 6.0)
    # Above the final ties-to-even boundary (464), the reference E4M3 scale is NaN.
    safe = tl.minimum(scaled, 480.0)
    field = tl.cast(safe, tl.int32, bitcast=True) & 0x7F80_0000
    bias = tl.cast(tl.maximum(field, 0x3C80_0000) + 0x0A00_0000, tl.float32, bitcast=True)
    rounded = (bias + safe) - bias
    scale = tl.where((scaled > 464.0) | (maximum_word > 0x7F80), float("nan"), rounded)
    tl.store(fp32_states + offset, source.to(tl.float32), valid[:, None])
    tl.store(scales + group, scale, valid)


@triton.jit(do_not_specialize=["groups"])
def _finish_kernel(normalized, scales, output, groups, BLOCK: tl.constexpr):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = group[:, None] * 16 + tl.arange(0, 16)[None, :]
    valid = group < groups
    values = tl.load(normalized + offset, valid[:, None], 0)
    magnitude = tl.abs(values)
    rounded = tl.where(magnitude <= 0.25, 0.0, 0.5)
    rounded = tl.where(magnitude >= 0.75, 1.0, rounded)
    rounded = tl.where(magnitude > 1.25, 1.5, rounded)
    rounded = tl.where(magnitude >= 1.75, 2.0, rounded)
    rounded = tl.where(magnitude > 2.5, 3.0, rounded)
    rounded = tl.where(magnitude >= 3.5, 4.0, rounded)
    rounded = tl.where(magnitude > 5.0, 6.0, rounded)
    rounded = tl.where(magnitude != magnitude, magnitude, rounded)
    sign = tl.cast(values, tl.int32, bitcast=True) & -0x8000_0000
    signed = tl.cast((tl.cast(rounded, tl.int32, bitcast=True) & 0x7FFF_FFFF) | sign, tl.float32, bitcast=True)
    scale = tl.load(scales + group, valid, 0)
    tl.store(output + offset, (signed * scale[:, None]).to(tl.bfloat16), valid[:, None])


def prepare_fp4_e4m3(states):
    groups = states.numel() // _GROUP_SIZE
    fp32_states = torch.empty_like(states, dtype=torch.float32)
    scales = torch.empty((groups,), device=states.device, dtype=torch.float32)
    _prepare_kernel[(triton.cdiv(groups, _GROUPS_PER_PROGRAM),)](
        states, fp32_states, scales, groups=groups, BLOCK=_GROUPS_PER_PROGRAM, enable_fp_fusion=False
    )
    return fp32_states, scales


def fp4_e4m3_qdq(states, inplace=False):
    fp32_states, scales = prepare_fp4_e4m3(states)
    # Keep the original FP32 tensor division, including its device rounding behavior.
    normalized = fp32_states.reshape(-1, _GROUP_SIZE) / scales[:, None]
    output = states if inplace else torch.empty_like(states)
    groups = scales.numel()
    _finish_kernel[(triton.cdiv(groups, _GROUPS_PER_PROGRAM),)](
        normalized, scales, output, groups=groups, BLOCK=_GROUPS_PER_PROGRAM, enable_fp_fusion=False
    )
    if inplace:
        torch.autograd.graph.increment_version(states)
    return output
