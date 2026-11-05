# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One A3 vector pass over complete BF16 groups for V4.1 window/output E4M3 QDQ."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_QUALIFIED_VECTOR_CORES = 48
_GROUP_SIZE = 32


@triton.jit
def _bias_qdq(src):
    bits = tl.cast(tl.cast(src, tl.int16, bitcast=True), tl.int32)
    max_word = tl.max(bits & 0x7FFF, axis=1)
    # For positive finite BF16 words, ceil(log2(amax/448)) crosses when
    # mantissa > 96. Adding 31 encodes that comparison plus exponent carry.
    exp = tl.maximum(-22, ((max_word + 31) >> 7) - 135)
    inf_group = max_word == 0x7F80
    nan_group = max_word > 0x7F80
    # IEEE FP32 bit patterns for signed zero, infinity, exponent and sign masks.
    scale_word = tl.where(nan_group, -0x8000_0000, tl.where(inf_group, 0x7F80_0000, (127 + exp) << 23))
    reciprocal_word = tl.where(nan_group, -0x0080_0000, tl.where(inf_group, 0, (127 - exp) << 23))
    scale = tl.cast(scale_word, tl.float32, bitcast=True)
    inv = tl.cast(reciprocal_word, tl.float32, bitcast=True)
    scaled = tl.cast(src, tl.float32) * inv[:, None]
    mag = tl.abs(scaled)
    # Keep the NaN class through clamp. For finite groups scale ensures |x/scale| <= 448.
    mag = tl.where(mag > 448.0, 448.0, mag)
    field = tl.cast(mag, tl.int32, bitcast=True) & 0x7F80_0000
    # A normal exponent field of 121 is E4M3's minimum. Twenty more FP32
    # exponent steps give the addition exactly three retained fraction bits.
    bias = tl.cast(tl.maximum(field, 0x3C80_0000) + 0x0A00_0000, tl.float32, bitcast=True)
    rounded = (bias + mag) - bias
    sign = tl.cast(scaled, tl.int32, bitcast=True) & -0x8000_0000
    signed = tl.cast((tl.cast(rounded, tl.int32, bitcast=True) & 0x7FFF_FFFF) | sign, tl.float32, bitcast=True)
    return tl.cast(signed * scale[:, None], tl.bfloat16)


@triton.jit
def _kernel(x, out, NTILES: tl.constexpr, G: tl.constexpr):
    for tile in range(tl.program_id(0), NTILES, tl.num_programs(0)):
        groups = tile * G + tl.arange(0, G)
        ix = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
        src = tl.load(x + ix)
        tl.store(out + ix, _bias_qdq(src))


def decode_fp8_qdq(x: torch.Tensor, inplace: bool):
    init_device_properties_triton()
    cores = get_vectorcore_num()
    if cores != _QUALIFIED_VECTOR_CORES:
        return None
    total_groups = x.numel() // _GROUP_SIZE
    groups = min(64, max(8, triton.next_power_of_2(triton.cdiv(total_groups, cores))))
    ntiles = total_groups // groups
    out = x if inplace else torch.empty_like(x)
    _kernel[(min(cores, ntiles),)](x, out, NTILES=ntiles, G=groups)
    if inplace:
        # A raw Triton store bypasses the version change made by Tensor.copy_.
        torch.autograd.graph.increment_version(x)
    return out


@triton.jit(do_not_specialize=["tiles"])
def _dynamic_kernel(x, out, tiles, G: tl.constexpr):
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        groups = tile * G + tl.arange(0, G)
        ix = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
        src = tl.load(x + ix)
        tl.store(out + ix, _bias_qdq(src))


def dynamic_fp8_qdq(x: torch.Tensor, inplace: bool):
    init_device_properties_triton()
    cores = get_vectorcore_num()
    if cores != _QUALIFIED_VECTOR_CORES:
        return None
    # Every supported row has 16 or 256 complete groups. Tile sizes divide
    # those counts, so neither padding nor an extra tail kernel is needed.
    groups = 16 if x.ndim == 2 else 64
    tiles = x.numel() // (_GROUP_SIZE * groups)
    out = x if inplace else torch.empty_like(x)
    _dynamic_kernel[(min(cores, tiles),)](x, out, tiles, G=groups)
    if inplace:
        torch.autograd.graph.increment_version(x)
    return out
