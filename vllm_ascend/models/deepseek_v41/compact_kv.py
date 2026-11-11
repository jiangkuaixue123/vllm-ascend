# SPDX-License-Identifier: Apache-2.0
"""The original V4.1 cache encodings. One row contains payload bytes followed by scale bytes."""

import torch

from vllm_ascend.models.deepseek_v41.eager_kernels import (
    _fp4_level,
    block_scale_bytes,
    cache_scale_values,
    quantize_fp8_blocks,
)


def cache_row_bytes(dim, bits, block):
    return (dim * bits + 7) // 8 + (dim + block - 1) // block


def pack_window(x):
    codes, _, raw = quantize_fp8_blocks(x)
    return torch.cat((codes.view(torch.uint8), raw), dim=-1)


def unpack_window(data, dim):
    codes = data[..., :dim].view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
    return (codes * cache_scale_values(data[..., dim:]).unsqueeze(-1)).flatten(-2).bfloat16()


def pack_fp4(x, block, e4scale):
    groups = x.float().unflatten(-1, (-1, block))
    maximum = groups.abs().amax(-1, keepdim=True)
    if e4scale:
        scales_fp8 = (maximum.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn)
        scales, raw = scales_fp8.float(), scales_fp8.view(torch.uint8)
    else:
        raw = block_scale_bytes(maximum.clamp_min(6 * 2.0**-126), 6.0)
        scales = cache_scale_values(raw)
    codes = _fp4_level((groups / scales).clamp(-6.0, 6.0)).to(torch.uint8).flatten(-2)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    return torch.cat((packed, raw.squeeze(-1)), dim=-1)


def unpack_fp4(data, dim, block, e4scale, lut):
    packed, raw = data.split((dim // 2, dim // block), dim=-1)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    scale = raw.view(torch.float8_e4m3fn).float() if e4scale else cache_scale_values(raw)
    values = lut[codes.long()].unflatten(-1, (-1, block))
    return (values * scale.unsqueeze(-1)).flatten(-2).bfloat16()
