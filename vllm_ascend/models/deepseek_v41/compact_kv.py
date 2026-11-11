# SPDX-License-Identifier: Apache-2.0
"""The original V4.1 cache encodings. One row contains payload bytes followed by scale bytes."""

from enum import IntEnum

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

class CacheEncoding(IntEnum):
    WINDOW_MXFP8 = 0
    COMPRESSED_E4_FP4 = 1
    INDEX_MXFP4 = 2


def store_compact_kv(cache, x, slots, encoding):
    """Write a complete post-RoPE BF16 row at a scheduler-owned or scratch slot."""
    if x.device.type == "npu":
        from vllm_ascend.device.device_config import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() == AscendDeviceType.A5:
            torch.ops._C_ascend.kv_compress_epilog_v2(cache, x.contiguous(), slots.contiguous(), int(encoding))
            return
    if encoding == CacheEncoding.WINDOW_MXFP8:
        packed = pack_window(x)
    elif encoding == CacheEncoding.COMPRESSED_E4_FP4:
        packed = pack_fp4(x, 16, True)
    elif encoding == CacheEncoding.INDEX_MXFP4:
        packed = pack_fp4(x, 32, False)
    else:
        raise ValueError(f"Unsupported compact KV encoding {encoding}")
    if x.device.type == "npu":
        # The scheduler maps inactive lanes to allocated scratch slots. Keep
        # the original graph-safe writer on devices without the A5 epilog.
        cache.index_copy_(0, slots, packed)
        return
    valid = (slots >= 0) & (slots < cache.size(0))
    cache.index_copy_(0, slots[valid].long(), packed[valid])


def store_mixed_kv(cache, x, slots):
    """Store BF16 rows in the existing Mixed MLA block-split MXFP8 cache."""
    if cache.ndim != 4 or cache.shape[2:] != (1, 584) or not cache.is_contiguous():
        raise ValueError("Mixed MLA cache must be contiguous [pages, page_size, 1, 584]")
    if (cache.dtype not in (torch.uint8, torch.float8_e4m3fn) or x.ndim != 2 or x.size(1) != 512
            or x.dtype != torch.bfloat16):
        raise ValueError("Expected a byte-addressable E4M3 cache and 512-channel BF16 rows")
    torch.ops._C_ascend.kv_compress_epilog(
        cache.view(torch.float8_e4m3fn), x.contiguous(), slots.contiguous(), 64, 2, True, 2
    )
