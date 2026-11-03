# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind test buffers with the exact per-layer shapes of the Ascend allocator."""

import torch


def allocate(cache, blocks, block_size, device=None):
    spec = cache.get_kv_cache_spec(None)
    cache.kv_cache = tuple(
        torch.zeros((blocks, block_size, 1, width), dtype=spec.dtype, device=device)
        for width in (spec.head_size, spec.head_size_v)
    )


def bits(cache, blocks=None):
    # FP32 compressor words can look like NaNs when interpreted as BF16.
    return tuple((bank if blocks is None else bank[blocks]).view(torch.int16) for bank in cache.kv_cache)


def snapshot(cache, blocks=None):
    return tuple(bank.clone() for bank in bits(cache, blocks))
