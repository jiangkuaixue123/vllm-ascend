# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refresh a dense logical IndexPool view from remapped or rewritten physical pages."""

import torch
from torch import nn
from vllm.triton_utils import tl, triton

from .cache_pages import PAGE_BYTES, PAGE_TOKENS, POOL_RATIO, POOL_START


@triton.jit
def _refresh(
    cache,
    pages,
    previous,
    keys,
    first,
    end,
    PAGES: tl.constexpr,
    GROUPS: tl.constexpr,
    PAGE_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
    BASE: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = index < PAGES
    now = tl.load(pages + index, live, 0)
    old = tl.load(previous + index, live, -1)
    start = tl.load(first)
    stop = tl.load(end)
    written = (start < stop) & (index * GROUPS < stop) & ((index + 1) * GROUPS > start)
    dirty = live & ((now != old) | written)
    if tl.max(dirty.to(tl.int32), axis=0):
        lane = tl.arange(0, PAGE_ELEMENTS)
        source = now.to(tl.int64)[:, None] * STRIDE + BASE + lane[None, :]
        target = index[:, None] * PAGE_ELEMENTS + lane[None, :]
        data = tl.load(cache + source, dirty[:, None], 0)
        tl.store(keys + target, data, dirty[:, None])
        tl.store(previous + index, now, dirty)


_REFRESH_PAGES = 4


class ResidentPool(nn.Module):
    """Derived keys for one target layer, invalidated on other writes to its cache."""

    def __init__(self, max_len, width, device):
        super().__init__()
        count = triton.cdiv(max_len // POOL_RATIO, PAGE_TOKENS // POOL_RATIO)
        self.register_buffer("physical", torch.full((count,), -1, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer(
            "keys",
            torch.empty((count * (PAGE_TOKENS // POOL_RATIO), width), dtype=torch.bfloat16, device=device),
            persistent=False,
        )

    def invalidate(self):
        self.physical.fill_(-1)

    def read(self, cache, pages, first, end):
        groups = PAGE_TOKENS // POOL_RATIO
        _refresh[(triton.cdiv(pages.numel(), _REFRESH_PAGES),)](
            cache.view(self.keys.dtype),
            pages,
            self.physical,
            self.keys,
            first,
            end,
            pages.numel(),
            groups,
            groups * self.keys.shape[-1],
            _REFRESH_PAGES,
            PAGE_BYTES // 2,
            POOL_START // 2,
            num_warps=1,
        )
        return self.keys
