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


@triton.jit
def _refresh_batch_pages(
    cache,
    block_table,
    visible,
    written_groups,
    previous_pages,
    keys,
    MAX_PAGES: tl.constexpr,
    TABLE_LENGTH: tl.constexpr,
    QUERIES: tl.constexpr,
    BUDGET: tl.constexpr,
    GROUPS_PER_PAGE: tl.constexpr,
    POOL_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_WRITES: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    POOL_OFFSET: tl.constexpr,
):
    page_indices = tl.program_id(0) * PAGES_PER_TILE + tl.arange(0, PAGES_PER_TILE)
    in_bounds = page_indices < MAX_PAGES
    query = tl.arange(0, triton.next_power_of_2(QUERIES))
    last_group = tl.max(tl.load(visible + query, query < QUERIES, -1), 0) // POOL_RATIO
    valid_page = (page_indices < TABLE_LENGTH) & (last_group > BUDGET) & (page_indices * GROUPS_PER_PAGE < last_group)
    physical = tl.load(block_table + page_indices, in_bounds & valid_page, 0).to(tl.int32)
    previous = tl.load(previous_pages + page_indices, in_bounds, -1)

    # A different request may have written this same physical page at another logical position.
    write = tl.arange(0, triton.next_power_of_2(NUM_WRITES))
    destination = tl.load(written_groups + write, write < NUM_WRITES, -1)
    matches = (destination[None, :] >= 0) & (physical[:, None] == destination[None, :] // GROUPS_PER_PAGE)
    rewritten = tl.max(matches.to(tl.int32), axis=1)
    dirty = in_bounds & ((previous != physical) | (rewritten != 0))
    if tl.max(dirty.to(tl.int32), axis=0):
        lane = tl.arange(0, GROUPS_PER_PAGE * HEAD_DIM)
        source = physical.to(tl.int64)[:, None] * PAGE_STRIDE + POOL_OFFSET + lane[None, :]
        target = page_indices[:, None] * GROUPS_PER_PAGE * HEAD_DIM + lane[None, :]
        data = tl.load(cache + source, dirty[:, None], 0)
        tl.store(keys + target, data, dirty[:, None])
        tl.store(previous_pages + page_indices, physical, dirty)


_REFRESH_PAGES = 4
_BATCH_REFRESH_PAGES = 2


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

    def read_batch(self, cache, table, visible, destinations, budget):
        """Refresh the owning request using every physical page updated by this batch."""
        count = self.physical.numel()
        _refresh_batch_pages[(triton.cdiv(count, _BATCH_REFRESH_PAGES),)](
            cache.view(self.keys.dtype),
            table,
            visible,
            destinations,
            self.physical,
            self.keys,
            count,
            table.numel(),
            visible.numel(),
            budget,
            PAGE_TOKENS // POOL_RATIO,
            POOL_RATIO,
            self.keys.shape[-1],
            destinations.numel(),
            _BATCH_REFRESH_PAGES,
            PAGE_BYTES // 2,
            POOL_START // 2,
            num_warps=1,
        )
        return self.keys
