# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM
from vllm.triton_utils import tl, triton

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES

PAGE_TOKENS = 64
POOL_RATIO = 4
RAW_ROW_BYTES = INDEX_DIM * torch.bfloat16.itemsize + 3 * torch.int64.itemsize
POOL_KEY_BYTES = INDEX_DIM * torch.bfloat16.itemsize
TOKEN_BYTES = C8_ROW_BYTES + RAW_ROW_BYTES + POOL_KEY_BYTES // POOL_RATIO
# Keep complete C8 records addressable across pages in the same allocation.
# The typed fields are word-aligned within each page, including sliding windows.
_ROW_ALIGNMENT = C8_ROW_BYTES // math.gcd(PAGE_TOKENS, C8_ROW_BYTES)
ALLOCATED_ROW_BYTES = (TOKEN_BYTES + _ROW_ALIGNMENT - 1) // _ROW_ALIGNMENT * _ROW_ALIGNMENT
PAGE_BYTES = PAGE_TOKENS * ALLOCATED_ROW_BYTES
C8_ROWS_PER_PAGE = PAGE_BYTES // C8_ROW_BYTES
RAW_START = PAGE_TOKENS * C8_ROW_BYTES
POOL_START = PAGE_BYTES - PAGE_TOKENS // POOL_RATIO * POOL_KEY_BYTES


@triton.jit
def c8_byte_offset(
    page,
    token,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    LAYOUT_PAGE_BYTES: tl.constexpr = PAGE_BYTES,
    LAYOUT_C8_ROW_BYTES: tl.constexpr = C8_ROW_BYTES,
):
    if WIDTH == LAYOUT_PAGE_BYTES:
        return page * LAYOUT_PAGE_BYTES + (token % PAGE_SIZE) * LAYOUT_C8_ROW_BYTES
    return (page * PAGE_SIZE + token % PAGE_SIZE) * WIDTH


@triton.jit
def _gather_raw(
    cache,
    slots,
    result,
    COUNT: tl.constexpr,
    WORDS: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT_PAGE_TOKENS: tl.constexpr = PAGE_TOKENS,
    LAYOUT_PAGE_BYTES: tl.constexpr = PAGE_BYTES,
    LAYOUT_RAW_START: tl.constexpr = RAW_START,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    slot = tl.load(slots + row, row < COUNT, 0).to(tl.int64)
    word = tl.arange(0, triton.next_power_of_2(WORDS))
    start = (
        slot // LAYOUT_PAGE_TOKENS * (LAYOUT_PAGE_BYTES // 4)
        + LAYOUT_RAW_START // 4
        + slot % LAYOUT_PAGE_TOKENS * WORDS
    )
    mask = (row[:, None] < COUNT) & (word[None, :] < WORDS)
    data = tl.load(cache + start[:, None] + word[None, :], mask, 0)
    tl.store(result + row[:, None] * WORDS + word[None, :], data, mask)


def gather_page_field(pages, blocks, positions, columns):
    within = positions % PAGE_TOKENS
    if columns.start in (0, None) and columns.stop == C8_ROW_BYTES:
        rows = blocks.long() * C8_ROWS_PER_PAGE + within
        selected = pages.view(-1, C8_ROW_BYTES).index_select(0, rows.flatten())
        return selected.view(*positions.shape, C8_ROW_BYTES)
    assert (columns.start, columns.stop) == (C8_ROW_BYTES, C8_ROW_BYTES + RAW_ROW_BYTES)
    slots = blocks.long() * PAGE_TOKENS + within
    result = pages.new_empty((*positions.shape, RAW_ROW_BYTES))
    if slots.numel():
        tile = 16
        _gather_raw[(triton.cdiv(slots.numel(), tile),)](
            pages.view(torch.int32),
            slots.contiguous(),
            result.view(torch.int32),
            slots.numel(),
            RAW_ROW_BYTES // torch.int32.itemsize,
            tile,
            num_warps=1,
        )
    return result


@triton.jit
def _physical_indices(
    table,
    selected,
    indices,
    COUNT: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT_PAGE_TOKENS: tl.constexpr = PAGE_TOKENS,
    LAYOUT_C8_ROWS_PER_PAGE: tl.constexpr = C8_ROWS_PER_PAGE,
):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    pos = tl.load(selected + i, i < COUNT, -1).to(tl.int32)
    page = tl.load(table + pos // LAYOUT_PAGE_TOKENS, pos >= 0, 0)
    slot = page * LAYOUT_C8_ROWS_PER_PAGE + pos % LAYOUT_PAGE_TOKENS
    tl.store(indices + i, tl.where(pos >= 0, slot, -1), i < COUNT)


def physical_c8_indices(table, selected):
    indices = torch.empty_like(selected, dtype=torch.int32, memory_format=torch.contiguous_format)
    tile = 256
    _physical_indices[(triton.cdiv(indices.numel(), tile),)](
        table,
        selected.contiguous(),
        indices,
        indices.numel(),
        tile,
        num_warps=1,
    )
    return indices


@triton.jit
def _map_compact(
    tables,
    indices,
    result,
    counts,
    TABLE_STRIDE: tl.constexpr,
    QUERIES: tl.constexpr,
    COUNT: tl.constexpr,
    TILE: tl.constexpr,
    PAGE: tl.constexpr,
    PHYSICAL_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, TILE)
    pos = tl.load(indices + row * COUNT + lane, lane < COUNT, -1).to(tl.int32)
    valid = pos >= 0
    before = tl.cumsum(valid.to(tl.int32), axis=0) - valid.to(tl.int32)
    count = tl.sum(valid.to(tl.int32), axis=0)
    dest = tl.where(valid, before, count + lane - before)
    page = tl.load(tables + (row // QUERIES) * TABLE_STRIDE + pos // PAGE, valid, 0).to(tl.int32)
    address = page * PHYSICAL_ROWS + pos % PAGE
    # Supply one in-bounds row if the logical set is empty; the final context is zeroed.
    entry = tl.where(valid, address, tl.where((count == 0) & (lane == 0), 0, -1))
    tl.store(result + row * COUNT + dest, entry, lane < COUNT)
    tl.store(counts + row, count)


def compact_physical_c8_indices(tables, logical):
    """Map each request’s logical indices, with valid C8 rows packed stably before padding."""
    requests, queries, width = logical.shape
    physical = torch.empty((requests * queries, width), device=logical.device, dtype=torch.int32)
    counts = physical.new_empty((requests, queries))
    _map_compact[(requests * queries,)](
        tables,
        logical.contiguous(),
        physical,
        counts,
        tables.stride(0),
        queries,
        width,
        triton.next_power_of_2(width),
        PAGE_TOKENS,
        C8_ROWS_PER_PAGE,
        num_warps=1,
    )
    return physical, counts
