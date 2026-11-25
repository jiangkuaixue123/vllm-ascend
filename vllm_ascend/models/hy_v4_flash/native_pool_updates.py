# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and write incremental IndexPool keys in request order for native decode pages."""

import torch
from torch.nn import functional as F
from vllm.models.hy_v4_flash.reference import INDEX_DIM
from vllm.triton_utils import tl, triton

from .cache_pages import PAGE_BYTES, PAGE_TOKENS, POOL_RATIO, POOL_START, RAW_ROW_BYTES, RAW_START

_POSITION_AXES = 3


@triton.jit
def _pool_update_slots(
    lengths,
    tables,
    raw_slots,
    key_slots,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    RAW: tl.constexpr,
    GROUPS: tl.constexpr,
    PAGE: tl.constexpr,
):
    request = tl.program_id(0)
    length = tl.load(lengths + request).to(tl.int32)
    first = tl.maximum(length - WIDTH, 0) // RATIO
    end = length // RATIO
    lane = tl.arange(0, RAW)
    history = first * RATIO + lane
    logical = tl.where(history < end * RATIO, history, 0)
    page = tl.load(tables + request * TABLE_STRIDE + logical // PAGE)
    tl.store(raw_slots + request * RAW + lane, page * PAGE + logical % PAGE)
    g = tl.arange(0, GROUPS)
    group = first + g
    live = group < end
    p = tl.load(tables + request * TABLE_STRIDE + group * RATIO // PAGE, live, 0)
    dest = p * (PAGE // RATIO) + group % (PAGE // RATIO)
    tl.store(key_slots + request * GROUPS + g, tl.where(live, dest, -1))


def plan_pool_refresh(lengths, tables, width):
    """Resolve the raw-key slots and writable pooled groups for this forward."""
    ratio = POOL_RATIO
    batch = lengths.shape[0]
    groups = triton.cdiv(width, ratio)
    raw = groups * ratio
    source = tables.new_empty(batch * raw, dtype=torch.int32)
    target = tables.new_empty(batch * groups, dtype=torch.int32)
    _pool_update_slots[(batch,)](
        lengths, tables, source, target, tables.stride(0), width, ratio, raw, groups, PAGE_TOKENS, num_warps=1
    )
    return source, target


@triton.jit
def _read_float_pool_inputs(
    cache,
    coordinates,
    slots,
    raw,
    positions,
    GROUPS: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    BASE: tl.constexpr,
    RECORD: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row).to(tl.int64)
    address = slot // PAGE * STRIDE + BASE + (slot % PAGE) * RECORD
    lane = tl.arange(0, WIDTH)
    raw_value = tl.load(cache + address // 2 + lane).to(tl.float32)
    tl.store(raw + row * WIDTH + lane, raw_value)
    if row % RATIO == 0:
        axis = tl.arange(0, 4)
        pos = tl.load(coordinates + (address + WIDTH * 2) // 8 + axis, axis < 3, 0)
        tl.store(positions + axis * GROUPS + row // RATIO, pos, axis < 3)


@triton.jit
def _write_last_pool_update(
    cache,
    keys,
    slots,
    COUNT: tl.constexpr,
    TILE: tl.constexpr,
    WIDTH: tl.constexpr,
    PER_PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    BASE: tl.constexpr,
):
    group = tl.program_id(0)
    physical = tl.load(slots + group).to(tl.int64)
    other = tl.arange(0, TILE)
    candidate = tl.load(slots + other, other < COUNT, -1)
    replaced = tl.max(((candidate == physical) & (other > group) & (other < COUNT)).to(tl.int32), 0)
    lane = tl.arange(0, WIDTH)
    start = physical // PER_PAGE * STRIDE + BASE + (physical % PER_PAGE) * WIDTH
    value = tl.load(keys + group * WIDTH + lane)
    tl.store(cache + start + lane, value, (physical >= 0) & (replaced == 0))


def refresh_pool_keys(index, cache, slots, position_ndim):
    """Read and project all groups together, preserving request order for aliased destinations."""
    projection = index.pool_projection
    source, destinations = slots
    groups = destinations.numel()
    raw = cache.new_empty((source.numel(), INDEX_DIM), dtype=torch.float32)
    positions = cache.new_empty((_POSITION_AXES, groups), dtype=torch.int64)
    _read_float_pool_inputs[(source.numel(),)](
        cache.view(torch.bfloat16),
        cache.view(torch.int64),
        source,
        raw,
        positions,
        groups,
        index.ratio,
        PAGE_TOKENS,
        PAGE_BYTES,
        RAW_START,
        RAW_ROW_BYTES,
        INDEX_DIM,
        num_warps=1,
    )
    product = F.linear(raw, projection).view(groups, index.ratio, 2 * INDEX_DIM)
    values, gates = product.split(INDEX_DIM, dim=-1)
    pooled = (values * (gates + index.intra_group_ape).softmax(dim=1)).sum(dim=1)
    pooled = F.rms_norm(pooled, (INDEX_DIM,), index.key_norm.weight, index.eps).bfloat16()
    keys = index._encode_rotated(pooled, positions[0] if position_ndim == 1 else positions, full_graph=True)
    _write_last_pool_update[(groups,)](
        cache.view(torch.bfloat16),
        keys,
        destinations,
        groups,
        triton.next_power_of_2(groups),
        INDEX_DIM,
        PAGE_TOKENS // index.ratio,
        PAGE_BYTES // 2,
        POOL_START // 2,
        num_warps=1,
    )
