# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 cache addressing without materializing a complete field plane."""

import torch
from vllm.triton_utils import tl, triton


def field_storage(cache, bank, offset, width):
    """Return the unsliced physical bank and column offset in its element dtype."""
    key, value = cache.pages()
    if bank == 0 and 0 <= offset < cache.dim and offset + width <= cache.dim:
        return key, offset
    if bank == 0 and offset == cache.dim and cache.source_ratio:
        return (value.view(torch.float32) if cache.source_ratio > 1 else value), 0
    if bank == 1 and offset == 0 and cache.source_ratio > 1:
        return value.view(torch.float32), cache.dim
    if bank == 1 and offset == cache.dim and cache.source_ratio:
        return (key if cache.source_ratio > 1 else value), cache.dim
    if bank == 1 and offset == cache.dim + cache.index_dim and width == 1 and cache.token_history:
        if cache.source_ratio:
            storage = key if cache.source_ratio > 1 else value
            return storage.view(torch.float32), offset // 2
        return value.view(torch.float32), 0
    raise ValueError(f"Unallocated V4.1 cache field: bank={bank}, offset={offset}, width={width}")


@triton.jit
def _decode_read_kernel(
    page,
    positions,
    blocks,
    active,
    lengths,
    output,
    ROWS: tl.constexpr,
    COUNT: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    POS_S0: tl.constexpr,
    POS_S1: tl.constexpr,
    BLOCK_S0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_LEN: tl.constexpr,
    TILE: tl.constexpr,
    CHANNELS: tl.constexpr,
):
    position_tiles: tl.constexpr = triton.cdiv(COUNT, TILE)
    channels = tl.arange(0, CHANNELS)
    for tile in range(tl.program_id(0), ROWS * position_tiles, tl.num_programs(0)):
        row = tile // position_tiles
        column = (tile % position_tiles) * TILE + tl.arange(0, TILE)
        pos = tl.load(positions + row * POS_S0 + column * POS_S1, column < COUNT, other=-1)
        clamped = tl.minimum(tl.maximum(pos, 0), MAX_LEN - 1)
        block = tl.load(blocks + row * BLOCK_S0 + clamped // BLOCK_SIZE).to(tl.int64)
        slot = block * BLOCK_SIZE + clamped % BLOCK_SIZE
        valid = (column < COUNT) & (tl.load(active + row) != 0) & (pos >= 0) & (pos < tl.load(lengths + row))
        valid = valid & (slot >= 0) & (slot < PAGE_ROWS)
        value = tl.load(
            page + slot[:, None] * PAGE_STRIDE + OFFSET + channels[None, :],
            valid[:, None] & (channels[None, :] < WIDTH),
            other=0,
        )
        tl.store(
            output + (row * COUNT + column[:, None]) * WIDTH + channels[None, :],
            value,
            (column[:, None] < COUNT) & (channels[None, :] < WIDTH),
        )


def decode_read(cache, md, positions, bank, offset, width):
    page, column = field_storage(cache, bank, offset, width)
    rows, count = positions.shape
    output = torch.empty((rows, count, width), dtype=page.dtype, device=page.device)
    if rows and count:
        _decode_read_kernel[(min(rows * triton.cdiv(count, 8), 64),)](
            page,
            positions,
            md.block_table_tensor,
            md.active,
            md.seq_lens,
            output,
            ROWS=rows,
            COUNT=count,
            WIDTH=width,
            PAGE_ROWS=page.shape[0],
            PAGE_STRIDE=page.stride(0),
            OFFSET=column,
            POS_S0=positions.stride(0),
            POS_S1=positions.stride(1),
            BLOCK_S0=md.block_table_tensor.stride(0),
            BLOCK_SIZE=cache.block_size,
            MAX_LEN=md.max_model_len,
            TILE=8,
            CHANNELS=triton.next_power_of_2(width),
        )
    return output


@triton.jit
def _decode_write_kernel(
    page,
    values,
    slots,
    active,
    enabled,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    VALUE_S0: tl.constexpr,
    VALUE_S1: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row).to(tl.int64)
    valid = (tl.load(active + row) != 0) & (tl.load(enabled + row) != 0) & (slot >= 0) & (slot < PAGE_ROWS)
    # Keep sequential last-writer semantics even for duplicate live slots.
    others = tl.arange(0, ROW_TILE)
    other_slots = tl.load(slots + others, others < ROWS, other=-1)
    other_valid = tl.load(active + others, others < ROWS, other=0) != 0
    other_valid = other_valid & (tl.load(enabled + others, others < ROWS, other=0) != 0)
    later = (others > row) & (others < ROWS) & other_valid & (other_slots == slot)
    valid = valid & (tl.sum(later.to(tl.int32), axis=0) == 0)
    channels = tl.arange(0, CHANNELS)
    value = tl.load(values + row * VALUE_S0 + channels * VALUE_S1, channels < WIDTH, other=0)
    tl.store(page + slot * PAGE_STRIDE + OFFSET + channels, value, valid & (channels < WIDTH))


def decode_write(cache, md, values, bank, offset, enabled):
    page, column = field_storage(cache, bank, offset, values.shape[-1])
    rows, width = values.shape
    if rows:
        _decode_write_kernel[(rows,)](
            page,
            values,
            md.slot_mapping,
            md.active,
            enabled,
            ROWS=rows,
            WIDTH=width,
            PAGE_ROWS=page.shape[0],
            PAGE_STRIDE=page.stride(0),
            OFFSET=column,
            VALUE_S0=values.stride(0),
            VALUE_S1=values.stride(1),
            CHANNELS=triton.next_power_of_2(width),
            ROW_TILE=triton.next_power_of_2(rows),
        )
