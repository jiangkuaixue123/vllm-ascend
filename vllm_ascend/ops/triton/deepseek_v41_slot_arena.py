# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage bounded raw windows and selected compressed values for the A3 target native attention operator."""

import os

import torch
from vllm.logger import logger
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

QWIDTH = 6
WIN = 128
TOP = 512
DIM = 512
UNION = WIN + QWIDTH - 1
RAW_PAD = 144
WIDTH = WIN + TOP
SLOT_ROWS = 4
SLOT_COLUMNS = 64


@triton.jit
def _gather(
    field,
    table,
    lens,
    active,
    topk,
    out,
    GROUP_ROWS: tl.constexpr,
    ROWS: tl.constexpr,
    COUNT: tl.constexpr,
    MAXLEN: tl.constexpr,
    PAGE: tl.constexpr,
    PHYS: tl.constexpr,
    STRIDE: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TOP_ROW: tl.constexpr,
    RATIO: tl.constexpr,
    RAW: tl.constexpr,
    PROMOTE: tl.constexpr,
    TILE: tl.constexpr,
    D: tl.constexpr,
):
    dims = tl.arange(0, D)
    for block in range(tl.program_id(0), tl.cdiv(ROWS * COUNT, TILE), tl.num_programs(0)):
        flat = block * TILE + tl.arange(0, TILE)
        ok = flat < ROWS * COUNT
        row = flat // COUNT
        col = flat % COUNT
        if RAW:
            table_row = row * GROUP_ROWS
            first = tl.load(lens + table_row, mask=ok, other=0)
            limit = tl.load(lens + table_row + GROUP_ROWS - 1, mask=ok, other=0)
            pos = tl.maximum(0, first - 128) + col
            live = ok & (pos < limit)
        else:
            table_row = row
            idx = tl.load(topk + row * TOP_ROW + col, mask=ok, other=-1)
            pos = (idx + 1) * RATIO - 1
            live = (
                ok
                & (idx >= 0)
                & (tl.load(active + row, mask=ok, other=0) != 0)
                & (pos < tl.load(lens + row, mask=ok, other=0))
            )
        safe = tl.maximum(0, tl.minimum(pos, MAXLEN - 1))
        page = tl.load(table + table_row * TABLE_ROW + safe // PAGE, mask=live, other=0).to(tl.int64)
        slot = page * PAGE + safe % PAGE
        valid = live & (pos >= 0) & (slot >= 0) & (slot < PHYS)
        address = tl.where(valid, slot, 0) * STRIDE
        value = tl.load(field + address[:, None] + dims[None, :], mask=valid[:, None], other=0)
        if PROMOTE:
            value = value.to(tl.int32) << 16
        tl.store(out + flat[:, None] * D + dims[None, :], value, mask=ok[:, None])


@triton.jit
def _slots(
    lens,
    active,
    topk,
    out,
    GROUP_ROWS: tl.constexpr,
    ROWS: tl.constexpr,
    CMP_BASE: tl.constexpr,
    TOP_ROW: tl.constexpr,
    TILE: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    # Keep row metadata vectorized while columns stay in contiguous small tiles.
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    row_valid = row < ROWS
    ok = row_valid[:, None] & (col[None, :] < 640)
    group = row // GROUP_ROWS
    # Qualified decode metadata is bounded by the 153600-token model limit.
    # Physical indices and every intermediate therefore fit signed int32.
    length = tl.load(lens + row, row_valid, 0).to(tl.int32)
    first = tl.load(lens + group * GROUP_ROWS, row_valid, 0).to(tl.int32)
    active_row = tl.load(active + row, row_valid, 0) != 0
    width = tl.minimum(128, tl.maximum(0, length))
    ordinal = col[None, :] - width[:, None]
    is_cmp = ok & (ordinal >= 0) & (ordinal < 512)
    idx = tl.load(topk + row[:, None] * TOP_ROW + tl.where(is_cmp, ordinal, 0), mask=is_cmp, other=-1)
    raw_begin = group * 133 + tl.maximum(0, length - 128) - tl.maximum(0, first - 128)
    raw_slot = raw_begin[:, None] + col[None, :]
    cmp_slot = CMP_BASE + row[:, None] * 512 + ordinal
    slot = tl.where(col[None, :] < width[:, None], raw_slot, tl.where(is_cmp & (idx >= 0), cmp_slot, -1))
    tl.store(out + row[:, None] * 640 + col[None, :], tl.where(active_row[:, None], slot, -1).to(tl.int32), mask=ok)


def read(field, metadata, ids, ratio, raw, dtype, destination):
    count = UNION if raw else TOP
    group_rows = min(QWIDTH, ids.shape[0])
    rows = ids.shape[0] // group_rows if raw else ids.shape[0]
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    tmp = torch.empty((rows, count, DIM), dtype=bits, device=field.device)
    grid = (min(triton.cdiv(rows * count, 8), get_vectorcore_num()),)
    _gather[grid](
        field.view(torch.int16 if raw or dtype == torch.bfloat16 else torch.int32),
        metadata.block_table_tensor,
        metadata.seq_lens,
        metadata.active,
        ids,
        tmp,
        group_rows,
        rows,
        count,
        metadata.max_model_len,
        128,
        field.shape[0],
        field.stride(0),
        metadata.block_table_tensor.stride(0),
        ids.stride(0),
        ratio,
        raw,
        raw and dtype == torch.float32,
        8,
        DIM,
    )
    # Keep S's stock Where and stock final FP32 -> BF16 conversion, including NPU NaN handling.
    live = metadata.active.view(-1, group_rows).any(-1) if raw else metadata.active
    destination.copy_(torch.where(live[:, None, None], tmp.view(dtype), 0.0).to(torch.bfloat16))


def stage(arena, raw, comp, md, src, topk, ratio, refresh):
    rows = topk.shape[0]
    group_rows = min(QWIDTH, rows)
    groups = rows // group_rows
    assert rows in (3, 6, 12) and topk.shape == (rows, TOP) and topk.stride(1) == 1
    assert raw.shape[1] == comp.shape[1] == DIM and raw.stride(1) == comp.stride(1) == 1
    assert (
        md.seq_lens.shape == src.seq_lens.shape == (rows,)
        and md.block_table_tensor.stride(1) == src.block_table_tensor.stride(1) == 1
    )
    assert (
        arena.shape == (rows * WIDTH, DIM)
        and arena.is_contiguous()
        and arena.dtype == torch.bfloat16
        and ratio in (1, 2)
    )
    cmp_base = groups * RAW_PAD
    read(raw, md, topk, ratio, True, comp.dtype, arena[: groups * UNION].view(groups, UNION, DIM))
    if refresh:
        read(comp, src, topk, ratio, False, comp.dtype, arena[cmp_base : cmp_base + rows * TOP].view(rows, TOP, DIM))
    ids = torch.empty((rows, 1, WIDTH), dtype=torch.int32, device=raw.device)
    _slots[(triton.cdiv(rows, SLOT_ROWS), triton.cdiv(WIDTH, SLOT_COLUMNS))](
        md.seq_lens, md.active, topk, ids, group_rows, rows, cmp_base, topk.stride(0), SLOT_COLUMNS, SLOT_ROWS
    )
    logger.info_once("TARGET_SLOT_ROWS_RUNTIME pid=%d rows=%d", os.getpid(), rows, scope="process")
    return ids
