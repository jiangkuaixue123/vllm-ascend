# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build draft attention's final KV pages and physical slots in one device pass."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _stage_draft_pages(
    field,
    query,
    table,
    start,
    pages,
    slots,
    FIELD_ROWS: tl.constexpr,
    FIELD_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    DEVICE_HISTORY: tl.constexpr,
):
    batch, tile = tl.program_id(0), tl.program_id(1)
    rows = tile * 8 + tl.arange(0, 8)
    columns = tl.arange(0, 512)
    begin = tl.load(start + batch).to(tl.int64)
    history = begin - 1 - ((begin - 1 - rows) & 127)
    if DEVICE_HISTORY:
        history = tl.maximum(history, 0)
    page = tl.load(table + batch * TABLE_STRIDE + history // 128, rows < 128, 0).to(tl.int64)
    source = page * 128 + history % 128
    source = tl.where(source < 0, source + FIELD_ROWS, source)
    context = tl.load(field + source[:, None] * FIELD_STRIDE + columns[None, :], rows[:, None] < 128, 0)
    proposed = tl.load(
        query + (batch * 5 + rows[:, None] - 128) * 512 + columns[None, :],
        (rows[:, None] >= 128) & (rows[:, None] < 133),
        0,
    )
    values = tl.where(rows[:, None] < 128, context, proposed)
    tl.store(pages + (batch * 144 + rows[:, None]) * 512 + columns[None, :], values)
    if tile == 0:
        lanes = tl.arange(0, 256)
        length = tl.minimum(tl.maximum(begin, 0), 128) if DEVICE_HISTORY else 128
        logical = tl.where(lanes < length, lanes, 128 + lanes - length)
        physical = tl.where(lanes < length + 5, batch * 144 + logical, -1).to(tl.int32)
        for step in tl.static_range(5):
            tl.store(slots + (batch * 5 + step) * 256 + lanes, physical)


def stage_draft_pages(field, query, table, start, *, device_history):
    """Caller qualifies A3 graph geometry; all output rows have one writer."""
    batch = start.numel()
    if (
        field.device.type != "npu"
        or field.dtype != torch.bfloat16
        or field.ndim != 2
        or field.shape[1] != 512
        or field.stride(1) != 1
        or query.shape != (batch * 5, 512)
        or query.dtype != field.dtype
        or query.device != field.device
        or not query.is_contiguous()
        or table.ndim != 2
        or table.shape[0] != batch
        or table.stride(1) != 1
        or table.dtype not in (torch.int32, torch.int64)
        or table.device != field.device
        or start.shape != (batch,)
        or start.dtype not in (torch.int32, torch.int64)
        or start.device != field.device
        or not start.is_contiguous()
    ):
        return None
    pages = torch.empty((batch, 144, 512), dtype=field.dtype, device=field.device)
    slots = torch.empty((batch * 5, 1, 256), dtype=torch.int32, device=field.device)
    _stage_draft_pages[(batch, 18)](
        field,
        query,
        table,
        start,
        pages,
        slots,
        field.shape[0],
        field.stride(0),
        table.stride(0),
        device_history,
        multibuffer=False,
    )
    return pages[:, :133], slots, pages
