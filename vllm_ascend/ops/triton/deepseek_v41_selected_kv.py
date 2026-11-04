# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Joint gather for V4.1 target decode compressed layers. Return BF16 selected KV and original logical IDs."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

TILE = 8
DIM = 512


@triton.jit
def _page_slot(
    table,
    row,
    pos,
    length,
    live,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    PHYS: tl.constexpr,
):
    clamped = tl.maximum(0, tl.minimum(pos, MAX_LEN - 1))
    logical = clamped // BLOCK
    block = tl.cast(
        tl.load(table + row * TABLE_ROW + logical * TABLE_COL, mask=live, other=0),
        tl.int64,
    )
    slot = block * BLOCK + clamped % BLOCK
    valid = live & (pos >= 0) & (pos < length) & (slot >= 0) & (slot < PHYS)
    return tl.where(valid, slot, 0), valid


@triton.jit
def _joint(
    raw,
    comp,
    raw_table,
    comp_table,
    raw_len,
    comp_len,
    raw_active,
    comp_active,
    topk,
    selected,
    ids,
    ROWS: tl.constexpr,
    WINDOW: tl.constexpr,
    EXTRA: tl.constexpr,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    RM: tl.constexpr,
    RB: tl.constexpr,
    RR: tl.constexpr,
    RC: tl.constexpr,
    RP: tl.constexpr,
    RS: tl.constexpr,
    CM: tl.constexpr,
    CB: tl.constexpr,
    CR: tl.constexpr,
    CC: tl.constexpr,
    CP: tl.constexpr,
    CS: tl.constexpr,
    TR: tl.constexpr,
    TC: tl.constexpr,
    FP32_SRC: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
):
    dims = tl.arange(0, D)
    for tile in range(tl.program_id(0), tl.cdiv(ROWS * WIDTH, N), tl.num_programs(0)):
        flat = tile * N + tl.arange(0, N)
        in_bounds = flat < ROWS * WIDTH
        row = flat // WIDTH
        col = flat % WIDTH
        length = tl.load(raw_len + row, mask=in_bounds, other=0)
        active = tl.load(raw_active + row, mask=in_bounds, other=0) != 0
        width = tl.minimum(WINDOW, tl.maximum(length, 0))
        ordinal = col - width
        eligible = in_bounds & (ordinal >= 0) & (ordinal < EXTRA)
        idx = tl.load(
            topk + row * TR + tl.where(eligible, ordinal, 0) * TC,
            mask=eligible,
            other=-1,
        )
        compressed = eligible & (idx >= 0)
        position = tl.maximum(length - WINDOW, 0) + col
        src_position = (idx + 1) * RATIO - 1
        slot, raw_live = _page_slot(
            raw_table,
            row,
            position,
            length,
            in_bounds & active & ~compressed,
            RM,
            RB,
            RR,
            RC,
            RP,
        )
        src_len = tl.load(comp_len + row, mask=in_bounds, other=0)
        src_active = tl.load(comp_active + row, mask=in_bounds, other=0) != 0
        src_slot, src_live = _page_slot(
            comp_table,
            row,
            src_position,
            src_len,
            compressed & src_active,
            CM,
            CB,
            CR,
            CC,
            CP,
        )
        r = tl.load(raw + slot[:, None] * RS + dims[None, :], mask=raw_live[:, None], other=0)
        c = tl.load(
            comp + src_slot[:, None] * CS + dims[None, :],
            mask=src_live[:, None],
            other=0,
        )
        if FP32_SRC:
            # C promotes the BF16 raw selection for its FP32 tensor Where.
            r = tl.cast(r, tl.int32) << 16
        chosen = tl.where(compressed[:, None], c, r)
        tl.store(
            selected + flat[:, None] * D + dims[None, :],
            chosen,
            mask=in_bounds[:, None],
        )
        # Membership and cache read validity are intentionally independent.
        logical_id = tl.where(active & ((col < width) | compressed), col, -1)
        tl.store(ids + flat, logical_id, mask=in_bounds)


def try_prepare_selected_kv(raw, comp, md, source_md, topk, window, ratio, raw_block, comp_block):
    """Return the staged latent rows and logical IDs, or None for the ordinary cache reader."""
    if not (
        raw.device.type == "npu"
        and get_ascend_device_type() == AscendDeviceType.A3
        and raw.dtype == torch.bfloat16
        and comp.dtype in (torch.bfloat16, torch.float32)
        and comp.device == raw.device
        and raw.ndim == comp.ndim == 2
        and raw.shape[1] == comp.shape[1] == DIM
        and raw.stride(1) == comp.stride(1) == 1
        and not (raw.requires_grad or comp.requires_grad)
        and topk.ndim == 2
        and topk.dtype == torch.int64
        and topk.device == raw.device
        and ratio in (1, 2)
        and window > 0
    ):
        return None
    rows, extra = topk.shape
    if rows <= 0 or extra <= 0:
        return None
    for m, block in ((md, raw_block), (source_md, comp_block)):
        table = m.block_table_tensor
        if not (
            m.seq_lens.shape == m.active.shape == (rows,)
            and table.ndim == 2
            and table.shape[0] == rows
            and m.seq_lens.dtype == torch.int64
            and m.active.dtype == torch.bool
            and m.seq_lens.stride(0) == m.active.stride(0) == 1
            and table.dtype == torch.int32
            and 0 < m.max_model_len <= table.shape[1] * block
            and block > 0
            and m.seq_lens.device == m.active.device == table.device == raw.device
        ):
            return None
    width = min(window, md.max_model_len) + extra
    if rows * width * DIM > torch.iinfo(torch.int32).max:
        return None
    fp32 = comp.dtype == torch.float32
    packed = torch.empty(
        (rows, width, DIM),
        dtype=torch.int32 if fp32 else torch.int16,
        device=raw.device,
    )
    ids = torch.empty((rows, 1, width), dtype=torch.int32, device=raw.device)
    t, st = md.block_table_tensor, source_md.block_table_tensor
    grid = (min(triton.cdiv(rows * width, TILE), get_vectorcore_num()),)
    _joint[grid](
        raw.view(torch.int16),
        comp.view(packed.dtype),
        t,
        st,
        md.seq_lens,
        source_md.seq_lens,
        md.active,
        source_md.active,
        topk,
        packed,
        ids,
        rows,
        window,
        extra,
        width,
        ratio,
        md.max_model_len,
        raw_block,
        *t.stride(),
        raw.shape[0],
        raw.stride(0),
        source_md.max_model_len,
        comp_block,
        *st.stride(),
        comp.shape[0],
        comp.stride(0),
        *topk.stride(),
        fp32,
        DIM,
        TILE,
    )
    # Native Where preserves the parent's A3 treatment of NaN payloads. A live cross-builder
    # source may supply values on a dead consumer, so membership IDs cannot mask this tensor.
    any_live = (md.active | source_md.active)[:, None, None]
    values = torch.where(any_live, packed.view(comp.dtype), 0.0)
    return values.to(torch.bfloat16), ids
