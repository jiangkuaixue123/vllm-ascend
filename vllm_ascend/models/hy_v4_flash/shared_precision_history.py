# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch
from vllm.models.hy_v4_flash.reference import mla_attention_probs
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES

from .c8_read import _decode_e4m3fn
from .cache_pages import PAGE_BYTES, PAGE_TOKENS, c8_byte_offset

DRAFT_HISTORY_ROWS = 2048
DRAFT_VERIFY_WIDTH = 6
_LATENT_DIM = 512
_ROPE_DIM = 64
_HEAD_DIM = _LATENT_DIM + _ROPE_DIM
_TILE_ROWS = 32


class SharedPrecisionHistory(NamedTuple):
    target_rows: torch.Tensor
    active: torch.Tensor


@triton.jit
def _read_draft_history(
    cache,
    cache_rope,
    cache_scale,
    draft,
    draft_rope,
    draft_scale,
    table,
    target,
    anchor,
    active,
    keys,
    values,
    candidates,
    WIDTH: tl.constexpr,
    HISTORY: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    PAGE: tl.constexpr,
    DRAFT_BYTES: tl.constexpr,
    LATENT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE_START: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    LANES: tl.constexpr,
):
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    group = tl.program_id(1)
    col = group * LANES + tl.arange(0, LANES)
    count = HISTORY + WIDTH
    is_history = row < HISTORY
    start = tl.load(anchor)
    position = tl.load(target + row, is_history, -1)
    enabled = tl.load(active) != 0
    live = is_history & (position >= 0) & (position < start) & enabled
    page = tl.load(table + position // PAGE, live, 0).to(tl.int64)
    byte_offset = c8_byte_offset(page, position, PAGE, ROW_BYTES)
    is_draft = (row >= HISTORY) & (row < count) & enabled
    offset = row - HISTORY
    if group < LATENT // LANES:
        cached_bits = tl.load(cache + byte_offset[:, None] + col[None, :], live[:, None], 0)
        draft_bits = tl.load(draft + offset[:, None] * DRAFT_BYTES + col[None, :], is_draft[:, None], 0)
        cached_scale = tl.load(cache_scale + byte_offset // 4 + SCALE_START + group, live, 0.0)
        local_scale = tl.load(draft_scale + offset * (DRAFT_BYTES // 4) + SCALE_START + group, is_draft, 0.0)
        data = _decode_e4m3fn(tl.where(is_history[:, None], cached_bits, draft_bits))
        data *= tl.where(is_history, cached_scale, local_scale)[:, None]
    else:
        inside = col[None, :] < HEAD_DIM
        rope_col = LATENT // 2 + (col - LATENT)[None, :]
        cached = tl.load(cache_rope + (byte_offset // 2)[:, None] + rope_col, live[:, None] & inside, 0.0)
        local = tl.load(draft_rope + offset[:, None] * (DRAFT_BYTES // 2) + rope_col, is_draft[:, None] & inside, 0.0)
        data = tl.where(is_history[:, None], cached, local).to(tl.float32)
    if group == 0:
        for query in tl.static_range(WIDTH):
            logical = tl.where(live, position, tl.where(is_draft & (offset <= query), start + offset, -1))
            tl.store(candidates + query * count + row, logical, row < count)
    tl.store(keys + row[:, None] * HEAD_DIM + col[None, :], data, (row[:, None] < count) & (col[None, :] < HEAD_DIM))
    if group < LATENT // LANES:
        tl.store(values + row[:, None] * LATENT + col[None, :], data, row[:, None] < count)


def prepare_draft_operands(cache, table, draft, history, anchor, page_size):
    """Gather and dequantize a single recall snapshot plus its causal draft block."""
    count = history.target_rows.numel()
    width = draft.shape[0]
    keys = cache.new_empty((count + width, _HEAD_DIM), dtype=torch.float32)
    values = cache.new_empty((count + width, _LATENT_DIM), dtype=torch.float32)
    selected = cache.new_empty((width, count + width), dtype=torch.int32)
    groups = triton.cdiv(_HEAD_DIM, SFA_QSFA_TILE_SIZE)
    _read_draft_history[(triton.cdiv(count + width, _TILE_ROWS), groups)](
        cache,
        cache.view(torch.bfloat16),
        cache.view(torch.float32),
        draft,
        draft.view(torch.bfloat16),
        draft.view(torch.float32),
        table,
        history.target_rows,
        anchor,
        history.active,
        keys,
        values,
        selected,
        width,
        count,
        cache.shape[-1],
        page_size,
        C8_ROW_BYTES,
        _LATENT_DIM,
        _HEAD_DIM,
        (_LATENT_DIM + _ROPE_DIM * torch.bfloat16.itemsize) // torch.float32.itemsize,
        _TILE_ROWS,
        SFA_QSFA_TILE_SIZE,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return keys, values, selected


def attend_draft_history(query, cache, table, draft, history, anchor, sink, scale, page_size):
    keys, values, selected = prepare_draft_operands(cache, table, draft, history, anchor, page_size)
    queries = query.shape[0]
    expanded = keys.unsqueeze(0).expand(queries, -1, -1).contiguous()
    probabilities = mla_attention_probs(query, expanded, selected, sink, scale)
    values = values.unsqueeze(0).expand(queries, -1, -1)
    return torch.bmm(probabilities, values).to(query.dtype)


def supports_shared_history(source, table, snapshot, positions, window):
    pages = source.kv_cache
    return (
        window == 0
        and source.page_planes
        and pages.device.type == "npu"
        and pages.dtype == torch.uint8
        and pages.is_contiguous()
        and pages.shape[2] == PAGE_TOKENS
        and pages.numel() >= PAGE_BYTES
        and source.kv_storage_width == C8_ROW_BYTES
        and table.stride(-1) == 1
        and positions.shape == (1, DRAFT_VERIFY_WIDTH)
        and snapshot.shape == (1, DRAFT_HISTORY_ROWS)
        and snapshot.is_contiguous()
    )
