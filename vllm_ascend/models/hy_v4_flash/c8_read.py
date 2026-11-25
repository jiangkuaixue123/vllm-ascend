# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE

from .cache_pages import c8_byte_offset

_LATENT = 512
_ROPE = 64
_ROWS_PER_PROGRAM = 8
_WINDOW_ROW_TILE = 16
_QK_HEAD_TILE = 16
_QK_CANDIDATE_TILE = 32


@triton.jit
def _decode_e4m3fn(bits):
    bits = bits.to(tl.int32)
    exponent = (bits >> 3) & 15
    fraction = bits & 7
    # E4M3FN normal numbers embed exactly in FP32. The smallest
    # E4M3FN subnormal is 2**-9.
    normal = (((exponent + 120) << 23) | (fraction << 20)).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, fraction.to(tl.float32) * (2.0**-9), normal)
    value = tl.where(bits & 128 != 0, -magnitude, magnitude)
    return tl.where(bits & 127 == 127, float("nan"), value)


@triton.jit(do_not_specialize=["count"])
def _read_c8_rows(
    cache,
    rope_cache,
    scale_cache,
    block_table,
    positions,
    output,
    count,
    ROW_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
    ROWS: tl.constexpr,
):
    i = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    in_bounds = i < count
    position = tl.load(positions + i, in_bounds, -1)
    live = in_bounds & (position >= 0)
    page = tl.load(block_table + position // PAGE_SIZE, live, 0).to(tl.int64)
    offset = c8_byte_offset(page, position, PAGE_SIZE, ROW_BYTES)
    lane = tl.arange(0, LATENT)
    bits = tl.load(cache + offset[:, None] + lane[None, :], live[:, None], 0)
    value = _decode_e4m3fn(bits)
    scale = tl.load(
        scale_cache + (offset // 4)[:, None] + (LATENT + 2 * ROPE) // 4 + lane[None, :] // SCALE_TILE,
        live[:, None],
        0.0,
    )
    dst = i[:, None] * (LATENT + ROPE)
    tl.store(output + dst + lane[None, :], value * scale, in_bounds[:, None])
    r = tl.arange(0, ROPE)
    rope = tl.load(rope_cache + (offset // 2)[:, None] + LATENT // 2 + r[None, :], live[:, None], 0.0).to(tl.float32)
    tl.store(output + dst + LATENT + r[None, :], rope, in_bounds[:, None])


def gather_c8_rows(cache, block_table, block_size, positions):
    output = torch.empty((*positions.shape, _LATENT + _ROPE), dtype=torch.float32, device=cache.device)
    count = positions.numel()
    if count:
        _read_c8_rows[(triton.cdiv(count, _ROWS_PER_PROGRAM),)](
            cache,
            cache.view(torch.bfloat16),
            cache.view(torch.float32),
            block_table,
            positions.contiguous(),
            output,
            count,
            cache.shape[-1],
            block_size,
            _LATENT,
            _ROPE,
            SFA_QSFA_TILE_SIZE,
            _ROWS_PER_PROGRAM,
            num_warps=1,
        )
    return output


@triton.jit
def _read_window_operands(
    cache,
    rope_cache,
    scale_cache,
    block_table,
    history,
    keys,
    values,
    TABLE_STRIDE: tl.constexpr,
    SEPARATE_VALUES: tl.constexpr,
    QUERIES: tl.constexpr,
    WINDOW: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    ROWS: tl.constexpr,
    SCALE_TILE: tl.constexpr,
):
    request = tl.program_id(2)
    block_table += request * TABLE_STRIDE
    history += request * (WINDOW + QUERIES - 1)
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    lane = group * SCALE_TILE + tl.arange(0, SCALE_TILE)
    in_bounds = row < WINDOW + QUERIES - 1
    position = tl.load(history + row, in_bounds, -1)
    live = in_bounds & (position >= 0)
    page = tl.load(block_table + position // PAGE_SIZE, live, 0).to(tl.int64)
    offset = c8_byte_offset(page, position, PAGE_SIZE, ROW_BYTES)
    if group * SCALE_TILE < LATENT:
        bits = tl.load(cache + offset[:, None] + lane[None, :], live[:, None], 0)
        scale = tl.load(scale_cache + offset // 4 + (LATENT + 2 * ROPE) // 4 + group, live, 0.0)
        data = _decode_e4m3fn(bits) * scale[:, None]
    else:
        data = tl.load(
            rope_cache + (offset // 2)[:, None] + LATENT // 2 + (lane - LATENT)[None, :],
            live[:, None] & (lane[None, :] < LATENT + ROPE),
            0.0,
        ).to(tl.float32)
    # Write each native GEMM batch independently. Both contractions retain the
    # reduction order of the corresponding single-query window.
    for query in tl.static_range(QUERIES):
        local = row - query
        keep = in_bounds & (local >= 0) & (local < WINDOW)
        dst = (request * QUERIES + query) * WINDOW + local
        tl.store(
            keys + dst[:, None] * (LATENT + ROPE) + lane[None, :], data, keep[:, None] & (lane[None, :] < LATENT + ROPE)
        )
        if SEPARATE_VALUES and group * SCALE_TILE < LATENT:
            tl.store(values + dst[:, None] * LATENT + lane[None, :], data, keep[:, None])


def gather_c8_window_operands(
    cache, block_table, block_size, history, queries, window, *, row_tile=_WINDOW_ROW_TILE, separate_values=True
):
    """Build contiguous native QK and PV operands from the union of consecutive C8 windows."""
    requests = history.shape[0] if history.ndim == 2 else 1
    rows = requests * queries
    keys = cache.new_empty((rows, window, _LATENT + _ROPE), dtype=torch.float32)
    values = cache.new_empty((rows, window, _LATENT), dtype=torch.float32) if separate_values else keys[:, :, :_LATENT]
    _read_window_operands[
        (triton.cdiv(history.shape[-1], row_tile), triton.cdiv(_LATENT + _ROPE, SFA_QSFA_TILE_SIZE), requests)
    ](
        cache,
        cache.view(torch.bfloat16),
        cache.view(torch.float32),
        block_table,
        history,
        keys,
        values,
        block_table.stride(0) if block_table.ndim == 2 else 0,
        separate_values,
        queries,
        window,
        _LATENT,
        _ROPE,
        block_size,
        cache.shape[-1],
        row_tile,
        SFA_QSFA_TILE_SIZE,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return keys.transpose(1, 2), values


@triton.jit
def _read_c8_qk(
    query,
    cache,
    rope_cache,
    scale_cache,
    block_table,
    positions,
    scores,
    values,
    ROW_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    CANDIDATE_TILE: tl.constexpr,
):
    qi = tl.program_id(0)
    row = tl.program_id(1) * CANDIDATE_TILE + tl.arange(0, CANDIDATE_TILE)
    head = tl.arange(0, HEAD_TILE)
    position = tl.load(positions + qi * TOPK + row, row < TOPK, -1)
    live = position >= 0
    page = tl.load(block_table + position // PAGE_SIZE, live, 0).to(tl.int64)
    offset = c8_byte_offset(page, position, PAGE_SIZE, ROW_BYTES)
    q_base = (qi * HEADS + head[:, None]) * (LATENT + ROPE)
    value_base = (qi * TOPK + row[:, None]) * LATENT
    acc = tl.zeros((HEAD_TILE, CANDIDATE_TILE), tl.float32)

    lane = tl.arange(0, SCALE_TILE)
    for group in tl.static_range(LATENT // SCALE_TILE):
        col = group * SCALE_TILE + lane
        q = tl.load(query + q_base + col[None, :], head[:, None] < HEADS, 0).to(tl.float32)
        bits = tl.load(cache + offset[:, None] + col[None, :], live[:, None], 0)
        scale = tl.load(scale_cache + offset // 4 + (LATENT + 2 * ROPE) // 4 + group, live, 0)
        k = _decode_e4m3fn(bits) * scale[:, None]
        tl.store(values + value_base + col[None, :], k, row[:, None] < TOPK)
        acc = tl.dot(q, tl.trans(k), acc, input_precision="ieee")

    rope = tl.arange(0, ROPE)
    q_rope = tl.load(query + q_base + LATENT + rope[None, :], head[:, None] < HEADS, 0).to(tl.float32)
    k_rope = tl.load(rope_cache + (offset // 2)[:, None] + LATENT // 2 + rope[None, :], live[:, None], 0).to(tl.float32)
    acc = tl.dot(q_rope, tl.trans(k_rope), acc, input_precision="ieee")
    tl.store(
        scores + (qi * HEADS + head[:, None]) * TOPK + row[None, :],
        acc,
        (head[:, None] < HEADS) & (row[None, :] < TOPK),
    )


def gather_c8_qk(query, cache, block_table, block_size, positions):
    """Return FP32 QK and the 512 latent values needed by PV for each candidate."""
    queries, heads, _ = query.shape
    topk = positions.shape[1]
    scores = torch.empty((queries, heads, topk), dtype=torch.float32, device=query.device)
    values = torch.empty((queries, topk, _LATENT), dtype=torch.float32, device=query.device)
    _read_c8_qk[(queries, triton.cdiv(topk, _QK_CANDIDATE_TILE))](
        query,
        cache,
        cache.view(torch.bfloat16),
        cache.view(torch.float32),
        block_table,
        positions.contiguous(),
        scores,
        values,
        cache.shape[-1],
        block_size,
        _LATENT,
        _ROPE,
        SFA_QSFA_TILE_SIZE,
        heads,
        topk,
        _QK_HEAD_TILE,
        _QK_CANDIDATE_TILE,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return scores, values
