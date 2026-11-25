# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton

from vllm_ascend.models.deepseek_v41.engram.npu import CHUNK_ROWS

_HASH_TILE_ROWS = 8
_HEADS_PER_ORDER = 8
_VERIFY_LAYERS = 4
_HISTORY_TOKENS = 2


def supports_verify_hash_geometry(state, hashes, history):
    columns = _HISTORY_TOKENS * _HEADS_PER_ORDER
    return (
        hashes.ndim == 3
        and hashes.shape[0] == _VERIFY_LAYERS
        and hashes.shape[1] > 0
        and hashes.shape[2] == columns
        and hashes.stride(2) == 1
        and state.multipliers.shape == (_VERIFY_LAYERS, _HISTORY_TOKENS + 1)
        and state.primes.shape == (_VERIFY_LAYERS, _HISTORY_TOKENS, _HEADS_PER_ORDER)
        and history.ndim == 2
        and history.shape[0] > 0
        and history.shape[1] == _HISTORY_TOKENS
        and history.is_contiguous()
    )


@triton.jit
def _verify_ngram_hashes(
    ids,
    history,
    token_map,
    factors,
    primes,
    offsets,
    output,
    starts,
    live_rows,
    map_size,
    mapped_pad,
    output_layer_stride,
    output_row_stride,
    PAD_ID: tl.constexpr,
    VISUAL_IDS: tl.constexpr,
    ROWS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    REQUESTS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    HISTORY: tl.constexpr,
    HEADS: tl.constexpr,
):
    layer = tl.program_id(0)
    row = tl.program_id(1) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    request = tl.full((TILE_ROWS,), 0, tl.int32)
    if REQUESTS > 1:
        for stage in tl.static_range(SEARCH_STEPS):
            candidate = request + (1 << (SEARCH_STEPS - stage - 1))
            offset = tl.load(starts + candidate, candidate < REQUESTS, 2147483647)
            request = tl.where((candidate < REQUESTS) & (offset <= row), candidate, request)
        first = tl.load(starts + request)
    else:
        first = 0
    head = tl.arange(0, HEADS)
    current = tl.load(ids + row, row < live_rows, -1)
    bad_output = (row >= live_rows) | (current < 0) | (current >= map_size) | (current == PAD_ID)
    for i in tl.static_range(len(VISUAL_IDS)):
        bad_output |= current == VISUAL_IDS[i]
    blocked = tl.full((TILE_ROWS,), False, tl.int1)
    rolling = tl.full((TILE_ROWS,), 0, tl.int64)
    for shift in tl.static_range(3):
        pos = row - first - shift
        present = (pos >= 0) & (row < live_rows)
        recent = tl.load(ids + row - shift, present, -1)
        past = tl.load(history + request * HISTORY - pos - 1, (pos < 0) & (row < live_rows), -1)
        token = tl.where(pos >= 0, recent, past)
        known = (token >= 0) & (token < map_size)
        mapped = tl.load(token_map + token, known, 0)
        blocked |= ~known
        value = tl.where(blocked, mapped_pad, mapped).to(tl.int64)
        rolling ^= value * tl.load(factors + layer * 3 + shift)
        if shift > 0:
            column = (shift - 1) * HEADS + head
            param = layer * (2 * HEADS) + column
            prime = tl.load(primes + param)
            rem = rolling[:, None] % prime[None, :]
            rem = tl.where(rem < 0, rem + prime[None, :], rem)
            hashed = rem + tl.load(offsets + param)[None, :]
            dest = layer * output_layer_stride + row[:, None] * output_row_stride + column[None, :]
            tl.store(output + dest, tl.where(bad_output[:, None], -1, hashed), row[:, None] < ROWS)


def fill_verify_hashes(state, input_ids, lookback, live_rows, output, starts=None):
    """Write per-layer indices directly into the strided replay input."""
    requests = lookback.shape[0]
    _verify_ngram_hashes[(output.shape[0], triton.cdiv(output.shape[1], _HASH_TILE_ROWS))](
        input_ids,
        lookback,
        state.token_map,
        state.multipliers,
        state.primes,
        state.offsets,
        output,
        lookback if starts is None else starts,
        live_rows,
        state.token_map.numel(),
        int(state.pad_id),
        output.stride(0),
        output.stride(1),
        state.layout.pad_token_id,
        state.visual_token_ids,
        output.shape[1],
        _HASH_TILE_ROWS,
        requests,
        (requests - 1).bit_length(),
        _HISTORY_TOKENS,
        _HEADS_PER_ORDER,
        num_warps=1,
    )


@triton.jit
def _read_host_rows(
    addresses,
    ids,
    output,
    id_stride,
    head_start,
    vocab_start,
    row_count,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    LANES: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    index = tl.load(ids + token * id_stride + head_start + head).to(tl.int64)
    # FlashEmbedding clamps each shard-local row, including invalid tokens.
    # The Engram layer applies the invalid-token mask after reading the rows.
    local = tl.minimum(tl.maximum(index - vocab_start, 0), row_count - 1)
    base = tl.load(addresses + local // CHUNK).to(tl.pointer_type(tl.bfloat16))
    col = tl.arange(0, LANES)
    values = tl.load(base + (local % CHUNK) * WIDTH + col, col < WIDTH, 0)
    tl.store(output + row * WIDTH + col, values.to(tl.float32), col < WIDTH)


def fill_host_feature_rows(table, hashes, output):
    """Read only the rank-local verification rows into the existing replay input."""
    heads = table.head_end - table.head_start
    _read_host_rows[(output.shape[0] * heads,)](
        table.host_weight.ptrs,
        hashes,
        output,
        hashes.stride(0),
        table.head_start,
        table.vocab_start,
        table.vocab_end - table.vocab_start,
        heads,
        table.dim,
        triton.next_power_of_2(table.dim),
        CHUNK_ROWS,
        num_warps=1,
    )
