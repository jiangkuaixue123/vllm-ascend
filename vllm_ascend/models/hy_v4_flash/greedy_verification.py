# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select each greedy verification row over TP-local vocabulary shards."""

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton


@triton.jit
def _partial_winners(logits, maxima, indices, WIDTH: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    column = part * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(logits + row * WIDTH + column, column < WIDTH, -float("inf")).to(tl.float32)
    val = tl.where(val == val, val, -float("inf"))
    peak = tl.max(val, 0)
    index = tl.min(tl.where((column < WIDTH) & (val == peak), column, 2147483647), 0)
    tl.store(maxima + row * TILES + part, peak)
    tl.store(indices + row * TILES + part, index)


@triton.jit
def _pack_row_winners(maxima, indices, packet, OFFSET: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    value = tl.load(maxima + row * TILES + part, part < TILES, -float("inf"))
    index = tl.load(indices + row * TILES + part, part < TILES, 2147483647)
    peak = tl.max(value, 0)
    idx = tl.min(tl.where(value == peak, index, 2147483647), 0)
    tl.store(packet + row * 2, peak.to(tl.int32, bitcast=True))
    tl.store(packet + row * 2 + 1, OFFSET + idx)


@triton.jit
def _merge_and_verify(
    packets,
    starts,
    drafts,
    sampled,
    lengths,
    ROWS: tl.constexpr,
    RANKS: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    req = tl.program_id(0)
    first = tl.load(starts + req).to(tl.int32)
    stop = tl.load(starts + req + 1).to(tl.int32)
    column = tl.arange(0, TILE)
    row = first + column
    valid = row < stop
    rank = tl.arange(0, RANKS)
    addr = (rank[None, :] * ROWS + row[:, None]) * 2
    value = tl.load(packets + addr, valid[:, None], 0).to(tl.float32, bitcast=True)
    token = tl.load(packets + addr + 1, valid[:, None], 0)
    # A finite maximum admits the smallest ID on ties, independently in every row.
    value = tl.where(value == value, value, -float("inf"))
    peak = tl.max(value, 1)
    ids = tl.min(tl.where(value == peak[:, None], token, 2147483647), 1)
    next_id = tl.load(drafts + row + 1, row + 1 < stop, -1)
    accept = (column < stop - first - 1) & (ids == next_id)
    length = tl.min(tl.where(accept, WIDTH, column), 0) + 1
    tl.store(sampled + req * WIDTH + column, tl.where(column < length, ids, -1), column < WIDTH)
    tl.store(lengths + req, length)


def choose(logits, group, start, nranks, starts, drafts, width):
    rows, width_local = logits.shape
    tile = 512
    tiles = triton.cdiv(width_local, tile)
    vals = logits.new_empty((rows, tiles))
    indices = logits.new_empty((rows, tiles), dtype=torch.int32)
    _partial_winners[(rows, tiles)](logits, vals, indices, width_local, tiles, tile, num_warps=1)
    send = logits.new_empty(rows * 2, dtype=torch.int32)
    gathered = send.new_empty(nranks * rows * 2)
    _pack_row_winners[(rows,)](vals, indices, send, start, tiles, triton.next_power_of_2(tiles), num_warps=1)
    dist.all_gather_into_tensor(gathered, send, group=group)
    requests = starts.shape[0] - 1
    sampled = send.new_empty((requests, width), dtype=torch.int64)
    lengths = send.new_empty(requests)
    _merge_and_verify[(requests,)](
        gathered, starts, drafts, sampled, lengths, rows, nranks, width, triton.next_power_of_2(width), num_warps=1
    )
    return sampled, lengths
