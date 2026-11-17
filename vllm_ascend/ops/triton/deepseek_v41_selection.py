# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize selection consumers without sorting or scanning integer indices."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element

_BLOCK_COLUMNS = 128


@triton.jit
def _block_tie_counts(
    scores, selected, counts, WIDTH: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr, TILE: tl.constexpr
):
    row, tile = tl.program_id(0), tl.program_id(1)
    columns = tile * TILE + tl.arange(0, TILE)
    cutoff = tl.load(selected + row * K + K - 1).to(tl.float32)
    values = tl.load(scores + row * WIDTH + columns, columns < WIDTH, 0).to(tl.float32)
    own = tl.sum(((columns < WIDTH) & (values == cutoff)).to(tl.int32), 0)
    tl.store(counts + row * (TILES + 1) + tile, own)
    if tile == 0:
        chosen = tl.load(selected + row * K + tl.arange(0, K)).to(tl.float32)
        tl.store(counts + row * (TILES + 1) + TILES, tl.sum((chosen == cutoff).to(tl.int32), 0))


@triton.jit
def _candidate_block_mask(
    scores,
    selected,
    counts,
    output,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    COUNT_PAD: tl.constexpr,
    TILE: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    lanes = tl.arange(0, TILE)
    columns = tile * TILE + lanes
    cutoff = tl.load(selected + row * K + K - 1).to(tl.float32)
    values = tl.load(scores + row * WIDTH + columns, columns < WIDTH, -float("inf")).to(tl.float32)
    earlier = tl.arange(0, COUNT_PAD)
    prior = tl.sum(tl.load(counts + row * (TILES + 1) + earlier, earlier < tile, 0), 0)
    own = tl.load(counts + row * (TILES + 1) + tile)
    needed = tl.load(counts + row * (TILES + 1) + TILES)
    tied = (values == cutoff) & (columns < WIDTH)
    keep = (values > cutoff) | (tied & (prior + own <= needed))
    # The original gather/scatter excludes NaNs and -inf, including a NaN cutoff.
    keep = keep & (values > -float("inf"))
    tl.store(output + row * WIDTH + columns, keep, columns < WIDTH)
    if (cutoff > -float("inf")) & (prior < needed) & (needed < prior + own):
        # Only the boundary tile needs stable, lower-column tie selection.
        ordinal = prior
        lane = 0
        tied_i32 = tied.to(tl.int32)
        while (lane < TILE) & (ordinal < needed):
            if get_element(tied_i32, (lane,)) != 0:
                tl.store(output + row * WIDTH + tile * TILE + lane, True)
                ordinal += 1
            lane += 1


def candidate_block_mask(scores, selected):
    rows, width = scores.shape
    k = selected.shape[-1]
    tiles = triton.cdiv(width, _BLOCK_COLUMNS)
    counts = torch.empty((rows, tiles + 1), dtype=torch.int32, device=scores.device)
    output = torch.empty_like(scores, dtype=torch.bool)
    _block_tie_counts[(rows, tiles)](scores, selected, counts, width, k, tiles, _BLOCK_COLUMNS, multibuffer=False)
    _candidate_block_mask[(rows, tiles)](
        scores,
        selected,
        counts,
        output,
        width,
        k,
        tiles,
        triton.next_power_of_2(tiles),
        _BLOCK_COLUMNS,
        multibuffer=False,
    )
    return output
