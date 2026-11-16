# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact cutoff ties without a second full-width TopK or an int64 scan."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element

_TILE_COLUMNS = 2048
_SELECTED_COLUMNS = 512


@triton.jit
def _finalize_prefill_ties(selected, indices, first, output, K: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, K)
    values = tl.load(selected + row * K + lanes).to(tl.float32)
    cutoff = tl.load(selected + row * K + K - 1).to(tl.float32)
    tied = values == cutoff
    # Sorted TopK puts finite cutoff ties in its final suffix; NaN ties are all false.
    ordinal = lanes - (K - tl.sum(tied.to(tl.int32), 0))
    choices = tl.load(first + row * K + lanes)
    replacement = tl.gather(choices, tl.maximum(ordinal, 0), axis=0).to(tl.int64)
    original = tl.load(indices + row * K + lanes)
    tl.store(output + row * K + lanes, tl.where(tied, replacement, original))


def finalize_prefill_ties(selected, indices, first):
    if (
        selected.shape != indices.shape
        or selected.shape != first.shape
        or selected.ndim != 2
        or selected.shape[1] != _SELECTED_COLUMNS
        or not all(x.is_contiguous() for x in (selected, indices, first))
    ):
        return None
    output = torch.empty_like(indices)
    _finalize_prefill_ties[(selected.shape[0],)](selected, indices, first, output, K=_SELECTED_COLUMNS)
    return output


@triton.jit
def _count_ties(
    scores, selected, counts, WIDTH: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr, TILE: tl.constexpr
):
    row, tile = tl.program_id(0), tl.program_id(1)
    columns = tile * TILE + tl.arange(0, TILE)
    cutoff = tl.load(selected + row * K + K - 1).to(tl.float32)
    values = tl.load(scores + row * WIDTH + columns, columns < WIDTH, 0).to(tl.float32)
    tied = (columns < WIDTH) & (values == cutoff)
    tl.store(counts + row * TILES + tile, tl.sum(tied.to(tl.int32), 0))


@triton.jit
def _compact_ties(
    scores, selected, counts, ties, WIDTH: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr, TILE: tl.constexpr
):
    row = tl.program_id(0)
    cutoff = tl.load(selected + row * K + K - 1).to(tl.float32)
    chosen = tl.load(selected + row * K + tl.arange(0, K)).to(tl.float32)
    needed = tl.sum((chosen == cutoff).to(tl.int32), 0)
    tl.store(ties + row * K + tl.arange(0, K), 0)
    written = 0
    tile = 0
    while (tile < TILES) & (written < needed):
        own = tl.load(counts + row * TILES + tile)
        if own > 0:
            columns = tile * TILE + tl.arange(0, TILE)
            values = tl.load(scores + row * WIDTH + columns, columns < WIDTH, 0).to(tl.float32)
            tied = ((columns < WIDTH) & (values == cutoff)).to(tl.int32)
            offset = 0
            # Use the same scalar compaction primitive as sparse_index_remap.
            while (offset < TILE) & (written < needed):
                if get_element(tied, (offset,)) != 0:
                    tl.store(ties + row * K + written, tile * TILE + offset)
                    written += 1
                offset += 1
        tile += 1


@triton.jit
def _replace_ties(selected, indices, ties, output, K: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, K)
    chosen = tl.load(selected + row * K + lanes).to(tl.float32)
    cutoff = tl.load(selected + row * K + K - 1).to(tl.float32)
    tied = chosen == cutoff
    # Native TopK is sorted: all equal cutoff entries form its final suffix.
    ordinal = lanes - (K - tl.sum(tied.to(tl.int32), 0))
    original = tl.load(indices + row * K + lanes)
    replacement = tl.load(ties + row * K + tl.maximum(ordinal, 0))
    tl.store(output + row * K + lanes, tl.where(tied, replacement.to(tl.int64), original))


def compact_tied_columns(scores, selected, indices):
    """Keep native non-cutoff/NaN choices and replace only cutoff ties by column."""
    if not scores.is_contiguous() or scores.dtype not in (torch.bfloat16, torch.float32):
        return None
    rows, width = scores.shape
    tiles = triton.cdiv(width, _TILE_COLUMNS)
    counts = torch.empty((rows, tiles), dtype=torch.int32, device=scores.device)
    ties = torch.empty((rows, _SELECTED_COLUMNS), dtype=torch.int32, device=scores.device)
    output = torch.empty_like(indices)
    grid = (rows, tiles)
    _count_ties[grid](scores, selected, counts, width, _SELECTED_COLUMNS, tiles, _TILE_COLUMNS, multibuffer=False)
    _compact_ties[(rows,)](
        scores,
        selected,
        counts,
        ties,
        width,
        _SELECTED_COLUMNS,
        tiles,
        _TILE_COLUMNS,
        multibuffer=False,
    )
    _replace_ties[(rows,)](selected, indices, ties, output, _SELECTED_COLUMNS, multibuffer=False)
    return output
