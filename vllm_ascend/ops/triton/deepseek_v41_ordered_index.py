# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain native TopK choices and fill cutoff ties in ascending column order."""

import torch
from vllm.triton_utils import tl, triton

_BLOCK = 1024
_LANES_PER_GROUP = 32
_OUTPUT_TILE = 4


@triton.jit
def _clear_kernel(original, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    column = block * BLOCK + tl.arange(0, BLOCK)
    tl.store(original + row * WIDTH + column, 0, column < WIDTH)


@triton.jit
def _mark_kernel(indices, original, WIDTH: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    column = tl.load(indices + row * K + slot, slot < K, 0)
    tl.store(original + row * WIDTH + column, 1, slot < K)


@triton.jit
def _population_count(word):
    value = word.to(tl.uint32)
    value = value - ((value >> 1) & 0x5555_5555)
    value = (value & 0x3333_3333) + ((value >> 2) & 0x3333_3333)
    value = (value + (value >> 4)) & 0x0F0F_0F0F
    value = value + (value >> 8)
    value = value + (value >> 16)
    return (value & 0x3F).to(tl.int32)


@triton.jit
def _exclusive_prefix(flags, one, GROUPS: tl.constexpr):
    lanes = tl.arange(0, 32)
    words = tl.sum(flags.to(tl.uint32) << lanes[None, :], axis=1)
    lower = (one.to(tl.uint32) << lanes) - 1
    within = _population_count(words[:, None] & lower[None, :])
    totals = tl.sum(flags.to(tl.int32), axis=1)
    groups = tl.arange(0, GROUPS)
    earlier = tl.sum(tl.where(groups[:, None] * one > groups[None, :], totals[None, :], 0), axis=1)
    return earlier[:, None] + within


@triton.jit
def _count_kernel(
    scores, original, cutoff, counts, WIDTH: tl.constexpr, K: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr
):
    row, block = tl.program_id(0), tl.program_id(1)
    column = block * BLOCK + tl.arange(0, BLOCK)
    valid = column < WIDTH
    value = tl.load(scores + row * WIDTH + column, valid, 0).to(tl.float32)
    selected = tl.load(original + row * WIDTH + column, valid, 0).to(tl.int1)
    boundary = tl.load(cutoff + row * K + K - 1).to(tl.float32)
    # Keep precisely the original non-cutoff selections, including any NaNs.
    above = selected & (value != boundary) & valid
    tied = (value == boundary) & valid
    offset = (row * BLOCKS + block) * 2
    tl.store(counts + offset, tl.sum(above.to(tl.int32), axis=0))
    tl.store(counts + offset + 1, tl.sum(tied.to(tl.int32), axis=0))


@triton.jit
def _rank_kernel(
    scores,
    original,
    indices,
    cutoff,
    counts,
    ranks,
    ends,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    # Use a device-resident one so prefix masks need no host constant transfer.
    first_selected = tl.load(indices + row * K)
    one = tl.load(original + row * WIDTH + first_selected)
    blocks = tl.arange(0, COUNT_BLOCK)
    offset = (row * BLOCKS + blocks) * 2
    above_counts = tl.load(counts + offset, blocks < BLOCKS, 0)
    tie_counts = tl.load(counts + offset + 1, blocks < BLOCKS, 0)
    above_before = tl.sum(tl.where(blocks < block, above_counts, 0), axis=0)
    ties_before = tl.sum(tl.where(blocks < block, tie_counts, 0), axis=0)
    needed = K - tl.sum(above_counts, axis=0)
    column = block * BLOCK + tl.arange(0, GROUPS)[:, None] * 32 + tl.arange(0, 32)[None, :]
    valid = column < WIDTH
    value = tl.load(scores + row * WIDTH + column, valid, 0).to(tl.float32)
    original_selected = tl.load(original + row * WIDTH + column, valid, 0).to(tl.int1)
    boundary = tl.load(cutoff + row * K + K - 1).to(tl.float32)
    tied = (value == boundary) & valid
    tie_rank = ties_before + _exclusive_prefix(tied, one, GROUPS)
    selected = (original_selected & (value != boundary) & valid) | (tied & (tie_rank < needed))
    base = above_before + tl.minimum(ties_before, needed)
    rank = base + _exclusive_prefix(selected, one, GROUPS)
    # Contiguous writes avoid the cross-block locks inserted for scattered
    # output indices. The next kernel gathers each final index by its rank.
    tl.store(ranks + row * WIDTH + column, tl.where(selected, rank, K), valid)
    count = tl.sum(tl.sum(selected.to(tl.int32), axis=1), axis=0)
    tl.store(ends + row * BLOCKS + block, base + count)


@triton.jit
def _emit_kernel(
    ranks,
    ends,
    output,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    OUTPUT_TILE: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tile * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
    lower = tl.full((OUTPUT_TILE,), 0, tl.int32)
    upper = tl.full((OUTPUT_TILE,), BLOCKS, tl.int32)
    for _ in range(SEARCH_STEPS):
        middle = (lower + upper) // 2
        end = tl.load(ends + row * BLOCKS + middle, middle < BLOCKS, K)
        before = end <= slot
        lower = tl.where(before, middle + 1, lower)
        upper = tl.where(before, upper, middle)
    column = lower[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    rank = tl.load(ranks + row * WIDTH + column, column < WIDTH, K)
    selected = tl.min(tl.where(rank == slot[:, None], column, WIDTH), axis=1)
    # Four int64 columns form an aligned, contiguous 32-byte output write.
    tl.store(output + row * K + slot, selected.to(tl.int64))


def ordered_index_columns(scores, k):
    rows, width = scores.shape
    top = scores.topk(k, dim=-1)
    # Keep native outputs contiguous and build the selection map on device.
    # Tensor scatter and a noncontiguous cutoff view are unnecessary here.
    original = torch.empty_like(scores, dtype=torch.int32)
    cutoff = top.values
    blocks = triton.cdiv(width, _BLOCK)
    counts = torch.empty((rows, blocks, 2), dtype=torch.int32, device=scores.device)
    ranks = torch.empty_like(original)
    ends = torch.empty((rows, blocks), dtype=torch.int32, device=scores.device)
    output = torch.empty((rows, k), dtype=torch.int64, device=scores.device)
    _clear_kernel[(rows, blocks)](original, WIDTH=width, BLOCK=_BLOCK)
    _mark_kernel[(rows,)](top.indices, original, WIDTH=width, K=k, BLOCK=triton.next_power_of_2(k))
    _count_kernel[(rows, blocks)](scores, original, cutoff, counts, WIDTH=width, K=k, BLOCKS=blocks, BLOCK=_BLOCK)
    _rank_kernel[(rows, blocks)](
        scores,
        original,
        top.indices,
        cutoff,
        counts,
        ranks,
        ends,
        WIDTH=width,
        K=k,
        BLOCKS=blocks,
        COUNT_BLOCK=triton.next_power_of_2(blocks),
        BLOCK=_BLOCK,
        GROUPS=_BLOCK // _LANES_PER_GROUP,
    )
    _emit_kernel[(rows, k // _OUTPUT_TILE)](
        ranks,
        ends,
        output,
        WIDTH=width,
        K=k,
        BLOCKS=blocks,
        BLOCK=_BLOCK,
        SEARCH_STEPS=blocks.bit_length(),
        OUTPUT_TILE=_OUTPUT_TILE,
    )
    return output
