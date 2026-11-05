# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain native TopK choices and fill cutoff ties in ascending column order."""

import torch
from vllm.triton_utils import tl, triton

_BLOCK = 1024
_BITS_PER_WORD = 32


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
    boundary = tl.load(cutoff + row * K).to(tl.float32)
    # Keep precisely the original non-cutoff selections, including any NaNs.
    above = selected & (value != boundary) & valid
    tied = (value == boundary) & valid
    offset = (row * BLOCKS + block) * 2
    tl.store(counts + offset, tl.sum(above.to(tl.int32), axis=0))
    tl.store(counts + offset + 1, tl.sum(tied.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["one"])
def _select_kernel(
    scores,
    original,
    cutoff,
    counts,
    output,
    one,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
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
    boundary = tl.load(cutoff + row * K).to(tl.float32)
    tied = (value == boundary) & valid
    tie_rank = ties_before + _exclusive_prefix(tied, one, GROUPS)
    selected = (original_selected & (value != boundary) & valid) | (tied & (tie_rank < needed))
    base = above_before + tl.minimum(ties_before, needed)
    rank = base + _exclusive_prefix(selected, one, GROUPS)
    tl.store(output + row * K + rank, column.to(tl.int64), selected)


def ordered_index_columns(scores, k):
    rows, width = scores.shape
    top = scores.topk(k, dim=-1)
    original = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, True)
    cutoff = top.values[:, -1]
    blocks = triton.cdiv(width, _BLOCK)
    counts = torch.empty((rows, blocks, 2), dtype=torch.int32, device=scores.device)
    output = torch.empty((rows, k), dtype=torch.int64, device=scores.device)
    _count_kernel[(rows, blocks)](scores, original, cutoff, counts, WIDTH=width, K=k, BLOCKS=blocks, BLOCK=_BLOCK)
    _select_kernel[(rows, blocks)](
        scores,
        original,
        cutoff,
        counts,
        output,
        1,
        WIDTH=width,
        K=k,
        BLOCKS=blocks,
        COUNT_BLOCK=triton.next_power_of_2(blocks),
        BLOCK=_BLOCK,
        GROUPS=_BLOCK // _BITS_PER_WORD,
    )
    return output
