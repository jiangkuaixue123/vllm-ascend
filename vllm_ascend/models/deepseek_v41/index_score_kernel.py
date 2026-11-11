# SPDX-License-Identifier: Apache-2.0
"""Device-resident scoring of the V4.1 paged E2M1/UE8M0 index."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rounded_bf16(values):
    # A5 may fuse floating extf/truncf with the following multiply. Preserve the
    # checkpoint's separately rounded BF16 intermediate in the FP32 bit pattern.
    bits = values.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _score_paged(
    Q,
    Weights,
    Records,
    Pages,
    Lengths,
    Blocks,
    Scores,
    PAGE_STRIDE,
    SCORE_STRIDE,
    WIDTH,
    RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STRIPES: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    stripe = tl.program_id(1)
    live = tl.minimum(tl.load(Lengths + row).to(tl.int32), WIDTH)
    features = tl.arange(0, 128)
    heads = tl.arange(0, 32)
    q = tl.load(Q + (row * 32 + heads[:, None]) * 128 + features[None, :])
    weights = tl.load(Weights + row * 32 + heads)
    if N_BLOCKS:
        # _candidate_blocks sorts top-k values and then turns its -inf tail into -1.
        # Every nonempty logical block is visible in at most ceil(live / BLOCK_SIZE) slots.
        stop = tl.minimum(tl.cdiv(live, BLOCK_SIZE), N_BLOCKS) * BLOCK_SIZE
    else:
        stop = live
    for tile in range(stripe, tl.cdiv(stop, TILE), STRIPES):
        seq = tile * TILE + tl.arange(0, TILE)
        if N_BLOCKS:
            block = tl.load(Blocks + row * N_BLOCKS + seq // BLOCK_SIZE, mask=seq < stop, other=-1)
            logical = block * BLOCK_SIZE + seq % BLOCK_SIZE
        else:
            logical = seq
        valid = (seq < stop) & (logical >= 0) & (logical < live)
        safe = tl.where(valid, logical, 0)
        last = (safe + 1) * RATIO - 1
        page = tl.load(Pages + row * PAGE_STRIDE + last // PAGE_SIZE, mask=valid, other=0).to(tl.int32)
        address = (page * PAGE_SIZE + last % PAGE_SIZE) // RATIO
        base = Records + address[:, None] * 68
        packed = tl.load(base + features[None, :] // 2, mask=valid[:, None], other=0).to(tl.uint32)
        nibble = (packed >> ((features[None, :] & 1) * 4)) & 15
        m = nibble & 7
        magnitude = tl.where(
            m < 4, m.to(tl.float32) * 0.5, tl.where(m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0)))
        )
        level = tl.where(nibble >= 8, -magnitude, magnitude)
        exponent = tl.load(base + 64 + features[None, :] // 32, mask=valid[:, None], other=0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        keys = (level * scale).to(tl.bfloat16)
        dots = _rounded_bf16(tl.dot(q, tl.trans(keys), out_dtype=tl.float32))
        product = _rounded_bf16(tl.maximum(dots, 0.0) * weights[:, None].to(tl.float32))
        score = _rounded_bf16(tl.sum(product, axis=0)).to(tl.bfloat16)
        tl.store(Scores + row * SCORE_STRIDE + safe, score, mask=valid)


def paged_index_scores(
    query, weight, records, pages, lengths, width, ratio, page_size, candidate_blocks=None, block_size=8
):
    """Return original-order scores for the existing backend's full-width top-k and tie handling."""
    rows = query.shape[0]
    if (
        query.shape[1:] != (32, 128)
        or weight.shape != (rows, 32)
        or records.shape[1:] != (68,)
        or page_size % ratio
        or ratio not in (1, 2)
        or block_size != 8
        or width < 1
        or not query.is_contiguous()
        or not weight.is_contiguous()
        or not records.is_contiguous()
        or pages.stride(-1) != 1
        or lengths.numel() != rows
        or not lengths.is_contiguous()
        or query.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or records.dtype != torch.uint8
        or pages.dtype != torch.int64
        or lengths.dtype != torch.int64
    ):
        raise ValueError("Unexpected V4.1 paged index input layout")
    # Mask the unvisited tail and all non-candidate columns before independent blocks scatter.
    # The subsequent launch reads the original sorted block list and writes each valid position once.
    scores = query.new_full((rows, width), -torch.inf)
    n_blocks = 0 if candidate_blocks is None else candidate_blocks.shape[1]
    if candidate_blocks is not None and (
        candidate_blocks.shape[0] != rows
        or not candidate_blocks.is_contiguous()
        or candidate_blocks.dtype != torch.int32
        or n_blocks < 1
    ):
        raise ValueError("Unexpected V4.1 candidate block layout")
    tile = 32
    stripes = 32
    _score_paged[(rows, stripes)](
        query,
        weight,
        records,
        pages,
        lengths,
        candidate_blocks if n_blocks else lengths,
        scores,
        pages.stride(0),
        width,
        width,
        ratio,
        page_size,
        n_blocks,
        block_size,
        stripes,
        tile,
    )
    return scores
