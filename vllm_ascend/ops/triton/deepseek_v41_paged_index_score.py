# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental A3 per-query BF16 index score over a physical latent-cache field."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_aicore_num

_TILE_KEYS = 64
_BLOCK_SIZE = 128
_INDEX_HEADS = 32
_INDEX_DIM = 128


@triton.jit
def _paged_index_score(
    q,
    keys,
    weights,
    source_lens,
    source_active,
    source_table,
    query_lens,
    query_active,
    output,
    PHYSICAL_ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    LANES: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    PAGE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // LANES
    lane = pid % LANES
    length = tl.load(query_lens + row).to(tl.int32)
    count = tl.where(tl.load(query_active + row) != 0, tl.minimum(tl.maximum(length, 0) // RATIO, CAPACITY), 0)
    source_length = tl.load(source_lens + row).to(tl.int32)
    live_source = tl.load(source_active + row) != 0
    heads = tl.arange(0, H)
    dims = tl.arange(0, D)
    # Query and head weights are constant across this program's key tiles.
    queries = tl.load(q + row * H * D + heads[:, None] * D + dims[None, :])
    w = tl.load(weights + row * H + heads)
    # No Cube/field-read iterations for an idle row or for its padded 16K suffix.
    for start in range(lane * N, count, LANES * N):
        ids = start + tl.arange(0, N)
        ends = (ids + 1) * RATIO - 1
        # A 64-key tile at ratio 1 or 2 is entirely inside one logical page
        # of 128 tokens. It therefore has a single physical page mapping.
        page = tl.load(source_table + row * TABLE_STRIDE + (start * RATIO) // PAGE).to(tl.int64)
        slots = page * PAGE + ends % PAGE
        valid = live_source & (ends < source_length) & (slots >= 0) & (slots < PHYSICAL_ROWS) & (ids < count)
        addresses = tl.where(valid, slots, 0) * ROW_STRIDE
        k = tl.load(keys + addresses[:, None] + dims[None, :], mask=valid[:, None], other=0)
        dots = tl.dot(queries, tl.trans(k), out_dtype=tl.float32)
        # C rounds the BF16 einsum result, then rounds the product of two BF16
        # operands before its head reduction; both conversion points are necessary.
        rounded = dots.to(tl.bfloat16).to(tl.float32)
        relu = tl.where(rounded < 0.0, 0.0, rounded)
        products = (relu * w.to(tl.float32)[:, None]).to(tl.bfloat16).to(tl.float32)
        scores = tl.sum(products, axis=0).to(tl.bfloat16)
        tl.store(output + row * CAPACITY + ids, scores, mask=ids < count)


def supports_paged_index_scores(q, field, weights, source_md, query_md, ratio, block_size):
    """Match layouts exercised by the A3 score kernel before dispatch."""
    if q.ndim != 3:
        return False
    rows, heads, dim = q.shape
    if (
        q.device.type != "npu"
        or q.dtype != torch.bfloat16
        or field.device != q.device
        or weights.device != q.device
        or source_md.block_table_tensor.device != q.device
        or source_md.block_table_tensor.dtype not in (torch.int32, torch.int64)
        or field.dtype != q.dtype
        or weights.dtype != q.dtype
        or not q.is_contiguous()
        or weights.shape != (rows, heads)
        or not weights.is_contiguous()
        or (heads, dim) != (_INDEX_HEADS, _INDEX_DIM)
        or field.ndim != 2
        or field.shape[1] != dim
        or field.stride(1) != 1
        or field.stride(0) < dim
        or block_size != _BLOCK_SIZE
        or ratio not in (1, 2)
        or not 0 < query_md.max_model_len <= torch.iinfo(torch.int32).max
        or source_md.max_model_len != query_md.max_model_len
        or source_md.block_table_tensor.shape[0] != rows
        or source_md.block_table_tensor.shape[1] * block_size < source_md.max_model_len
        or source_md.block_table_tensor.stride(1) != 1
    ):
        return False
    for tensor in (source_md.seq_lens, source_md.active, query_md.seq_lens, query_md.active):
        if tensor.device != q.device or tensor.shape != (rows,) or not tensor.is_contiguous():
            return False
    return (
        source_md.active.dtype == query_md.active.dtype == torch.bool
        and source_md.seq_lens.dtype in (torch.int32, torch.int64)
        and query_md.seq_lens.dtype in (torch.int32, torch.int64)
    )


def paged_index_scores(q, field, weights, source_md, query_md, ratio, block_size):
    """Produce a fixed-shape index score matrix for an already-qualified FULL bucket."""
    rows, heads, dim = q.shape
    capacity = query_md.max_model_len // ratio
    # The original full-size score matrix and all masked -inf values are maintained
    # even if a replay has fewer active rows or keys than its preceding replay.
    output = torch.full((rows, capacity), -torch.inf, dtype=q.dtype, device=q.device)
    if rows and capacity:
        lanes = max(1, get_aicore_num() // rows)
        _paged_index_score[(rows * lanes,)](
            q,
            field,
            weights,
            source_md.seq_lens,
            source_md.active,
            source_md.block_table_tensor,
            query_md.seq_lens,
            query_md.active,
            output,
            PHYSICAL_ROWS=field.shape[0],
            CAPACITY=capacity,
            ROW_STRIDE=field.stride(0),
            TABLE_STRIDE=source_md.block_table_tensor.stride(0),
            RATIO=ratio,
            LANES=lanes,
            D=dim,
            H=heads,
            N=_TILE_KEYS,
            PAGE=block_size,
        )
    return output
