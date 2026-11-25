# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather four registered CPU feature tables with one independent grid dimension."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_layers(
    ptr0,
    ptr1,
    ptr2,
    ptr3,
    hashes,
    out0,
    out1,
    out2,
    out3,
    layer_stride,
    token_stride,
    HEAD_STARTS: tl.constexpr,
    VOCAB_STARTS: tl.constexpr,
    ROW_COUNTS: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    LANES: tl.constexpr,
    CHUNK: tl.constexpr,
):
    addresses = (ptr0, ptr1, ptr2, ptr3)
    results = (out0, out1, out2, out3)
    layer = tl.program_id(1)
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    col = tl.arange(0, LANES)
    for i in tl.static_range(4):
        if layer == i:
            index = tl.load(hashes + layer * layer_stride + token * token_stride + HEAD_STARTS[i] + head).to(tl.int64)
            local = tl.minimum(tl.maximum(index - VOCAB_STARTS[i], 0), ROW_COUNTS[i] - 1)
            base = tl.load(addresses[i] + local // CHUNK).to(tl.pointer_type(tl.bfloat16))
            value = tl.load(base + (local % CHUNK) * WIDTH + col, col < WIDTH, 0)
            tl.store(results[i] + row * WIDTH + col, value.to(tl.float32), col < WIDTH)


def supports_grouped_feature_rows(tables, hashes, results):
    if len(tables) != 4 or len(results) != 4 or hashes.ndim != 3 or hashes.shape[0] != 4 or hashes.stride(2) != 1:
        return False
    heads = tables[0].head_end - tables[0].head_start
    width = tables[0].dim
    return (
        heads > 0
        and width > 0
        and all(
            table.host_weight is not None
            and table.head_end - table.head_start == heads
            and 0 <= table.head_start < table.head_end <= hashes.shape[2]
            and table.dim == width
            and output.dtype == torch.float32
            and output.device == hashes.device
            and output.shape == (hashes.shape[1], heads, width)
            and output.is_contiguous()
            for table, output in zip(tables, results, strict=True)
        )
    )


def fill_grouped_feature_rows(tables, hashes, results):
    # Resolve the host allocator after the worker has installed its device operators.
    from vllm_ascend.models.deepseek_v41.engram.npu import CHUNK_ROWS

    first = tables[0]
    heads = first.head_end - first.head_start
    width = first.dim
    _gather_layers[(hashes.shape[1] * heads, 4)](
        tables[0].host_weight.ptrs,
        tables[1].host_weight.ptrs,
        tables[2].host_weight.ptrs,
        tables[3].host_weight.ptrs,
        hashes,
        results[0],
        results[1],
        results[2],
        results[3],
        hashes.stride(0),
        hashes.stride(1),
        tuple(t.head_start for t in tables),
        tuple(t.vocab_start for t in tables),
        tuple(t.vocab_end - t.vocab_start for t in tables),
        heads,
        width,
        triton.next_power_of_2(width),
        CHUNK_ROWS,
        num_warps=1,
    )
