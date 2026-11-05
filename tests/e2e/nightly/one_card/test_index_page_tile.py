# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare page-sized key tiles against the frozen parent scoring kernel."""

import statistics
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.deepseek_v41_paged_index_score import paged_index_scores, supports_paged_index_scores
from vllm_ascend.ops.triton.triton_utils import get_aicore_num, init_device_properties_triton

MODEL_LIMIT = 153600
PAGE_SIZE = 128
PHYSICAL_PAGES = 16
HEADS = 32
DIM = 128
FIELD_WIDTH = 1024
FIELD_OFFSET = 512
TILE_KEYS = 64
BENCHMARK_CALLS = 16
BENCHMARK_REPEATS = 7


# Frozen pre-change kernel: query and weights are loaded inside each key tile.
@triton.jit
def _original_paged_index_score(
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
        queries = tl.load(q + row * H * D + heads[:, None] * D + dims[None, :])
        w = tl.load(weights + row * H + heads)
        dots = tl.dot(queries, tl.trans(k), out_dtype=tl.float32)
        # C rounds the BF16 einsum result, then rounds the product of two BF16
        # operands before its head reduction; both conversion points are necessary.
        rounded = dots.to(tl.bfloat16).to(tl.float32)
        relu = tl.where(rounded < 0.0, 0.0, rounded)
        products = (relu * w.to(tl.float32)[:, None]).to(tl.bfloat16).to(tl.float32)
        scores = tl.sum(products, axis=0).to(tl.bfloat16)
        tl.store(output + row * CAPACITY + ids, scores, mask=ids < count)


def inputs(rows, ratio):
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(20261106)
    q = torch.randn(rows, HEADS, DIM, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(rows, HEADS, device="npu", dtype=torch.bfloat16)
    field = torch.randn(PHYSICAL_PAGES * PAGE_SIZE, FIELD_WIDTH, device="npu", dtype=torch.bfloat16)[
        :, FIELD_OFFSET : FIELD_OFFSET + DIM
    ]
    table = torch.arange(MODEL_LIMIT // PAGE_SIZE, device="npu", dtype=torch.int32) % PHYSICAL_PAGES
    source = SimpleNamespace(
        active=torch.ones(rows, device="npu", dtype=torch.bool),
        seq_lens=torch.full((rows,), 8192, device="npu", dtype=torch.int64),
        block_table_tensor=table.expand(rows, -1).contiguous(),
        max_model_len=MODEL_LIMIT,
    )
    query = SimpleNamespace(active=source.active.clone(), seq_lens=source.seq_lens.clone(), max_model_len=MODEL_LIMIT)
    assert supports_paged_index_scores(q, field, weights, source, query, ratio, PAGE_SIZE)
    return q, field, weights, source, query


def original(args, ratio):
    q, field, weights, source, query = args
    rows, heads, dim = q.shape
    capacity = query.max_model_len // ratio
    output = torch.full((rows, capacity), -torch.inf, dtype=q.dtype, device=q.device)
    lanes = max(1, get_aicore_num() // rows)
    _original_paged_index_score[(rows * lanes,)](
        q,
        field,
        weights,
        source.seq_lens,
        source.active,
        source.block_table_tensor,
        query.seq_lens,
        query.active,
        output,
        PHYSICAL_ROWS=field.shape[0],
        CAPACITY=capacity,
        ROW_STRIDE=field.stride(0),
        TABLE_STRIDE=source.block_table_tensor.stride(0),
        RATIO=ratio,
        LANES=lanes,
        D=dim,
        H=heads,
        N=TILE_KEYS,
        PAGE=PAGE_SIZE,
    )
    return output


def candidate(args, ratio):
    return paged_index_scores(*args, ratio, PAGE_SIZE)


@pytest.mark.parametrize("rows,ratio", [(6, 1), (6, 2), (12, 1), (12, 2)])
@torch.inference_mode()
def test_page_tiles_preserve_refreshed_scores(rows, ratio):
    args = inputs(rows, ratio)
    q, _, weights, source, query = args
    candidate(args, ratio)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = candidate(args, ratio)
    address = actual.data_ptr()
    for step, length in enumerate((0, 127, 129, 8193, MODEL_LIMIT, 513, 1, 0)):
        lengths = torch.arange(rows, device="npu") + length - 2
        query.seq_lens.copy_(lengths)
        query.active.copy_((torch.arange(rows, device="npu") + step) % 3 != 0)
        source.seq_lens.copy_(lengths.clamp(0, MODEL_LIMIT))
        source.active.copy_((torch.arange(rows, device="npu") + step) % 4 != 0)
        q.copy_(torch.randn_like(q))
        weights.copy_(torch.randn_like(weights))
        if step == 3:
            q[-1, 0, 0] = torch.nan
        elif step == 4:
            weights[-1, 0] = torch.inf
        expected = original(args, ratio)
        graph.replay()
        torch.npu.synchronize()
        assert actual.data_ptr() == address
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), step


def timed_graph(function, args, ratio):
    function(args, ratio)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(BENCHMARK_CALLS):
            function(args, ratio)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(BENCHMARK_REPEATS):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@pytest.mark.parametrize("rows,ratio", [(6, 1), (6, 2), (12, 1), (12, 2)])
@torch.inference_mode()
def test_complete_score_component_latency(rows, ratio):
    args = inputs(rows, ratio)
    reference_ms = timed_graph(original, args, ratio)
    candidate_ms = timed_graph(candidate, args, ratio)
    print(
        {
            "rows": rows,
            "ratio": ratio,
            "calls": BENCHMARK_CALLS,
            "reference_ms": reference_ms,
            "candidate_ms": candidate_ms,
        }
    )
    assert reference_ms > 0 and candidate_ms > 0
