# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the native score producer's query-mask guarantee across graph reuse."""

import statistics
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.deepseek_v41_paged_index_score import paged_index_scores, supports_paged_index_scores
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

MODEL_LIMIT = 153600
PAGE_SIZE = 128
PHYSICAL_PAGES = 16
HEADS = 32
DIM = 128
BENCHMARK_CALLS = 16
BENCHMARK_REPEATS = 7


def inputs(rows, ratio):
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(20261106)
    q = torch.randn(rows, HEADS, DIM, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(rows, HEADS, device="npu", dtype=torch.bfloat16)
    field = torch.randn(PHYSICAL_PAGES * PAGE_SIZE, DIM * 2, device="npu", dtype=torch.bfloat16)[:, DIM:]
    table = torch.arange(MODEL_LIMIT // PAGE_SIZE, device="npu", dtype=torch.int32) % PHYSICAL_PAGES
    source = SimpleNamespace(
        active=torch.ones(rows, device="npu", dtype=torch.bool),
        seq_lens=torch.full((rows,), MODEL_LIMIT, device="npu", dtype=torch.int64),
        block_table_tensor=table.expand(rows, -1).contiguous(),
        max_model_len=MODEL_LIMIT,
    )
    query = SimpleNamespace(active=source.active.clone(), seq_lens=source.seq_lens.clone(), max_model_len=MODEL_LIMIT)
    assert supports_paged_index_scores(q, field, weights, source, query, ratio, PAGE_SIZE)
    return q, field, weights, source, query


def produce(args, ratio, repeat_mask):
    q, field, weights, source, query = args
    scores = paged_index_scores(q, field, weights, source, query, ratio, PAGE_SIZE)
    lengths = torch.where(query.active, query.seq_lens // ratio, 0)[:, None]
    if repeat_mask:
        columns = torch.arange(scores.shape[-1], device=scores.device).unsqueeze(0)
        scores.masked_fill_(columns >= lengths, -torch.inf)
    return scores, lengths


@pytest.mark.parametrize("rows,ratio", [(6, 1), (6, 2), (12, 1), (12, 2)])
@torch.inference_mode()
def test_mask_contract_survives_graph_context_changes(rows, ratio):
    args = inputs(rows, ratio)
    q, _, weights, source, query = args
    produce(args, ratio, False)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scores, lengths = produce(args, ratio, False)
    address = scores.data_ptr()
    # Include growth, shrinkage, incomplete groups, capacity clipping and idle rows.
    for step, length in enumerate((0, 127, 129, 8193, MODEL_LIMIT, 513, 1, 0)):
        current = torch.arange(rows, device="npu") + length - 2
        query.seq_lens.copy_(current)
        query.active.copy_((torch.arange(rows, device="npu") + step) % 3 != 0)
        source.active.copy_((torch.arange(rows, device="npu") + step) % 4 != 0)
        source.seq_lens.copy_(current.clamp(0, MODEL_LIMIT))
        q.copy_(torch.randn_like(q))
        weights.copy_(torch.randn_like(weights))
        expected_scores, expected_lengths = produce(args, ratio, True)
        graph.replay()
        torch.npu.synchronize()
        assert scores.data_ptr() == address
        assert torch.equal(scores.view(torch.int16), expected_scores.view(torch.int16)), step
        assert torch.equal(lengths, expected_lengths), step


def timed_graph(args, ratio, repeat_mask):
    produce(args, ratio, repeat_mask)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(BENCHMARK_CALLS):
            produce(args, ratio, repeat_mask)
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
def test_native_score_and_mask_component_latency(rows, ratio):
    args = inputs(rows, ratio)
    args[-1].seq_lens.fill_(8192)
    reference_ms = timed_graph(args, ratio, True)
    candidate_ms = timed_graph(args, ratio, False)
    print(
        {
            "rows": rows,
            "ratio": ratio,
            "capacity": MODEL_LIMIT // ratio,
            "calls": BENCHMARK_CALLS,
            "reference_ms": reference_ms,
            "candidate_ms": candidate_ms,
        }
    )
    assert reference_ms > 0 and candidate_ms > 0
