# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete context graph replay with native W8A8 projections and strided cache."""

import json
import statistics
import time

import pytest
import torch
from torch import nn

from vllm_ascend.models.common.ops.draft_context_graph import DraftContextGraph

pytest_plugins = ("tests.e2e.nightly.one_card.test_draft_attention_graph",)

CALLS = 12
REPEATS = 5


@pytest.fixture(autouse=True)
def reset_context_graph(model):
    model._context_graph = None
    yield
    model._context_graph = None


def inputs(rows, position_dtype=torch.int32, slot_dtype=torch.int64, step=0):
    states = torch.randn(rows + 1, 5120, dtype=torch.bfloat16, device="npu")[1:]
    positions = torch.arange(rows, dtype=position_dtype, device="npu") + (0, 128, 8192, 153550)[step % 4]
    slots = (torch.arange(rows, dtype=slot_dtype, device="npu") * 7 + step) % 1024
    slots[::3] = -1
    if step == 2:
        slots.fill_(-1)
    return states, positions, slots


def parent_store(model, args):
    saved = model._context_graph
    model._context_graph = None
    try:
        model.store_context(*args)
    finally:
        model._context_graph = saved


@pytest.mark.parametrize("rows", [6, 24])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_replay_refreshes_inputs_and_preserves_full_cache_bits(model, rows, position_dtype, slot_dtype):
    graph = model._context_graph = DraftContextGraph()
    original = model.cache.kv_cache
    # Inference tensors need not expose a view's base. Own the backing explicitly.
    storage = torch.randn(1024, 528, dtype=torch.bfloat16, device="npu")
    model.cache.kv_cache = (storage[:, 8:520], original[1])
    try:
        entry = None
        for step in range(4):
            args = inputs(rows, position_dtype, slot_dtype, step)
            initial = storage.clone()
            parent_store(model, args)
            expected = storage.clone()
            storage.copy_(initial)
            model.store_context(*args)
            torch.testing.assert_close(storage.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
            assert rows in graph.entries
            if entry is None:
                entry = graph.entries[rows]
            else:
                assert graph.entries[rows] is entry
    finally:
        model.cache.kv_cache = original


@torch.inference_mode()
def test_storage_stream_and_integer_dtype_changes_fall_back(model):
    graph = model._context_graph = DraftContextGraph()
    args = inputs(6)
    assert graph(model, *args)
    cache = model.cache.kv_cache
    model.cache.kv_cache = (cache[0].clone(), cache[1])
    assert not graph(model, *args)
    model.cache.kv_cache = cache
    weight = model.kv_norm.weight
    model.kv_norm.weight = nn.Parameter(weight.clone(), requires_grad=False)
    assert not graph(model, *args)
    model.kv_norm.weight = weight
    assert not graph(model, args[0], args[1].long(), args[2])
    assert not graph(model, args[0], args[1], args[2].int())
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        assert not graph(model, *args)
    torch.npu.current_stream().wait_stream(stream)


@pytest.mark.parametrize("rows", [6, 24])
@torch.inference_mode()
def test_actual_1024_column_cache_stride_preserves_neighbor_fields(model, rows):
    original = model.cache.kv_cache
    storage = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
    model.cache.kv_cache = (storage[:, :512], original[1])
    model._context_graph = DraftContextGraph()
    try:
        args = inputs(rows)
        initial = storage.clone()
        parent_store(model, args)
        expected = storage.clone()
        storage.copy_(initial)
        model.store_context(*args)
        assert rows in model._context_graph.entries
        torch.testing.assert_close(storage.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
    finally:
        model.cache.kv_cache = original


@torch.inference_mode()
def test_unsupported_rows_and_missing_slots_keep_parent(model):
    model._context_graph = DraftContextGraph()
    field = model.cache.field(0, 0, 512)
    args = inputs(3)
    initial = field.clone()
    parent_store(model, args)
    expected = field.clone()
    field.copy_(initial)
    model.store_context(*args)
    assert not model._context_graph.entries
    torch.testing.assert_close(field.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
    model.store_context(args[0], args[1], None)
    torch.testing.assert_close(field.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@torch.inference_mode()
def test_outer_capture_does_not_nest_context_graph(model):
    candidate = DraftContextGraph()
    args = inputs(6)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        assert not candidate(model, *args)
        output = args[0] + 1
    graph.replay()
    torch.testing.assert_close(output, args[0] + 1, rtol=0, atol=0)
    assert not candidate.entries


@pytest.mark.parametrize("rows", [6, 24])
@torch.inference_mode()
def test_complete_context_graph_latency(model, rows):
    args = inputs(rows)
    result = {"rows": rows, "complete_context_calls": CALLS}
    for name in ("parent", "candidate"):
        model._context_graph = None if name == "parent" else DraftContextGraph()
        for _ in range(3):
            model.store_context(*args)
        torch.npu.synchronize()
        samples = []
        for _ in range(REPEATS):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter()
            for _ in range(CALLS):
                model.store_context(*args)
            submitted = time.perf_counter()
            end.record()
            end.synchronize()
            samples.append(((submitted - begin) * 1000, (time.perf_counter() - begin) * 1000, start.elapsed_time(end)))
        result[name] = dict(zip(("host_ms", "ready_ms", "device_ms"), map(statistics.median, zip(*samples))))
    print("DSPARK_CONTEXT_GRAPH_TIMING " + json.dumps(result), flush=True)
