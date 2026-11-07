# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Masked context writes preserve full cache storage and native projection output."""

import json
import statistics
import time

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import dspark
from vllm_ascend.ops.triton.deepseek_v41_draft_context import try_write_draft_context
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

pytest_plugins = ("tests.e2e.nightly.one_card.test_draft_attention_graph",)

WIDTH = 512
PHYSICAL_ROWS = 256
CALLS = 12
REPEATS = 5


@pytest.fixture(autouse=True)
def device():
    torch.npu.set_device(0)
    init_device_properties_triton()


@pytest.mark.parametrize("rows", [1, 6, 24, 48])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_full_storage_bits_and_invalid_slots(rows, slot_dtype):
    storage = torch.randint(-32768, 32768, (PHYSICAL_ROWS + 2, WIDTH + 32), dtype=torch.int16, device="npu")
    field = storage.view(torch.bfloat16)[1:-1, 16 : 16 + WIDTH]
    source = torch.randint(-32768, 32768, (rows + 1, WIDTH + 16), dtype=torch.int16, device="npu")
    values = source.view(torch.bfloat16)[1:, 8 : 8 + WIDTH]
    mapped = [row * 3 if row % 3 else -2 for row in range(rows)]
    if rows > 1:
        mapped[1], mapped[-1] = 0, PHYSICAL_ROWS - 1
    slots = torch.tensor(mapped, dtype=slot_dtype, device="npu")
    reference = storage.clone()
    active = (slots >= 0).nonzero().flatten()
    reference.view(torch.bfloat16)[1:-1, 16 : 16 + WIDTH].index_copy_(
        0, slots.index_select(0, active).long(), values.index_select(0, active)
    )
    assert try_write_draft_context(field, slots, values)
    torch.testing.assert_close(storage, reference, rtol=0, atol=0)
    # A later all-padding invocation must leave row zero and every other lane untouched.
    slots.fill_(-1)
    values.fill_(2)
    assert try_write_draft_context(field, slots, values)
    torch.testing.assert_close(storage, reference, rtol=0, atol=0)
    # Reusing the same storage with new valid slots must consume the current mapping.
    slots.copy_(torch.arange(rows, device="npu", dtype=slot_dtype))
    reference.view(torch.bfloat16)[1:-1, 16 : 16 + WIDTH].index_copy_(0, slots.long(), values)
    assert try_write_draft_context(field, slots, values)
    torch.testing.assert_close(storage, reference, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [6, 24, 48])
@torch.inference_mode()
def test_complete_project_and_store_matches_parent(model, monkeypatch, rows):
    states = torch.randn(rows, 5120, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(rows, dtype=torch.int32, device="npu") + 8192
    slots = torch.arange(rows, dtype=torch.int64, device="npu") * 3
    slots[::3] = -1
    field = model.cache.field(0, 0, WIDTH)
    initial = field.clone()
    with monkeypatch.context() as patch:
        patch.setattr(dspark, "try_write_draft_context", None)
        model.store_context(states, positions, slots)
    expected = field.clone()
    field.copy_(initial)
    executed = []

    def write(*args):
        used = try_write_draft_context(*args)
        executed.append(used)
        return used

    monkeypatch.setattr(dspark, "try_write_draft_context", write)
    model.store_context(states, positions, slots)
    assert executed == [True]
    torch.testing.assert_close(field.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@torch.inference_mode()
def test_outer_capture_keeps_parent():
    field = torch.zeros(PHYSICAL_ROWS, WIDTH, dtype=torch.bfloat16, device="npu")
    slots = torch.zeros(6, dtype=torch.int64, device="npu")
    values = torch.ones(6, WIDTH, dtype=torch.bfloat16, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        assert not try_write_draft_context(field, slots, values)
        output = values + 1
    graph.replay()
    torch.testing.assert_close(field, torch.zeros_like(field), rtol=0, atol=0)
    torch.testing.assert_close(output, torch.full_like(values, 2), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [6, 24])
@torch.inference_mode()
def test_complete_context_latency(model, monkeypatch, rows):
    states = torch.randn(rows, 5120, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(rows, dtype=torch.int32, device="npu") + 8192
    slots = torch.arange(rows, dtype=torch.int64, device="npu") * 3
    slots[::3] = -1
    result = {"rows": rows, "complete_context_calls": CALLS}
    for name, write in (("parent", None), ("candidate", try_write_draft_context)):
        monkeypatch.setattr(dspark, "try_write_draft_context", write)
        for _ in range(3):
            model.store_context(states, positions, slots)
        torch.npu.synchronize()
        samples = []
        for _ in range(REPEATS):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter()
            for _ in range(CALLS):
                model.store_context(states, positions, slots)
            submitted = time.perf_counter()
            end.record()
            end.synchronize()
            samples.append(((submitted - begin) * 1000, (time.perf_counter() - begin) * 1000, start.elapsed_time(end)))
        result[name] = dict(zip(("host_ms", "ready_ms", "device_ms"), map(statistics.median, zip(*samples))))
    print("DSPARK_CONTEXT_WRITE_TIMING " + json.dumps(result), flush=True)
