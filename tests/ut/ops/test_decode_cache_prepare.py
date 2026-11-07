# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact preparation, full native cache writes and dynamic-address graph replay."""

import json
import statistics
import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.common.ops.decode_cache_prepare import try_prepare_decode_raw_write
from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.utils import enable_custom_op

WIDTH = 512
BF16_PATTERNS = 1 << 16
SMALL_PHYSICAL_ROWS = 137
SERVICE_PHYSICAL_ROWS = 184576
TIMED_CALLS = 40
SPECIAL_BITS = (0, -32768, 1, -32767, 32640, -128, 32641, 32704, 32767, -127, -64, -1, 127, -32641, 32703, -65)


def reference_prepare(page, values, slots, active, enabled):
    valid = enabled & active & (slots >= 0) & (slots < page.shape[0])
    slots = torch.where(valid, slots, 0)
    zero_is_live = valid & (slots == 0)
    zero_row = zero_is_live.long().argmax().reshape(1)
    zero_value = torch.where(zero_is_live.any(), values.index_select(0, zero_row), page[:1])
    return slots, torch.where(slots[:, None] == 0, zero_value, values)


def exact(actual, expected):
    if actual.dtype == torch.bfloat16:
        actual, expected = actual.view(torch.int16), expected.view(torch.int16)
    if actual.device.type == "npu" and torch.equal(actual, expected):
        return
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


def require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    assert enable_custom_op(), "The retained native cache scatter must be available"


def set_case(slots, active, enabled, physical_rows, case):
    rows = slots.numel()
    index = torch.arange(rows, device=slots.device)
    slots.copy_(index * 2 + 1)
    active.fill_(True)
    enabled.fill_(True)
    if case == 1:
        active.copy_(index % 3 != 0)
        slots[0] = 0  # Inactive slot 0 must not replace its old value.
        slots[2] = 0  # The live zero must also supply all inactive aliases.
    elif case == 2:
        slots[0] = -1
        slots[1] = physical_rows
        enabled.copy_(index % 2 == 0)
    elif case == 3:
        active.fill_(False)
        slots.zero_()
    elif case == 4:
        slots[1] = 0
        slots[-1] = 0  # Even multiple live-zero rows follow the first witness.
        enabled.copy_(index % 3 != 0)


def make_cache(backing, metadata):
    cache = attention.DeepseekV41Cache.__new__(attention.DeepseekV41Cache)
    nn.Module.__init__(cache)
    cache.prefix = "raw-prepare-native-test"
    cache.source_ratio = 2 if backing.shape[1] > WIDTH else 0
    cache.field = lambda bank, offset, width: backing[:, :width]
    return cache


def test_cpu_returns_to_original_path():
    values = torch.zeros(24, WIDTH, dtype=torch.bfloat16)
    page = torch.zeros(SMALL_PHYSICAL_ROWS, WIDTH, dtype=torch.bfloat16)
    slots = torch.arange(24)
    active = torch.ones(24, dtype=torch.bool)
    assert try_prepare_decode_raw_write(page, values, slots, active, active) is None


@pytest.mark.parametrize("rows", [24, 48])
@pytest.mark.parametrize("stride", [WIDTH, WIDTH * 2])
@torch.inference_mode()
def test_all_bits_aliases_and_entire_native_cache(rows, stride, monkeypatch):
    require_npu()
    backing = torch.empty(SMALL_PHYSICAL_ROWS, stride, device="npu", dtype=torch.bfloat16)
    parent = torch.empty_like(backing)
    values = torch.empty(rows, WIDTH, device="npu", dtype=torch.bfloat16)
    slots = torch.arange(rows, device="npu", dtype=torch.int64)
    active = torch.ones(rows, device="npu", dtype=torch.bool)
    enabled = torch.ones_like(active)
    metadata = SimpleNamespace(query_width=6, grouped_row_start=0, slot_mapping=slots, active=active)
    cache = make_cache(backing, metadata)
    original = make_cache(parent, metadata)
    # Invoke both real stock write variants without constructing a model runner.
    monkeypatch.setattr(attention, "_supports_decode_scatter", lambda *args: stride > WIDTH)
    initial = (torch.arange(backing.numel(), dtype=torch.int32) % BF16_PATTERNS).to(torch.int16).view(torch.bfloat16)
    initial = initial.reshape(backing.shape)
    initial.view(torch.int16)[0, :WIDTH] = torch.tensor(SPECIAL_BITS, dtype=torch.int16).repeat(
        WIDTH // len(SPECIAL_BITS)
    )
    for offset in range(0, BF16_PATTERNS, values.numel()):
        bits = ((torch.arange(values.numel(), dtype=torch.int32) + offset) % BF16_PATTERNS).to(torch.int16)
        values.copy_(bits.view(torch.bfloat16).reshape(values.shape))
        for case in range(5):
            set_case(slots, active, enabled, SMALL_PHYSICAL_ROWS, case)
            backing.copy_(initial)
            parent.copy_(initial)
            expected_slots, expected_updates = reference_prepare(parent[:, :WIDTH], values, slots, active, enabled)
            prepared = try_prepare_decode_raw_write(backing[:, :WIDTH], values, slots, active, enabled)
            assert prepared is not None
            exact(prepared[0], expected_slots)
            exact(prepared[1], expected_updates)
            with monkeypatch.context() as patch:
                patch.setattr(attention, "try_prepare_decode_raw_write", lambda *args: None)
                original.decode_write(metadata, values, 0, 0, enabled)
            cache.decode_write(metadata, values, 0, 0, enabled)
            exact(backing, parent)  # Includes untouched rows and adjacent fields.

    assert try_prepare_decode_raw_write(backing[:, :WIDTH], values.float(), slots, active, enabled) is None
    assert try_prepare_decode_raw_write(backing[:, :WIDTH], values, slots.int(), active, enabled) is None
    assert try_prepare_decode_raw_write(backing[:, :WIDTH], values[:1], slots[:1], active[:1], enabled[:1]) is None


@pytest.mark.parametrize("rows", [24, 48])
@pytest.mark.parametrize("stride", [WIDTH, WIDTH * 2])
@torch.inference_mode()
def test_complete_native_write_graph_and_timing(rows, stride, monkeypatch):
    require_npu()
    backing = [torch.ones(SERVICE_PHYSICAL_ROWS, stride, device="npu", dtype=torch.bfloat16) for _ in range(2)]
    values = torch.randn(rows, WIDTH, device="npu", dtype=torch.bfloat16)
    slots = torch.arange(rows, device="npu", dtype=torch.int64)
    active = torch.ones(rows, device="npu", dtype=torch.bool)
    enabled = torch.ones_like(active)
    metadata = SimpleNamespace(query_width=6, grouped_row_start=0, slot_mapping=slots, active=active)
    caches = [make_cache(page, metadata) for page in backing]
    monkeypatch.setattr(attention, "_supports_decode_scatter", lambda *args: stride > WIDTH)
    graphs = []
    for index, cache in enumerate(caches):
        with monkeypatch.context() as patch:
            if index == 0:
                patch.setattr(attention, "try_prepare_decode_raw_write", lambda *args: None)
            cache.decode_write(metadata, values, 0, 0, enabled)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                cache.decode_write(metadata, values, 0, 0, enabled)
            graphs.append(graph)
    for case in range(5):
        set_case(slots, active, enabled, SERVICE_PHYSICAL_ROWS, case)
        values.normal_()
        values.view(torch.int16)[0, : len(SPECIAL_BITS)] = torch.tensor(SPECIAL_BITS, dtype=torch.int16, device="npu")
        for graph in graphs:
            graph.replay()
            torch.npu.synchronize()
        exact(backing[0], backing[1])
    set_case(slots, active, enabled, SERVICE_PHYSICAL_ROWS, 0)
    values.normal_()
    timed_graphs = []
    for index, cache in enumerate(caches):
        with monkeypatch.context() as patch:
            if index == 0:
                patch.setattr(attention, "try_prepare_decode_raw_write", lambda *args: None)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(TIMED_CALLS):
                    cache.decode_write(metadata, values, 0, 0, enabled)
            timed_graphs.append(graph)
    samples = [[], []]
    for repeat in range(5):
        for index in (repeat % 2, 1 - repeat % 2):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            timed_graphs[index].replay()
            torch.npu.synchronize()
            samples[index].append((time.perf_counter_ns() - start) / 1e6)
    print(
        "TARGET_CACHE_PREPARE_COMPONENT "
        + json.dumps(
            {
                "rows": rows,
                "row_stride": stride,
                "physical_rows": SERVICE_PHYSICAL_ROWS,
                "calls": TIMED_CALLS,
                "parent_ms": statistics.median(samples[0]),
                "candidate_ms": statistics.median(samples[1]),
                "status": "PASS",
                "scope": "full decode_write including unchanged native scatter",
            }
        ),
        flush=True,
    )
