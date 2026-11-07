# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native BF16 checkpoint sinks, FP32 fallback and full gathered attention timing."""

import json
import statistics
import time

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.selected_fia import try_selected_fia
from vllm_ascend.models.deepseek_v41.attention_ops import sparse_attn
from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention
from vllm_ascend.utils import enable_custom_op

HEADS = 64
DIM = 512
WIDTH = 640
CALLS = 40
REPEATS = 5
RTOL = 0.025
ATOL = 0.02


def fixture(rows):
    torch.npu.set_device(0)
    assert enable_custom_op(), "native attention custom operators must be available"
    generator = torch.Generator(device="cpu").manual_seed(2207)
    q = torch.randn(rows, 1, HEADS, DIM, generator=generator).bfloat16().npu()
    kv = torch.randn(rows, WIDTH, DIM, generator=generator).bfloat16().npu()
    sink = torch.linspace(-16, 16, HEADS, dtype=torch.float32).bfloat16().float().npu()
    slots = torch.randint(rows * WIDTH, (rows, 1, WIDTH), generator=generator, dtype=torch.int32)
    for row, count in enumerate((0, 1, 127, 512, WIDTH - 1, WIDTH)):
        slots[row, 0, count:] = -1
    slots[-1, 0, :3] = slots[-1, 0, 3]
    slots = slots.npu()
    runner = DeepseekV41NativeAttention()
    runner.pin_decode_plan(q.device, HEADS, rows, WIDTH)
    runner.sink_source_dtype = torch.bfloat16
    return runner, q, kv, sink, slots


def reference(q, kv, sink, slots):
    result = sparse_attn(
        q.cpu().reshape(1, -1, HEADS, DIM),
        kv.cpu().reshape(1, -1, DIM),
        sink.cpu(),
        slots.cpu().reshape(1, -1, WIDTH),
        DIM**-0.5,
    )
    return result.reshape_as(q)


def parent(runner, q, kv, sink, slots):
    return try_selected_fia(q, kv, sink, slots, DIM**-0.5)


def verify(candidate, expected):
    assert candidate is not None
    torch.testing.assert_close(candidate.float().cpu(), expected.float().cpu(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_native_selected_attention_and_graph_updates(rows):
    runner, q, kv, sink, slots = fixture(rows)
    original = parent(runner, q, kv, sink, slots)
    output = try_selected_fia(q, kv, sink, slots, DIM**-0.5, sink_source_dtype=torch.bfloat16)
    verify(original, reference(q, kv, sink, slots))
    verify(output, reference(q, kv, sink, slots))
    verify(output, original)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = runner.decode_with_physical_slots(q, kv, sink, slots, DIM**-0.5)
    for scale in (0.0, 0.625, 2.0):
        q.copy_(torch.randn_like(q) * scale)
        kv.copy_(torch.randn_like(kv))
        sink.add_(0.125)
        slots[-1, 0, :3] = WIDTH + 7
        graph.replay()
        expected = reference(q, kv, sink, slots)
        verify(output, expected)
        verify(output, parent(runner, q, kv, sink, slots))
    assert torch.count_nonzero(output[0]).item() == 0


def elapsed(call):
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    before = time.perf_counter_ns()
    call()
    host = (time.perf_counter_ns() - before) / 1e6
    end.record()
    end.synchronize()
    return host, start.elapsed_time(end)


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_complete_selected_attention_timing(rows):
    runner, q, kv, sink, slots = fixture(rows)
    slots.copy_(torch.arange(rows * WIDTH, device=q.device, dtype=torch.int32).view(rows, 1, WIDTH))
    functions = []
    for candidate in (False, True):

        def run(candidate=candidate):
            for _ in range(CALLS):
                if candidate:
                    runner.decode_with_physical_slots(q, kv, sink, slots, DIM**-0.5)
                else:
                    parent(runner, q, kv, sink, slots)

        run()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        functions.append(graph.replay)
    samples = [[], []]
    for repeat in range(REPEATS):
        for i in (repeat % 2, 1 - repeat % 2):
            samples[i].append(elapsed(functions[i]))
    medians = [list(map(statistics.median, zip(*sample))) for sample in samples]
    print(
        "TARGET_NATIVE_SINK_COMPONENT "
        + json.dumps(
            {
                "rows": rows,
                "calls": CALLS,
                "parent_host_ready_ms": medians[0],
                "candidate_host_ready_ms": medians[1],
                "scope": "complete gather, masks and native BF16 sink attention; repeated same KV",
            }
        )
    )


@torch.inference_mode()
def test_unsupported_geometry_uses_existing_native_path():
    _, q, kv, sink, slots = fixture(6)
    for other in (q.float(), q[:1], q[:, :, :16], q.cpu()):
        assert try_selected_fia(other, kv, sink, slots, DIM**-0.5) is None
    assert try_selected_fia(q, kv, sink, slots[..., :128], DIM**-0.5) is None


@pytest.mark.parametrize("source_dtype", [None, torch.float32, torch.float16])
@torch.inference_mode()
def test_other_source_dtypes_preserve_full_precision_sink(source_dtype):
    runner, q, kv, sink, slots = fixture(6)
    runner.sink_source_dtype = source_dtype
    sink.add_(0.0031)
    expected = try_selected_fia(q, kv, sink, slots, DIM**-0.5)
    actual = runner.decode_with_physical_slots(q, kv, sink, slots, DIM**-0.5)
    assert torch.equal(actual, expected)
