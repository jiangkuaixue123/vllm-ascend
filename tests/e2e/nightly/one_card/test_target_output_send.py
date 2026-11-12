# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare packed output bits to native inverse RoPE and retained FP8 QDQ."""

import statistics
from types import SimpleNamespace

import pytest
import torch
import torch_npu
import vllm.forward_context as forward_context

from vllm_ascend.models.common.ops.target_output_send import try_target_output_send
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Attention, _quantize_projection_input

CALLS = 20
SAMPLES = 7


def layer():
    return SimpleNamespace(
        dsa_cp=True,
        tp=4,
        sequence_group=SimpleNamespace(world_size=4),
        heads=64,
        local_heads=16,
        dim=512,
        groups=2,
        rope_dim=64,
        frequencies=torch.exp(torch.linspace(0, -11, 32, dtype=torch.float32, device="npu")),
    )


def reference(model, values, positions, *, copy_input=True):
    if copy_input:
        values = values.clone()
    DeepseekV41Attention.rotate(model, values.unsqueeze(0), positions, inverse=True)
    send = values.reshape(-1, 4, 16, 512).transpose(0, 1).contiguous()
    # The collective only exchanges whole 32-element quantization groups.
    return _quantize_projection_input(send.view(-1, 2, 4096)).view(-1, 16, 512)


def exact(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.fixture(autouse=True)
def target_context(monkeypatch):
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: SimpleNamespace(is_draft_model=False))


@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("npu_format", [0, 2])
@torch.inference_mode()
def test_exact_native_rotary_quantization_and_packing(rows, npu_format):
    torch.npu.set_device(0)
    model = layer()
    values = torch_npu.npu_format_cast(torch.randn(rows, 64, 512, device="npu", dtype=torch.bfloat16), npu_format)
    positions = torch.arange(rows, device="npu")
    assert torch_npu.get_npu_format(values) == npu_format
    for start in (0, 127, 8192, 153580):
        positions.copy_(torch.arange(rows, device="npu") + start)
        actual = try_target_output_send(model, values, positions)
        assert actual is not None and torch_npu.get_npu_format(actual) == 2
        exact(actual, reference(model, values, positions))
    # Cover all BF16 words, including special groups and halfway values.
    words = torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16)
    values.copy_(words.repeat(values.numel() // words.numel()).reshape_as(values).npu())
    exact(try_target_output_send(model, values, positions), reference(model, values, positions))
    for zero in (0.0, -0.0):
        values.fill_(zero)
        exact(try_target_output_send(model, values, positions), reference(model, values, positions))


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_graph_refresh_and_latency(rows):
    torch.npu.set_device(0)
    model = layer()
    values = torch.randn(rows, 64, 512, device="npu", dtype=torch.bfloat16)
    positions = torch.arange(rows, device="npu")
    assert try_target_output_send(model, values, positions) is not None
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = try_target_output_send(model, values, positions)
    address = actual.data_ptr()
    for step in range(16):
        values.normal_().mul_(2.0 ** (step - 8))
        if step == 8:
            values.zero_()
        elif step == 9:
            values.fill_(-0.0)
        positions.add_(127)
        expected = reference(model, values, positions)
        original = values.clone()
        graph.replay()
        torch.npu.synchronize()
        exact(actual, expected)
        exact(values, original)
        assert actual.data_ptr() == address
    graphs = []

    def consumed_reference(model, values, positions):
        # The production parent consumes its attention output in place. Avoid
        # charging the comparison-only input clone to its timed operator chain.
        return reference(model, values, positions, copy_input=False)

    for function in (consumed_reference, try_target_output_send):
        function(model, values, positions)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(CALLS):
                function(model, values, positions)
        graphs.append(graph)
    times = [[], []]
    for repeat in range(SAMPLES):
        for index in (repeat % 2, 1 - repeat % 2):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            times[index].append(start.elapsed_time(end))
    print(
        {
            "rows": rows,
            "calls": CALLS,
            "parent_ms": statistics.median(times[0]),
            "candidate_ms": statistics.median(times[1]),
        }
    )
