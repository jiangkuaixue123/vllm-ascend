# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact rotary consumers, changing graph inputs and a 38-layer decode bench."""

import statistics
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.rotary_reuse import RotaryFrequencyReference
from vllm_ascend.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41Runtime,
    rotary_frequencies,
)
from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig

REPLAY_REPEATS = 128
LAYERS = 38
BENCHMARK_REPEATS = 7


def attention(compressed):
    with torch.inference_mode(False):
        cpu = rotary_frequencies(DeepseekV41TextConfig(), compressed)
        frequencies = cpu.to("npu")
    return SimpleNamespace(
        frequencies=frequencies,
        rotary_reference=RotaryFrequencyReference(frequencies, tuple(cpu.tolist()), frequencies._version),
        rope_dim=64,
    )


def rotate(module, x, positions, runtime=None, inverse=False):
    return DeepseekV41Attention.rotate(module, x, positions, inverse=inverse, runtime=runtime)


@pytest.mark.parametrize("rows", [6, 12, 24, 48])
@pytest.mark.parametrize("compressed", [False, True])
@torch.inference_mode()
def test_consumers_are_bitwise_equal(rows, compressed):
    torch.npu.set_device(0)
    module = attention(compressed)
    runtime = DeepseekV41Runtime([])
    lengths = torch.arange(rows, device="npu") * (153600 // rows)
    positions = runtime.rotary_cache.positions(lengths)
    for heads, dim in ((1, 512), (32, 128), (64, 512)):
        for inverse in (False, True):
            value = torch.randn(1, rows, heads, dim, device="npu", dtype=torch.bfloat16)
            expected = rotate(module, value.clone(), positions, inverse=inverse)
            actual = rotate(module, value.clone(), positions, runtime, inverse)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("change", ["mutation", "replacement", "missing_reference", "noncontiguous", "float32"])
@torch.inference_mode()
def test_ordinary_rotation_fallback(change):
    torch.npu.set_device(0)
    module = attention(True)
    runtime = DeepseekV41Runtime([])
    positions = torch.arange(6, device="npu")
    value = torch.randn(1, 6, 2, 512, device="npu", dtype=torch.bfloat16)
    if change == "mutation":
        module.frequencies.mul_(0.75)
    elif change == "replacement":
        module.frequencies = module.frequencies.clone()
    elif change == "missing_reference":
        del module.rotary_reference
    elif change == "noncontiguous":
        value = value.transpose(1, 2).contiguous().transpose(1, 2)
    else:
        value = value.float()
    expected = rotate(module, value.clone(), positions)
    actual = rotate(module, value.clone(), positions, runtime)
    assert torch.equal(actual, expected)


def repeated_layers(modules, lengths, source, reuse):
    runtime = DeepseekV41Runtime([]) if reuse else None
    outputs = []
    for index, module in enumerate(modules):
        if reuse:
            positions = runtime.rotary_cache.positions(lengths)
            positions = runtime.rotary_cache.shard(positions, 0, lengths.shape[0] // 4)
        else:
            positions = (lengths - 1).clamp_min(0)[: lengths.shape[0] // 4]
        value = source[index].clone()
        rotate(module, value, positions, runtime)
        rotate(module, value, positions, runtime, inverse=True)
        outputs.append(value)
    return outputs


@pytest.mark.parametrize("rows", [24, 48])
@torch.inference_mode()
def test_graph_refresh_and_latency(rows):
    torch.npu.set_device(0)
    modules = [attention(index % 2 == 1) for index in range(LAYERS)]
    lengths = torch.arange(rows, device="npu")
    source = torch.randn(LAYERS, 1, rows // 4, 64, 512, device="npu", dtype=torch.bfloat16)
    graphs, outputs = [], []
    for reuse in (False, True):
        repeated_layers(modules, lengths, source, reuse)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = repeated_layers(modules, lengths, source, reuse)
        graphs.append(graph)
        outputs.append(result)
    for step in range(REPLAY_REPEATS):
        lengths.copy_((torch.arange(rows, device="npu") * 7919 + step * 1237) % 153601)
        source.copy_(torch.randn_like(source))
        for graph in graphs:
            graph.replay()
        torch.npu.synchronize()
        for expected, actual in zip(*outputs):
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), step
    samples = [[], []]
    for _ in range(BENCHMARK_REPEATS):
        for index, graph in enumerate(graphs):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
    reference, candidate = (statistics.median(sample) for sample in samples)
    print({"rows": rows, "layers": LAYERS, "reference_ms": reference, "candidate_ms": candidate})
    assert candidate < reference
