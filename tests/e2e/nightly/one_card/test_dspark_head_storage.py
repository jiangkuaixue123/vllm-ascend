# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual draft checkpoint logits and removal of repeated static weight conversion."""

import importlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod

from vllm_ascend.models.deepseek_v41.model import read_parameter_slice

WIDTH = 5120
SHARD = 32320
CALLS = 8
REPEATS = 5


class Reader:
    def __init__(self, value):
        self.value = value

    def get_slice(self, key):
        return self

    def get_shape(self):
        return self.value.shape

    def __getitem__(self, index):
        return self.value[index]


def load_checkpoint(head, checkpoint):
    read_parameter_slice(Reader(checkpoint), "head.weight", head.weight, head, 0, 1)


def loaded_head(checkpoint):
    head = object.__new__(ParallelLMHead)
    nn.Module.__init__(head)
    head.weight = nn.Parameter(torch.empty(checkpoint.shape, device="npu", dtype=torch.bfloat16), requires_grad=False)
    head.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=checkpoint.shape[0])
    head.quant_method = UnquantizedEmbeddingMethod()
    load_checkpoint(head, checkpoint)
    return head


@pytest.fixture(scope="module")
def heads():
    importlib.import_module("torch_npu")
    torch.npu.set_device(0)
    path = Path("/home/admin/model")
    if not (path / "quant_model_weights.safetensors.index.json").exists():
        pytest.skip("actual DSpark checkpoint unavailable")
    index = json.loads((path / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    key = "mtp.2.head.weight"
    with safe_open(path / index[key], framework="pt", device="cpu") as reader:
        cpu_weight = reader.get_slice(key)[:SHARD, :]
    assert cpu_weight.shape == (SHARD, WIDTH) and cpu_weight.dtype == torch.float32
    checkpoint = cpu_weight.npu()
    parent = SimpleNamespace(weight=checkpoint, quant_method=UnquantizedEmbeddingMethod())
    # The indexed loader copies checkpoint slices from CPU into the parameter.
    candidate = loaded_head(cpu_weight)
    same_words(candidate.weight, parent.weight.to(torch.bfloat16))
    processor = object.__new__(LogitsProcessor)
    nn.Module.__init__(processor)
    processor.head_dtype = torch.bfloat16
    return processor, parent, candidate


def hidden(rows):
    # Preserve the model's BF16 -> FP32 -> configured head dtype sequence.
    return torch.randn(rows, WIDTH, device="npu", dtype=torch.bfloat16).float()


def same_words(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@torch.inference_mode()
def test_cpu_checkpoint_copy_matches_device_conversion_for_special_values(heads):
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = words.view(torch.bfloat16).float()
    boundaries = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, 2**-134, 2**-133, -(2**-134)], dtype=torch.float32)
    checkpoint = torch.cat((values, boundaries))
    head = loaded_head(checkpoint.unsqueeze(-1))
    same_words(head.weight, checkpoint.to("npu").to(torch.bfloat16).unsqueeze(-1))


@pytest.mark.parametrize("rows", [1, 5, 20, 40, 80, 160])
@torch.inference_mode()
def test_actual_processor_outputs_match_for_fresh_hidden_states(heads, rows):
    processor, parent, candidate = heads
    for _ in range(3):
        states = hidden(rows)
        same_words(processor._apply_head(candidate, states, None), processor._apply_head(parent, states, None))


@torch.inference_mode()
def test_nonfinite_checkpoint_values_preserve_output_bits(heads):
    processor, parent, candidate = heads
    saved = parent.weight[0].clone()
    try:
        parent.weight[0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0], device="npu")
        load_checkpoint(candidate, parent.weight.cpu())
        states = hidden(6)
        same_words(processor._apply_head(candidate, states, None), processor._apply_head(parent, states, None))
    finally:
        parent.weight[0].copy_(saved)
        load_checkpoint(candidate, parent.weight.cpu())


@torch.inference_mode()
def test_graph_replay_uses_refreshed_inputs_and_reloaded_weight(heads):
    processor, parent, candidate = heads
    states = hidden(20)
    processor._apply_head(candidate, states, None)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = processor._apply_head(candidate, states, None)
    saved = parent.weight[0].clone()
    try:
        for step in range(3):
            states.copy_(hidden(20))
            if step:
                parent.weight[0].add_(0.003)
                load_checkpoint(candidate, parent.weight.cpu())
            graph.replay()
            same_words(actual, processor._apply_head(parent, states, None))
    finally:
        parent.weight[0].copy_(saved)
        load_checkpoint(candidate, parent.weight.cpu())


def elapsed(processor, head, states):
    torch.npu.synchronize()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    before = time.perf_counter()
    begin.record()
    for _ in range(CALLS):
        processor._apply_head(head, states, None)
    end.record()
    submitted = time.perf_counter()
    end.synchronize()
    return 1000 * (submitted - before), 1000 * (time.perf_counter() - before), begin.elapsed_time(end)


@pytest.mark.parametrize("rows", [5, 20, 40, 160])
@torch.inference_mode()
def test_full_projection_latency_includes_original_cast(heads, rows):
    processor, parent, candidate = heads
    states = hidden(rows)
    same_words(processor._apply_head(candidate, states, None), processor._apply_head(parent, states, None))
    measurements = {"parent": [], "candidate": []}
    for repeat in range(REPEATS):
        order = [("parent", parent), ("candidate", candidate)]
        if repeat % 2:
            order.reverse()
        for name, head in order:
            measurements[name].append(elapsed(processor, head, states))
    medians = {
        name: tuple(statistics.median(values) for values in zip(*samples)) for name, samples in measurements.items()
    }
    print({"rows": rows, "calls": CALLS, "metrics": ["host_submit_ms", "wall_ready_ms", "device_ms"], **medians})
    assert all(value > 0 for values in medians.values() for value in values)
