# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact NPU Markov graph outputs and full copy-in/replay/copy-out timings."""

import importlib
import statistics
import time

import pytest
import torch

from vllm_ascend.models.common.ops.dspark_markov_graph import DSparkMarkovGraph, greedy_markov_chain

VOCAB = 129280
RANK = 256
BLOCK = 5
CALLS = 16
REPEATS = 5


@pytest.fixture(scope="module")
def weights():
    importlib.import_module("torch_npu")
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    embedding = torch.nn.Embedding(VOCAB, RANK, device="npu", dtype=torch.bfloat16)
    projection = torch.nn.Linear(RANK, VOCAB, bias=False, device="npu", dtype=torch.float32)
    return embedding, projection


def inputs(batch, dtype=torch.bfloat16):
    logits = torch.randn(batch, BLOCK, VOCAB, device="npu", dtype=dtype)
    ids = torch.zeros(batch, BLOCK + 1, device="npu", dtype=torch.int64)
    ids[:, 0] = torch.randint(VOCAB, (batch,), device="npu")
    return logits, ids


def same_words(actual, expected):
    assert torch.equal(actual.cpu().view(torch.int32), expected.cpu().view(torch.int32))


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_refreshed_logits_seeds_and_weights_match_eager_bitwise(weights, batch, dtype):
    embedding, projection = weights
    sampler = DSparkMarkovGraph()
    for step in range(3):
        logits, ids = inputs(batch, dtype)
        if step == 1:
            logits[..., :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0], device="npu")
        if step == 2:
            projection.weight.add_(0.000001)
        expected_logits, expected_ids = logits.clone(), ids.clone()
        greedy_markov_chain(expected_logits, expected_ids, embedding, projection)
        assert sampler(logits, ids, embedding, projection)
        same_words(logits, expected_logits)
        assert torch.equal(ids.cpu(), expected_ids.cpu())
    assert len(sampler.entries) == 1


@torch.inference_mode()
def test_tied_argmax_retains_first_token_and_graph_entries_are_bounded(weights):
    embedding, projection = weights
    saved = projection.weight.clone()
    projection.weight.zero_()
    sampler = DSparkMarkovGraph()
    try:
        for batch in (1, 2, 4, 8):
            logits, ids = inputs(batch)
            logits.zero_()
            assert sampler(logits, ids, embedding, projection)
            assert torch.count_nonzero(ids[:, 1:]).item() == 0
        assert len(sampler.entries) == 4
        for batch in (0, 3, 5, 6, 7, 9):
            logits, ids = inputs(batch)
            before = logits.clone()
            assert not sampler(logits, ids, embedding, projection)
            same_words(logits, before)
        assert len(sampler.entries) == 4
    finally:
        projection.weight.copy_(saved)


@torch.inference_mode()
def test_changed_parameter_storage_and_other_stream_fall_back(weights):
    embedding, projection = weights
    sampler = DSparkMarkovGraph()
    logits, ids = inputs(1)
    assert sampler(logits, ids, embedding, projection)
    before = logits.clone()
    other = torch.nn.Linear(RANK, VOCAB, bias=False, device="npu", dtype=torch.float32)
    other.weight.copy_(projection.weight)
    assert not sampler(logits, ids, embedding, other)
    same_words(logits, before)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        assert not sampler(logits, ids, embedding, projection)
    same_words(logits, before)


@torch.inference_mode()
def test_changed_logits_dtype_keeps_original_rounding_and_falls_back(weights):
    embedding, projection = weights
    sampler = DSparkMarkovGraph()
    logits, ids = inputs(1)
    assert sampler(logits, ids, embedding, projection)
    wider_logits = logits.float()
    before, before_ids = wider_logits.clone(), ids.clone()
    assert not sampler(wider_logits, ids, embedding, projection)
    same_words(wider_logits, before)
    assert torch.equal(ids.cpu(), before_ids.cpu())


@torch.inference_mode()
def test_noncontiguous_and_training_inputs_use_original_path(weights):
    embedding, projection = weights
    logits, ids = inputs(1)
    sampler = DSparkMarkovGraph()
    backing = torch.zeros(1, BLOCK, VOCAB * 2, device="npu")
    backing[..., ::2].copy_(logits)
    assert not sampler(backing[..., ::2], ids, embedding, projection)
    with torch.inference_mode(False):
        assert not sampler(logits, ids, embedding, projection)
    assert not sampler.entries


def elapsed(function, template, seeds):
    logits, ids = template.clone(), seeds.clone()
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    before = time.perf_counter()
    start.record()
    for _ in range(CALLS):
        function(logits, ids)
    end.record()
    submitted = time.perf_counter()
    end.synchronize()
    return 1000 * (submitted - before), 1000 * (time.perf_counter() - before), start.elapsed_time(end)


@pytest.mark.parametrize("batch", [1, 4, 8])
@torch.inference_mode()
def test_complete_greedy_chain_latency_includes_all_copies(weights, batch):
    embedding, projection = weights
    logits, ids = inputs(batch)
    sampler = DSparkMarkovGraph()
    parent = lambda logits, ids: greedy_markov_chain(logits, ids, embedding, projection)
    candidate = lambda logits, ids: sampler(logits, ids, embedding, projection)
    expected_logits, expected_ids = logits.clone(), ids.clone()
    parent(expected_logits, expected_ids)
    assert candidate(logits, ids)
    same_words(logits, expected_logits)
    assert torch.equal(ids.cpu(), expected_ids.cpu())
    measurements = {"parent": [], "candidate": []}
    for repeat in range(REPEATS):
        order = [("parent", parent), ("candidate", candidate)]
        if repeat % 2:
            order.reverse()
        for name, function in order:
            measurements[name].append(elapsed(function, logits, ids))
    medians = {
        name: tuple(statistics.median(values) for values in zip(*samples)) for name, samples in measurements.items()
    }
    print({"batch": batch, "calls": CALLS, "metrics": ["host_submit_ms", "wall_ready_ms", "device_ms"], **medians})
    assert all(value > 0 for values in medians.values() for value in values)
