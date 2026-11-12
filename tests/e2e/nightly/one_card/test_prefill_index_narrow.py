# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact target prefill selection with bounded floating-point column IDs."""

import statistics
import time

import pytest
import test_index_tie_fp32 as previous
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import attention

SELECTED = 512
TIMING_SAMPLES = 5


def reference(scores, k):
    return previous.reference_topk_columns(scores, k).sort(-1).values


@pytest.mark.parametrize("rows,width", [(128, 4096), (64, 8192), (17, 513), (3, 63), (1, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_exact_prefill_indices(rows, width, dtype):
    torch.npu.set_device(0)
    torch.manual_seed(20261113)
    k = min(SELECTED, width)
    for mode in ("random", "few_values", "zeros", "invalid", "infinities", "few_nan", "many_nan"):
        scores = torch.randn(rows, width, device="npu", dtype=dtype)
        if mode == "few_values":
            scores.round_()
        elif mode == "zeros":
            scores.zero_()
        elif mode == "invalid":
            scores.fill_(-torch.inf)
        elif mode == "infinities":
            scores[:, ::3] = torch.inf
            scores[:, ::5] = -torch.inf
        elif mode == "few_nan":
            scores[:, :3] = torch.nan
        elif mode == "many_nan":
            scores[:, : SELECTED * 2] = torch.nan
        saved = scores.clone()
        actual = attention.sorted_topk_columns(scores, k, prefill=True)
        expected = reference(scores, k)
        assert actual.dtype == torch.int64 and torch.equal(actual, expected), (mode, rows, width, dtype)
        assert torch.equal(scores.view(torch.int16), saved.view(torch.int16))
    assert attention.sorted_topk_columns(scores, 0, prefill=True).shape == (rows, 0)


@pytest.mark.parametrize("width,block_size", [(4096, 64), (8192, 64), (517, 64), (127, 128)])
@torch.inference_mode()
def test_latest_candidate_blocks_and_invalid_prefix(width, block_size):
    torch.npu.set_device(0)
    lengths = torch.tensor([0, 1, 63, 127, 128, width - 1, width], device="npu")[:, None]
    columns = torch.arange(width, device="npu")[None]
    for tied in (False, True):
        scores = torch.randn(lengths.numel(), width, device="npu", dtype=torch.bfloat16)
        if tied:
            scores.round_()
        scores.masked_fill_(columns >= lengths, -torch.inf)
        for per_token in (False, True):
            expected = attention.select_candidate_blocks(scores, lengths, 4, block_size, per_token=per_token)
            actual = attention.select_candidate_blocks(
                scores, lengths, 4, block_size, per_token=per_token, prefill=True
            )
            assert torch.equal(actual, expected)


def elapsed(function):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    function()
    submitted = time.perf_counter_ns()
    torch.npu.synchronize()
    return (submitted - start) / 1e6, (time.perf_counter_ns() - start) / 1e6


@pytest.mark.parametrize("rows,width", [(128, 4096), (64, 8192)])
@torch.inference_mode()
def test_eager_component_time(rows, width):
    torch.npu.set_device(0)
    scores = torch.randn(rows, width, device="npu", dtype=torch.bfloat16).round()
    functions = [
        lambda: reference(scores, SELECTED),
        lambda: attention.sorted_topk_columns(scores, SELECTED, prefill=True),
    ]
    for function in functions:
        function()
    samples = [[], []]
    for _ in range(TIMING_SAMPLES):
        for index, function in enumerate(functions):
            samples[index].append(elapsed(function))
    medians = [[statistics.median(sample[column] for sample in group) for column in (0, 1)] for group in samples]
    print(
        {
            "rows": rows,
            "width": width,
            "parent": medians[0],
            "candidate": medians[1],
            "metrics": ["host_ms", "ready_ms"],
        }
    )
