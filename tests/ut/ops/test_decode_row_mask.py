# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact copy semantics, fallback gradients, and changed-input NPU graph replay."""

import json
import statistics
import time

import pytest
import torch

from vllm_ascend.models.common.ops.decode_row_mask import mask_decode_rows

WIDTH = 5120
BF16_PATTERNS = 1 << 16
TIMED_CALLS = 120


def assert_bits(actual, expected):
    torch.testing.assert_close(actual.view(torch.int16).cpu(), expected.view(torch.int16).cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_fallback_preserves_gradients(dtype):
    values = torch.randn(6, WIDTH, dtype=dtype, requires_grad=True)
    active = torch.tensor([True, False, True, False, True, False])
    result = mask_decode_rows(values, active)
    torch.testing.assert_close(result, torch.where(active[:, None], values, 0.0), rtol=0, atol=0)
    result.float().sum().backward()
    torch.testing.assert_close(values.grad, active[:, None].expand_as(values).to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_npu_all_bf16_bits_and_dynamic_graph_mask(rows):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_ascend.ops.triton.decode_row_mask import decode_row_mask

    assert decode_row_mask is not None
    values = torch.empty(rows, WIDTH, device="npu", dtype=torch.bfloat16)
    active = torch.ones(rows, device="npu", dtype=torch.bool)
    mask_decode_rows(values, active)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = mask_decode_rows(values, active)
    # Every BF16 bit pattern, including signed zeros and NaN payloads, appears
    # in an active row. Mask updates also exercise all-idle and alternating rows.
    for offset in range(0, BF16_PATTERNS, values.numel()):
        bits = ((torch.arange(values.numel(), dtype=torch.int32) + offset) % BF16_PATTERNS).to(torch.int16)
        values.copy_(bits.view(torch.bfloat16).reshape_as(values))
        for mask in (
            torch.ones(rows, dtype=torch.bool),
            torch.arange(rows) % 2 == 0,
            torch.zeros(rows, dtype=torch.bool),
        ):
            active.copy_(mask)
            expected = torch.where(active[:, None], values, 0.0)
            assert_bits(decode_row_mask(values, active), expected)
            graph.replay()
            torch.npu.synchronize()
            assert_bits(result, expected)
    # Original behavior remains available outside the specialized layout.
    for other in (values.float(), values[:1], values[:, ::2], values.T.contiguous().T):
        other_active = active[: other.shape[0]]
        actual = mask_decode_rows(other, other_active)
        expected = torch.where(other_active[:, None], other, 0.0)
        if other.dtype == torch.bfloat16:
            assert_bits(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    # Three masks in each of 40 target layers; include completion, not just launch.
    values.normal_()
    active.fill_(True)
    graphs = []
    for operation in (lambda value, mask: torch.where(mask[:, None], value, 0.0), mask_decode_rows):

        def complete(operation=operation):
            value = values
            for _ in range(TIMED_CALLS):
                value = operation(value, active)
            return value

        complete()
        torch.npu.synchronize()
        timed_graph = torch.npu.NPUGraph()
        with torch.npu.graph(timed_graph):
            complete()
        graphs.append(timed_graph)
    samples = [[], []]
    for repeat in range(5):
        for index in (repeat % 2, 1 - repeat % 2):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            graphs[index].replay()
            torch.npu.synchronize()
            samples[index].append((time.perf_counter_ns() - start) / 1e6)
    print(
        "TARGET_ROW_MASK_COMPONENT "
        + json.dumps(
            {
                "rows": rows,
                "calls": TIMED_CALLS,
                "parent_ms": statistics.median(samples[0]),
                "candidate_ms": statistics.median(samples[1]),
                "status": "PASS",
            }
        ),
        flush=True,
    )
