# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare staged serial and batched DSpark attention on the actual A3 operator."""

import statistics
import time
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.dspark import try_stage_block_inputs
from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention

QUERY_ROWS = 5
WINDOW = 128
HEADS = 16
HEAD_DIM = 512
REPEATS = 7


def make_case(batch):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(185 + batch)
    backing = torch.randn(12 * WINDOW, HEAD_DIM * 2, generator=generator, dtype=torch.float32).bfloat16().npu()
    field = backing[:, :HEAD_DIM]
    table = torch.randint(12, (batch, 1200), dtype=torch.int32, generator=generator).npu()
    table[:, 0] = 0
    histories = [128, 129, 255, 256, 8192, 8193, 153593, 153594][:batch]
    q = (
        torch.randn(
            batch * QUERY_ROWS,
            HEADS,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        .bfloat16()
        .npu()
    )
    query = torch.randn(batch * QUERY_ROWS, HEAD_DIM, generator=generator, dtype=torch.float32).bfloat16().npu()
    sink = torch.randn(HEADS, generator=generator, dtype=torch.float32).npu()
    metadata = SimpleNamespace(
        requests=[
            SimpleNamespace(begin=i * QUERY_ROWS, end=(i + 1) * QUERY_ROWS, start_pos=h)
            for i, h in enumerate(histories)
        ],
        num_actual_tokens=batch * QUERY_ROWS,
        block_table_tensor=table,
    )
    cache = SimpleNamespace(block_size=WINDOW, field=lambda *args: field)
    return q, query, sink, metadata, cache, backing


def run(case, kernel, *, batched):
    q, query, sink, metadata, cache, _ = case
    kv, indices = try_stage_block_inputs(
        q,
        query,
        metadata,
        cache,
        window_size=WINDOW,
        block_size=QUERY_ROWS,
        dtype=torch.bfloat16,
    )
    batch = len(metadata.requests)
    if batched:
        return kernel(
            q.view(batch, QUERY_ROWS, HEADS, HEAD_DIM),
            kv,
            sink,
            indices.expand(batch, -1, -1),
            HEAD_DIM**-0.5,
            mode="SWA draft bidirectional",
        ).reshape_as(q)
    result = torch.zeros_like(q)
    for row in range(batch):
        rows = slice(row * QUERY_ROWS, (row + 1) * QUERY_ROWS)
        result[rows] = kernel(
            q[rows].unsqueeze(0),
            kv[row : row + 1],
            sink,
            indices,
            HEAD_DIM**-0.5,
            mode="SWA draft bidirectional",
        )[0]
    return result


@pytest.mark.parametrize("batch", [2, 4, 8])
@torch.inference_mode()
def test_batched_native_output_and_request_isolation(batch):
    case = make_case(batch)
    q, query, _, _, _, backing = case
    before = backing.clone()
    serial, batched = DeepseekV41NativeAttention(), DeepseekV41NativeAttention()
    expected = run(case, serial, batched=False)
    actual = run(case, batched, batched=True)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    # Refresh only the final request. Other requests must keep identical output.
    q[-QUERY_ROWS:].neg_()
    query[-QUERY_ROWS:].add_(1)
    refreshed = run(case, batched, batched=True)
    reference = run(case, serial, batched=False)
    assert torch.equal(refreshed.view(torch.int16), reference.view(torch.int16))
    assert torch.equal(
        refreshed[:-QUERY_ROWS].view(torch.int16),
        actual[:-QUERY_ROWS].view(torch.int16),
    )
    assert torch.equal(backing.view(torch.int16), before.view(torch.int16))


@pytest.mark.parametrize("batch", [2, 4, 8])
@torch.inference_mode()
def test_staging_and_attention_latency(batch):
    case = make_case(batch)
    kernels = [DeepseekV41NativeAttention(), DeepseekV41NativeAttention()]
    for batched in (False, True):
        run(case, kernels[int(batched)], batched=batched)
    torch.npu.synchronize()
    elapsed = {False: [], True: []}
    for repeat in range(REPEATS):
        for batched in (False, True) if repeat % 2 == 0 else (True, False):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            run(case, kernels[int(batched)], batched=batched)
            torch.npu.synchronize()
            elapsed[batched].append((time.perf_counter_ns() - start) / 1e6)
    print(
        {
            "batch": batch,
            "serial_wall_ms": statistics.median(elapsed[False]),
            "batched_wall_ms": statistics.median(elapsed[True]),
            "includes_staging": True,
        }
    )
