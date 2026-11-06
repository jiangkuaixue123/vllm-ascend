# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check signed ring slots, changing host inputs and full staging latency on A3."""

import statistics
import time
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.dspark_ring_indices import try_build_ring_indices
from vllm_ascend.models.deepseek_v41.dspark import try_stage_block_inputs

WINDOW = 128
QUERY_ROWS = 5
DIM = 512
REPEATS = 7
HISTORIES = (128, 129, 255, 256, 8192, 8193, 153593, 153594)


def make_table(batch, dtype, layout="strided"):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(120 + batch)
    cpu = torch.randint(-1, 12, (batch * 2, 1200), dtype=dtype, generator=generator)
    device = cpu.npu()
    if layout == "expanded":
        return cpu[:1].expand(batch, -1), device[:1].expand(batch, -1)
    return cpu[::2], device[::2]


def scalar_slots(starts, table):
    return torch.tensor(
        [
            [
                int(table[row, (start - 1 - (start - 1 - lane) % WINDOW) // WINDOW]) * WINDOW + lane
                for lane in range(WINDOW)
            ]
            for row, start in enumerate(starts)
        ],
        dtype=torch.int64,
    )


@pytest.mark.parametrize("batch", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["strided", "expanded"])
@torch.inference_mode()
def test_exact_slots_and_logical_indices(batch, dtype, layout):
    cpu, table = make_table(batch, dtype, layout)
    before = table.clone()
    starts = HISTORIES[:batch]
    slots, indices = try_build_ring_indices(starts, table, window_size=WINDOW, block_size=QUERY_ROWS)
    assert slots.dtype == torch.int64
    assert torch.equal(slots.cpu(), scalar_slots(starts, cpu))
    assert indices.dtype == torch.int32 and indices.stride() == (133, 0, 1)
    expected = torch.arange(WINDOW + QUERY_ROWS, dtype=torch.int32).view(1, 1, -1).expand(1, QUERY_ROWS, -1)
    assert torch.equal(indices.cpu(), expected)
    assert torch.equal(table, before)


@torch.inference_mode()
def test_queued_changing_host_inputs_keep_ownership():
    cpu, table = make_table(4, torch.int32)
    queued = []
    for iteration in range(32):
        starts = [8192 + iteration * 5 + row for row in range(4)]
        result = try_build_ring_indices(starts, table, window_size=WINDOW, block_size=QUERY_ROWS)
        queued.append((list(starts), result))
        starts[:] = [128] * 4
    torch.npu.synchronize()
    for starts, (slots, _) in queued:
        assert torch.equal(slots.cpu(), scalar_slots(starts, cpu))


@pytest.mark.parametrize("kind", ["cpu", "single", "short", "overflow", "dtype", "columns", "window", "queries"])
@torch.inference_mode()
def test_unsupported_inputs_use_reference(kind):
    cpu, table = make_table(2, torch.int32)
    starts, window, queries = [8192, 8193], WINDOW, QUERY_ROWS
    if kind == "cpu":
        table = cpu
    elif kind == "single":
        starts = starts[:1]
    elif kind == "short":
        starts[0] = WINDOW - 1
    elif kind == "overflow":
        starts[0] = table.shape[1] * WINDOW + 1
    elif kind == "dtype":
        table = table.float()
    elif kind == "columns":
        table = table[:, ::2]
    elif kind == "window":
        window //= 2
    elif kind == "queries":
        queries -= 1
    assert try_build_ring_indices(starts, table, window_size=window, block_size=queries) is None


def eager_staging(starts, table, field, query):
    device_starts = torch.tensor(starts, dtype=torch.long, device=table.device)
    lanes = torch.arange(WINDOW, device=table.device)
    history = device_starts[:, None] - 1 - (device_starts[:, None] - 1 - lanes).remainder(WINDOW)
    pages = table[: len(starts)].gather(1, history // WINDOW).long()
    slots = pages * WINDOW + history % WINDOW
    context = field[slots]
    kv = torch.cat((context, query.view(len(starts), QUERY_ROWS, DIM)), dim=1)
    indices = torch.arange(WINDOW + QUERY_ROWS, dtype=torch.int32, device=table.device)
    return kv, indices.view(1, 1, -1).expand(1, QUERY_ROWS, -1)


@pytest.mark.parametrize("batch", [2, 4, 8])
@torch.inference_mode()
def test_full_staging_contents_and_latency(batch):
    _, table = make_table(batch, torch.int32)
    generator = torch.Generator().manual_seed(321 + batch)
    backing = torch.randn(12 * WINDOW, DIM * 2, dtype=torch.float32, generator=generator).bfloat16().npu()
    field = backing[:, :DIM]
    before = backing.clone()
    query = torch.randn(batch * QUERY_ROWS, DIM, dtype=torch.float32, generator=generator).bfloat16().npu()
    q = torch.zeros(batch * QUERY_ROWS, 16, DIM, dtype=torch.bfloat16, device=table.device)
    starts = HISTORIES[:batch]
    metadata = SimpleNamespace(
        requests=[
            SimpleNamespace(begin=i * QUERY_ROWS, end=(i + 1) * QUERY_ROWS, start_pos=s) for i, s in enumerate(starts)
        ],
        num_actual_tokens=batch * QUERY_ROWS,
        block_table_tensor=table,
    )
    cache = SimpleNamespace(block_size=WINDOW, field=lambda *args: field)

    def run(fused):
        if fused:
            return try_stage_block_inputs(
                q, query, metadata, cache, window_size=WINDOW, block_size=QUERY_ROWS, dtype=torch.bfloat16
            )
        return eager_staging(starts, table, field, query)

    expected, actual = run(False), run(True)
    assert torch.equal(actual[0].view(torch.int16), expected[0].view(torch.int16))
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(backing.view(torch.int16), before.view(torch.int16))
    elapsed = {False: [], True: []}
    for repeat in range(REPEATS):
        for fused in (False, True) if repeat % 2 == 0 else (True, False):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            run(fused)
            torch.npu.synchronize()
            elapsed[fused].append((time.perf_counter_ns() - start) / 1e6)
    print(
        {
            "batch": batch,
            "eager_wall_ms": statistics.median(elapsed[False]),
            "fused_wall_ms": statistics.median(elapsed[True]),
            "includes_host_copy_and_kv": True,
        }
    )
