# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check target slot ordering, changing graph inputs and parent kernel latency."""

import statistics

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.deepseek_v41_slot_arena import SLOT_COLUMNS, SLOT_ROWS, _slots
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

WIDTH = 640
TOP = 512
QUERY_ROWS = 6
RAW_PAD = 144
CALLS = 80
REPEATS = 5


# The retained flattened kernel is kept here solely as the timing baseline.
@triton.jit
def _parent_slots(
    lens, active, topk, out, ROWS: tl.constexpr, CMP_BASE: tl.constexpr, TOP_ROW: tl.constexpr, TILE: tl.constexpr
):
    for block in range(tl.program_id(0), tl.cdiv(ROWS * 640, TILE), tl.num_programs(0)):
        flat = block * TILE + tl.arange(0, TILE)
        ok = flat < ROWS * 640
        row = flat // 640
        col = flat % 640
        group = row // 6
        length = tl.load(lens + row, mask=ok, other=0)
        first = tl.load(lens + group * 6, mask=ok, other=0)
        active_row = tl.load(active + row, mask=ok, other=0) != 0
        width = tl.minimum(128, tl.maximum(0, length))
        ordinal = col - width
        is_cmp = ok & (ordinal >= 0) & (ordinal < 512)
        idx = tl.load(topk + row * TOP_ROW + tl.where(is_cmp, ordinal, 0), mask=is_cmp, other=-1)
        raw_slot = group * 133 + tl.maximum(0, length - 128) - tl.maximum(0, first - 128) + col
        cmp_slot = CMP_BASE + row * 512 + ordinal
        slot = tl.where(col < width, raw_slot, tl.where(is_cmp & (idx >= 0), cmp_slot, -1))
        tl.store(out + flat, tl.where(active_row, slot, -1).to(tl.int32), mask=ok)


def make_inputs(rows, dtype, base):
    torch.npu.set_device(0)
    init_device_properties_triton()
    lengths = torch.tensor(
        [min(base + (row // QUERY_ROWS) * 7, 153594) + row % QUERY_ROWS for row in range(rows)], dtype=torch.int64
    )
    active = torch.arange(rows) % 3 != 0
    backing = torch.arange(rows * 520, dtype=dtype).reshape(rows, 520)
    topk = backing[:, 4:516]
    topk[:, [0, 1, 63, 128, 511]] = -1
    topk[:, 7:10] = 17
    device_backing = backing.npu()
    return (lengths, active, topk), (lengths.npu(), active.npu(), device_backing[:, 4:516])


def oracle(lengths, active, topk):
    lengths, active, topk = lengths.tolist(), active.tolist(), topk.tolist()
    rows = len(lengths)
    result = []
    for row, length in enumerate(lengths):
        group = row // QUERY_ROWS
        width = min(128, max(0, length))
        raw_begin = group * 133 + max(0, length - 128) - max(0, lengths[group * QUERY_ROWS] - 128)
        values = []
        for col in range(WIDTH):
            ordinal = col - width
            if not active[row]:
                slot = -1
            elif col < width:
                slot = raw_begin + col
            elif 0 <= ordinal < TOP and topk[row][ordinal] >= 0:
                slot = rows // QUERY_ROWS * RAW_PAD + row * TOP + ordinal
            else:
                slot = -1
            values.append(slot)
        result.append(values)
    return torch.tensor(result, dtype=torch.int32).unsqueeze(1)


def run(inputs, parent=False):
    lengths, active, topk = inputs
    rows = lengths.numel()
    output = torch.empty((rows, 1, WIDTH), dtype=torch.int32, device=lengths.device)
    if parent:
        tile = 128
        grid = (min(triton.cdiv(output.numel(), tile), get_vectorcore_num()),)
        _parent_slots[grid](lengths, active, topk, output, rows, rows // QUERY_ROWS * RAW_PAD, topk.stride(0), tile)
    else:
        grid = (triton.cdiv(rows, SLOT_ROWS), triton.cdiv(WIDTH, SLOT_COLUMNS))
        _slots[grid](
            lengths, active, topk, output, rows, rows // QUERY_ROWS * RAW_PAD, topk.stride(0), SLOT_COLUMNS, SLOT_ROWS
        )

    return output


@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_complete_slot_words(rows, dtype):
    for base in (-5, 0, 120, 8192, 153594):
        cpu, device = make_inputs(rows, dtype, base)
        saved = tuple(value.clone() for value in device)
        expected = oracle(*cpu)
        assert torch.equal(run(device).cpu(), expected)
        assert torch.equal(run(device, parent=True).cpu(), expected)
        for before, after in zip(saved, device):
            assert torch.equal(before, after)


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_graph_refreshes_lengths_masks_and_selected_indices(rows):
    _, device = make_inputs(rows, torch.int64, 8192)
    run(device)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run(device)
    address = output.data_ptr()
    for base in (0, 120, 153594, 8192):
        cpu, fresh = make_inputs(rows, torch.int64, base)
        for original, replacement in zip(device, fresh):
            original.copy_(replacement)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), oracle(*cpu))
        assert output.data_ptr() == address


def captured_chain(inputs, parent):
    run(inputs, parent)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(CALLS):
            run(inputs, parent)
    graph.replay()
    torch.npu.synchronize()
    return graph


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_slot_graph_latency(rows):
    _, inputs = make_inputs(rows, torch.int64, 8192)
    graphs = [captured_chain(inputs, parent) for parent in (True, False)]
    elapsed = [[], []]
    for repeat in range(REPEATS):
        for choice in (0, 1) if repeat % 2 == 0 else (1, 0):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graphs[choice].replay()
            end.record()
            end.synchronize()
            elapsed[choice].append(start.elapsed_time(end))
    print(
        {
            "rows": rows,
            "calls": CALLS,
            "parent_ms": statistics.median(elapsed[0]),
            "row_program_ms": statistics.median(elapsed[1]),
        }
    )
