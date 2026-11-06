# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve complete cache words while removing target arena intermediates."""

import statistics
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.deepseek_v41_slot_arena import read
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

QWIDTH = 6
WIN = 128
TOP = 512
DIM = 512
UNION = WIN + QWIDTH - 1
PHYSICAL = 1024
LIMIT = 153600
CALLS = 40
REPEATS = 5


# These two retained functions are the behavioral and timing reference.
@triton.jit
def _parent_gather(
    field,
    table,
    lens,
    active,
    topk,
    out,
    ROWS: tl.constexpr,
    COUNT: tl.constexpr,
    MAXLEN: tl.constexpr,
    PAGE: tl.constexpr,
    PHYS: tl.constexpr,
    STRIDE: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TOP_ROW: tl.constexpr,
    RATIO: tl.constexpr,
    RAW: tl.constexpr,
    PROMOTE: tl.constexpr,
    TILE: tl.constexpr,
    D: tl.constexpr,
):
    dims = tl.arange(0, D)
    for block in range(tl.program_id(0), tl.cdiv(ROWS * COUNT, TILE), tl.num_programs(0)):
        flat = block * TILE + tl.arange(0, TILE)
        ok = flat < ROWS * COUNT
        row = flat // COUNT
        col = flat % COUNT
        if RAW:
            table_row = row * 6
            first = tl.load(lens + table_row, mask=ok, other=0)
            limit = tl.load(lens + table_row + 5, mask=ok, other=0)
            pos = tl.maximum(0, first - 128) + col
            live = ok & (pos < limit)
        else:
            table_row = row
            idx = tl.load(topk + row * TOP_ROW + col, mask=ok, other=-1)
            pos = (idx + 1) * RATIO - 1
            live = (
                ok
                & (idx >= 0)
                & (tl.load(active + row, mask=ok, other=0) != 0)
                & (pos < tl.load(lens + row, mask=ok, other=0))
            )
        safe = tl.maximum(0, tl.minimum(pos, MAXLEN - 1))
        page = tl.load(table + table_row * TABLE_ROW + safe // PAGE, mask=live, other=0).to(tl.int64)
        slot = page * PAGE + safe % PAGE
        valid = live & (pos >= 0) & (slot >= 0) & (slot < PHYS)
        address = tl.where(valid, slot, 0) * STRIDE
        value = tl.load(field + address[:, None] + dims[None, :], mask=valid[:, None], other=0)
        if PROMOTE:
            value = value.to(tl.int32) << 16
        tl.store(out + flat[:, None] * D + dims[None, :], value, mask=ok[:, None])


def parent_read(field, metadata, ids, ratio, raw, dtype, destination):
    count = UNION if raw else TOP
    rows = ids.shape[0] // QWIDTH if raw else ids.shape[0]
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    tmp = torch.empty((rows, count, DIM), dtype=bits, device=field.device)
    grid = (min(triton.cdiv(rows * count, 8), get_vectorcore_num()),)
    _parent_gather[grid](
        field.view(torch.int16 if raw or dtype == torch.bfloat16 else torch.int32),
        metadata.block_table_tensor,
        metadata.seq_lens,
        metadata.active,
        ids,
        tmp,
        rows,
        count,
        metadata.max_model_len,
        128,
        field.shape[0],
        field.stride(0),
        metadata.block_table_tensor.stride(0),
        ids.stride(0),
        ratio,
        raw,
        raw and dtype == torch.float32,
        8,
        DIM,
    )
    # Keep S's stock Where and stock final FP32 -> BF16 conversion, including NPU NaN handling.
    live = metadata.active.view(-1, QWIDTH).any(-1) if raw else metadata.active
    destination.copy_(torch.where(live[:, None, None], tmp.view(dtype), 0.0).to(torch.bfloat16))


def inputs(rows, dtype, raw, base=8192, idle=False):
    torch.npu.set_device(0)
    init_device_properties_triton()
    cache_dtype = torch.bfloat16 if raw else dtype
    generator = torch.Generator().manual_seed(607 + base)
    backing = torch.randn(PHYSICAL, DIM + 16, generator=generator).to(cache_dtype)
    field = backing[:, 8 : 8 + DIM]
    bits = torch.int16 if cache_dtype == torch.bfloat16 else torch.int32
    pattern = (
        [-32768, 0x7FC2, 0x7F80, -128, 1, 0]
        if bits == torch.int16
        else [-2147483648, 0x7FC20001, 0x7F800000, -8388608, 1, 0]
    )
    field.view(bits)[:, : len(pattern)] = torch.tensor(pattern, dtype=bits)
    field = backing.npu()[:, 8 : 8 + DIM]
    lengths = (torch.arange(rows) % QWIDTH + base).clamp(max=LIMIT)
    active = torch.arange(rows) % 3 != 0
    if idle:
        active[:QWIDTH] = False
    table = torch.arange(LIMIT // WIN).repeat(rows, 1).int() % (PHYSICAL // WIN)
    table[:, 0], table[:, 2] = -1, PHYSICAL // WIN
    topk_storage = torch.arange(rows * 520).reshape(rows, 520).long() % TOP
    ids = topk_storage[:, 4:516]
    ids[:, [0, 1, 63, 127, 511]] = -1
    ids[:, 8:11] = 17
    ids[:, 13] = LIMIT
    md = SimpleNamespace(
        seq_lens=lengths.npu(), active=active.npu(), block_table_tensor=table.npu(), max_model_len=LIMIT
    )
    return field, md, topk_storage.npu()[:, 4:516]


def output_for(rows, raw):
    return torch.full(
        (rows // QWIDTH if raw else rows, UNION if raw else TOP, DIM), float("nan"), dtype=torch.bfloat16, device="npu"
    )


def words(tensor):
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("raw", [False, True])
@torch.inference_mode()
def test_exact_masked_cache_words_and_unchanged_inputs(rows, dtype, raw):
    ratio = 2 if dtype == torch.float32 else 1
    for base in (-5, 0, 120, 8192, 153594):
        field, md, ids = inputs(rows, dtype, raw, base, idle=base in (0, 8192))
        saved = [value.clone() for value in (field, md.seq_lens, md.active, md.block_table_tensor, ids)]
        expected, actual = output_for(rows, raw), output_for(rows, raw)
        parent_read(field, md, ids, ratio, raw, dtype, expected)
        read(field, md, ids, ratio, raw, dtype, actual)
        assert torch.equal(words(actual), words(expected))
        assert torch.equal(words(saved[0]), words(field))
        for before, after in zip(saved[1:], (md.seq_lens, md.active, md.block_table_tensor, ids)):
            assert torch.equal(before, after)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("raw", [False, True])
@torch.inference_mode()
def test_graph_refreshes_fields_and_active_rows(dtype, raw):
    rows = 12
    ratio = 2 if dtype == torch.float32 else 1
    field, md, ids = inputs(rows, dtype, raw)
    expected, actual = output_for(rows, raw), output_for(rows, raw)
    read(field, md, ids, ratio, raw, dtype, actual)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        read(field, md, ids, ratio, raw, dtype, actual)
    for base in (0, 120, 153594, 8192):
        fresh_field, fresh_md, fresh_ids = inputs(rows, dtype, raw, base, idle=base in (0, 8192))
        for target, fresh in zip(
            (field, md.seq_lens, md.active, md.block_table_tensor, ids),
            (fresh_field, fresh_md.seq_lens, fresh_md.active, fresh_md.block_table_tensor, fresh_ids),
        ):
            target.copy_(fresh)
        parent_read(field, md, ids, ratio, raw, dtype, expected)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(words(actual), words(expected))


def capture_reads(rows, dtype, parent):
    values = [inputs(rows, dtype, raw) for raw in (True, False)]
    outputs = [output_for(rows, raw) for raw in (True, False)]
    operation = parent_read if parent else read
    ratio = 2 if dtype == torch.float32 else 1
    for raw, (field, md, ids), output in zip((True, False), values, outputs):
        operation(field, md, ids, ratio, raw, dtype, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(CALLS):
            for raw, (field, md, ids), output in zip((True, False), values, outputs):
                operation(field, md, ids, ratio, raw, dtype, output)
    graph.replay()
    torch.npu.synchronize()
    return graph, values, outputs


@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_raw_and_compressed_gather_latency(rows, dtype):
    graphs = [capture_reads(rows, dtype, parent) for parent in (True, False)]
    elapsed = [[], []]
    for repeat in range(REPEATS):
        for choice in (0, 1) if repeat % 2 == 0 else (1, 0):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graphs[choice][0].replay()
            end.record()
            end.synchronize()
            elapsed[choice].append(start.elapsed_time(end))
    print(
        {
            "rows": rows,
            "dtype": str(dtype),
            "read_pairs": CALLS,
            "parent_ms": statistics.median(elapsed[0]),
            "direct_ms": statistics.median(elapsed[1]),
        }
    )


@torch.inference_mode()
def test_every_bf16_word_preserves_native_where_semantics():
    torch.npu.set_device(0)
    init_device_properties_triton()
    rows = QWIDTH
    # One cache page covers every BF16 encoding, including both signs of every NaN.
    field = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).npu().view(torch.bfloat16).reshape(WIN, DIM)
    metadata = SimpleNamespace(
        seq_lens=torch.full((rows,), WIN, dtype=torch.int64, device="npu"),
        active=torch.tensor([False, True, True, False, True, True], device="npu"),
        block_table_tensor=torch.zeros((rows, LIMIT // WIN), dtype=torch.int32, device="npu"),
        max_model_len=LIMIT,
    )
    ids = torch.arange(TOP, dtype=torch.int64, device="npu").repeat(rows, 1)
    expected, actual = output_for(rows, False), output_for(rows, False)
    parent_read(field, metadata, ids, 1, False, torch.bfloat16, expected)
    read(field, metadata, ids, 1, False, torch.bfloat16, actual)
    assert torch.equal(words(actual), words(expected))
