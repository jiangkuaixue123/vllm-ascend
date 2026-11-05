# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig


def builder(rows, width, max_model_len=16384):
    b = attention.DeepseekV41MetadataBuilder.__new__(attention.DeepseekV41MetadataBuilder)
    b.decode_graph_enabled = True
    b.capture_sizes = {rows}
    b.max_model_len = max_model_len
    b.query_width = width
    b.block_size = 128
    b.decode_buffers = {}
    b.native_layers = []
    b.native_decode_plans = {}
    count = rows // width
    columns = (max_model_len + b.block_size - 1) // b.block_size
    common = SimpleNamespace(
        num_input_tokens=rows,
        num_actual_tokens=rows,
        num_reqs=count,
        max_query_len=width,
        max_seq_len=1029 + count - 1,
        seq_lens=torch.arange(count, dtype=torch.long) + 1029,
        slot_mapping=torch.arange(rows),
        block_table_tensor=torch.arange(count * columns).view(count, columns).int(),
    )
    return b, common


@pytest.mark.parametrize(("full", "width", "expected"), [(24, 6, 0), (48, 6, 0), (12, 6, None), (24, 1, None)])
def test_uniform_builder_is_the_origin_of_grouped_page_table(full, width, expected):
    b, common = builder(full, width)
    empty = b._update_decode(common, capture=True)
    assert empty.grouped_row_start == expected and not empty.active.any()
    assert b._update_decode(common, bounds=list(range(0, full + 1, width))) is empty
    assert torch.equal(empty.block_table_tensor, common.block_table_tensor.repeat_interleave(width, 0))
    pos = torch.arange(full)
    assert torch.equal(empty.seq_lens, common.seq_lens[pos // width] + pos % width + 1 - width)
    common.slot_mapping[0] = -1
    b._update_decode(common, bounds=list(range(0, full + 1, width)))
    assert not empty.active[0] and empty.active[1]
    assert (attention.DeepseekV41Runtime.for_decode(SimpleNamespace(decode=empty)).slot_arena is not None) == (
        expected == 0
    )


@pytest.mark.parametrize("start", [0, 12, 24, 36])
def test_actual_tp_shard_preserves_origin(start):
    b, common = builder(48, 6)
    md = b._update_decode(common, capture=True).shard(start, 12)
    assert md.grouped_row_start == start
    assert attention._supports_arena_metadata(md, torch.device("cpu"), 12)


@pytest.mark.parametrize("max_model_len", [16384, 153600])
@pytest.mark.parametrize("full", [24, 48])
@pytest.mark.parametrize("rank", range(4))
def test_both_builder_tables_reach_last_page_on_each_shard(max_model_len, full, rank):
    rows = full // 4
    start = rank * rows
    shard_tables = []
    for page_offset in (0, 11):
        b, common = builder(full, 6, max_model_len)
        common.max_seq_len = max_model_len
        common.seq_lens.fill_(max_model_len)
        common.block_table_tensor.add_(page_offset).remainder_(32)
        positions = common.seq_lens[:, None] + torch.arange(6) - 6
        pages = common.block_table_tensor.gather(1, positions // b.block_size)
        common.slot_mapping.copy_((pages * b.block_size + positions % b.block_size).flatten())
        common.slot_mapping[::6] = -1
        captured = b._update_decode(common, capture=True)
        assert not captured.active.any()
        assert b._update_decode(common, bounds=list(range(0, full + 1, 6))) is captured
        md = captured.shard(start, rows)
        assert md.grouped_row_start == start and md.active[1] and not md.active[0]
        assert torch.equal(md.seq_lens, torch.arange(max_model_len - 5, max_model_len + 1).repeat(rows // 6))
        assert torch.equal(
            md.block_table_tensor[:, -1], common.block_table_tensor[:, -1].repeat_interleave(6)[start : start + rows]
        )
        shard_tables.append(md)
    md, src = shard_tables
    assert not torch.equal(md.block_table_tensor, src.block_table_tensor)
    assert attention._supports_arena_pair(md, src, md.active.device, rows, max_model_len)
    for short, other in ((md, src), (src, md)):
        short = replace(short, block_table_tensor=short.block_table_tensor[:, :-1])
        assert not attention._supports_arena_pair(short, other, md.active.device, rows, max_model_len)
        assert not attention._supports_arena_pair(other, short, md.active.device, rows, max_model_len)


def test_arena_table_pair_requires_matching_run_and_shard_but_allows_padded_pitch():
    b, common = builder(48, 6, 153600)
    full_md = b._update_decode(common, capture=True)
    md = full_md.shard(0, 12)
    padded = torch.zeros((12, 1217), dtype=torch.int32)[:, :1200]
    src = replace(md, block_table_tensor=padded)
    device = md.active.device
    assert attention._supports_arena_pair(md, src, device, 12, 153600)
    assert not attention._supports_arena_pair(md, full_md.shard(12, 12), device, 12, 153600)
    assert not attention._supports_arena_pair(md, replace(src, max_model_len=16384), device, 12, 153600)
    assert not attention._supports_arena_pair(md, src, device, 12, 16384)
    assert not attention._supports_arena_pair(md, replace(src, grouped_row_start=None), device, 12, 153600)
    assert not attention._supports_arena_pair(md, replace(src, grouped_row_start=False), device, 12, 153600)
    assert not attention._supports_arena_pair(md, replace(src, query_width=1), device, 12, 153600)
    assert not attention._supports_arena_pair(
        md, replace(src, block_table_tensor=padded[:1].expand_as(padded)), device, 12, 153600
    )
    for length in (8192, 16385, 153599, 153601, 153600.0):
        assert not attention._supports_arena_metadata(replace(md, max_model_len=length), device, 12)


def test_long_table_row_offset_cannot_overflow_device_index():
    device = torch.device("meta")
    rows, columns = 12, 1200
    max_pitch = (2**31 - columns) // (rows - 1)
    md = attention.DeepseekV41DecodeMetadata(
        torch.empty(rows, dtype=torch.int64, device=device),
        torch.empty(rows, dtype=torch.int64, device=device),
        torch.empty_strided((rows, columns), (max_pitch, 1), dtype=torch.int32, device=device),
        torch.empty(rows, dtype=torch.bool, device=device),
        153600,
        6,
        0,
    )
    assert attention._supports_arena_metadata(md, device, rows)
    over = torch.empty_strided((rows, columns), (max_pitch + 1, 1), dtype=torch.int32, device=device)
    assert not attention._supports_arena_metadata(replace(md, block_table_tensor=over), device, rows)


def test_compressed_layer_owners_advance_real_source_and_index_generations(monkeypatch):
    # Small CPU projections and cache fields isolate the two actual generation writers.
    config = DeepseekV41TextConfig(index_head_dim=4, index_n_heads=1, index_topk=4)
    b, common = builder(24, 6)
    full = b._update_decode(common, capture=True)
    md = replace(full.shard(0, 6), max_model_len=32)
    md.seq_lens.copy_(torch.arange(6) + 3)
    md.active.fill_(True)
    runtime = attention.DeepseekV41Runtime.for_decode(SimpleNamespace(decode=full))
    states = torch.ones(6, 4)
    indexer = SimpleNamespace(
        wk=lambda x: x, k_norm=lambda x: x, wq_b=lambda x: (x,), weights_proj=lambda x: (x[:, :1],)
    )
    compressor = SimpleNamespace(wkv=lambda x: x, wgate=lambda x: x, norm=lambda x: x)
    monkeypatch.setattr(attention, "fp4_quant_dequant", lambda x, *args, **kwargs: x)
    observed_sources, observed_indexes = [], []
    for lid in range(2, config.num_hidden_layers):
        cache = SimpleNamespace(
            decode_write=lambda *args: None,
            decode_read=lambda md, positions, bank, offset, width: torch.zeros(*positions.shape, width),
        )
        layer = SimpleNamespace(
            config=config,
            layer_id=lid,
            ratio=config.compress_ratios[lid],
            owns_kv=lid in config.kv_source_layer_ids,
            owns_index=lid in config.index_source_layer_ids,
            dim=4,
            index_heads=1,
            tp=4,
            dsa_cp=True,
            compressor=compressor,
            indexer=indexer,
            rotate=lambda *args: None,
            cache=cache,
        )
        arena = runtime.slot_arena
        previous = (arena.source_generation, arena.index_generation)
        previous_cache, previous_topk = runtime.source_cache, runtime.decode_topk
        attention.DeepseekV41Attention._decode_compressed(layer, states, md.seq_lens - 1, md, runtime)
        attention.DeepseekV41Attention._decode_index(layer, states, states, md.seq_lens - 1, md, runtime)
        if arena.source_generation != previous[0]:
            assert arena.source_generation == previous[0] + 1 and runtime.source_cache is cache
            observed_sources.append(lid)
        else:
            assert runtime.source_cache is previous_cache
        if arena.index_generation != previous[1]:
            assert arena.index_generation == previous[1] + 1 and runtime.decode_topk is not previous_topk
            observed_indexes.append(lid)
        else:
            assert runtime.decode_topk is previous_topk
    assert observed_sources == [2, 8, 14, 20]
    assert observed_indexes == [2, 8, 14, 20, 24, 28, 32, 36]


@pytest.mark.parametrize("reason", ["missing", "misaligned", "rows", "table_stride", "lens_type", "short_table"])
def test_unproved_metadata_cannot_use_arena(reason):
    b, common = builder(48, 6)
    md = b._update_decode(common, capture=True).shard(0, 12)
    changes = {
        "missing": {"grouped_row_start": None},
        "misaligned": {"grouped_row_start": 1},
        "rows": {"active": md.active[:6]},
        "table_stride": {"block_table_tensor": md.block_table_tensor[:, ::2]},
        "lens_type": {"seq_lens": md.seq_lens.int()},
        "short_table": {"block_table_tensor": md.block_table_tensor[:, :127]},
    }
    assert not attention._supports_arena_metadata(replace(md, **changes[reason]), torch.device("cpu"), 12)


def test_private_runtime_and_legacy_fallback():
    b, common = builder(48, 6)
    md = b._update_decode(common, capture=True)
    make = lambda obj: attention.DeepseekV41Runtime.for_decode(SimpleNamespace(decode=obj))
    first, second = make(md), make(md)
    assert first.slot_arena is not second.slot_arena
    first.slot_arena.source_generation = 2
    assert second.slot_arena.source_generation == 0
    assert make(replace(md, grouped_row_start=None)).slot_arena is None
    first.slot_arena.staged_generation = (1, 1, 1)
    assert attention.DeepseekV41Attention._try_decode_slot_arena(None, torch.zeros(12, 64, 512), md, first, 512) is None
    assert first.slot_arena.staged_generation is None


def test_arena_storage_is_private_to_forward_shard_stream_and_bucket():
    b, common = builder(24, 6, 153600)
    metadata = SimpleNamespace(decode=b._update_decode(common, capture=True))
    first = attention.DeepseekV41Runtime.for_decode(metadata).slot_arena
    second = attention.DeepseekV41Runtime.for_decode(metadata).slot_arena
    q = torch.empty((6, 64, 512), dtype=torch.bfloat16)
    stream = object()
    values = first.values_for_shard(q, stream, 6)
    assert values.shape == (6 * 640, 512)
    assert first.values_for_shard(q, stream, 6) is values
    assert first.values_for_shard(q, stream, 12) is None
    assert first.values_for_shard(q, object(), 6) is None
    assert first.values_for_shard(q.repeat(2, 1, 1), stream, 6) is None
    assert first.values_for_shard(q.to("meta"), stream, 6) is None
    assert second.values_for_shard(q, stream, 6).data_ptr() != values.data_ptr()


def test_disjoint_fields_checks_actual_byte_ranges():
    base = torch.empty(32, 1024, dtype=torch.bfloat16)
    comp = base[12:, :512]
    assert attention._disjoint_cache_fields(base[:12, :512], comp)
    assert not attention._disjoint_cache_fields(base[11:14, :512], comp)
    assert not attention._disjoint_cache_fields(comp, comp.view(torch.float32))
    assert attention._disjoint_cache_fields(base, torch.empty_like(base))
