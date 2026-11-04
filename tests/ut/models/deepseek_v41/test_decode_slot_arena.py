# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention


def builder(rows, width):
    b = attention.DeepseekV41MetadataBuilder.__new__(attention.DeepseekV41MetadataBuilder)
    b.decode_graph_enabled = True
    b.capture_sizes = {rows}
    b.max_model_len = 16384
    b.query_width = width
    b.block_size = 128
    b.decode_buffers = {}
    b.native_layers = []
    b.native_decode_plans = {}
    count = rows // width
    common = SimpleNamespace(
        num_input_tokens=rows,
        num_actual_tokens=rows,
        num_reqs=count,
        max_query_len=width,
        max_seq_len=1029,
        seq_lens=torch.arange(count, dtype=torch.long) + 1029,
        slot_mapping=torch.arange(rows),
        block_table_tensor=torch.arange(count * 128).view(count, 128).int(),
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


def test_disjoint_fields_checks_actual_byte_ranges():
    base = torch.empty(32, 1024, dtype=torch.bfloat16)
    comp = base[12:, :512]
    assert attention._disjoint_cache_fields(base[:12, :512], comp)
    assert not attention._disjoint_cache_fields(base[11:14, :512], comp)
    assert not attention._disjoint_cache_fields(comp, comp.view(torch.float32))
    assert attention._disjoint_cache_fields(base, torch.empty_like(base))
