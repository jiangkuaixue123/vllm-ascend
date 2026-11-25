# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_joined_refresh_precedes_distributed_recall(monkeypatch):
    from vllm_ascend.models.hy_v4_flash import attention as module
    from vllm_ascend.models.hy_v4_flash.cache_pages import PAGE_BYTES, PAGE_TOKENS

    requests, width, heads, dim = 4, 6, 4, 2
    layer = module.AscendHYV4FlashAttention.__new__(module.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(layer)
    layer.share_verify_layout = layer.share_swa_c8_reads = layer.overlap_verify_projections = False
    layer.page_planes = layer.incremental_index_pool = layer.fuse_indexpool_select = layer.shard_indexpool_scores = True
    layer.native_sparse_c8_verify = layer.batch_verify_attention = True
    layer.max_model_len, layer.window, layer.local_heads = 131584, 2048, heads
    layer.request_recall_rank, layer.request_recall_group = 3, object()
    layer.paired_recall, layer.resident_pool = (1, True), Mock()
    layer.topk_snapshot = torch.full((requests * width, 2048), -3)
    layer.learnable_sink_param, layer.scale = torch.zeros(heads), 1.0
    lengths = torch.tensor([130048, 513, 65, 0])
    logical = lengths[:, None] - width + torch.arange(width)
    logical[-1].fill_(-1)
    tables = torch.arange(requests, dtype=torch.int32).view(-1, 1)
    slots = (logical >= 0).int() - 1
    meta = SimpleNamespace(decode_seq_lens=lengths, slot_mapping=slots.flatten(), block_table=tables)
    q = torch.zeros(requests * width, heads, dim, dtype=torch.bfloat16)
    out = q.new_empty((requests * width, heads * dim))
    identity = torch.eye(dim, dtype=q.dtype).repeat(heads, 1, 1)
    index_query = torch.ones(requests * width, 16, 128)
    index_weight = torch.ones(requests * width, 16)
    layer.indexer = SimpleNamespace(
        pool_projection=torch.empty(0),
        budget=511,
        ratio=4,
        num_heads=16,
        encode_queries=Mock(return_value=(index_query, index_weight)),
    )
    cache = SimpleNamespace(
        dtype=torch.uint8,
        device=SimpleNamespace(type="npu"),
        shape=(32, PAGE_BYTES),
        view=Mock(return_value=torch.zeros(1)),
    )
    calls = []
    layer._update_decode_pool = Mock(side_effect=AssertionError("duplicate per-request cache update"))

    def plan(seq, table, qwidth):
        assert table is tables and qwidth == width
        torch.testing.assert_close(seq, lengths)
        return "raw_slots", "written_groups"

    def refresh(index, live_cache, slots, coords):
        assert (
            index is layer.indexer and live_cache is cache and slots == ("raw_slots", "written_groups") and coords == 1
        )
        calls.append("refreshed")

    monkeypatch.setattr(module, "plan_pool_refresh", plan)
    monkeypatch.setattr(module, "refresh_pool_keys", refresh)

    def select(*args, resident, updated_slots):
        assert resident is layer.resident_pool and updated_slots == "written_groups"
        assert calls == ["refreshed"] and args[0] is index_query and args[2] is cache
        assert args[3] is tables and args[-2] is layer.request_recall_group and args[-1] == 3
        torch.testing.assert_close(args[4], logical)
        raise StopAfterRecall

    class StopAfterRecall(Exception):
        pass

    pending = []
    monkeypatch.setattr(module, "get_forward_context", object)
    monkeypatch.setattr(module, "recall_rows_for_forward", lambda *args: (None, pending))
    monkeypatch.setattr(module, "select_request_partitioned_pool", select)
    with pytest.raises(StopAfterRecall):
        layer._forward_decode_graph(meta, cache, PAGE_TOKENS, out, out, torch.arange(q.shape[0]), q, identity, out)
    assert calls == ["refreshed"]
    layer._update_decode_pool.assert_not_called()


def test_layout_replans_when_a_different_physical_table_is_supplied(monkeypatch):
    from vllm_ascend.models.hy_v4_flash import decode_layout

    lengths = torch.tensor([127, 512])
    slots = torch.arange(12)
    meta = SimpleNamespace(decode_seq_lens=lengths, slot_mapping=slots)
    layout = decode_layout.VerificationLayout.for_forward(SimpleNamespace(additional_kwargs={}), meta, 6)
    first = torch.arange(20).view(2, 10)
    other = first.roll(1, 1)
    calls = []

    def plan(seq, table, width):
        calls.append(table)
        return (table, object())

    monkeypatch.setattr(decode_layout, "plan_pool_refresh", plan)
    original = layout.get_native_update_slots(first)
    assert layout.get_native_update_slots(first) is original
    replaced = layout.get_native_update_slots(other)
    assert replaced is not original and replaced[0] is other
    assert len(calls) == 2
