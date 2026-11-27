# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_partitioned_recall_publishes_rows_after_updating_each_local_cache(monkeypatch):
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
        budget=511, ratio=4, num_heads=16, encode_queries=Mock(return_value=(index_query, index_weight))
    )
    cache = SimpleNamespace(
        device=SimpleNamespace(type="npu"), shape=(32, PAGE_BYTES), view=Mock(return_value=torch.zeros(1))
    )
    expected = logical[:, :, None].expand(-1, -1, 2048)
    calls = []
    layer._update_decode_pool = lambda local, tab, size, update, dims: calls.append(int(tab[0]))

    def select(*args):
        assert calls == list(range(requests)) and args[0] is index_query and args[2] is cache
        assert args[3] is tables and args[-2] is layer.request_recall_group and args[-1] == 3
        torch.testing.assert_close(args[4], logical)
        return expected

    def compact(table, selected):
        assert table is tables
        torch.testing.assert_close(selected, expected)
        counts = (logical >= 0).int()
        return torch.arange(requests * width)[:, None], counts

    pending = []
    monkeypatch.setattr(module, "get_forward_context", object)
    monkeypatch.setattr(module, "recall_rows_for_forward", lambda *a: (None, pending))
    monkeypatch.setattr(module, "supports_sharded_recall", Mock(return_value=True))
    monkeypatch.setattr(module, "select_request_partitioned_pool", select)
    monkeypatch.setattr(module, "compact_physical_c8_indices", compact)

    def attend(*args, decode_rows):
        positions, counts = decode_rows
        return torch.ones_like(q).masked_fill(((positions < 0) | (counts == 0)).reshape(-1, 1, 1), 0)

    monkeypatch.setattr(module, "attend_packed_c8", attend)
    layer._forward_decode_graph(meta, cache, PAGE_TOKENS, out, out, torch.arange(q.shape[0]), q, identity, out)
    assert calls == list(range(requests))
    torch.testing.assert_close(torch.stack(pending), expected)
    torch.testing.assert_close(layer.topk_snapshot, expected.reshape(requests * width, 2048))
    torch.testing.assert_close(out, (logical >= 0).reshape(-1, 1).to(out.dtype).expand_as(out))
    layer.resident_pool.invalidate.assert_called_once_with()


@pytest.mark.parametrize("batched", [False, True])
def test_tp1_paged_recall_keeps_each_requests_page_table(monkeypatch, batched):
    from vllm_ascend.models.hy_v4_flash import attention as module
    from vllm_ascend.models.hy_v4_flash.cache_pages import PAGE_BYTES, PAGE_TOKENS

    width, index_heads, ratio, budget = module.PAGED_POOL_DECODE_SHAPE
    attention_heads = 32
    candidates_per_query = (budget + 1) * ratio
    layer = module.AscendHYV4FlashAttention.__new__(module.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(layer)
    layer.share_verify_layout = layer.share_swa_c8_reads = layer.overlap_verify_projections = False
    layer.fuse_indexpool_select = layer.incremental_index_pool = layer.page_planes = True
    layer.batch_indexpool_scores = batched
    layer.shard_indexpool_scores = layer.native_sparse_c8_verify = layer.batch_verify_attention = False
    layer.fuse_c8_qk = layer.pack_window_operands = False
    layer.max_model_len, layer.window, layer.local_heads = 131584, 512, attention_heads
    layer.paired_recall = layer.resident_pool = layer.topk_snapshot = None
    index_query = torch.arange(2 * width)[:, None, None].expand(-1, index_heads, 128).contiguous().float()
    index_weight = torch.ones(2 * width, index_heads)
    layer.indexer = SimpleNamespace(
        budget=budget,
        ratio=ratio,
        num_heads=index_heads,
        encode_queries=Mock(return_value=(index_query, index_weight)),
        pool_projection=object() if batched else None,
    )
    lengths = torch.tensor([130048, 61])
    slots = torch.zeros(2, width, dtype=torch.int32)
    slots[1, -2:] = -1
    tables = torch.arange(10, dtype=torch.int32).view(2, 5)[:, :3]
    meta = SimpleNamespace(decode_seq_lens=lengths, slot_mapping=slots.flatten(), block_table=tables)
    _, _, logical = module.decode_positions(lengths, meta.slot_mapping, width)
    query = torch.zeros(2 * width, attention_heads, 2, dtype=torch.bfloat16)
    out = query.new_empty((2 * width, attention_heads))
    cache = SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(16, PAGE_BYTES), dtype=torch.uint8)
    layer._update_decode_pool = Mock()
    pools = [object(), object()]
    layer._read_decode_pool = Mock(side_effect=pools)
    layer._gather_attention_rows = Mock(return_value=object())
    layer._attend_decoded = Mock()
    recalled = torch.full((width, candidates_per_query), 17, dtype=torch.long)
    calls = []
    pool_slots = object()
    refresh = Mock()
    if batched:
        monkeypatch.setattr(module, "plan_pool_refresh", Mock(return_value=pool_slots))
        monkeypatch.setattr(module, "refresh_pool_keys", refresh)

    def select_dense(keys, weights, pool, visible, actual_ratio, actual_budget):
        request = len(calls)
        first = request * width
        torch.testing.assert_close(keys, index_query[first : first + width])
        torch.testing.assert_close(weights, index_weight[first : first + width])
        torch.testing.assert_close(visible, logical[request])
        assert pool is pools[request]
        assert (actual_ratio, actual_budget) == (ratio, budget)
        calls.append(request)
        return recalled

    paged_select = Mock(side_effect=AssertionError("TP1 keeps its original single-request dense scoring."))
    monkeypatch.setattr(module, "select_paged_pool", paged_select)
    monkeypatch.setattr(module, "sparse_candidates", select_dense)

    def select_batch(keys, weights, storage, page_tables, visible, block_size, max_groups, actual_ratio, actual_budget):
        refresh.assert_called_once_with(layer.indexer, cache, pool_slots, 1)
        assert keys is index_query and weights is index_weight and storage is cache
        torch.testing.assert_close(page_tables, tables)
        torch.testing.assert_close(visible, logical)
        assert page_tables.stride(0) == tables.stride(0)
        assert (block_size, max_groups, actual_ratio, actual_budget) == (
            PAGE_TOKENS,
            layer.max_model_len // ratio,
            ratio,
            budget,
        )
        return torch.stack((recalled, module.dense_candidates(visible[1], candidates_per_query)))

    batch_select = Mock(side_effect=select_batch)
    monkeypatch.setattr(module, "select_batched_paged_pool", batch_select)
    layer._forward_decode_graph(meta, cache, PAGE_TOKENS, out, out, torch.arange(query.shape[0]), query, None, out)

    if batched:
        assert calls == []
        batch_select.assert_called_once()
        layer._update_decode_pool.assert_not_called()
        layer._read_decode_pool.assert_not_called()
    else:
        assert calls == [0, 1]
        assert layer._read_decode_pool.call_count == 2
        batch_select.assert_not_called()
    paged_select.assert_not_called()
    selected0, selected1 = [call.args[1] for call in layer._attend_decoded.call_args_list]
    torch.testing.assert_close(selected0, recalled)
    torch.testing.assert_close(selected1, module.dense_candidates(logical[1], candidates_per_query))
