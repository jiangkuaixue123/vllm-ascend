# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

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
