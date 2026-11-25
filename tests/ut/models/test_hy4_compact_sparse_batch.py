# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_native_decode_keeps_request_rows_and_masks_padding(monkeypatch, batch):
    from vllm_ascend.models.hy_v4_flash import attention as module
    from vllm_ascend.models.hy_v4_flash.cache_pages import PAGE_BYTES, PAGE_TOKENS

    queries, heads, window, dimension = 6, 4, 2048, 2
    layer = module.AscendHYV4FlashAttention.__new__(module.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(layer)
    layer.indexer = SimpleNamespace(budget=511, ratio=4)
    layer.native_sparse_c8_verify = layer.batch_verify_attention = True
    layer.share_verify_layout = layer.share_swa_c8_reads = layer.incremental_index_pool = False
    layer.local_heads = heads
    layer.window = layer.max_model_len = window
    layer.topk_snapshot = layer.resident_pool = None
    layer.learnable_sink_param = torch.zeros(heads)
    layer.scale = 1.0
    lengths = torch.arange(batch) + 19
    slots = torch.zeros(batch, queries, dtype=torch.int32)
    slots[::2, -1] = -1
    tables = torch.arange(batch).reshape(-1, 1).int()
    meta = SimpleNamespace(decode_seq_lens=lengths, slot_mapping=slots.flatten(), block_table=tables)
    q = torch.zeros(batch * queries, heads, dimension, dtype=torch.bfloat16)
    out = q.new_empty((batch * queries, heads * dimension))
    weights = torch.eye(dimension, dtype=q.dtype).repeat(heads, 1, 1)
    cache = torch.zeros(1, PAGE_BYTES, dtype=torch.uint8)
    calls = []

    def map_rows(table, selected):
        assert table is tables and selected.shape == (batch, queries, window)
        counts = (selected >= 0).sum(-1)
        actual = lengths[:, None] - queries + 1 + torch.arange(queries)
        torch.testing.assert_close(counts, actual.masked_fill(slots < 0, 0))
        counts[:, 0] = 0
        calls.append(1)
        return table.repeat_interleave(queries, 0), counts

    def native(query, cache, rows, sink, scale):
        assert query is q
        return (rows + 1).to(q.dtype).view(-1, 1, 1).expand_as(q).clone()

    monkeypatch.setattr(module, "compact_physical_c8_indices", map_rows)
    monkeypatch.setattr(module, "attend_packed_c8", native)
    layer._forward_decode_graph(meta, cache, PAGE_TOKENS, None, None, None, q, weights, out)
    expected = (tables + 1).repeat(1, queries).masked_fill(slots < 0, 0)
    expected[:, 0] = 0
    torch.testing.assert_close(out, expected.to(out.dtype).reshape(-1, 1).expand_as(out))
    assert calls == [1]
