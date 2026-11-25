# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.hy_v4_flash import dfly
from vllm_ascend.models.hy_v4_flash.shared_precision_history import SharedPrecisionHistory


@pytest.mark.parametrize("batch", [1, 4, 8, 16])
def test_draft_history_keeps_request_tables_and_private_rows(monkeypatch, batch):
    width, heads, hidden = 6, 4, 3
    table = torch.arange(batch * 7).view(batch, 7).flip(0)
    positions = torch.arange(batch * width).view(batch, width) + 33
    history = SharedPrecisionHistory(torch.arange(batch * 2048).view(batch, 2048), torch.arange(batch) % 3 != 1)
    draft = torch.arange(batch * width * 8).reshape(batch, width, 8).flip(0)
    q = torch.rand(batch * width, heads, hidden)
    w = torch.rand(heads, hidden, 2)
    expected = torch.randn_like(q)
    source = SimpleNamespace(cache_rows=lambda: table)
    layer = SimpleNamespace(
        context=(64, table, history, positions),
        target_attention=lambda: source,
        _pack_kv=lambda *_: draft,
        learnable_sink_param=torch.zeros(heads),
        scale=0.0625,
        absorption_weights=SimpleNamespace(value=w),
    )
    consumed = []

    def attend(query, cache, local, private, selected, anchor, sink, scale, page):
        first = sum(consumed)
        count = len(local)
        region = slice(first, first + count)
        tokens = slice(first * width, (first + count) * width)
        assert cache is table and page == 64
        torch.testing.assert_close(query, q[tokens])
        torch.testing.assert_close(local, table[region])
        torch.testing.assert_close(private, draft[region])
        torch.testing.assert_close(selected.target_rows, history.target_rows[region])
        torch.testing.assert_close(selected.active, history.active[region])
        torch.testing.assert_close(anchor, positions[region, 0])
        consumed.append(count)
        return expected[tokens]

    monkeypatch.setattr(dfly, "attend_shared_fp32", attend)
    out = torch.empty(batch * width, heads * 2)
    dfly.DFlyAttention._forward_paged(layer, None, None, None, None, None, None, None, out, ready_query=q)
    assert sum(consumed) == batch
    torch.testing.assert_close(out, dfly._head_bmm(expected, w).reshape_as(out))
