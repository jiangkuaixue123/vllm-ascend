# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES
from vllm_ascend.models.hy_v4_flash import dfly
from vllm_ascend.models.hy_v4_flash.cache_pages import C8_ROWS_PER_PAGE, PAGE_BYTES
from vllm_ascend.models.hy_v4_flash.native_draft_history import prepare_native_draft_history
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
        native_c8_history=False,
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


def test_native_draft_stages_request_local_c8_and_keeps_causal_candidates():
    cache = torch.zeros(4, PAGE_BYTES, dtype=torch.uint8)
    rows = cache.view(-1, C8_ROW_BYTES)
    table = torch.tensor([[3, 2], [1, 0], [0, 1]], dtype=torch.int32)
    rows[3 * C8_ROWS_PER_PAGE + 5].fill_(11)
    rows[2 * C8_ROWS_PER_PAGE + 1].fill_(12)
    rows[1 * C8_ROWS_PER_PAGE + 5].fill_(21)
    positions = torch.tensor([[70, 71], [20, 21], [15, 16]])
    state = SharedPrecisionHistory(
        torch.tensor([[5, -1, 65, 70], [5, 20, -1, 65], [3, -1, 1, 2]]), torch.tensor([True, True, False])
    )
    draft = torch.randint(256, (3, 2, C8_ROW_BYTES), dtype=torch.uint8)
    packed, selected = prepare_native_draft_history(cache, table, draft, state, positions)

    assert packed.shape == (3, 6, C8_ROW_BYTES)
    assert [packed[0, 0, 0].item(), packed[0, 2, 0].item(), packed[1, 0, 0].item()] == [11, 12, 21]
    torch.testing.assert_close(packed[:, -2:], draft)
    assert selected.tolist() == [
        [[0, -1, 2, -1, 4, -1], [0, -1, 2, -1, 4, 5]],
        [[0, -1, -1, -1, 4, -1], [0, -1, -1, -1, 4, 5]],
        [[-1] * 6, [-1] * 6],
    ]
