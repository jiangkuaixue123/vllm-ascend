# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("requests", [2, 7, 8, 13, 16])
def test_request_partitions_preserve_per_request_scores_and_output_order(monkeypatch, requests):
    from vllm_ascend.models.hy_v4_flash import index_pool_select as pool

    width, budget, ratio = 6, 3, 4
    group = Mock()
    group.size.return_value = 8
    query = torch.arange(requests * width).reshape(-1, 1, 1)
    weights = query.flatten(1).clone()
    tables = torch.arange(requests, dtype=torch.int32).view(requests, 1)
    visible = query.view(requests, width)
    expected = visible.unsqueeze(-1).expand(requests, width, (budget + 1) * ratio)
    received = []

    def select(q, w, cache, table, ends, *args):
        r = int(table[0])
        torch.testing.assert_close(q[:, 0, 0], expected[r, :, 0])
        torch.testing.assert_close(w[:, 0], ends)
        return expected[r]

    def gather(dst, src, group):
        received.append(src.clone())
        dst.zero_()
        dst[:requests].copy_(expected)

    monkeypatch.setattr(pool, "select_native_page_pool", select)
    monkeypatch.setattr(pool.dist, "all_gather_into_tensor", gather)
    for rank in range(8):
        result = pool.select_request_partitioned_pool(
            query, weights, None, tables, visible, 64, 511, ratio, budget, group, rank
        )
        torch.testing.assert_close(result, expected)
    all_sent = torch.cat(received)
    torch.testing.assert_close(all_sent[:requests], expected.int())
    assert not all_sent[requests:].count_nonzero()
