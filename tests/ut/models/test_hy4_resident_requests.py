# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize(("requests", "rank", "owners"), [(4, 0, [0]), (4, 7, []), (16, 3, [6, 7]), (13, 6, [12])])
def test_resident_plane_only_serves_its_first_local_request(monkeypatch, requests, rank, owners):
    from vllm_ascend.models.hy_v4_flash import index_pool_select as module

    queries, ratio, budget = 6, 4, 3
    inputs = torch.zeros(requests * queries, 1, 1)
    tables = torch.arange(requests, dtype=torch.int32).view(requests, 1)
    logical = torch.arange(queries).expand(requests, -1)
    destination = torch.arange(requests * 2)
    native, refreshed = [], []
    group = SimpleNamespace(size=lambda: 8)
    cache, dense = object(), object()
    resident = Mock()

    def refresh(pages, table, visible, writes, count):
        assert pages is cache and writes is destination and count == budget
        refreshed.append(int(table[0]))
        return dense

    resident.read_batch = refresh

    def score(q, w, visible, keys, max_groups, r, limit):
        assert keys is dense and r == ratio and limit == budget
        return torch.full((queries, (limit + 1) * r), refreshed[-1], dtype=torch.int32)

    def ordinary(q, w, pages, table, visible, page_size, max_groups, r, limit):
        native.append(int(table[0]))
        return torch.full((queries, (limit + 1) * r), native[-1], dtype=torch.int32)

    def gather(out, packet, *, group):
        out.fill_(-1)
        out[rank * len(packet) : (rank + 1) * len(packet)].copy_(packet)

    monkeypatch.setattr(module, "_score_native_keys", score)
    monkeypatch.setattr(module, "select_native_page_pool", ordinary)
    monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather)
    selected = module.select_request_partitioned_pool(
        inputs,
        inputs,
        cache,
        tables,
        logical,
        64,
        256,
        ratio,
        budget,
        group,
        rank,
        resident=resident,
        updated_slots=destination,
    )
    assert refreshed == owners[:1] and native == owners[1:]
    assert resident.invalidate.call_count == int(not owners)
    for request in owners:
        assert torch.all(selected[request] == request)
