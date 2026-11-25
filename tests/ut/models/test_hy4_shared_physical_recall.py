# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch


def test_paired_mapping_is_local_to_rows_forward_and_exact_table():
    from vllm_ascend.models.hy_v4_flash.layer_recall import recall_rows_for_forward

    metadata = object()
    table = torch.arange(6).reshape(2, 3)
    context = SimpleNamespace(additional_kwargs={})

    def populate(ctx):
        _, rows = recall_rows_for_forward(ctx, metadata, (4, True), 2, 6, 4, 511)
        for i in range(2):
            rows.append(torch.tensor([i, i + 1]))
        return rows

    owner = populate(context)
    shared, _ = recall_rows_for_forward(context, metadata, (4, False), 2, 6, 4, 511)
    calls = []

    def resolve(blocks, rows):
        calls.append((blocks, rows))
        return torch.stack(rows) + blocks[:, :1], torch.ones(len(rows))

    first = owner.physical_for(table, resolve)
    assert shared.physical_for(table, resolve) is first
    view = table.view_as(table)
    assert shared.physical_for(view, resolve) is not first
    assert calls[0][0] is table and calls[1][0] is view
    other = populate(SimpleNamespace(additional_kwargs={}))
    assert other.physical_for(view, resolve) is not shared.physical_for(view, resolve)
    assert len(calls) == 3 and calls[-1][1] is other
