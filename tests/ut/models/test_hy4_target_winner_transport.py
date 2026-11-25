# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize(
    ("requests", "rows", "capacity", "copy_available", "uses_copy"),
    [
        (1, 6, 6, True, True),
        (4, 15, 24, True, True),
        (8, 34, 48, True, True),
        (8, 34, 24, True, False),
        (4, 17, 24, False, False),
    ],
)
def test_target_packets_follow_decode_capacity(monkeypatch, requests, rows, capacity, copy_available, uses_copy):
    from vllm_ascend.models.hy_v4_flash import greedy_sampling as sample

    base, copy = object(), object()
    mapping = torch.arange(requests)
    row_ids = torch.arange(rows)
    batch = SimpleNamespace(
        input_ids=row_ids,
        positions=row_ids,
        logits_indices=row_ids,
        expanded_idx_mapping=row_ids,
        expanded_local_pos=row_ids,
        cu_num_logits=torch.zeros(requests + 1),
        seq_lens=mapping,
        idx_mapping=mapping,
        num_reqs=requests,
    )
    vocabulary = SimpleNamespace(
        project=lambda x: x, start=0, max_decode_tokens=capacity, copy_group=copy if copy_available else None
    )
    runner = SimpleNamespace(
        sampler=SimpleNamespace(req_states=SimpleNamespace(prefill_len=SimpleNamespace(gpu=mapping))),
        rejection_sampler=SimpleNamespace(num_speculative_steps=5),
    )
    choose = Mock(return_value=(row_ids, mapping))
    monkeypatch.setattr(sample, "choose", choose)
    monkeypatch.setattr(sample, "gather_draft_sampled", lambda *args: (row_ids, None))
    monkeypatch.setattr(sample, "get_tp_group", lambda: SimpleNamespace(device_group=base, world_size=8))
    monkeypatch.setattr(sample, "get_num_sampled_and_rejected", lambda *args: (mapping, mapping))
    sample.sample(runner, torch.zeros(rows, 2), batch, vocabulary)
    assert choose.call_args.args[1] is (copy if uses_copy else base)
