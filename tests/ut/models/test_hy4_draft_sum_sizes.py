# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock


def test_checked_sizes_are_shared_by_matching_draft_projections(monkeypatch):
    from vllm_ascend.models.hy_v4_flash import draft_projection_reduce as sums

    def projection():
        return SimpleNamespace(custom_op=None)

    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(o_proj=projection()), mlp=SimpleNamespace(down_proj=projection()))
        for _ in range(2)
    ]
    correction = projection()
    draft = SimpleNamespace(layers=layers, hidden_correction=SimpleNamespace(down_proj=correction))
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=48),
        compilation_config=SimpleNamespace(decode_cudagraph_capture_request_sizes=[1, 4, 6, 8, 16, 128]),
    )
    check = Mock(side_effect=lambda rows, *args: None if rows == 4 else rows)
    monkeypatch.setattr(sums, "get_tp_group", object)
    monkeypatch.setattr(sums, "ordinary_projection", lambda layer: True)
    monkeypatch.setattr(sums, "check_plan", check)
    monkeypatch.setattr(sums, "DraftRowParallelOp", lambda layer, reducers: reducers)
    sums.configure_draft_sums(draft, object(), config, 6)
    limit = min(48, sums._MAX_ORDERED_ROWS)
    eligible = lambda sizes: {r for r in sizes if r <= limit}
    before = eligible({1, 4, 6, 8, 16})
    verify = eligible({6, 24, 36, 48})
    assert [c.args[0] for c in check.call_args_list] == sorted(before | verify)
    assert set(correction.custom_op) == before - {4}
    for layer in layers:
        assert set(layer.self_attn.o_proj.custom_op) == verify
        assert layer.self_attn.o_proj.custom_op is layer.mlp.down_proj.custom_op
