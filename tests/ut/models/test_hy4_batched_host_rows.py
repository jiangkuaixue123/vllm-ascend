# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch


def test_hy4_registered_rows_use_request_local_boundaries(monkeypatch):
    from vllm_ascend.models.hy_v4_flash import model_state

    tables = [SimpleNamespace(host_weight=object()) for _ in range(4)]
    engram = SimpleNamespace(
        layout=SimpleNamespace(lookup_table=object()),
        multipliers=torch.ones(4, 3),
        primes=torch.ones(4, 2, 8),
    )
    model = SimpleNamespace(_flash_model=SimpleNamespace(engram_hash=engram), _engram_embeddings=tables)
    token_ids = torch.zeros(24, dtype=torch.int32)
    boundaries = torch.tensor([0, 2, 2, 7, 7], dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=3,
        num_reqs_after_padding=4,
        input_ids=token_ids,
        query_start_loc=boundaries,
        query_start_loc_np=boundaries.numpy(),
    )
    history = torch.zeros(3, 2, dtype=torch.int32)
    hashes = torch.full((4, 32, 16), -7, dtype=torch.int32)[:, :24]
    observed = []
    reads = []
    monkeypatch.setattr(model_state, "fill_verify_hashes", lambda *args: observed.append(args))
    monkeypatch.setattr(model_state, "fill_host_feature_rows", lambda *args: reads.append(args))
    monkeypatch.setattr(model_state, "supports_grouped_feature_rows", lambda *args: False)
    prepare = model_state.FlashModelState._prepare_registered_features
    assert prepare(model, batch, history, hashes, [object() for _ in tables])
    assert observed[0][1] is token_ids and observed[0][2] is history
    assert observed[0][3] == 7 and observed[0][-1] is boundaries
    assert len(reads) == 4
    grouped = []
    monkeypatch.setattr(model_state, "supports_grouped_feature_rows", lambda *args: True)
    monkeypatch.setattr(model_state, "fill_grouped_feature_rows", lambda *args: grouped.append(args))
    outputs = [object() for _ in tables]
    assert prepare(model, batch, history, hashes, outputs)
    assert len(reads) == 4 and len(grouped) == 1
    assert grouped[0][0] is tables and grouped[0][1] is hashes and grouped[0][2] is outputs
    tables[0].host_weight = None
    assert not prepare(model, batch, history, hashes, [object() for _ in tables])
