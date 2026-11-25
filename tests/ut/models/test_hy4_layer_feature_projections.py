# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


@pytest.mark.parametrize("tokens", [1, 6, 24, 48, 96])
def test_cross_layer_decode_reductions_preserve_row_and_layer_order(tokens, monkeypatch):
    from vllm_ascend.models.hy_v4_flash import engram

    layers = []
    for i in range(4):
        local = SimpleNamespace(head_start=0, head_end=2)
        layer = SimpleNamespace(
            channel_weights=torch.arange(4 * 3 * 4).reshape(4, 3, 4).float() * 0.0003 + i,
            channel_biases=torch.zeros(4, 3) + 0.25 * i,
            lookup_per_head_norms=[SimpleNamespace(weight=torch.ones(2)) for _ in range(2)],
            multi_head_embedding=local,
        )
        layer.normalize_lookup_rows = lambda hashes, rows, dtype, m=layer: engram.FlashEngram.normalize_lookup_rows(
            m, hashes, rows, dtype
        )
        layers.append(layer)
    hashes = torch.arange(4 * tokens * 2).view(4, tokens, 2)
    hashes[::2, ::3, 0] = -1
    rows = tuple(torch.arange(tokens * 4).reshape(tokens, 2, 2).float() * 0.01 + i for i in range(4))
    calls = []

    def reduce(tensor):
        calls.append(tensor.shape)
        return tensor + 1

    monkeypatch.setattr(engram, "tensor_model_parallel_all_reduce", reduce)
    got = engram.join_feature_projections(layers, hashes, rows)
    size = engram._LAYERS_PER_FEATURE_REDUCTION
    assert calls == [(size, tokens, 12)] * (4 // size)
    for layer_index, (result, head) in enumerate(zip(got, layers)):
        inputs = head.normalize_lookup_rows(hashes[layer_index], rows[layer_index], torch.float32)
        want = F.linear(inputs, head.channel_weights.flatten(0, 1), head.channel_biases.flatten()) + 1
        assert isinstance(result, engram.ProjectedFeatureRows)
        torch.testing.assert_close(result.channels, want)
