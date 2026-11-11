"""CPU regression for native W1/W3 ordering and preserving E2M1/E8M0 payload bytes."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.expert_weights import install_expert_weights


def matrices(n, k, code, scale):
    return SimpleNamespace(
        weight=torch.full((n, k // 2), code, dtype=torch.uint8),
        scale=torch.full((n, k // 32), scale, dtype=torch.uint8),
    )


def fixture():
    dim, hidden = 128, 64
    experts = [None, None]
    for i in range(2):
        experts.append(
            SimpleNamespace(
                w1=matrices(hidden, dim, 0xF0 + i, 0 + i),
                w3=matrices(hidden, dim, 0x07 + i, 254 + i),
                w2=matrices(dim, hidden, 0x88 + i, 126 + i),
            )
        )
    source = SimpleNamespace(experts=experts, experts_start_idx=2, experts_end_idx=4, dim=dim)
    layer = SimpleNamespace(
        expert_map=torch.tensor([-1, -1, 0, 1]),
        w13_weight=torch.empty(2, 2 * hidden, dim // 2, dtype=torch.uint8),
        w13_weight_scale=torch.empty(2, 2 * hidden, dim // 32, dtype=torch.float8_e8m0fnu),
        w2_weight=torch.empty(2, dim, hidden // 2, dtype=torch.uint8),
        w2_weight_scale=torch.empty(2, dim, hidden // 32, dtype=torch.float8_e8m0fnu),
    )
    return source, layer


def test_compact_expert_payload_and_placement():
    source, layer = fixture()
    receipt = install_expert_weights(layer, source, process=False)
    assert receipt["expert_range"] == [2, 4]
    for i, e in enumerate(source.experts[2:]):
        for target, field in (("weight", "weight"), ("weight_scale", "scale")):
            gate, up = getattr(layer, "w13_" + target)[i].view(torch.uint8).chunk(2)
            assert torch.equal(gate, getattr(e.w1, field))
            assert torch.equal(up, getattr(e.w3, field))
            assert torch.equal(getattr(layer, "w2_" + target)[i].view(torch.uint8), getattr(e.w2, field))


def test_refuse_other_rank_or_malformed_geometry():
    source, layer = fixture()
    layer.expert_map = torch.tensor([0, 1, -1, -1])
    with pytest.raises(ValueError, match="placement"):
        install_expert_weights(layer, source, process=False)
    _, layer = fixture()
    layer.w2_weight = layer.w2_weight[:, :-1]
    with pytest.raises(ValueError, match="layout"):
        install_expert_weights(layer, source, process=False)
