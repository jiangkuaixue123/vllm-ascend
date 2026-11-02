# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention


def dense_index(layer, x, qr, positions, shared):
    """Unchunked eager formula with an out-of-place multiply, as before tiling."""
    count = shared.index_keys.shape[0]
    q = layer.indexer.wq_b(qr)[0].view(-1, layer.index_heads, layer.config.index_head_dim)
    layer.rotate(q.unsqueeze(0), positions)
    q = attention.fp4_quant_dequant(q, 32, scale_format="e8m0")
    scale = layer.config.index_head_dim**-0.5 * layer.config.index_n_heads**-0.5
    weights = layer.indexer.weights_proj(x)[0] * scale
    scores = (torch.einsum("shd,td->sht", q, shared.index_keys).relu_() * weights.unsqueeze(-1)).sum(1)
    if layer.tp > 1 and not layer.dsa_cp:
        scores = attention.tensor_model_parallel_all_reduce(scores)
    lengths = ((positions + 1) // layer.ratio).unsqueeze(-1)
    scores.masked_fill_(torch.arange(count) >= lengths, -torch.inf)
    if layer.layer_id == layer.config.candidate_source_layer_id:
        shared.candidates = attention.select_candidate_blocks(
            scores, lengths, layer.config.candidate_topk_blocks, layer.config.candidate_block_size
        )
    elif 0 <= layer.config.candidate_source_layer_id < layer.layer_id:
        scores.masked_fill_(~shared.candidates, -torch.inf)
    indices = attention.topk_columns(scores, min(layer.config.index_topk, count)).sort(-1).values
    return torch.where(indices < lengths, indices, -1)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dsa_cp", [False, True])
@pytest.mark.parametrize("stage", ["no_source", "before_source", "source", "after_source"])
@torch.inference_mode()
def test_tiled_index_exactly_matches_dense(ratio, dsa_cp, stage, monkeypatch):
    # Seven local query rows, starting mid-request. Two-row tiles leave an uneven tail.
    rows, heads, dim = 7, 2, 32
    positions = torch.arange(2, 2 + rows)
    count = (2 + rows) // ratio
    generator = torch.Generator(device="cpu").manual_seed(210)
    qr = torch.randn((rows, heads * dim), generator=generator, device="cpu").bfloat16().to(positions.device)
    x = torch.randn((rows, heads), generator=generator, device="cpu").bfloat16().to(positions.device)
    keys = torch.randn((count, dim), generator=generator, device="cpu").bfloat16().to(positions.device)
    keys[1::2] = keys[0]  # Exact ties can cross compressed blocks.
    source, layer_id = {"no_source": (-1, 2), "before_source": (2, 1), "source": (2, 2), "after_source": (2, 3)}[stage]
    layer = SimpleNamespace(
        owns_index=True,
        index_heads=heads,
        tp=2,
        dsa_cp=dsa_cp,
        layer_id=layer_id,
        ratio=ratio,
        config=SimpleNamespace(
            index_head_dim=dim,
            index_n_heads=heads if dsa_cp else heads * 2,
            index_topk=2,
            candidate_source_layer_id=source,
            candidate_topk_blocks=1,
            candidate_block_size=2,
        ),
        indexer=SimpleNamespace(wq_b=lambda v: (v.clone(), None), weights_proj=lambda v: (v, None)),
        frequencies=torch.linspace(0.01, 1, dim // 2),
        rope_dim=dim,
    )
    layer.rotate = lambda tensor, pos: attention.DeepseekV41Attention.rotate(layer, tensor, pos)
    mask = (torch.arange(rows)[:, None] + torch.arange(count)) % 3 != 0 if stage == "after_source" else None
    full = attention.SharedRequestAttention(compressed=keys, index_keys=keys, candidates=mask)
    tiled = attention.SharedRequestAttention(compressed=keys, index_keys=keys, candidates=mask)
    reduce = Mock(side_effect=lambda scores: scores * 2)
    monkeypatch.setattr(attention, "tensor_model_parallel_all_reduce", reduce)
    expected = dense_index(layer, x, qr, positions, full)
    reduce.reset_mock()
    monkeypatch.setattr(attention, "_INDEX_SCORE_TILE_BYTES", 2 * heads * max(dim * 4, count * qr.element_size()))
    einsum = Mock(wraps=torch.einsum)
    monkeypatch.setattr(torch, "einsum", einsum)
    actual = attention.DeepseekV41Attention._index(layer, x, qr, positions, tiled)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [args[1].shape[0] for args, _ in einsum.call_args_list] == [2, 2, 2, 1]
    assert reduce.call_count == (0 if dsa_cp else 4)
    if full.candidates is None:
        assert tiled.candidates is None
    else:
        torch.testing.assert_close(tiled.candidates, full.candidates, rtol=0, atol=0)
    layer.owns_index = False
    assert attention.DeepseekV41Attention._index(layer, x, qr, positions, tiled) is actual


def test_missing_shared_index_and_empty_keys():
    x = torch.zeros(3, 32)
    positions = torch.arange(3)
    shared = attention.SharedRequestAttention(compressed=x[:0], index_keys=x[:0])
    layer = SimpleNamespace(owns_index=False)
    with pytest.raises(RuntimeError, match="index source did not execute"):
        attention.DeepseekV41Attention._index(layer, x, x, positions, shared)
    layer.owns_index = True
    empty = attention.DeepseekV41Attention._index(layer, x, x, positions, shared)
    assert empty.shape == (3, 0) and empty.dtype == torch.long
