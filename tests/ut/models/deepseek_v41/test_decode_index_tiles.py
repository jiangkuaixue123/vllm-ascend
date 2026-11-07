# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tensor_device", ["cpu", "npu"])
@pytest.mark.parametrize("precomputed", [False, True])
def test_decode_index_selects_across_key_tiles(monkeypatch, ratio, tensor_device, precomputed):
    if tensor_device == "npu" and not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    device = torch.device(tensor_device)
    rows, heads, dim, capacity = 3, 2, 32, 17
    generator = torch.Generator(device="cpu").manual_seed(197)
    keys = torch.randn(rows, capacity, dim, generator=generator).bfloat16().to(device)
    keys[:, 1::2] = keys[:, :1]  # Tie scores across page and tile boundaries.
    q = torch.randn(rows, heads, dim, generator=generator).bfloat16().to(device)
    weights = torch.randn(rows, heads, generator=generator).to(device)
    md = SimpleNamespace(
        active=torch.tensor([True, False, True], device=device),
        seq_lens=torch.tensor([9 * ratio, 0, capacity * ratio], device=device),
        max_model_len=capacity * ratio + ratio - 1,
    )
    widths = []

    def read(metadata, positions, bank, offset, width):
        assert metadata is md and (bank, offset, width) == (1, dim, dim)
        widths.append(positions.shape[-1])
        return keys.gather(1, (positions // ratio).unsqueeze(-1).expand(-1, -1, dim))

    cfg = SimpleNamespace(
        index_head_dim=dim,
        index_n_heads=heads,
        candidate_source_layer_id=2,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        index_topk=5,
    )
    layer = SimpleNamespace(
        owns_index=True,
        ratio=ratio,
        layer_id=2,
        dim=dim,
        config=cfg,
        tp=1,
        index_heads=heads,
        rotate=lambda *args: None,
        indexer=SimpleNamespace(wq_b=lambda v: (v, None), weights_proj=lambda v: (weights, None)),
    )
    runtime = SimpleNamespace(source_cache=SimpleNamespace(decode_read=read), source_decode=md)
    monkeypatch.setattr(attention, "_DECODE_INDEX_TILE_KEYS", 4)
    prepared = None
    if precomputed:
        prepared = (
            attention.fp4_quant_dequant(q.clone(), 32, scale_format="e8m0"),
            weights * (dim**-0.5 * heads**-0.5),
        )
    attention.DeepseekV41Attention._decode_index(layer, q, q.clone(), md.seq_lens - 1, md, runtime, prepared)

    query = attention.fp4_quant_dequant(q, 32, scale_format="e8m0")
    per_head = torch.einsum("bhd,btd->bht", query, keys).relu_()
    scores = (per_head * (weights * (dim**-0.5 * heads**-0.5)).unsqueeze(-1)).sum(1)
    lengths = torch.where(md.active, md.seq_lens // ratio, 0)[:, None]
    scores.masked_fill_(torch.arange(capacity, device=device) >= lengths, -torch.inf)
    expected = attention.topk_columns(scores, cfg.index_topk).sort(-1).values
    torch.testing.assert_close(runtime.decode_topk, torch.where(expected < lengths, expected, -1), rtol=0, atol=0)
    expected_mask = attention.select_candidate_blocks(scores, lengths, 2, 4)
    torch.testing.assert_close(runtime.decode_candidate_blocks, expected_mask[..., ::4])
    actual_scores = scores.clone()
    attention.mask_candidate_columns_(
        actual_scores, runtime.decode_candidate_blocks, cfg.candidate_block_size, runtime.decode_candidate_width
    )
    torch.testing.assert_close(actual_scores, scores.masked_fill(~expected_mask, -torch.inf), rtol=0, atol=0)
    assert widths == [4, 4, 4, 4, 1]


@pytest.mark.parametrize(("width", "block_size", "budget"), [(0, 8, 0), (1, 8, 1), (17, 8, 2), (32, 8, 0), (32, 8, 9)])
def test_block_candidate_mask_broadcast_preserves_follower_bits(width, block_size, budget):
    # The pinned last block need not have a high score; two other equal blocks
    # must still choose the earlier column at the cutoff.
    lengths = torch.tensor([[0], [min(width, 9)], [width]])
    source = torch.ones(3, width)
    if width:
        source[:, -1] = -5
    source.masked_fill_(torch.arange(width) >= lengths, -torch.inf)
    dense = attention.select_candidate_blocks(source, lengths, budget, block_size)
    compact = attention.select_candidate_blocks(source, lengths, budget, block_size, per_token=False)
    torch.testing.assert_close(compact, dense[..., ::block_size])
    follower = torch.randn(3, width)
    if width >= 4:
        follower[:, :4] = torch.tensor([-0.0, torch.inf, -torch.inf, torch.nan])
    expected = follower.masked_fill(~dense, -torch.inf)
    attention.mask_candidate_columns_(follower, compact, block_size, width)
    # Floating-point equality does not distinguish -0 from +0 or NaN payloads.
    torch.testing.assert_close(follower.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)


def test_block_mask_keeps_original_cross_width_broadcast():
    keep = torch.tensor([[True], [False]])
    follower = torch.arange(30).view(2, 15).float()
    expected = follower.masked_fill(~keep, -torch.inf)
    attention.mask_candidate_columns_(follower, keep, block_size=8, source_width=1)
    torch.testing.assert_close(follower, expected)
    with pytest.raises(RuntimeError):
        attention.mask_candidate_columns_(follower, keep, block_size=8, source_width=7)


def test_decode_index_empty_capacity_does_not_access_pages():
    layer = SimpleNamespace(owns_index=True, ratio=2)
    queries = torch.zeros(3, 128, device="cpu")
    metadata = SimpleNamespace(max_model_len=1)
    runtime = SimpleNamespace()
    attention.DeepseekV41Attention._decode_index(layer, queries, None, None, metadata, runtime)
    assert runtime.decode_topk.shape == (3, 0)
    assert runtime.decode_topk.dtype == torch.int64


@pytest.mark.parametrize(
    ("source_ratio", "source_limit", "follower_ratio", "follower_limit", "can_broadcast"),
    [
        (1, 17, 1, 17, True),
        (2, 35, 2, 35, True),
        (1, 17, 2, 35, True),
        (2, 35, 1, 17, True),
        (2, 3, 1, 17, True),
        (1, 17, 2, 17, False),
        (2, 17, 1, 17, False),
    ],
)
def test_decode_candidate_chain_matches_dense_ratio_contract(
    source_ratio, source_limit, follower_ratio, follower_limit, can_broadcast
):
    generator = torch.Generator().manual_seed(209)
    rows, heads, dim, block, budget = 3, 2, 32, 4, 2
    query = torch.randn(rows, heads, dim, generator=generator).bfloat16()
    weights = torch.randn(rows, heads, generator=generator).bfloat16()
    keys = torch.randn(rows, max(source_limit, follower_limit), dim, generator=generator).bfloat16()
    keys[0].zero_()

    def read(_metadata, positions, _bank, _offset, _width):
        return keys.gather(1, positions.unsqueeze(-1).expand(-1, -1, dim))

    config = SimpleNamespace(
        candidate_source_layer_id=20,
        candidate_block_size=block,
        candidate_topk_blocks=budget,
        index_topk=5,
        index_n_heads=heads,
        index_head_dim=dim,
    )
    runtime = SimpleNamespace(source_cache=SimpleNamespace(decode_read=read))
    dense = None
    for ordinal, layer_id in enumerate((20, 24, 28, 32, 36)):
        ratio, limit = (source_ratio, source_limit) if ordinal == 0 else (follower_ratio, follower_limit)
        capacity = limit // ratio
        metadata = SimpleNamespace(
            active=torch.tensor([True, False, True]),
            seq_lens=torch.tensor([limit, 0, max(1, limit - 5)]),
            max_model_len=limit,
        )
        runtime.source_decode = metadata
        layer = SimpleNamespace(
            config=config,
            owns_index=True,
            layer_id=layer_id,
            ratio=ratio,
            dim=dim,
            tp=1,
            index_heads=heads,
            rotate=lambda *args: None,
            indexer=SimpleNamespace(wq_b=lambda x: (x, None), weights_proj=lambda x: (weights, None)),
        )
        q = query.roll(ordinal, dims=0)
        columns = torch.arange(capacity)
        ends = (columns + 1) * ratio - 1
        quant = attention.fp4_quant_dequant(q, 32, scale_format="e8m0")
        dots = torch.einsum("bhd,btd->bht", quant, keys[:, ends]).relu_()
        scores = (dots * (weights * (dim**-0.5 * heads**-0.5)).unsqueeze(-1)).sum(1)
        lengths = torch.where(metadata.active, metadata.seq_lens // ratio, 0)[:, None]
        scores.masked_fill_(columns >= lengths, -torch.inf)
        if ordinal and not can_broadcast:
            with pytest.raises(RuntimeError):
                scores.masked_fill_(~dense, -torch.inf)
            with pytest.raises(RuntimeError):
                attention.DeepseekV41Attention._decode_index(layer, q, q, metadata.seq_lens - 1, metadata, runtime)
            return
        if ordinal:
            scores.masked_fill_(~dense, -torch.inf)
        else:
            dense = attention.select_candidate_blocks(scores, lengths, budget, block)
        indices = attention.topk_columns(scores, min(config.index_topk, capacity)).sort(-1).values
        expected = torch.where(indices < lengths, indices, -1)
        attention.DeepseekV41Attention._decode_index(layer, q, q, metadata.seq_lens - 1, metadata, runtime)
        assert torch.equal(runtime.decode_topk, expected)
        assert runtime.decode_candidate_width == source_limit // source_ratio
        assert torch.equal(runtime.decode_candidate_blocks, dense[..., ::block])
