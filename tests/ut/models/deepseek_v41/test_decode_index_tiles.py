# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tensor_device", ["cpu", "npu"])
def test_decode_index_selects_across_key_tiles(monkeypatch, ratio, tensor_device):
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
    attention.DeepseekV41Attention._decode_index(layer, q, q.clone(), md.seq_lens - 1, md, runtime)

    query = attention.fp4_quant_dequant(q, 32, scale_format="e8m0")
    per_head = torch.einsum("bhd,btd->bht", query, keys).relu_()
    scores = (per_head * (weights * (dim**-0.5 * heads**-0.5)).unsqueeze(-1)).sum(1)
    lengths = torch.where(md.active, md.seq_lens // ratio, 0)[:, None]
    scores.masked_fill_(torch.arange(capacity, device=device) >= lengths, -torch.inf)
    expected = attention.topk_columns(scores, cfg.index_topk).sort(-1).values
    torch.testing.assert_close(runtime.decode_topk, torch.where(expected < lengths, expected, -1), rtol=0, atol=0)
    torch.testing.assert_close(runtime.decode_candidates, attention.select_candidate_blocks(scores, lengths, 2, 4))
    assert widths == [4, 4, 4, 4, 1]


def test_decode_index_empty_capacity_does_not_access_pages():
    layer = SimpleNamespace(owns_index=True, ratio=2)
    queries = torch.zeros(3, 128, device="cpu")
    metadata = SimpleNamespace(max_model_len=1)
    runtime = SimpleNamespace()
    attention.DeepseekV41Attention._decode_index(layer, queries, None, None, metadata, runtime)
    assert runtime.decode_topk.shape == (3, 0)
    assert runtime.decode_topk.dtype == torch.int64
