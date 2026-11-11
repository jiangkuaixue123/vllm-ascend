# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the actual A3 paged score op with B's BF16 indexer arithmetic under changing ACLGraph inputs."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Cache, select_candidate_blocks, topk_columns
from vllm_ascend.models.deepseek_v41.attention_ops import fp4_quant_dequant
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

if HAS_TRITON:
    from vllm_ascend.ops.triton.deepseek_v41_paged_index_score import (
        paged_index_scores,
        supports_grouped_index_scores,
        supports_paged_index_scores,
    )


@pytest.mark.skipif(not HAS_TRITON, reason="Requires Ascend Triton")
@pytest.mark.parametrize(
    "rows,ratio,group_start",
    [(6, 2, None), (12, 1, None)]
    + [(rows, ratio, start) for rows, start in ((3, 0), (3, 3), (6, 0), (12, 0)) for ratio in (1, 2)],
)
@torch.inference_mode()
def test_paged_index_score_dynamic_replay(rows, ratio, group_start):
    if get_ascend_device_type() != AscendDeviceType.A3:
        pytest.skip("A3 only")
    init_device_properties_triton()
    assert enable_custom_op()
    gen = torch.Generator().manual_seed(83)
    limit = 153600 if group_start is not None and (rows, ratio) in ((3, 2), (12, 1)) else 16384
    block, dim, heads = 128, 128, 32
    capacity = limit // ratio
    field = torch.empty(limit * 2, 1024, dtype=torch.bfloat16, device="npu")[:, 512:640]
    latent = fp4_quant_dequant(torch.randn(len(field), dim, generator=gen).bfloat16().npu(), 32)
    field.copy_(latent)
    q = fp4_quant_dequant(torch.randn(rows, heads, dim, generator=gen).bfloat16().npu(), 32)
    q[0].zero_()  # Cutoff ties across physical pages.
    weights = torch.randn(rows, heads, generator=gen).bfloat16().npu()
    table = torch.zeros(rows, limit // block, dtype=torch.int32, device="npu")
    src = SimpleNamespace(
        active=torch.zeros(rows, dtype=torch.bool, device="npu"),
        seq_lens=torch.zeros(rows, dtype=torch.int64, device="npu"),
        block_table_tensor=table,
        max_model_len=limit,
    )
    dst = SimpleNamespace(active=src.active.clone(), seq_lens=src.seq_lens.clone(), max_model_len=limit)
    if group_start is not None:
        src.query_width = dst.query_width = 6
        src.grouped_row_start = dst.grouped_row_start = group_start
        dst.seq_lens, dst.active = src.seq_lens, src.active
    assert supports_grouped_index_scores(q, src, dst) == (group_start is not None)
    assert supports_paged_index_scores(q, field, weights, src, dst, ratio, block)
    cache = SimpleNamespace(block_size=block, field=lambda *args: field)
    cols = torch.arange(capacity, device="npu")[None, :]
    ends = ((cols + 1) * ratio - 1).expand(rows, -1)

    def original():
        pieces = []
        for tile in ends.split(8192, dim=1):
            keys = DeepseekV41Cache.decode_read(cache, src, tile, 1, 512, 128)
            dots = torch.einsum("bhd,btd->bht", q, keys)
            pieces.append((dots.relu_() * weights.unsqueeze(-1)).sum(1))
        return torch.cat(pieces, 1).masked_fill_(
            ~dst.active[:, None] | (cols >= dst.seq_lens[:, None] // ratio), -torch.inf
        )

    def selected(scores):
        lens = torch.where(dst.active, dst.seq_lens // ratio, 0)[:, None]
        ids = topk_columns(scores, 512).sort(-1).values
        return scores.view(torch.int16), torch.where(ids < lens, ids, -1), select_candidate_blocks(scores, lens, 64, 8)

    def fused():
        return selected(paged_index_scores(q, field, weights, src, dst, ratio, block))

    fused()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = fused()
    for step, length in enumerate((0, 127, 129, 1029, 2050, limit, 513, 0)):
        tpl = torch.randperm(limit * 2 // block, generator=gen)[: limit // block].expand(rows, -1).clone().int()
        if group_start is not None:
            for begin in range(6, rows, 6):
                tpl[begin : begin + 6] = torch.randperm(limit * 2 // block, generator=gen)[: limit // block].int()
        tpl[:, 0] = 0
        if step == 4:
            tpl[:, 1] = -1
            tpl[:, 2] = limit * 2 // block
        table.copy_(tpl)
        lengths = (torch.arange(rows) % 6 + length - 5).clamp_min(0)
        dst.seq_lens.copy_(lengths)
        src.seq_lens.copy_(lengths if group_start is not None else (lengths - 7).clamp_min(0))
        dst.active.copy_(lengths > 0)
        src.active.copy_(lengths > 0)
        src.active[1] = False
        dst.active[2] = False
        eager = fused()
        ref = selected(original())
        graph.replay()
        torch.npu.synchronize()
        for part, (x, y, z) in enumerate(zip(out, eager, ref)):
            torch.testing.assert_close(
                x,
                y,
                rtol=0,
                atol=0,
                msg=lambda message, step=step, part=part: f"step={step}, part={part}, graph/eager\n{message}",
            )
            torch.testing.assert_close(
                x,
                z,
                rtol=0,
                atol=0,
                msg=lambda message, step=step, part=part: f"step={step}, part={part}, graph/reference\n{message}",
            )


@pytest.mark.skipif(not HAS_TRITON, reason="Requires Ascend Triton")
@pytest.mark.parametrize("changed", ["query_width", "grouped_row_start", "seq_lens", "active", "dtype"])
def test_grouped_index_requires_matching_source_inputs(changed):
    q = torch.zeros(3, 32, 128)
    md = dict(
        query_width=6,
        grouped_row_start=3,
        seq_lens=torch.ones(3, dtype=torch.int64),
        active=torch.ones(3, dtype=torch.bool),
    )
    src, dst = SimpleNamespace(**md), SimpleNamespace(**md)
    assert supports_grouped_index_scores(q, src, dst)
    if changed == "dtype":
        src.seq_lens = src.seq_lens.view(torch.int32)[:3]
    else:
        value = getattr(src, changed)
        setattr(src, changed, value.clone() if isinstance(value, torch.Tensor) else value + 1)
    assert not supports_grouped_index_scores(q, src, dst)


@pytest.mark.parametrize("tensor_device", ["cpu", "npu"])
@pytest.mark.parametrize("rows,tp", [(3, 8), (6, 4), (6, 8), (12, 4)])
@torch.inference_mode()
def test_paged_decode_dispatch_matches_reference(tensor_device, rows, tp):
    if tensor_device == "npu" and (
        not torch.npu.is_available()
        or attention.paged_index_scores is None
        or attention.get_ascend_device_type() != attention.AscendDeviceType.A3
    ):
        pytest.skip("Requires A3 and Ascend Triton")
    if tensor_device == "npu":
        init_device_properties_triton()
    heads, dim, block = 32, 128, 128
    gen = torch.Generator(device="cpu").manual_seed(42)

    def values(shape):
        return torch.randn(shape, generator=gen, device="cpu").bfloat16().to(tensor_device)

    q = values((rows, heads, dim))
    weights = values((rows, heads))
    field = values((block * 2, dim * 2))[:, dim:]
    field.copy_(attention.fp4_quant_dequant(field.contiguous(), 32, scale_format="e8m0"))
    cache = SimpleNamespace(block_size=block, field=lambda *args: field)
    cache.decode_read = lambda *args: attention.DeepseekV41Cache.decode_read(cache, *args)
    md = SimpleNamespace(
        active=torch.tensor([True, True, False, True, True, True] * 2, device=tensor_device)[:rows],
        seq_lens=torch.tensor([1, 27, 0, 128, 133, 255] * 2, device=tensor_device)[:rows],
        max_model_len=block * 2,
        block_table_tensor=torch.tensor([[1, 0]] * rows, dtype=torch.int32, device=tensor_device),
        query_width=6,
        grouped_row_start=3 if rows == 3 else 0,
    )
    runtime = SimpleNamespace(source_cache=cache, source_decode=md)
    cfg = SimpleNamespace(
        index_head_dim=dim,
        index_n_heads=heads,
        candidate_source_layer_id=0,
        candidate_topk_blocks=4,
        candidate_block_size=8,
        index_topk=16,
    )
    layer = SimpleNamespace(
        owns_index=True,
        ratio=2,
        layer_id=0,
        dim=512,
        tp=tp,
        dsa_cp=True,
        index_heads=heads,
        config=cfg,
        rotate=lambda *args: None,
        indexer=SimpleNamespace(wq_b=lambda x: (x, None), weights_proj=lambda x: (weights, None)),
    )

    def run():
        outputs = []
        for layer_id in (0, 4, 8, 12, 16):
            layer.layer_id = layer_id
            layer_query = q.roll(layer_id // 4, dims=0)
            attention.DeepseekV41Attention._decode_index(layer, layer_query, layer_query, md.seq_lens - 1, md, runtime)
            if layer_id == 0:
                assert runtime.decode_candidate_width == md.max_model_len // layer.ratio
                outputs.append(runtime.decode_candidate_blocks)
            outputs.append(runtime.decode_topk)
        return outputs

    with patch.object(attention, "paged_index_scores", None):
        expected = run()
    with patch.object(attention, "paged_index_scores", wraps=attention.paged_index_scores) as fused:
        actual = run()
        assert fused.call_count == 5 * int(tensor_device == "npu")
    for found, wanted in zip(actual, expected):
        assert torch.equal(found, wanted)
