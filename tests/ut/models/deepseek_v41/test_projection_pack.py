# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep BF16 words and TP ownership identical across the exchange boundary."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.ops.triton.deepseek_v41_fp8_qdq import pack_prefill_projection_input


def reference(x, tp):
    send = x.reshape(x.shape[0], tp, 64 // tp, 512).transpose(0, 1).contiguous()
    send = send.reshape(-1, 8 // tp, 4096)
    return attention._quantize_projection_input(send).view(-1, 64 // tp, 512)


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("rows", [257, 513, 2040])
@torch.inference_mode()
def test_prefill_pack_exact_words_and_layout(rows, tp):
    n = rows * 64 * 512
    for pattern in ("finite", "words"):
        if pattern == "words":
            words = torch.arange(65536, dtype=torch.int32).short()
            x = words.repeat((n + 65535) // 65536)[:n].view(torch.bfloat16).reshape(rows, 64, 512).npu()
        else:
            x = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device="npu")
            x[0, 0, :8] = torch.tensor([0.0, -0.0, 448, -448, 1.0625, 1.1875, 2**-10, -3 * 2**-10]).npu()
        before = x.clone()
        expected = reference(x, tp)
        actual = pack_prefill_projection_input(x, tp)
        assert actual is not None and actual.shape == expected.shape
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(x.view(torch.int16), before.view(torch.int16))


@torch.inference_mode()
def test_prefill_pack_graph_changes_input():
    x = torch.randn(257, 64, 512, dtype=torch.bfloat16, device="npu")
    pack_prefill_projection_input(x, 4)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = pack_prefill_projection_input(x, 4)
    for step in range(8):
        x.copy_(torch.randn_like(x) * (step + 1))
        expected = reference(x, 4)
        graph.replay()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("case", ["decode", "strided", "dtype", "tp", "batch_invariant"])
@torch.inference_mode()
def test_prefill_pack_gate(case):
    x = torch.empty(257, 64, 512, dtype=torch.bfloat16, device="npu")
    tp = 4
    if case == "decode":
        x = x[:48]
    elif case == "strided":
        x = torch.empty(257, 64, 1024, dtype=x.dtype, device=x.device)[..., ::2]
    elif case == "dtype":
        x = x.float()
    elif case == "tp":
        tp = 2
    with patch("vllm_ascend.ops.triton.deepseek_v41_fp8_qdq.envs.VLLM_BATCH_INVARIANT", case == "batch_invariant"):
        assert pack_prefill_projection_input(x, tp) is None


@pytest.mark.parametrize("tp", [4, 8])
@torch.inference_mode()
def test_projection_consumer_preserves_bf16_result(tp):
    rows, groups, rank = 257, 8 // tp, 32
    x = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device="npu")
    a = torch.randn(groups * rank, 4096, dtype=x.dtype, device=x.device) / 64
    b = torch.randn(5120, groups * rank, dtype=x.dtype, device=x.device) / 8
    group = SimpleNamespace(device_group=object(), reduce_scatter=lambda value, dim: value.chunk(tp, dim)[1])
    layer = SimpleNamespace(
        dsa_cp=True,
        tp=tp,
        local_heads=64 // tp,
        dim=512,
        groups=groups,
        config=SimpleNamespace(o_lora_rank=rank),
        sequence_group=group,
        wo_a=SimpleNamespace(weight=a),
        wo_b=SimpleNamespace(quant_method=SimpleNamespace(apply=lambda layer, value, bias: value @ b.T)),
    )

    def exchange(out, send, **kwargs):
        assert kwargs["group"] is group.device_group
        # Four/eight identical producers, receiver 1 owns its consecutive head slice.
        out.copy_(send.chunk(tp, 0)[1].repeat(tp, 1, 1))

    with patch.object(attention.dist, "all_to_all_single", exchange):
        with patch.object(attention, "pack_prefill_projection_input", None):
            expected = attention.DeepseekV41Attention._project_output(layer, x, x.dtype)
        actual = attention.DeepseekV41Attention._project_output(layer, x, x.dtype)
        packed = pack_prefill_projection_input(x, tp)
        direct = attention.DeepseekV41Attention._project_output(layer, packed, x.dtype, prequantized=True)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(direct.view(torch.int16), expected.view(torch.int16))
