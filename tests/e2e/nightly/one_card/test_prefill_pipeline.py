# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 large-prefill producer/consumer equivalence, including BF16 boundaries."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.hc_prefill import hc_prefill_post, hc_prefill_prepare
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Attention
from vllm_ascend.models.deepseek_v41.core import hc_mixes, hc_post, hc_pre
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Block
from vllm_ascend.ops.triton.deepseek_v41_fp8_qdq import (
    pack_prefill_projection_input,
    pack_rotated_prefill_output,
)


def exact(actual, expected):
    assert actual is not None and actual.shape == expected.shape and actual.dtype == expected.dtype
    word = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.view(word), expected.view(word))


@pytest.mark.parametrize("rows", [257, 513, 2040])
@torch.inference_mode()
def test_hc_prefill_complete_chain(rows):
    torch.manual_seed(9041)
    x = torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device="npu")
    previous = torch.rand(rows, 4, device="npu")
    fn = torch.randn(24, 20480, device="npu") / 256
    scale = torch.rand(3, device="npu")
    base = torch.rand(24, device="npu")
    block = SimpleNamespace(mixes=lambda value, name: hc_mixes(value, fn, scale, base))
    runtime = SimpleNamespace(decode=None)
    for _ in range(2):
        before = x.clone()
        expected_pre = hc_pre(x, previous)
        expected_gates = hc_mixes(x, fn, scale, base)
        prepared, *gates = DeepseekV41Block.pre_and_mixes(block, x, previous, "attn", runtime)
        exact(prepared, expected_pre)
        for actual, expected in zip(gates, expected_gates):
            exact(actual, expected)
        sub = torch.randn(rows, 5120, dtype=x.dtype, device=x.device)
        expected = hc_post(sub, x, *expected_gates[1:])
        actual = DeepseekV41Block.post_residual(block, sub, x, *gates[1:], runtime)
        exact(actual, expected)
        exact(x, before)
        x, previous = actual, gates[0]


@torch.inference_mode()
def test_hc_prefill_cancellation_and_nonfinite():
    rows = 257
    x = torch.zeros(rows, 4, 5120, dtype=torch.bfloat16, device="npu")
    pre = torch.ones(rows, 4, device="npu")
    comb = torch.ones(rows, 4, 4, device="npu")
    sub = torch.zeros(rows, 5120, dtype=x.dtype, device=x.device)
    for pattern in ((2**26, 1, -(2**26), 1), (-0.0,) * 4, (float("inf"), 1, -float("inf"), 1)):
        x.copy_(torch.tensor(pattern, dtype=x.dtype, device=x.device)[None, :, None])
        mixed, states = hc_prefill_prepare(x, pre)
        exact(mixed, hc_pre(x, pre))
        exact(states, x.float())
        exact(hc_prefill_post(sub, x, pre, comb), hc_post(sub, x, pre, comb))


def layer():
    return SimpleNamespace(rope_dim=64, frequencies=(10000.0 ** (-torch.arange(0, 64, 2).float() / 64)).npu())


def rotate_and_pack(x, positions, tp, rotary):
    reference = x.clone()
    DeepseekV41Attention.rotate(rotary, reference[None], positions, inverse=True)
    return pack_prefill_projection_input(reference, tp)


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("pattern", ["finite", "words"])
@pytest.mark.parametrize("rows", [513, 2040])
@torch.inference_mode()
def test_rotary_qdq_send_spans(tp, pattern, rows):
    total, start = rows + 88, 37
    if pattern == "words":
        words = torch.arange(65536, dtype=torch.int32).short()
        x = words.repeat(rows // 2 + 1)[: rows * 64 * 512].view(torch.bfloat16).reshape(rows, 64, 512).npu()
        rotary_words = words.repeat((rows * 64 * 64 + 65535) // 65536)[: rows * 64 * 64]
        x[..., 448:] = rotary_words.view(torch.bfloat16).reshape(rows, 64, 64).npu()
    else:
        x = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(rows, device="npu") * 97
    rotary = layer()
    before = x.clone()
    expected = x.new_zeros((tp, total, 64 // tp, 512))
    expected[:, start : start + rows] = rotate_and_pack(x, positions, tp, rotary).view(tp, rows, 64 // tp, 512)
    actual = torch.zeros_like(expected).view(tp * total, 64 // tp, 512)
    for begin, end in ((0, 13), (13, 257), (257, rows)):
        pack_rotated_prefill_output(x[begin:end], positions[begin:end], rotary.frequencies, actual, start + begin, tp)
    exact(actual.view_as(expected), expected)
    exact(x, before)


@torch.inference_mode()
def test_prefill_producers_graph_refresh():
    rows, tp = 257, 4
    x = torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device="npu")
    pre = torch.rand(rows, 4, device="npu")
    sub = torch.randn(rows, 5120, dtype=x.dtype, device=x.device)
    comb = torch.rand(rows, 4, 4, device="npu")
    attended = torch.randn(rows, 64, 512, dtype=x.dtype, device=x.device)
    positions = torch.arange(rows, device="npu")
    rotary = layer()
    out = attended.new_zeros((tp * rows, 16, 512))

    def run():
        y, states = hc_prefill_prepare(x, pre)
        z = hc_prefill_post(sub, x, pre, comb)
        pack_rotated_prefill_output(attended, positions, rotary.frequencies, out, 0, tp)
        return y, states, z

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        y, states, z = run()
    for step in range(3):
        for tensor in (x, pre, sub, comb, attended):
            tensor.copy_(torch.randn_like(tensor) * (step + 1))
        positions.add_(513)
        graph.replay()
        exact(y, hc_pre(x, pre))
        exact(states, x.float())
        exact(z, hc_post(sub, x, pre, comb))
        exact(out, rotate_and_pack(attended, positions, tp, rotary))


@torch.inference_mode()
def test_hc_small_and_cpu_fallback():
    for device, rows in (("cpu", 257), ("npu", 48)):
        x = torch.zeros(rows, 4, 5120, dtype=torch.bfloat16, device=device)
        pre = torch.ones(rows, 4, device=device)
        assert hc_prefill_prepare(x, pre) is None
        assert hc_prefill_post(x[:, 0].contiguous(), x, pre, torch.ones(rows, 4, 4, device=device)) is None
