# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-layer phase ownership and exact inverse-RoPE/FP8/TP send fusion."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Attention
from vllm_ascend.ops.triton.deepseek_v41_fp8_qdq import (
    pack_decode_rotary_projection_input,
    pack_prefill_projection_input,
)
from vllm_ascend.utils import enable_custom_op


def layer():
    return SimpleNamespace(rope_dim=64, frequencies=(1 / (10000.0 ** (torch.arange(32).float() / 32))).npu())


def reference(x, positions, rotary, tp):
    rotated = x.clone()
    DeepseekV41Attention.rotate(rotary, rotated[None], positions, inverse=True)
    return pack_prefill_projection_input(rotated, tp)


def exact(a, b):
    assert a is not None
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("pattern", ["finite", "all_words"])
@torch.inference_mode()
def test_native_inverse_pack_matches_stock_bytes(rows, tp, pattern):
    assert enable_custom_op()
    if pattern == "all_words":
        n = rows * 64 * 512
        words = torch.arange(65536, dtype=torch.int32).short()
        x = words.repeat((n + 65535) // 65536)[:n].view(torch.bfloat16).reshape(rows, 64, 512).npu()
    else:
        torch.manual_seed(71116)
        x = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(rows, device="npu") * 17011
    rotary = layer()
    phases = DeepseekV41Attention._decode_rotary_factors(rotary, positions)
    saved = x.clone()
    actual = pack_decode_rotary_projection_input(x, tp, *phases)
    exact(actual, reference(x, positions, rotary, tp))
    exact(x, saved)
    # The query's positive rotation sees precisely the same phase tensors.
    a, b = x.clone(), x.clone()
    DeepseekV41Attention.rotate(rotary, a[None], positions, factors=phases)
    DeepseekV41Attention.rotate(rotary, b[None], positions)
    exact(a, b)


@pytest.mark.parametrize("tp", [4, 8])
@torch.inference_mode()
def test_graph_refreshes_data_and_phase_inputs(tp):
    assert enable_custom_op()
    x = torch.randn(6, 64, 512, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(6, device="npu")
    rotary = layer()
    cosine, sine = DeepseekV41Attention._decode_rotary_factors(rotary, positions)
    pack_decode_rotary_projection_input(x, tp, cosine, sine)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        pack_decode_rotary_projection_input(x, tp, cosine, sine)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = pack_decode_rotary_projection_input(x, tp, cosine, sine)
    for step in range(4):
        positions.add_(17011)
        x.copy_(torch.randn_like(x) * (step + 1))
        co, si = DeepseekV41Attention._decode_rotary_factors(rotary, positions)
        cosine.copy_(co)
        sine.copy_(si)
        graph.replay()
        exact(actual, reference(x, positions, rotary, tp))


@torch.inference_mode()
def test_unsupported_phase_and_rows_fall_back():
    x = torch.zeros(6, 64, 512, dtype=torch.bfloat16, device="npu")
    phase = torch.zeros(6, 1, 1, 64, dtype=torch.float32, device="npu")
    assert pack_decode_rotary_projection_input(x[:1], 4, phase[:1], phase[:1]) is None
    assert pack_decode_rotary_projection_input(x, 2, phase, phase) is None
    assert pack_decode_rotary_projection_input(x, 4, phase.bfloat16(), phase) is None
