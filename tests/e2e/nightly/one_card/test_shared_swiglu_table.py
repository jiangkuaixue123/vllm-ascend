# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the table against native SiLU over every BF16 gate bit pattern."""

import pytest
import torch

from vllm_ascend.models.common.ops.shared_swiglu import (
    BF16_SIGN_OFFSET,
    SHARED_INTERMEDIATE,
    SharedClippedSwiGLU,
)
from vllm_ascend.models.deepseek_v41.core import clipped_swiglu


def exact_finite(actual, expected):
    assert actual is not None
    torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected), rtol=0, atol=0)
    finite = ~torch.isnan(expected)
    torch.testing.assert_close(actual.view(torch.int16)[finite], expected.view(torch.int16)[finite], rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("limit", [1.0, 10.0])
def test_all_bf16_gate_values_and_up_edges(limit):
    torch.npu.set_device(0)
    words = torch.arange(-BF16_SIGN_OFFSET, BF16_SIGN_OFFSET, dtype=torch.int32).to(torch.int16)
    rows = (words.numel() + SHARED_INTERMEDIATE - 1) // SHARED_INTERMEDIATE
    gate = torch.zeros(rows * SHARED_INTERMEDIATE, dtype=torch.bfloat16)
    gate[: words.numel()] = words.view(torch.bfloat16)
    gate = gate.reshape(rows, SHARED_INTERMEDIATE).npu()
    helper = SharedClippedSwiGLU().eval()
    for value in (0.0, -0.0, 1.0, -1.0, 10.0, -10.0, float("inf"), -float("inf"), float("nan")):
        up = torch.full_like(gate, value)
        packed = torch.cat((gate, up), dim=-1)
        expected = clipped_swiglu(gate, up, limit).bfloat16()
        exact_finite(helper(packed, limit), expected)


@torch.inference_mode()
def test_table_lifecycle_and_fallbacks(monkeypatch):
    packed = torch.randn(5, 2 * SHARED_INTERMEDIATE, device="npu", dtype=torch.bfloat16)
    helper = SharedClippedSwiGLU().eval()
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    assert helper(packed, 10.0) is None
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    gate, up = packed.chunk(2, -1)
    exact_finite(helper(packed, 10.0), clipped_swiglu(gate, up, 10.0).bfloat16())
    table = helper.gate_silu
    helper(packed, 10.0)
    assert helper.gate_silu is table
    assert not helper.state_dict()
    helper.to(dtype=torch.bfloat16)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    assert helper(packed, 10.0) is None
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    exact_finite(helper(packed, 1.0), clipped_swiglu(gate, up, 1.0).bfloat16())
    assert helper.gate_silu.dtype == torch.float32
    assert helper._table_limit == 1.0
    for unsupported in (packed.float(), packed[:, ::2], packed[:0], packed.cpu()):
        assert helper(unsupported, 1.0) is None
    for limit in (0.0, -1.0, float("inf"), float("nan")):
        assert helper(packed, limit) is None
    helper.train()
    assert helper(packed, 1.0) is None


def test_grad_mode_uses_reference_path():
    helper = SharedClippedSwiGLU().eval()
    packed = torch.randn(2, 2 * SHARED_INTERMEDIATE, device="npu", dtype=torch.bfloat16)
    assert helper(packed, 10.0) is None
