# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

import pytest
import torch

from vllm_ascend.models.deepseek_v41.core import route_sqrtsoftplus


@pytest.mark.parametrize("experts,top_k,rows", [(384, 6, 6), (384, 6, 12), (128, 3, 5), (384, 6, 13)])
@pytest.mark.parametrize("inference", [False, True])
def test_cpu_router_preserves_reference_without_optional_kernel(monkeypatch, experts, top_k, rows, inference):
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.deepseek_v41_route_weights", None)
    generator = torch.Generator().manual_seed(211)
    logits = torch.randn(rows, experts, generator=generator, requires_grad=not inference)
    bias = torch.randn(experts, generator=generator)
    with torch.inference_mode(inference):
        expected = route_sqrtsoftplus(logits, bias, top_k)
        actual = route_sqrtsoftplus(logits, bias, top_k, fuse_weights=True)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        if not inference:
            actual[0].sum().backward()
            assert torch.isfinite(logits.grad).all()
