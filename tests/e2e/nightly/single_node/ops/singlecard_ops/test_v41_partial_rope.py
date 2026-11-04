# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention


@pytest.mark.parametrize("shape", [(1, 6, 32, 128), (1, 12, 64, 512), (1, 48, 128), (1, 24, 512)])
@pytest.mark.parametrize("inverse", [False, True])
@torch.inference_mode()
def test_partial_rope_replays_with_new_positions(shape, inverse):
    if not attention.enable_custom_op() or not hasattr(torch.ops._C_ascend, "inplace_partial_rotary_mul"):
        pytest.skip("Requires the Ascend partial rotary operator")
    rows = shape[1]
    freq = 160000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32, device="cpu") / 64)
    layer = SimpleNamespace(rope_dim=64, frequencies=freq.npu())
    source = torch.zeros(shape, device="npu", dtype=torch.bfloat16)
    work = source.clone()
    positions = torch.zeros(rows, device="npu", dtype=torch.int64)
    native = torch.ops._C_ascend.inplace_partial_rotary_mul

    def run():
        work.copy_(source)
        return attention.DeepseekV41Attention.rotate(layer, work, positions, inverse)

    with patch.object(torch.ops._C_ascend, "inplace_partial_rotary_mul", wraps=native) as op:
        assert run() is work
        op.assert_called_once()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    generator = torch.Generator(device="cpu").manual_seed(614)
    for last_position in (1023, 16383, 149999):
        source.copy_(torch.randn(shape, generator=generator, device="cpu").bfloat16())
        positions.copy_(torch.linspace(0, last_position, rows, device="cpu").long())
        expected = source.clone()
        with patch.object(attention, "enable_custom_op", return_value=False):
            attention.DeepseekV41Attention.rotate(layer, expected, positions, inverse)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(work.view(torch.int16), expected.view(torch.int16))
