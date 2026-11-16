# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The new unweighted producer preserves BF16 activation and INT8 quantization."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.deepseek_v41.core import clipped_swiglu
from vllm_ascend.utils import enable_custom_op


def check(x, output):
    activation = clipped_swiglu(*x.chunk(2, dim=-1)).bfloat16()
    quantized, scales = torch_npu.npu_dynamic_quant(activation)
    assert torch.equal(output[0], quantized)
    torch.testing.assert_close(output[1], scales, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("nonfinite", [False, True])
@torch.inference_mode()
def test_shared_native_producer_quantized_bytes_and_graph(rows, nonfinite):
    assert enable_custom_op()
    x = torch.randn(rows, 4608, dtype=torch.bfloat16, device="npu") * 20
    saved = x.clone()
    actual = torch.ops._C_ascend.shared_gate_up_quant(x)
    check(x, actual)
    assert torch.equal(x.view(torch.int16), saved.view(torch.int16))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        torch.ops._C_ascend.shared_gate_up_quant(x)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = torch.ops._C_ascend.shared_gate_up_quant(x)
    count = x.numel()
    for start in range(0, 65536 + count, count):
        words = ((torch.arange(count, dtype=torch.int32) + start) % 65536).short()
        if not nonfinite:
            words = torch.where((words.int() & 32767) < 32640, words, 0).short()
        x.copy_(words.view(torch.bfloat16).view_as(x))
        graph.replay()
        check(x, captured)
    x.zero_()
    graph.replay()
    check(x, captured)


@torch.inference_mode()
def test_shared_native_rejects_unsupported_input():
    x = torch.zeros(6, 4608, dtype=torch.bfloat16, device="npu")
    for bad in (x[:2], x.float(), x[:, ::2]):
        with pytest.raises(RuntimeError):
            torch.ops._C_ascend.shared_gate_up_quant(bad)
