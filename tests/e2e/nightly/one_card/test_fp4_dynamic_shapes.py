# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Changing prefill row counts must reuse an already compiled FP4 kernel."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import attention_ops
from vllm_ascend.ops.triton import deepseek_v41_fp4_e4m3_qdq as e4m3
from vllm_ascend.ops.triton import deepseek_v41_fp4_qdq as e8m0

ROWS = (1, 5, 6, 24, 37, 48, 511, 8192, 8186)


class ObservedKernel:
    def __init__(self, kernel):
        self.kernel = kernel
        self.executables = []

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            executable = self.kernel[grid](*args, **kwargs)
            self.executables.append(executable)
            return executable

        return invoke


@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
@torch.inference_mode()
def test_rows_reuse_the_same_compiled_executable(scale_format):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    if scale_format == "e8m0":
        module, names, width, block = e8m0, ("_kernel",), 128, 32
    else:
        module, names, width, block = e4m3, ("_prepare_kernel", "_finish_kernel"), 512, 16
    observed = [ObservedKernel(getattr(module, name)) for name in names]
    with ExitStack() as stack:
        for name, wrapper in zip(names, observed):
            stack.enter_context(patch.object(module, name, wrapper))
        for rows in ROWS:
            states = torch.randn(rows, width, device="npu", dtype=torch.bfloat16)
            actual = attention_ops.fp4_quant_dequant(states, block, scale_format=scale_format)
            with (
                patch.object(attention_ops, "_try_fp4_e8m0_aiv", return_value=None),
                patch.object(attention_ops, "_try_fp4_e4m3_aiv", return_value=None),
            ):
                expected = attention_ops.fp4_quant_dequant(states, block, scale_format=scale_format)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), rows
    for name, wrapper in zip(names, observed):
        assert len(wrapper.executables) == len(ROWS), name
        assert all(kernel is wrapper.executables[0] for kernel in wrapper.executables), name
