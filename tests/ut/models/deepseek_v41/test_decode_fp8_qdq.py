# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention_ops as ops


@pytest.mark.parametrize("rows", [24, 48])
def test_decode_arithmetic_matches_reference_bf16_words(rows):
    # Force only the device predicate on CPU. The real entry still groups, casts
    # once, restores shape and returns the original object for inplace=True.
    bits = torch.arange(rows * 2 * 4096, dtype=torch.int32).short().reshape(rows, 2, 4096)
    x = bits.view(torch.bfloat16)
    expected = ops.fp8_quant_dequant(x)
    with patch.object(ops, "_use_decode_fp8_qdq", return_value=True):
        result = ops.fp8_quant_dequant(x)
        torch.testing.assert_close(result.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
        assert result.data_ptr() != x.data_ptr()
        assert ops.fp8_quant_dequant(x, inplace=True) is x
        torch.testing.assert_close(x.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,block", [(torch.float16, 32), (torch.float32, 32), (torch.bfloat16, 16)])
def test_fp8_fallback_keeps_view_and_storage(dtype, block):
    storage = torch.linspace(-3, 4, 24 * 1024).to(dtype).reshape(24, 1024)
    x = storage[:, 1::2]
    untouched = storage[:, ::2].clone()
    with patch.object(ops, "_decode_fp8_qdq", side_effect=AssertionError("unexpected decode path")):
        expected = ops.fp8_quant_dequant(x.clone(), block)
        assert ops.fp8_quant_dequant(x, block, inplace=True) is x
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage[:, ::2], untouched, rtol=0, atol=0)


@pytest.mark.parametrize(
    "x,block,error",
    [
        (torch.empty(1, 31), 32, ValueError),
        (torch.empty(()), 32, ValueError),
        (torch.empty(1, 32), 0, ValueError),
        (torch.empty((1, 32), dtype=torch.int32), 32, TypeError),
    ],
)
def test_fp8_validation_precedes_dispatch(x, block, error):
    with (
        patch.object(ops, "_use_decode_fp8_qdq", side_effect=AssertionError("validation was skipped")),
        pytest.raises(error),
    ):
        ops.fp8_quant_dequant(x, block)
