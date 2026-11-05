# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native delayed-pre ABI and strict-FP32 diagnostics on the supported A3 shapes.

The task's frozen component harness additionally covers adversarial reductions,
all nonfinite classes, full graph replay and complete-path paired timings.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41.core import hc_mixes
from vllm_ascend.utils import enable_custom_op

HC_WIDTH = 4
HIDDEN_SIZE = 5120
MIX_WIDTH = 24
NORM_EPS = 1e-20
HC_EPS = 1e-6
SINKHORN_ITERS = 20


@pytest.fixture
def native_op():
    if get_ascend_device_type() != AscendDeviceType.A3:
        pytest.skip("delayed HC pre has an A3-only native implementation")
    assert enable_custom_op()
    return torch.ops._C_ascend.npu_hc_pre_delayed


def make_inputs(rows):
    generator = torch.Generator().manual_seed(20261105)
    x = torch.randn(rows, HC_WIDTH, HIDDEN_SIZE, generator=generator).bfloat16()
    fn = torch.randn(MIX_WIDTH, HC_WIDTH * HIDDEN_SIZE, generator=generator) / (HC_WIDTH * HIDDEN_SIZE) ** 0.5
    scale = torch.tensor([0.75, -0.5, 1.25])
    base = torch.randn(MIX_WIDTH, generator=generator)
    previous = torch.zeros(rows, HC_WIDTH)
    previous[:, 0] = 1
    return tuple(t.npu() for t in (x, fn, scale, base, previous))


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_delayed_pre_keeps_first_gate_and_returns_real_fp32_intermediates(native_op, rows):
    x, fn, scale, base, previous = make_inputs(rows)
    saved_hf32 = torch.npu.matmul.allow_hf32
    try:
        # This test is a strict-FP32 component diagnostic, not a serving-parent claim.
        torch.npu.matmul.allow_hf32 = False
        reference_projection = torch.nn.functional.linear(x.flatten(-2).float(), fn)
        reference_rstd = torch.rsqrt(x.flatten(-2).float().square().mean(-1, keepdim=True) + NORM_EPS)
        next_pre, post, comb = hc_mixes(x, fn, scale, base, NORM_EPS, HC_EPS, SINKHORN_ITERS)
        outputs = native_op(x, fn, scale, base, previous, NORM_EPS)
    finally:
        torch.npu.matmul.allow_hf32 = saved_hf32
    y, actual_pre, actual_post, actual_comb, projection, rstd = outputs
    expected_shapes = [
        (rows, HIDDEN_SIZE),
        (rows, HC_WIDTH),
        (rows, HC_WIDTH),
        (rows, HC_WIDTH, HC_WIDTH),
        (rows, MIX_WIDTH),
        (rows, 1),
    ]
    for index, (output, shape) in enumerate(zip(outputs, expected_shapes)):
        assert output.shape == shape and output.is_contiguous()
        assert output.dtype == (torch.bfloat16 if index == 0 else torch.float32)
        assert all(output.data_ptr() != source.data_ptr() for source in (x, fn, scale, base, previous))
    torch.testing.assert_close(y, x[:, 0, :], rtol=0, atol=0)
    abs_dot = x.flatten(-2).float().abs().cpu().double() @ fn.abs().cpu().double().T
    budget = 2e-5 * reference_projection.cpu().double().abs() + 2e-6 * abs_dot + 1e-7
    assert ((projection.cpu().double() - reference_projection.cpu().double()).abs() <= budget).all()
    torch.testing.assert_close(rstd, reference_rstd, rtol=5e-6, atol=1e-7)
    torch.testing.assert_close(actual_pre, next_pre, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(actual_post, post, rtol=5e-5, atol=1e-4)
    torch.testing.assert_close(actual_comb, comb, rtol=1e-4, atol=1e-4)
    # A different previous gate changes only y, not coefficients of the current x.
    previous.zero_()
    previous[:, 3] = 1
    updated = native_op(x, fn, scale, base, previous, NORM_EPS)
    torch.testing.assert_close(updated[0], x[:, 3, :], rtol=0, atol=0)
    for current, old in zip(updated[1:], outputs[1:]):
        torch.testing.assert_close(current, old, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["rows", "previous_shape", "previous_dtype", "fn_layout", "epsilon"])
@torch.inference_mode()
def test_direct_native_rejects_unsupported_contract(native_op, change):
    args = list(make_inputs(6))
    eps = NORM_EPS
    if change == "rows":
        args[0] = args[0][:1]
        args[4] = args[4][:1]
    elif change == "previous_shape":
        args[4] = args[4][:, :3]
    elif change == "previous_dtype":
        args[4] = args[4].bfloat16()
    elif change == "fn_layout":
        args[1] = args[1].T.contiguous().T
    else:
        eps = 0.0
    with pytest.raises(RuntimeError):
        native_op(*args, eps)
