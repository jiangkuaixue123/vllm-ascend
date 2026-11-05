# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded native HC against the eager DSpark path, including complete call costs."""

import statistics
import time
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.common.ops.hc_pre import hc_pre_draft
from vllm_ascend.models.deepseek_v41.core import hc_mixes, hc_pre
from vllm_ascend.models.deepseek_v41.dspark import DSparkStage
from vllm_ascend.utils import enable_custom_op

HC_WIDTH = 4
HIDDEN_SIZE = 5120
MIX_WIDTH = 24
NORM_EPS = 1e-20
HC_EPS = 1e-6
SINKHORN_ITERS = 20
STAGE_CALLS = 6
TIMING_REPEATS = 7
REFRESH_REPEATS = 8


@pytest.fixture(scope="module", autouse=True)
def native_context():
    if get_ascend_device_type() != AscendDeviceType.A3:
        pytest.skip("draft HC reuse requires the retained A3 native operation")
    torch.npu.set_device(0)
    assert enable_custom_op()


def make_inputs(rows, *, offset=False):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(20261106)
    allocated_rows = rows + int(offset)
    x = torch.randn(allocated_rows, HC_WIDTH, HIDDEN_SIZE, generator=generator).bfloat16()
    fn = torch.randn(MIX_WIDTH, HC_WIDTH * HIDDEN_SIZE, generator=generator) / (HC_WIDTH * HIDDEN_SIZE) ** 0.5
    scale = torch.tensor([0.75, -0.5, 1.25])
    base = torch.randn(MIX_WIDTH, generator=generator)
    previous = torch.rand(allocated_rows, HC_WIDTH, generator=generator)
    x, fn, scale, base, previous = (t.npu() for t in (x, fn, scale, base, previous))
    if offset:
        x, previous = x[1:], previous[1:]
    return x, fn, scale, base, previous


def reference(x, fn, scale, base, previous):
    next_pre, post, comb = hc_mixes(x, fn, scale, base, NORM_EPS, HC_EPS, SINKHORN_ITERS)
    return hc_pre(x, previous), next_pre, post, comb


def candidate(*args):
    result = hc_pre_draft(*args, NORM_EPS, HC_EPS, SINKHORN_ITERS)
    assert result is not None
    return result


def check_outputs(actual, expected):
    # Preserve the exact BF16 pre-mix boundary and the existing native HC
    # FP32 coefficient budgets. No matmul mode is changed by these tests.
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0, equal_nan=True)
    for value, ref, rtol, atol in zip(actual[1:], expected[1:], (5e-5, 5e-5, 1e-4), (5e-5, 1e-4, 1e-4)):
        torch.testing.assert_close(value, ref, rtol=rtol, atol=atol, equal_nan=True)
    assert all(a.shape == b.shape and a.dtype == b.dtype and a.is_contiguous() for a, b in zip(actual, expected))


@pytest.mark.parametrize("rows", [5, 10, 15, 20, 25, 30, 35, 40])
@torch.inference_mode()
def test_all_draft_sizes_keep_row_order_and_inputs(rows):
    args = make_inputs(rows, offset=True)
    saved = [t.clone() for t in args]
    check_outputs(candidate(*args), reference(*args))
    for current, original in zip(args, saved):
        assert torch.equal(current, original)


@pytest.mark.parametrize("mode", ["zero", "tiny", "large", "alternating", "nan", "infinity"])
@torch.inference_mode()
def test_padding_does_not_mix_rows_or_nonfinite_values(mode):
    args = list(make_inputs(20))
    x = args[0]
    if mode == "zero":
        x.zero_()
    elif mode == "tiny":
        x.mul_(2.0**-60)
    elif mode == "large":
        x.mul_(2.0**20)
    elif mode == "alternating":
        x[:, 1] = -x[:, 0]
        x[:, 3] = -x[:, 2]
        args[-1].fill_(1)
    elif mode == "nan":
        x[-1, 0, 0] = torch.nan
    else:
        x[-1, 0, 0] = torch.inf
    check_outputs(candidate(*args), reference(*args))


@pytest.mark.parametrize("rows", [5, 20, 40])
@torch.inference_mode()
def test_changing_gates_and_states_refresh_all_chunks(rows):
    args = list(make_inputs(rows))
    for step in range(REFRESH_REPEATS):
        args[0].add_(0.03125)
        args[-1].zero_()
        args[-1][:, step % HC_WIDTH] = 1
        actual = candidate(*args)
        expected = reference(*args)
        check_outputs(actual, expected)
        torch.testing.assert_close(actual[0], args[0][:, step % HC_WIDTH], rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 20])
@torch.inference_mode()
def test_real_draft_stage_uses_native_or_reference_with_position_runtime(rows):
    args = make_inputs(rows)
    x, fn, scale, base, previous = args
    stage = DSparkStage.__new__(DSparkStage)
    torch.nn.Module.__init__(stage)
    stage.config = SimpleNamespace(rms_norm_eps=NORM_EPS, hc_eps=HC_EPS, hc_sinkhorn_iters=SINKHORN_ITERS)
    for name, tensor in (("fn", fn), ("scale", scale), ("base", base)):
        setattr(stage, "hc_attn_" + name, torch.nn.Parameter(tensor))
    positions = torch.arange(rows, device=x.device)
    check_outputs(stage.pre_and_mixes(x, previous, "attn", positions), reference(*args))


def ready_wall_ms(function, args):
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(STAGE_CALLS):
        function(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000


@pytest.mark.parametrize("rows", [5, 10, 20, 40])
@torch.inference_mode()
def test_complete_eager_component_latency(rows):
    args = make_inputs(rows)
    reference(*args)
    candidate(*args)
    times = {"reference": [], "candidate": []}
    functions = {"reference": reference, "candidate": candidate}
    for iteration in range(TIMING_REPEATS):
        order = ("reference", "candidate") if iteration % 2 == 0 else ("candidate", "reference")
        for name in order:
            times[name].append(ready_wall_ms(functions[name], args))
    measured = {name + "_ms": statistics.median(values) for name, values in times.items()}
    print({"rows": rows, "calls": STAGE_CALLS, "timing": "eager_ready_wall", **measured})
    assert all(value > 0 for value in measured.values())
