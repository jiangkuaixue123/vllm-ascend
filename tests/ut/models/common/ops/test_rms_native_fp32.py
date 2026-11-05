# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native RMS dispatch boundaries without importing the device runtime."""

import ast
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple
    dtype: str = "bf16"
    device: object = "npu"
    contiguous: bool = True
    negative: bool = False
    conjugate: bool = False
    npu_format: int = 2

    def __post_init__(self):
        if isinstance(self.device, str):
            object.__setattr__(self, "device", SimpleNamespace(type=self.device))

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate

    def float(self):
        return replace(self, dtype="fp32")

    def to(self, dtype):
        return replace(self, dtype=dtype)


def load_dispatcher():
    path = ROOT / "vllm_ascend/models/common/ops/rms_native_fp32.py"
    tree = ast.parse(path.read_text())

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    calls = []

    def native(*args):
        calls.append(args)
        assert args[0].dtype == args[1].dtype == "fp32"
        return args[0], object()

    env = {
        "math": math,
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32", is_inference_mode_enabled=lambda: True),
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": SimpleNamespace(A3="A3"),
        "get_npu_format": lambda t: t.npu_format,
        "npu_rms_norm": native,
    }
    exec(compile(RemoveImports().visit(tree), str(path), "exec"), env)
    return env, calls, None


@pytest.mark.parametrize(
    "shape", [(1, 128), (5, 5120), (6, 5120), (12, 5120), (20, 5120), (24, 512), (24, 1024), (40, 5120), (48, 5120)]
)
@pytest.mark.parametrize("weight_dtype", ["bf16", "fp32"])
def test_supported_call_promotes_operands_and_restores_dtype(shape, weight_dtype):
    env, calls, _ = load_dispatcher()
    x, weight = TensorMetadata(shape), TensorMetadata((shape[-1],), dtype=weight_dtype)
    assert env["native_rms_norm_fp32"](x, weight, 1e-20) == x
    assert calls == [(x.float(), weight.float(), 1e-20)]


@pytest.mark.parametrize(
    "index,change",
    [
        (0, {"shape": (0, 5120)}),
        (0, {"shape": (49, 5120)}),
        (0, {"shape": (6, 4, 5120)}),
        (0, {"shape": (6, 129)}),
        (0, {"dtype": "fp32"}),
        (0, {"device": "cpu"}),
        (0, {"contiguous": False}),
        (0, {"negative": True}),
        (0, {"conjugate": True}),
        (0, {"npu_format": 29}),
        (1, {"shape": (1, 5120)}),
        (1, {"dtype": "fp16"}),
        (1, {"device": "cpu"}),
        (1, {"contiguous": False}),
        (1, {"negative": True}),
        (1, {"npu_format": 29}),
    ],
)
def test_unsupported_inputs_never_dispatch(index, change):
    env, calls, _ = load_dispatcher()
    inputs = [TensorMetadata((6, 5120)), TensorMetadata((5120,))]
    inputs[index] = replace(inputs[index], **change)
    assert env["native_rms_norm_fp32"](*inputs, 1e-20) is None
    assert not calls


@pytest.mark.parametrize("eps", [0, -1, math.nan, math.inf, 1e-200, 1e200])
def test_invalid_epsilon_uses_reference(eps):
    env, calls, _ = load_dispatcher()
    assert env["native_rms_norm_fp32"](TensorMetadata((6, 5120)), TensorMetadata((5120,)), eps) is None
    assert not calls


@pytest.mark.parametrize("training", [False, True])
def test_training_or_other_hardware_uses_reference(training):
    env, calls, _ = load_dispatcher()
    if training:
        env["torch"].is_inference_mode_enabled = lambda: False
    else:
        env["get_ascend_device_type"] = lambda: "A2"
    assert env["native_rms_norm_fp32"](TensorMetadata((6, 5120)), TensorMetadata((5120,)), 1e-20) is None
    assert not calls
