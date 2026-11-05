# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RMSNorm dispatch and fallback coverage without device imports."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def tensor(shape, dtype="bf16", **changes):
    metadata = dict(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=SimpleNamespace(type="npu"),
        contiguous=True,
        negative=False,
        conjugated=False,
        npu_format=2,
    )
    metadata.update(changes)
    value = SimpleNamespace(**metadata)
    value.is_contiguous = lambda: value.contiguous
    value.is_neg = lambda: value.negative
    value.is_conj = lambda: value.conjugated
    return value


def load_dispatcher():
    path = ROOT / "vllm_ascend/models/common/ops/rms_elementwise.py"

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    tree = RemoveImports().visit(ast.parse(path.read_text()))
    calls, output = [], object()
    env = {
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32", is_inference_mode_enabled=lambda: True),
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": SimpleNamespace(A3="A3"),
        "get_npu_format": lambda value: value.npu_format,
        "rms_elementwise": lambda *args: calls.append(args) or output,
    }
    exec(compile(tree, str(path), "exec"), env)
    return env, calls, output


@pytest.mark.parametrize("rows,hidden", [(1, 128), (5, 512), (6, 1280), (12, 5120), (48, 5120)])
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_supported_inputs(rows, hidden, dtype):
    env, calls, output = load_dispatcher()
    args = (tensor((rows, hidden)), tensor((hidden,), dtype), 1e-20)
    assert env["rms_elementwise_fused"](*args) is output
    assert calls == [args]


@pytest.mark.parametrize(
    "index,changes",
    [
        (0, {"device": SimpleNamespace(type="cpu")}),
        (0, {"dtype": "fp32"}),
        (0, {"shape": (6, 4096)}),
        (0, {"shape": (0, 5120)}),
        (0, {"shape": (49, 5120)}),
        (0, {"ndim": 3}),
        (0, {"contiguous": False}),
        (0, {"negative": True}),
        (1, {"shape": (1, 5120)}),
        (1, {"dtype": "fp16"}),
        (1, {"npu_format": 29}),
        (1, {"conjugated": True}),
        (1, {"device": SimpleNamespace(type="npu", index=1)}),
    ],
)
def test_unsupported_metadata(index, changes):
    env, calls, _ = load_dispatcher()
    args = [tensor((6, 5120)), tensor((5120,))]
    for name, value in changes.items():
        setattr(args[index], name, value)
    assert env["rms_elementwise_fused"](*args, 1e-20) is None
    assert not calls


def test_no_triton_autograd_hardware_and_other_epsilon_fall_back():
    args = (tensor((6, 5120)), tensor((5120,)), 1e-20)
    for setting in ("triton", "autograd", "hardware", "epsilon"):
        env, calls, _ = load_dispatcher()
        if setting == "triton":
            env["rms_elementwise"] = None
        elif setting == "autograd":
            env["torch"].is_inference_mode_enabled = lambda: False
        elif setting == "hardware":
            env["get_ascend_device_type"] = lambda: "A2"
        else:
            args = (*args[:2], 1e-6)
        assert env["rms_elementwise_fused"](*args) is None
        assert not calls


def test_kernel_errors_are_not_hidden():
    env, _, _ = load_dispatcher()

    def fail(*args):
        raise RuntimeError("kernel failure")

    env["rms_elementwise"] = fail
    with pytest.raises(RuntimeError, match="kernel failure"):
        env["rms_elementwise_fused"](tensor((6, 5120)), tensor((5120,)), 1e-20)


@pytest.mark.parametrize("supported", [False, True])
def test_actual_module_forward(supported):
    path = ROOT / "vllm_ascend/models/deepseek_v41/attention.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41RMSNorm"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    fused, reference, calls = object(), object(), []
    env = {
        "rms_elementwise_fused": lambda *args: calls.append("fused") or (fused if supported else None),
        "rms_norm": lambda *args: calls.append("reference") or reference,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    module = SimpleNamespace(weight=object(), eps=1e-20, _fused_witness_pending=False)
    assert env["forward"](module, object()) is (fused if supported else reference)
    assert calls == (["fused"] if supported else ["fused", "reference"])
