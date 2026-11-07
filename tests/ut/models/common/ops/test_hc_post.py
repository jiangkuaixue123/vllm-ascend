# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata fallbacks and the actual model dispatch, without requiring NPU imports."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def tensor(shape, dtype, **changes):
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
    path = ROOT / "vllm_ascend/models/common/ops/hc_post.py"

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    tree = RemoveImports().visit(ast.parse(path.read_text()))
    calls = []
    output = object()
    env = {
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32", is_inference_mode_enabled=lambda: True),
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": SimpleNamespace(A3="A3"),
        "get_npu_format": lambda value: value.npu_format,
        "hc_post4": lambda *args: calls.append(args) or output,
    }
    exec(compile(tree, str(path), "exec"), env)
    return env, calls, output


def inputs(rows=6):
    return [
        tensor((rows, 5120), "bf16"),
        tensor((rows, 4, 5120), "bf16"),
        tensor((rows, 4), "fp32"),
        tensor((rows, 4, 4), "fp32"),
    ]


@pytest.mark.parametrize("rows", [6, 12])
def test_supported_inputs_preserve_operand_order(rows):
    env, calls, output = load_dispatcher()
    args = inputs(rows)
    assert env["hc_post_fused"](*args) is output
    assert len(calls) == 1 and all(actual is expected for actual, expected in zip(calls[0], args))


@pytest.mark.parametrize("rows", [5, 20])
def test_draft_rows_are_explicit_and_preserve_operand_order(rows):
    env, calls, output = load_dispatcher()
    args = inputs(rows)
    assert env["hc_post_fused"](*args) is None
    assert env["hc_post_fused"](*args, draft=True) is output
    assert len(calls) == 1 and all(actual is expected for actual, expected in zip(calls[0], args))


@pytest.mark.parametrize("rows", [0, 1, 6, 10, 12, 24, 40])
def test_other_draft_rows_keep_parent(rows):
    env, calls, _ = load_dispatcher()
    assert env["hc_post_fused"](*inputs(rows), draft=True) is None
    assert not calls


@pytest.mark.parametrize(
    "index,changes",
    [
        (0, {"device": SimpleNamespace(type="cpu")}),
        (0, {"dtype": "fp32"}),
        (0, {"shape": (6, 4096)}),
        (0, {"ndim": 3}),
        (1, {"shape": (6, 3, 5120)}),
        (1, {"dtype": "fp32"}),
        (2, {"shape": (6, 3)}),
        (2, {"dtype": "bf16"}),
        (3, {"shape": (6, 16)}),
        (3, {"dtype": "bf16"}),
        (3, {"contiguous": False}),
        (1, {"npu_format": 29}),
        (2, {"negative": True}),
        (3, {"conjugated": True}),
    ],
)
def test_unsupported_metadata_falls_back(index, changes):
    env, calls, _ = load_dispatcher()
    args = inputs()
    for name, value in changes.items():
        setattr(args[index], name, value)
    assert env["hc_post_fused"](*args) is None
    assert not calls


@pytest.mark.parametrize("rows", [0, 1, 5, 24])
def test_other_shapes_fall_back(rows):
    env, calls, _ = load_dispatcher()
    assert env["hc_post_fused"](*inputs(rows)) is None
    assert not calls


def test_no_triton_and_autograd_fall_back():
    env, calls, _ = load_dispatcher()
    env["hc_post4"] = None
    assert env["hc_post_fused"](*inputs()) is None
    env["torch"].is_inference_mode_enabled = lambda: False
    assert env["hc_post_fused"](*inputs()) is None
    assert not calls


def test_kernel_failures_are_not_hidden():
    env, _, _ = load_dispatcher()

    def fail(*args):
        raise RuntimeError("kernel failure")

    env["hc_post4"] = fail
    with pytest.raises(RuntimeError, match="kernel failure"):
        env["hc_post_fused"](*inputs())


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_actual_model_method_only_fuses_decode(decode, supported):
    path = ROOT / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Block")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "post_residual")
    fused, reference = object(), object()
    calls = []
    env = {
        "hc_post_fused": lambda *args: calls.append("fused") or (fused if supported else None),
        "hc_post": lambda *args: calls.append("reference") or reference,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    runtime = SimpleNamespace(decode=object() if decode else None)
    result = env["post_residual"](object(), *inputs(), runtime)
    assert result is (fused if decode and supported else reference)
    expected = (["fused"] if supported else ["fused", "reference"]) if decode else ["reference"]
    assert calls == expected


@pytest.mark.parametrize("supported", [False, True])
def test_actual_draft_stage_uses_explicit_draft_dispatch_and_parent_fallback(supported):
    path = ROOT / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DSparkStage")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "post_residual"]
    fused, reference = object(), object()
    calls = []

    class Parent:
        def post_residual(self, *args):
            calls.append("parent")
            return reference

    def dispatch(*args, draft=False):
        assert draft is True
        calls.append("draft")
        return fused if supported else None

    env = {
        "DeepseekV41Block": Parent,
        "hc_post_fused": dispatch,
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": SimpleNamespace(getpid=lambda: 1),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), env)
    stage = env["DSparkStage"]()
    stage.attn = SimpleNamespace(cache=SimpleNamespace(prefix="mtp.0.attn.cache"))
    assert stage.post_residual(*inputs(5), object()) is (fused if supported else reference)
    assert calls == (["draft"] if supported else ["draft", "parent"])
