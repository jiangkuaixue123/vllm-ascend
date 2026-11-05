# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered index eligibility and reference fallback tests without device imports."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def tensor(shape=(6, 76800), **changes):
    fields = dict(
        device=SimpleNamespace(type="npu"),
        shape=shape,
        ndim=len(shape),
        dtype="bf16",
        contiguous=True,
        negative=False,
        conjugated=False,
        npu_format=2,
    )
    fields.update(changes)
    result = SimpleNamespace(**fields)
    result.is_contiguous = lambda: result.contiguous
    result.is_neg = lambda: result.negative
    result.is_conj = lambda: result.conjugated
    return result


def dispatcher():
    path = ROOT / "vllm_ascend/models/common/ops/ordered_index.py"

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
        "get_npu_format": lambda x: x.npu_format,
        "HAS_TRITON": True,
        "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": SimpleNamespace(getpid=lambda: 123),
        "ordered_index_columns": lambda *args: calls.append(args) or output,
    }
    exec(compile(tree, str(path), "exec"), env)
    return env, calls, output


@pytest.mark.parametrize("rows,width", [(6, 76800), (6, 153600), (12, 76800), (12, 153600)])
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_supported_inputs(rows, width, dtype):
    env, calls, output = dispatcher()
    scores = tensor((rows, width), dtype=dtype)
    assert env["ordered_index_fused"](scores, 512) is output
    assert calls == [(scores, 512)]


@pytest.mark.parametrize(
    "changes",
    [
        {"device": SimpleNamespace(type="cpu")},
        {"dtype": "fp16"},
        {"ndim": 3},
        {"shape": (0, 76800)},
        {"shape": (5, 76800)},
        {"shape": (6, 16384)},
        {"contiguous": False},
        {"negative": True},
        {"conjugated": True},
        {"npu_format": 29},
    ],
)
def test_other_metadata_falls_back(changes):
    env, calls, _ = dispatcher()
    assert env["ordered_index_fused"](tensor(**changes), 512) is None
    assert not calls


@pytest.mark.parametrize("mode", ["hardware", "triton", "batch_invariant", "training", "other_k"])
def test_unsupported_modes(mode):
    env, calls, _ = dispatcher()
    if mode == "hardware":
        env["get_ascend_device_type"] = lambda: "A2"
    elif mode == "triton":
        env["HAS_TRITON"] = False
    elif mode == "batch_invariant":
        env["envs"].VLLM_BATCH_INVARIANT = True
    elif mode == "training":
        env["torch"].is_inference_mode_enabled = lambda: False
    assert env["ordered_index_fused"](tensor(), 128 if mode == "other_k" else 512) is None
    assert not calls


def test_kernel_failures_propagate():
    env, _, _ = dispatcher()

    def fail(*args):
        raise RuntimeError("selection failure")

    env["ordered_index_columns"] = fail
    with pytest.raises(RuntimeError, match="selection failure"):
        env["ordered_index_fused"](tensor(), 512)


@pytest.mark.parametrize("supported", [False, True])
def test_model_consumer_keeps_sorted_reference_fallback(supported):
    path = ROOT / "vllm_ascend/models/deepseek_v41/attention.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ordered_topk_columns")
    fast, reference, calls = object(), object(), []
    env = {
        "ordered_index_fused": lambda *args: calls.append("fast") or (fast if supported else None),
        "topk_columns": lambda *args: (
            calls.append("reference") or SimpleNamespace(sort=lambda dim: SimpleNamespace(values=reference))
        ),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    assert env["ordered_topk_columns"](object(), 512) is (fast if supported else reference)
    assert calls == (["fast"] if supported else ["fast", "reference"])
