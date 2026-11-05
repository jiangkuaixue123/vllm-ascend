# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic FP8 dispatch and conservative unsupported-input fallback."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def tensor(shape=(5, 512), **changes):
    fields = dict(
        device=SimpleNamespace(type="npu"),
        shape=shape,
        ndim=len(shape),
        dtype="bf16",
        layout="strided",
        contiguous=True,
        negative=False,
        conjugated=False,
        requires_grad=False,
        npu_format=2,
        offset=0,
        address=64,
    )
    fields.update(changes)
    result = SimpleNamespace(**fields)
    result.is_contiguous = lambda: result.contiguous
    result.is_neg = lambda: result.negative
    result.is_conj = lambda: result.conjugated
    result.storage_offset = lambda: result.offset
    result.data_ptr = lambda: result.address
    result.element_size = lambda: 2
    return result


def dispatcher():
    path = ROOT / "vllm_ascend/models/common/ops/fp8_dynamic.py"

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    tree = RemoveImports().visit(ast.parse(path.read_text()))
    calls, output = [], object()
    env = {
        "torch": SimpleNamespace(
            bfloat16="bf16",
            strided="strided",
            is_inference_mode_enabled=lambda: True,
            compiler=SimpleNamespace(is_compiling=lambda: False),
        ),
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": SimpleNamespace(A3="A3"),
        "get_npu_format": lambda x: x.npu_format,
        "HAS_TRITON": True,
        "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": SimpleNamespace(getpid=lambda: 123),
        "dynamic_fp8_qdq": lambda *args: calls.append(args) or output,
    }
    exec(compile(tree, str(path), "exec"), env)
    return env, calls, output


@pytest.mark.parametrize("shape", [(1, 512), (20, 512), (8192, 512), (5, 2, 4096), (8192, 2, 4096)])
@pytest.mark.parametrize("inplace", [False, True])
def test_supported_inputs(shape, inplace):
    env, calls, output = dispatcher()
    value = tensor(shape)
    assert env["dynamic_fp8_fused"](value, 32, inplace) is output
    assert calls == [(value, inplace)]


@pytest.mark.parametrize(
    "changes",
    [
        {"device": SimpleNamespace(type="cpu")},
        {"dtype": "fp32"},
        {"shape": (0, 512)},
        {"shape": (8193, 512)},
        {"shape": (5, 128)},
        {"shape": (5, 1, 4096), "ndim": 3},
        {"shape": (512,), "ndim": 1},
        {"layout": "sparse"},
        {"contiguous": False},
        {"negative": True},
        {"conjugated": True},
        {"requires_grad": True},
        {"npu_format": 29},
        {"offset": 1},
        {"address": 66},
    ],
)
def test_other_metadata_falls_back(changes):
    env, calls, _ = dispatcher()
    assert env["dynamic_fp8_fused"](tensor(**changes), 32, False) is None
    assert not calls


@pytest.mark.parametrize("mode", ["hardware", "triton", "batch_invariant", "training", "compiling", "block"])
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
    elif mode == "compiling":
        env["torch"].compiler.is_compiling = lambda: True
    assert env["dynamic_fp8_fused"](tensor(), 16 if mode == "block" else 32, False) is None
    assert not calls


def test_kernel_errors_propagate():
    env, _, _ = dispatcher()

    def fail(*args):
        raise RuntimeError("dynamic QDQ failure")

    env["dynamic_fp8_qdq"] = fail
    with pytest.raises(RuntimeError, match="dynamic QDQ failure"):
        env["dynamic_fp8_fused"](tensor(), 32, False)
