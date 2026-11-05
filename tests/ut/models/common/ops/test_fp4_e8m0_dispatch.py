# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility and failure propagation for the V4.1 FP4 index fast path."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def tensor(shape=(6, 64, 128), **changes):
    fields = dict(
        device=SimpleNamespace(type="npu"),
        dtype="bf16",
        ndim=len(shape),
        shape=shape,
        contiguous=True,
        negative=False,
        conjugated=False,
        npu_format=2,
    )
    fields.update(changes)
    value = SimpleNamespace(**fields)
    value.is_contiguous = lambda: value.contiguous
    value.is_neg = lambda: value.negative
    value.is_conj = lambda: value.conjugated
    value.numel = lambda: math.prod(value.shape)
    return value


def dispatcher():
    path = ROOT / "vllm_ascend/models/deepseek_v41/attention_ops.py"
    method = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_try_fp4_e8m0_aiv"
    )

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    method = RemoveImports().visit(method)
    calls, output = [], object()
    env = {
        "torch": SimpleNamespace(
            Tensor=object,
            bfloat16="bf16",
            is_inference_mode_enabled=lambda: True,
            compiler=SimpleNamespace(is_compiling=lambda: False),
        ),
        "torch_npu": SimpleNamespace(Format=SimpleNamespace(ND=2), get_npu_format=lambda x: x.npu_format),
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": SimpleNamespace(A3="A3"),
        "HAS_TRITON": True,
        "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": SimpleNamespace(getpid=lambda: 123),
        "_FP4_MAX_ROWS": 8192,
        "_FP4_INDEX_DIM": 128,
        "_FP8_DECODE_GROUP_SIZE": 32,
        "fp4_e8m0_qdq": lambda *args: calls.append(args) or output,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    return env, calls, output


@pytest.mark.parametrize(
    "shape", [(1, 128), (5, 128), (48, 128), (6, 64, 128), (12, 64, 128), (8192, 128), (512, 64, 128)]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_supported_shapes_and_alias_request(shape, inplace):
    env, calls, output = dispatcher()
    states = tensor(shape)
    assert env["_try_fp4_e8m0_aiv"](states, 32, "e8m0", inplace) is output
    assert calls == [(states, inplace)]


@pytest.mark.parametrize(
    "changes",
    [
        {"device": SimpleNamespace(type="cpu")},
        {"dtype": "fp32"},
        {"shape": (0, 128)},
        {"shape": (6, 0, 128)},
        {"shape": (8193, 128)},
        {"shape": (6, 512)},
        {"ndim": 4},
        {"negative": True},
        {"conjugated": True},
        {"contiguous": False},
        {"npu_format": 29},
    ],
)
def test_unsupported_metadata_falls_back(changes):
    env, calls, _ = dispatcher()
    assert env["_try_fp4_e8m0_aiv"](tensor(**changes), 32, "e8m0", False) is None
    assert not calls


@pytest.mark.parametrize("mode", ["triton", "batch_invariant", "hardware", "training", "compiler", "scale", "block"])
def test_unsupported_execution_modes(mode):
    env, calls, _ = dispatcher()
    if mode == "triton":
        env["HAS_TRITON"] = False
    elif mode == "batch_invariant":
        env["envs"].VLLM_BATCH_INVARIANT = True
    elif mode == "hardware":
        env["get_ascend_device_type"] = lambda: "A2"
    elif mode == "training":
        env["torch"].is_inference_mode_enabled = lambda: False
    elif mode == "compiler":
        env["torch"].compiler.is_compiling = lambda: True
    assert (
        env["_try_fp4_e8m0_aiv"](tensor(), 16 if mode == "block" else 32, "e4m3" if mode == "scale" else "e8m0", False)
        is None
    )
    assert not calls


def test_kernel_failure_propagates():
    env, _, _ = dispatcher()

    def fail(*args):
        raise RuntimeError("FP4 kernel failure")

    env["fp4_e8m0_qdq"] = fail
    with pytest.raises(RuntimeError, match="FP4 kernel failure"):
        env["_try_fp4_e8m0_aiv"](tensor(), 32, "e8m0", False)
