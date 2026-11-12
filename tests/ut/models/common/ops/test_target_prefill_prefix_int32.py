# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound exact prefix counts and target prefill dispatch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def method(name, env):
    path = ROOT / "vllm_ascend/models/deepseek_v41/attention.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


@pytest.mark.parametrize(
    "tp,available,mode,draft,expected",
    [
        (4, True, "none", False, True),
        (8, True, "none", False, False),
        (4, False, "none", False, False),
        (4, True, "full", False, False),
        (4, True, "none", True, False),
        (4, True, "none", None, False),
    ],
)
def test_target_tp4_prefill_only(tp, available, mode, draft, expected):
    context = SimpleNamespace(cudagraph_runtime_mode=mode)
    if draft is not None:
        context.is_draft_model = draft
    env = {
        "_INDEX_OVERLAP_TP": 4,
        "CUDAGraphMode": SimpleNamespace(NONE="none"),
        "is_forward_context_available": lambda: available,
        "get_forward_context": lambda: context,
    }
    assert method("_target_prefill_prefix_int32", env)(tp) is expected


@pytest.mark.parametrize(
    "shape,device,dtype,inference,expected",
    [
        ((128, 512), "npu", "bool", True, True),
        ((8192, 1), "npu", "bool", True, True),
        ((8193, 512), "npu", "bool", True, False),
        ((128, 513), "npu", "bool", True, False),
        ((0, 512), "npu", "bool", True, False),
        ((128, 0), "npu", "bool", True, False),
        ((1, 128, 512), "npu", "bool", True, False),
        ((128, 512), "cpu", "bool", True, False),
        ((128, 512), "npu", "float", True, False),
        ((128, 512), "npu", "bool", False, False),
    ],
)
def test_cutoff_prefix_dtype_and_bounds(shape, device, dtype, inference, expected):
    calls, output = [], object()

    class Cutoff(SimpleNamespace):
        def int(self):
            calls.append("int")
            return self

        def long(self):
            calls.append("long")
            return self

        def cumsum(self, dimension, **kwargs):
            calls.append(("cumsum", dimension, kwargs.get("dtype")))
            return self

        def __sub__(self, value):
            assert value == 1
            return output

    env = {
        "_PREFILL_INDEX_MAX_ROWS": 8192,
        "_DECODE_INDEX_SORT_K": 512,
        "torch": SimpleNamespace(bool="bool", int32="int32", is_inference_mode_enabled=lambda: inference),
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": SimpleNamespace(getpid=lambda: 1),
    }
    cutoff = Cutoff(shape=shape, ndim=len(shape), device=SimpleNamespace(type=device), dtype=dtype)
    assert method("_prefill_cutoff_ordinals", env)(cutoff) is output
    assert calls == (["int", ("cumsum", -1, "int32"), "long"] if expected else ["long", ("cumsum", -1, None)])
