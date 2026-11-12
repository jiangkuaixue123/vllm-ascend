# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect target-only dispatch and the public reference default."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def load_function(filename, name, env):
    path = ROOT / "vllm_ascend/models/deepseek_v41" / filename
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


@pytest.mark.parametrize(
    "tp,available,mode,draft,expected",
    [
        (4, True, "full", False, True),
        (8, True, "full", False, False),
        (4, False, "full", False, False),
        (4, True, "none", False, False),
        (4, True, "piecewise", False, False),
        (4, True, "full", True, False),
        (4, True, "full", None, False),
    ],
)
def test_only_target_tp4_full_decode(tp, available, mode, draft, expected):
    context = SimpleNamespace(cudagraph_runtime_mode=mode)
    if draft is not None:
        context.is_draft_model = draft
    env = {
        "_FP4_DECODE_TP": 4,
        "is_forward_context_available": lambda: available,
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
    }
    eligible = load_function("attention.py", "_target_fp4_decode", env)
    assert eligible(tp) is expected


@pytest.mark.parametrize("opt_in", [False, True])
def test_public_api_requires_explicit_opt_in(opt_in):
    calls, output = [], object()

    def reference_conversion():
        raise RuntimeError("reference conversion reached")

    states = SimpleNamespace(float=reference_conversion)
    env = {
        "torch": SimpleNamespace(Tensor=object),
        "_group_shape": lambda *args: (24, 4, 32),
        "_try_fp4_e8m0_aiv": lambda *args: calls.append("e8m0") or output,
        "_try_fp4_e4m3_aiv": lambda *args: calls.append("e4m3") or output,
    }
    quantize = load_function("attention_ops.py", "fp4_quant_dequant", env)
    if opt_in:
        assert quantize(states, decode_fusion=True) is output
        assert calls == ["e8m0"]
    else:
        with pytest.raises(RuntimeError, match="reference conversion reached"):
            quantize(states)
        assert not calls


@pytest.mark.parametrize(
    "tp,available,mode,draft,expected",
    [
        (4, True, "none", False, True),
        (8, True, "none", False, False),
        (4, False, "none", False, False),
        (4, True, "full", False, False),
        (4, True, "piecewise", False, False),
        (4, True, "none", True, False),
        (4, True, "none", None, False),
    ],
)
def test_only_target_tp4_eager_prefill(tp, available, mode, draft, expected):
    context = SimpleNamespace(cudagraph_runtime_mode=mode)
    if draft is not None:
        context.is_draft_model = draft
    env = {
        "_FP4_DECODE_TP": 4,
        "is_forward_context_available": lambda: available,
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(NONE="none"),
    }
    eligible = load_function("attention.py", "_target_fp4_prefill", env)
    assert eligible(tp) is expected


@pytest.mark.parametrize(
    "shape,compressed,expected",
    [
        ((1, 128), False, True),
        ((8192, 128), False, True),
        ((127, 32, 128), False, True),
        ((1, 512), True, True),
        ((8192, 512), True, True),
        ((0, 128), False, False),
        ((8193, 128), False, False),
        ((8193, 512), True, False),
        ((2, 64, 128), False, False),
        ((2, 32, 512), True, False),
        ((512,), True, False),
        ((1, 2, 32, 128), False, False),
    ],
)
def test_prefill_shape_boundaries(shape, compressed, expected):
    env = {
        "torch": SimpleNamespace(Tensor=object),
        "_FP4_PREFILL_MAX_ROWS": 8192,
        "_FP4_PREFILL_INDEX_HEADS": 32,
        "_FP4_COMPRESSED_DIM": 512,
        "_FP4_INDEX_DIM": 128,
    }
    eligible = load_function("attention_ops.py", "_fp4_prefill_shape", env)
    assert eligible(SimpleNamespace(shape=shape, ndim=len(shape)), compressed=compressed) is expected


@pytest.mark.parametrize("scale", ["e8m0", "e4m3"])
def test_prefill_public_opt_in_dispatch(scale):
    calls, output = [], object()

    def dispatch(*args, **kwargs):
        calls.append((args[2], kwargs))
        return output

    env = {
        "torch": SimpleNamespace(Tensor=object),
        "_group_shape": lambda *args: (127, 4, 32),
        "_try_fp4_e8m0_aiv": dispatch if scale == "e8m0" else lambda *a, **kw: None,
        "_try_fp4_e4m3_aiv": dispatch,
    }
    quantize = load_function("attention_ops.py", "fp4_quant_dequant", env)
    assert quantize(object(), scale_format=scale, prefill_fusion=True) is output
    assert calls == [(scale, {"prefill": True})]
