# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound exact column representation and target prefill dispatch."""

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
    "shape,k,device,inference,expected",
    [
        ((128, 4096), 512, "npu", True, True),
        ((8192, 1 << 24), 512, "npu", True, True),
        ((128, 64), 16, "npu", True, True),
        ((0, 4096), 512, "npu", True, False),
        ((8193, 4096), 512, "npu", True, False),
        ((1, (1 << 24) + 1), 512, "npu", True, False),
        ((1, 1, 4096), 512, "npu", True, False),
        ((128, 4096), 0, "npu", True, False),
        ((128, 4096), 4097, "npu", True, False),
        ((128, 4096), 512, "cpu", True, False),
        ((128, 4096), 512, "npu", False, False),
    ],
)
def test_exact_id_range_and_execution(shape, k, device, inference, expected):
    env = {
        "_PREFILL_INDEX_MAX_ROWS": 8192,
        "_DECODE_INDEX_SORT_MAX_WIDTH": 1 << 24,
        "torch": SimpleNamespace(is_inference_mode_enabled=lambda: inference),
    }
    eligible = method("_use_prefill_narrow_columns", env)
    scores = SimpleNamespace(shape=shape, ndim=len(shape), device=SimpleNamespace(type=device))
    assert eligible(scores, k) is expected


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
    assert method("_target_prefill_index_narrow", env)(tp) is expected
