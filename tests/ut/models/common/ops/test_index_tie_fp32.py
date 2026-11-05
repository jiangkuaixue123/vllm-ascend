# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility for exact floating-point cutoff-column selection."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize(
    "device,shape,k,inference,expected",
    [
        ("npu", (6, 76800), 512, True, True),
        ("npu", (12, 153600), 512, True, True),
        ("npu", (6, 153600), 512, True, True),
        ("npu", (12, 76800), 512, True, True),
        ("cpu", (6, 76800), 512, True, False),
        ("npu", (24, 76800), 512, True, False),
        ("npu", (6, 16777217), 512, True, False),
        ("npu", (6, 1024), 512, True, False),
        ("npu", (1, 6, 76800), 512, True, False),
        ("npu", (6, 76800), 128, True, False),
        ("npu", (6, 76800), 512, False, False),
    ],
)
def test_exact_tie_column_domain(device, shape, k, inference, expected):
    path = ROOT / "vllm_ascend/models/deepseek_v41/attention.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_use_float32_tie_columns")
    env = {
        "torch": SimpleNamespace(is_inference_mode_enabled=lambda: inference),
        "_DECODE_INDEX_SORT_ROWS": (6, 12),
        "_DECODE_INDEX_SORT_K": 512,
        "_DECODE_INDEX_TIE_WIDTHS": (76800, 153600),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), env)
    scores = SimpleNamespace(device=SimpleNamespace(type=device), shape=shape, ndim=len(shape))
    assert env["_use_float32_tie_columns"](scores, k) is expected
