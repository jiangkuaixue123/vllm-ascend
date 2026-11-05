# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded native index conversion and fallback without device imports."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

MAX_EXACT_WIDTH = 1 << 24
ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize(
    "device,shape,k,inference,expected",
    [
        ("npu", (6, 76800), 512, True, ["float", "sort", "long"]),
        ("npu", (12, 153600), 512, True, ["float", "sort", "long"]),
        ("npu", (6, MAX_EXACT_WIDTH), 512, True, ["float", "sort", "long"]),
        ("npu", (6, MAX_EXACT_WIDTH + 1), 512, True, ["sort"]),
        ("npu", (24, 76800), 512, True, ["sort"]),
        ("npu", (6, 76800), 128, True, ["sort"]),
        ("npu", (1, 6, 76800), 512, True, ["sort"]),
        ("npu", (6, 76800), 512, False, ["sort"]),
        ("cpu", (6, 76800), 512, True, ["sort"]),
    ],
)
def test_sort_width_and_dtype_contract(device, shape, k, inference, expected):
    path = ROOT / "vllm_ascend/models/deepseek_v41/attention.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sorted_topk_columns")
    calls = []

    class Indices:
        def float(self):
            calls.append("float")
            return self

        def sort(self, dimension):
            assert dimension == -1
            calls.append("sort")
            return SimpleNamespace(values=self)

        def long(self):
            calls.append("long")
            return self

    indices = Indices()
    scores = SimpleNamespace(device=SimpleNamespace(type=device), shape=shape, ndim=len(shape))
    selections = []
    env = {
        "topk_columns": lambda *args: selections.append(args) or indices,
        "torch": SimpleNamespace(
            int32="int32",
            iinfo=lambda dtype: SimpleNamespace(max=MAX_EXACT_WIDTH),
            is_inference_mode_enabled=lambda: inference,
        ),
        "_DECODE_INDEX_SORT_ROWS": (6, 12),
        "_DECODE_INDEX_SORT_K": 512,
        "_DECODE_INDEX_SORT_MAX_WIDTH": MAX_EXACT_WIDTH,
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": SimpleNamespace(getpid=lambda: 1),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    assert env["sorted_topk_columns"](scores, k) is indices
    assert selections == [(scores, k)]
    assert calls == expected
