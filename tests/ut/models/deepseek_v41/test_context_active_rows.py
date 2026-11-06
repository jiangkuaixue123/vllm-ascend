# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-invocation reuse of identical context views without reading device data."""

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest


@dataclass
class Slots:
    values: list = field(default_factory=lambda: [0, -1, 7, -1])
    address: int = 100
    device: str = "npu:0"
    dtype: str = "int64"
    shape: tuple = (4,)
    strides: tuple = (1,)
    calls: list = field(default_factory=list)

    def data_ptr(self):
        return self.address

    def stride(self):
        return self.strides

    def __ge__(self, value):
        assert value == 0
        return self

    def nonzero(self):
        self.calls.append(self.address)
        return self

    def flatten(self):
        return tuple(i for i, v in enumerate(self.values) if v >= 0)


@pytest.fixture
def prepare():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_context_active_rows")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["prepare_context_active_rows"]


def test_identical_views_share_one_dynamic_index_result(prepare):
    slots = Slots()
    rows, count = prepare([slots, replace(slots), replace(slots)])
    assert count == 1 and len(slots.calls) == 1
    assert rows == [(0, 2)] * 3
    assert rows[0] is rows[1] is rows[2]


@pytest.mark.parametrize(
    "change",
    [
        {"address": 108},
        {"device": "npu:1"},
        {"dtype": "int32"},
        {"shape": (2, 2)},
        {"strides": (2,)},
    ],
)
def test_distinct_views_do_not_share_indices(prepare, change):
    slots = Slots()
    rows, count = prepare([slots, replace(slots, **change)])
    assert count == 2 and len(slots.calls) == 2
    assert rows == [(0, 2)] * 2


def test_refresh_and_empty_masks_are_not_cached_across_calls(prepare):
    slots = Slots()
    assert prepare([slots])[0] == [(0, 2)]
    slots.values[:] = [-1] * 4
    assert prepare([slots])[0] == [()]
    slots.values[:] = [0, 2, 3, 4]
    assert prepare([slots])[0] == [(0, 1, 2, 3)]
    assert len(slots.calls) == 3


def test_separate_groups_keep_different_reject_masks(prepare):
    first, second = Slots(), Slots(address=200, values=[-1, 0, -1, 9])
    rows, count = prepare([first, second, replace(first)])
    assert rows == [(0, 2), (1, 3), (0, 2)] and count == 2
