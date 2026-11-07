# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy request spans preserve exact lengths and synchronize only for eager use."""

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class Span:
    begin: int
    end: int
    seq_len: int


class Lengths:
    def __init__(self, values):
        self.values = values
        self.copies = 0

    def cpu(self):
        self.copies += 1
        return self

    def tolist(self):
        return list(self.values)


@pytest.fixture
def spans_class():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DSparkRequestSpans")
    namespace = {"Sequence": Sequence, "RequestSpan": Span}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["DSparkRequestSpans"]


def test_geometry_does_not_read_lengths_and_eager_materializes_once(spans_class):
    lengths = Lengths([11, 8])
    spans = spans_class([0, 5, 10], lengths)
    assert len(spans) == 2 and spans.query_bounds == (0, 5, 10)
    assert spans.device_seq_lens is lengths
    assert not spans.materialized and lengths.copies == 0
    assert list(spans) == [Span(0, 5, 11), Span(5, 10, 8)]
    assert spans.materialized and lengths.copies == 1
    lengths.values[:] = [99, 100]
    assert spans[:] == [Span(0, 5, 11), Span(5, 10, 8)]
    assert spans[-1].seq_len == 8 and lengths.copies == 1


def test_empty_and_out_of_range_spans(spans_class):
    lengths = Lengths([])
    spans = spans_class([0], lengths)
    assert len(spans) == 0 and lengths.copies == 0
    assert list(spans) == []
    with pytest.raises(IndexError):
        _ = spans[0]
    assert lengths.copies == 1
