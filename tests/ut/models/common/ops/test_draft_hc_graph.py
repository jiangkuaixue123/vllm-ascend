# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only coverage of the eager draft graph eligibility boundary."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple
    dtype: str
    device: object = "npu"
    contiguous: bool = True
    negative: bool = False
    conjugate: bool = False

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate


@pytest.fixture
def eligibility():
    root = Path(__file__).resolve().parents[5]
    source = root / "vllm_ascend/models/common/ops/draft_hc_graph.py"
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.Assign)]
    selected += [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "supports_draft_hc_graph"
    ]
    torch = SimpleNamespace(bfloat16="bf16", float32="fp32", is_inference_mode_enabled=lambda: True)
    namespace = {"torch": torch, "get_ascend_device_type": lambda: "a3", "AscendDeviceType": SimpleNamespace(A3="a3")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    device = SimpleNamespace(type="npu", index=0)
    tensors = [
        TensorMetadata(shape, dtype, device)
        for shape, dtype in [
            ((5, 4, 5120), "bf16"),
            ((5, 4), "fp32"),
            ((24, 20480), "fp32"),
            ((3,), "fp32"),
            ((24,), "fp32"),
        ]
    ]
    return namespace, tensors


@pytest.mark.parametrize("rows", [5])
def test_complete_supported_draft_blocks(eligibility, rows):
    namespace, tensors = eligibility
    tensors[0] = replace(tensors[0], shape=(rows, 4, 5120))
    tensors[1] = replace(tensors[1], shape=(rows, 4))
    assert namespace["supports_draft_hc_graph"](*tensors)


@pytest.mark.parametrize(
    "index,change",
    [
        (0, {"shape": (0, 4, 5120)}),
        (0, {"shape": (15, 4, 5120)}),
        (0, {"shape": (5, 4, 1024)}),
        (0, {"dtype": "fp32"}),
        (1, {"shape": (5, 3)}),
        (1, {"dtype": "bf16"}),
        (2, {"shape": (24, 10240)}),
        (2, {"dtype": "bf16"}),
        (3, {"shape": (4,)}),
        (4, {"shape": (25,)}),
        (0, {"contiguous": False}),
        (1, {"negative": True}),
        (2, {"conjugate": True}),
        (4, {"device": SimpleNamespace(type="npu", index=1)}),
        (0, {"device": SimpleNamespace(type="cpu", index=None)}),
    ],
)
def test_unsupported_metadata_keeps_parent(eligibility, index, change):
    namespace, tensors = eligibility
    tensors[index] = replace(tensors[index], **change)
    assert not namespace["supports_draft_hc_graph"](*tensors)


def test_training_and_other_device_generation_keep_parent(eligibility):
    namespace, tensors = eligibility
    namespace["torch"].is_inference_mode_enabled = lambda: False
    assert not namespace["supports_draft_hc_graph"](*tensors)
    namespace["torch"].is_inference_mode_enabled = lambda: True
    namespace["get_ascend_device_type"] = lambda: "a2"
    assert not namespace["supports_draft_hc_graph"](*tensors)


@pytest.mark.parametrize("rows", [10, 20, 40])
def test_multi_request_shapes_never_enter_graph(eligibility, rows):
    namespace, tensors = eligibility
    tensors[0] = replace(tensors[0], shape=(rows, 4, 5120))
    tensors[1] = replace(tensors[1], shape=(rows, 4))
    assert not namespace["supports_draft_hc_graph"](*tensors)
