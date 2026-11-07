# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only context graph eligibility checks."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple
    dtype: str
    device: object
    strides: tuple
    contiguous: bool = True
    negative: bool = False
    conjugate: bool = False

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, dimension):
        return self.strides[dimension]

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate


@pytest.fixture
def case():
    source = Path(__file__).resolve().parents[5] / "vllm_ascend/models/common/ops/draft_context_graph.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and node.name == "supports_draft_context_graph"
    ]
    namespace = {
        "torch": SimpleNamespace(bfloat16="bf16", int32="i32", int64="i64", is_inference_mode_enabled=lambda: True),
        "store_context_rows": object(),
        "get_ascend_device_type": lambda: "a3",
        "AscendDeviceType": SimpleNamespace(A3="a3"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    device = SimpleNamespace(type="npu", index=0)
    attention = SimpleNamespace(training=False, dim=512, rope_dim=64, tp=4, wkv=SimpleNamespace(gather_output=False))
    tensors = [
        TensorMetadata((6, 5120), "bf16", device, (5120, 1)),
        TensorMetadata((6,), "i32", device, (1,)),
        TensorMetadata((6,), "i64", device, (1,)),
        TensorMetadata((1024, 512), "bf16", device, (528, 1)),
    ]
    return namespace, attention, tensors


@pytest.mark.parametrize("rows", [6, 24])
@pytest.mark.parametrize("dtype", ["i32", "i64"])
def test_supported_rows_and_integer_inputs(case, rows, dtype):
    namespace, attention, tensors = case
    tensors[0] = replace(tensors[0], shape=(rows, 5120))
    tensors[1] = replace(tensors[1], shape=(rows,), dtype=dtype)
    tensors[2] = replace(tensors[2], shape=(rows,), dtype=dtype)
    assert namespace["supports_draft_context_graph"](attention, *tensors)


@pytest.mark.parametrize(
    "index,updates",
    [
        (0, {"shape": (5, 5120)}),
        (0, {"shape": (6, 4096)}),
        (0, {"dtype": "fp32"}),
        (0, {"contiguous": False}),
        (0, {"negative": True}),
        (1, {"shape": (5,)}),
        (1, {"dtype": "fp32"}),
        (1, {"contiguous": False}),
        (2, {"dtype": "fp32"}),
        (2, {"contiguous": False}),
        (3, {"shape": (0, 512)}),
        (3, {"shape": (1024, 511)}),
        (3, {"strides": (511, 1)}),
        (3, {"strides": (1024, 2)}),
        (3, {"dtype": "fp32"}),
        (3, {"conjugate": True}),
    ],
)
def test_unsupported_layouts_keep_parent(case, index, updates):
    namespace, attention, tensors = case
    tensors[index] = replace(tensors[index], **updates)
    assert not namespace["supports_draft_context_graph"](attention, *tensors)


@pytest.mark.parametrize("name,value", [("training", True), ("dim", 128), ("rope_dim", 32), ("tp", 8)])
def test_unsupported_model_geometry_keeps_parent(case, name, value):
    namespace, attention, tensors = case
    setattr(attention, name, value)
    assert not namespace["supports_draft_context_graph"](attention, *tensors)


def test_missing_kernel_cpu_autograd_and_hardware_keep_parent(case):
    namespace, attention, tensors = case
    original = namespace["store_context_rows"]
    namespace["store_context_rows"] = None
    assert not namespace["supports_draft_context_graph"](attention, *tensors)
    namespace["store_context_rows"] = original
    cpu = [replace(tensor, device=SimpleNamespace(type="cpu")) for tensor in tensors]
    assert not namespace["supports_draft_context_graph"](attention, *cpu)
    namespace["torch"].is_inference_mode_enabled = lambda: False
    assert not namespace["supports_draft_context_graph"](attention, *tensors)
    namespace["torch"].is_inference_mode_enabled = lambda: True
    namespace["get_ascend_device_type"] = lambda: "a2"
    assert not namespace["supports_draft_context_graph"](attention, *tensors)
