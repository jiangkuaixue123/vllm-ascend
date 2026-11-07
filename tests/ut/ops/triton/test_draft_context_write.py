# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host eligibility rejects layouts that cannot use masked draft context writes."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple
    dtype: str
    device: str = "npu"
    strides: tuple = (528, 1)
    address: int = 100000
    negative: bool = False
    conjugate: bool = False

    def __post_init__(self):
        object.__setattr__(self, "device", SimpleNamespace(type=self.device))

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, dimension):
        return self.strides[dimension]

    def data_ptr(self):
        return self.address

    def element_size(self):
        return 2

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate


@pytest.fixture
def case():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/deepseek_v41_draft_context.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and node.name == "supports_draft_context_write"
    ]
    namespace = {
        "torch": SimpleNamespace(bfloat16="bf16", int32="i32", int64="i64", is_inference_mode_enabled=lambda: True),
        "get_ascend_device_type": lambda: "a3",
        "AscendDeviceType": SimpleNamespace(A3="a3"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, [
        TensorMetadata((1024, 512), "bf16"),
        TensorMetadata((24,), "i64", strides=(1,), address=5000000),
        TensorMetadata((24, 512), "bf16", address=6000000),
    ]


def changed(tensor, **kwargs):
    return replace(tensor, device=tensor.device.type, **kwargs)


@pytest.mark.parametrize("rows", [1, 6, 24, 48])
def test_supported_decode_context(case, rows):
    namespace, tensors = case
    tensors[1] = changed(tensors[1], shape=(rows,))
    tensors[2] = changed(tensors[2], shape=(rows, 512))
    assert namespace["supports_draft_context_write"](*tensors)


@pytest.mark.parametrize(
    "index,updates",
    [
        (0, {"shape": (1024, 511)}),
        (0, {"strides": (511, 1)}),
        (0, {"strides": (1024, 2)}),
        (0, {"dtype": "fp32"}),
        (0, {"negative": True}),
        (1, {"shape": (23,)}),
        (1, {"dtype": "fp32"}),
        (1, {"strides": (2,)}),
        (2, {"shape": (49, 512)}),
        (2, {"shape": (0, 512)}),
        (2, {"dtype": "fp32"}),
        (2, {"address": 100002}),
        (2, {"conjugate": True}),
    ],
)
def test_unsupported_layout_keeps_parent(case, index, updates):
    namespace, tensors = case
    tensors[index] = changed(tensors[index], **updates)
    assert not namespace["supports_draft_context_write"](*tensors)


def test_cpu_autograd_and_other_hardware_keep_parent(case):
    namespace, tensors = case
    cpu = [replace(tensor, device="cpu") for tensor in tensors]
    assert not namespace["supports_draft_context_write"](*cpu)
    namespace["torch"].is_inference_mode_enabled = lambda: False
    assert not namespace["supports_draft_context_write"](*tensors)
    namespace["torch"].is_inference_mode_enabled = lambda: True
    namespace["get_ascend_device_type"] = lambda: "a2"
    assert not namespace["supports_draft_context_write"](*tensors)
