# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only eligibility checks for the singleton draft attention graph."""

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
    contiguous: bool = True
    negative: bool = False
    conjugate: bool = False
    column_stride: int = 1

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate

    def stride(self, index):
        assert index == 1
        return self.column_stride


@pytest.fixture
def case():
    source = Path(__file__).resolve().parents[5] / "vllm_ascend/models/common/ops/draft_attention_graph.py"
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.Assign)]
    selected += [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "supports_draft_attention_graph"
    ]
    torch = SimpleNamespace(bfloat16="bf16", int64="int64", int32="int32", is_inference_mode_enabled=lambda: True)
    namespace = {"torch": torch, "get_ascend_device_type": lambda: "a3", "AscendDeviceType": SimpleNamespace(A3="a3")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    device = SimpleNamespace(type="npu", index=0)
    attention = SimpleNamespace(
        training=False,
        config=SimpleNamespace(dspark_block_size=5, sliding_window=128),
        cache=SimpleNamespace(block_size=128),
        dim=512,
        heads=16,
        dsa_cp=False,
        wq_b=SimpleNamespace(gather_output=False),
    )
    metadata = SimpleNamespace(
        requests=[SimpleNamespace(begin=0, end=5, start_pos=8192)],
        num_actual_tokens=5,
        block_table_tensor=TensorMetadata((1, 1200), "int32", device),
    )
    return (
        namespace,
        attention,
        TensorMetadata((5, 5120), "bf16", device),
        TensorMetadata((5,), "int32", device),
        metadata,
    )


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_complete_singleton_and_boundary_history(case, dtype):
    namespace, *args = case
    args[2] = replace(args[2], dtype=dtype)
    for start in (128, 8192, 153595):
        args[-1].requests[0].start_pos = start
        assert namespace["supports_draft_attention_graph"](*args)


@pytest.mark.parametrize(
    "index,change",
    [
        (0, {"shape": (0, 5120)}),
        (0, {"shape": (20, 5120)}),
        (0, {"dtype": "fp32"}),
        (0, {"contiguous": False}),
        (0, {"negative": True}),
        (0, {"conjugate": True}),
        (0, {"device": SimpleNamespace(type="cpu", index=None)}),
        (1, {"shape": (6,)}),
        (1, {"dtype": "fp32"}),
        (1, {"contiguous": False}),
        (1, {"device": SimpleNamespace(type="npu", index=1)}),
    ],
)
def test_unsupported_tensor_metadata_keeps_parent(case, index, change):
    namespace, attention, hidden, positions, metadata = case
    tensors = [hidden, positions]
    tensors[index] = replace(tensors[index], **change)
    assert not namespace["supports_draft_attention_graph"](attention, *tensors, metadata)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "short",
        "partial",
        "padded",
        "multiple",
        "page_bound",
        "table_stride",
        "training",
        "gather",
        "global_heads",
        "device_generation",
        "autograd",
    ],
)
def test_unsupported_requests_and_execution_keep_parent(case, kind):
    namespace, attention, hidden, positions, metadata = case
    if kind == "missing":
        metadata = None
    elif kind == "short":
        metadata.requests[0].start_pos = 127
    elif kind == "partial":
        metadata.requests[0].end = 4
    elif kind == "padded":
        metadata.num_actual_tokens = 4
    elif kind == "multiple":
        metadata.requests.append(metadata.requests[0])
    elif kind == "page_bound":
        metadata.block_table_tensor = replace(metadata.block_table_tensor, shape=(1, 63))
    elif kind == "table_stride":
        metadata.block_table_tensor = replace(metadata.block_table_tensor, column_stride=2)
    elif kind == "training":
        attention.training = True
    elif kind == "gather":
        attention.wq_b.gather_output = True
    elif kind == "global_heads":
        attention.heads = 64
    elif kind == "device_generation":
        namespace["get_ascend_device_type"] = lambda: "a2"
    else:
        namespace["torch"].is_inference_mode_enabled = lambda: False
    assert not namespace["supports_draft_attention_graph"](attention, hidden, positions, metadata)


def test_four_requests_use_independent_history_and_page_bounds(case):
    namespace, attention, hidden, positions, metadata = case
    hidden = replace(hidden, shape=(20, 5120))
    positions = replace(positions, shape=(20,))
    metadata.requests = [
        SimpleNamespace(begin=row * 5, end=(row + 1) * 5, start_pos=start)
        for row, start in enumerate((128, 8192, 8191, 153595))
    ]
    metadata.block_table_tensor = replace(metadata.block_table_tensor, shape=(4, 1200))
    metadata.num_actual_tokens = 20
    assert namespace["supports_draft_attention_graph"](attention, hidden, positions, metadata)
    metadata.requests[-1].start_pos = 153601
    assert not namespace["supports_draft_attention_graph"](attention, hidden, positions, metadata)
