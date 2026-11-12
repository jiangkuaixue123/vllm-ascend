# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard unsupported output layouts before importing or launching the kernel."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


class Tensor:
    def __init__(self, shape, dtype="bf16", **changes):
        self.__dict__.update(
            shape=shape,
            dtype=dtype,
            ndim=len(shape),
            device=SimpleNamespace(type="npu"),
            layout="strided",
            contiguous=True,
            negative=False,
            conjugated=False,
            requires_grad=False,
            offset=0,
            address=64,
            npu_format=2,
        )
        self.__dict__.update(changes)

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugated

    def storage_offset(self):
        return self.offset

    def data_ptr(self):
        return self.address

    def float(self):
        return self

    def unsqueeze(self, *args):
        return self

    def view(self, *args):
        return self

    def sin(self):
        return self

    def cos(self):
        return self

    def __neg__(self):
        return self

    def __mul__(self, other):
        return self


def dispatcher(rows=6):
    path = ROOT / "vllm_ascend/models/common/ops/target_output_send.py"

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    tree = RemoveImports().visit(ast.parse(path.read_text()))
    calls, sent = [], object()
    env = {
        "torch": SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            int32="int32",
            int64="int64",
            strided="strided",
            is_inference_mode_enabled=lambda: True,
            compiler=SimpleNamespace(is_compiling=lambda: False),
            ops=SimpleNamespace(_C_ascend=SimpleNamespace(inplace_partial_rotary_mul=object())),
            stack=lambda values, dim: values[0],
        ),
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": SimpleNamespace(A3="A3"),
        "get_npu_format": lambda value: value.npu_format,
        "HAS_TRITON": True,
        "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: SimpleNamespace(is_draft_model=False),
        "enable_custom_op": lambda: True,
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": SimpleNamespace(getpid=lambda: 123),
        "prepare_output_send": lambda *args: calls.append(args) or sent,
    }
    model = SimpleNamespace(
        dsa_cp=True,
        tp=4,
        sequence_group=SimpleNamespace(world_size=4),
        heads=64,
        local_heads=16,
        dim=512,
        groups=2,
        rope_dim=64,
        frequencies=Tensor((32,), "fp32"),
    )
    exec(compile(tree, str(path), "exec"), env)
    return env, model, Tensor((rows, 64, 512)), Tensor((rows,), "int64"), calls, sent


@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("npu_format", [0, 2])
def test_target_decode_dispatch(rows, npu_format):
    env, model, values, positions, calls, sent = dispatcher(rows)
    values.npu_format = npu_format
    assert env["try_target_output_send"](model, values, positions) is sent
    assert len(calls) == 1 and calls[0][0] is values


@pytest.mark.parametrize(
    "changes",
    [
        {"dsa_cp": False},
        {"tp": 8},
        {"sequence_group": None},
        {"sequence_group": SimpleNamespace(world_size=8)},
        {"heads": 32},
        {"local_heads": 8},
        {"dim": 128},
        {"groups": 1},
        {"rope_dim": 32},
        {"frequencies": Tensor((32,), "bf16")},
    ],
)
def test_other_models_fall_back(changes):
    env, model, values, positions, calls, _ = dispatcher()
    model.__dict__.update(changes)
    assert env["try_target_output_send"](model, values, positions) is None
    assert not calls


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": (5, 64, 512)},
        {"shape": (24, 64, 512)},
        {"shape": (6, 32, 512)},
        {"ndim": 4},
        {"dtype": "fp32"},
        {"device": SimpleNamespace(type="cpu")},
        {"layout": "sparse"},
        {"contiguous": False},
        {"negative": True},
        {"conjugated": True},
        {"requires_grad": True},
        {"offset": 1},
        {"address": 66},
        {"npu_format": 29},
    ],
)
def test_other_output_metadata_falls_back(changes):
    env, model, values, positions, calls, _ = dispatcher()
    values.__dict__.update(changes)
    assert env["try_target_output_send"](model, values, positions) is None
    assert not calls


@pytest.mark.parametrize(
    "mode", ["draft", "no_context", "unknown_role", "triton", "hardware", "batch", "custom", "training", "compiling"]
)
def test_unsupported_execution_falls_back(mode):
    env, model, values, positions, calls, _ = dispatcher()
    if mode == "draft":
        env["get_forward_context"] = lambda: SimpleNamespace(is_draft_model=True)
    elif mode == "no_context":
        env["is_forward_context_available"] = lambda: False
    elif mode == "unknown_role":
        env["get_forward_context"] = SimpleNamespace
    elif mode == "triton":
        env["HAS_TRITON"] = False
    elif mode == "hardware":
        env["get_ascend_device_type"] = lambda: "A2"
    elif mode == "batch":
        env["envs"].VLLM_BATCH_INVARIANT = True
    elif mode == "custom":
        env["enable_custom_op"] = lambda: False
    elif mode == "training":
        env["torch"].is_inference_mode_enabled = lambda: False
    else:
        env["torch"].compiler.is_compiling = lambda: True
    assert env["try_target_output_send"](model, values, positions) is None
    assert not calls
