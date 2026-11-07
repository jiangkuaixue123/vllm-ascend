# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host metadata eligibility for complete local draft regions."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


def tensor(shape, dtype, device):
    value = SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device=device, contiguous=True)
    value.is_contiguous = lambda: value.contiguous
    value.is_neg = lambda: False
    value.is_conj = lambda: False
    return value


@pytest.fixture
def case():
    source = Path(__file__).resolve().parents[5] / "vllm_ascend/models/common/ops/draft_stage_graph.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and node.name == "supports_draft_stage_graph"
    ]
    env = {
        "math": math,
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32", is_inference_mode_enabled=lambda: True),
        "get_ascend_device_type": lambda: "a3",
        "AscendDeviceType": SimpleNamespace(A3="a3"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
    stage = SimpleNamespace(
        training=False,
        attn=SimpleNamespace(tp=4),
        config=SimpleNamespace(hc_mult=4, hidden_size=5120, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20),
    )
    return env, stage, SimpleNamespace(type="npu", index=0)


def inputs(rows, device, after_attention):
    values = (tensor((rows, 4, 5120), "bf16", device), tensor((rows, 4), "fp32", device))
    if after_attention:
        values = (tensor((rows, 5120), "bf16", device),) + values
        values += (tensor((rows, 4), "fp32", device), tensor((rows, 4, 4), "fp32", device))
    return values


@pytest.mark.parametrize("rows", [5, 20])
@pytest.mark.parametrize("after_attention", [False, True])
def test_supported_local_regions(case, rows, after_attention):
    env, stage, device = case
    assert env["supports_draft_stage_graph"](stage, inputs(rows, device, after_attention), after_attention)


@pytest.mark.parametrize("rows", [0, 1, 10, 40])
def test_other_batch_sizes_keep_parent(case, rows):
    env, stage, device = case
    assert not env["supports_draft_stage_graph"](stage, inputs(rows, device, False), False)


@pytest.mark.parametrize(
    "change", ["cpu", "training", "autograd", "generation", "tp", "dtype", "stride", "gate_shape", "eps", "count"]
)
def test_unsupported_metadata_keeps_parent(case, change):
    env, stage, device = case
    values = inputs(20, device, True)
    if change == "cpu":
        device.type = "cpu"
    elif change == "training":
        stage.training = True
    elif change == "autograd":
        env["torch"].is_inference_mode_enabled = lambda: False
    elif change == "generation":
        env["get_ascend_device_type"] = lambda: "a2"
    elif change == "tp":
        stage.attn.tp = 1
    elif change == "dtype":
        values[0].dtype = "fp32"
    elif change == "stride":
        values[-1].contiguous = False
    elif change == "gate_shape":
        values[2].shape = (20, 3)
    elif change == "eps":
        stage.config.rms_norm_eps = float("nan")
    elif change == "count":
        values = values[:-1]
    assert not env["supports_draft_stage_graph"](stage, values, True)
