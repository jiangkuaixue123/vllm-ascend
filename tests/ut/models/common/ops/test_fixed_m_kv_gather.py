# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-M projection scope and original TP row placement."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def projection():
    path = Path(__file__).resolve().parents[5] / "vllm_ascend/models/deepseek_v41/attention.py"
    tree = ast.parse(path.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Attention")
    method = next(
        node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_try_gather_fixed_m_kv"
    )

    class Unquantized:
        pass

    bf16, full = object(), object()
    context = SimpleNamespace(cudagraph_runtime_mode=full, is_draft_model=False)
    projected = Mock()
    local = projected.narrow.return_value
    local.contiguous.return_value = local
    padded = Mock()
    states = SimpleNamespace(
        shape=(6, 5120),
        ndim=2,
        dtype=bf16,
        device=SimpleNamespace(type="npu"),
        is_contiguous=lambda: True,
        new_zeros=Mock(return_value=padded),
    )
    wkv = Mock(return_value=(projected, None))
    wkv.quant_method = Unquantized()
    wkv.weight = SimpleNamespace(dtype=bf16, shape=(512, 5120))
    layer = SimpleNamespace(
        dsa_cp=True,
        tp=4,
        dim=512,
        owns_kv=False,
        wkv=wkv,
        sequence_group=SimpleNamespace(world_size=4, rank_in_group=0, all_gather=Mock()),
    )
    namespace = {
        "_FIXED_M_KV_LAYOUT": (4, 5120, 512),
        "_FIXED_M_KV_ROWS": (6, 12),
        "torch": SimpleNamespace(bfloat16=bf16, is_inference_mode_enabled=lambda: True),
        "AscendUnquantizedLinearMethod": Unquantized,
        "AscendDeviceType": SimpleNamespace(A3=3),
        "get_ascend_device_type": lambda: 3,
        "CUDAGraphMode": SimpleNamespace(FULL=full),
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context,
        "logger": SimpleNamespace(info_once=Mock()),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace, layer, states, context, padded, projected, local


@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_original_gemm_shape_and_row_offset(projection, rows, rank):
    namespace, layer, states, _, padded, projected, local = projection
    states.shape = (rows, 5120)
    layer.sequence_group.rank_in_group = rank
    result = namespace["_try_gather_fixed_m_kv"](layer, states)
    states.new_zeros.assert_called_once_with((rows * 4, 5120))
    padded.narrow.assert_called_once_with(0, rows * rank, rows)
    padded.narrow.return_value.copy_.assert_called_once_with(states)
    layer.wkv.assert_called_once_with(padded)
    projected.narrow.assert_called_once_with(0, rows * rank, rows)
    layer.sequence_group.all_gather.assert_called_once_with(local, dim=0)
    assert result is layer.sequence_group.all_gather.return_value


@pytest.mark.parametrize(
    "reason",
    [
        "source",
        "tp8",
        "dsa_disabled",
        "rows",
        "width",
        "dtype",
        "cpu",
        "noncontiguous",
        "method",
        "weight",
        "draft",
        "prefill",
        "context",
        "training",
        "hardware",
    ],
)
def test_unsupported_paths_keep_parent(projection, reason):
    namespace, layer, states, context, *_ = projection
    if reason == "source":
        layer.owns_kv = True
    elif reason == "tp8":
        layer.tp = layer.sequence_group.world_size = 8
    elif reason == "dsa_disabled":
        layer.dsa_cp = False
    elif reason == "rows":
        states.shape = (3, 5120)
    elif reason == "width":
        states.shape = (6, 4096)
    elif reason == "dtype":
        states.dtype = object()
    elif reason == "cpu":
        states.device.type = "cpu"
    elif reason == "noncontiguous":
        states.is_contiguous = lambda: False
    elif reason == "method":
        layer.wkv.quant_method = object()
    elif reason == "weight":
        layer.wkv.weight.dtype = object()
    elif reason == "draft":
        context.is_draft_model = True
    elif reason == "prefill":
        context.cudagraph_runtime_mode = object()
    elif reason == "context":
        namespace["is_forward_context_available"] = lambda: False
    elif reason == "training":
        namespace["torch"].is_inference_mode_enabled = lambda: False
    elif reason == "hardware":
        namespace["get_ascend_device_type"] = lambda: 2
    assert namespace["_try_gather_fixed_m_kv"](layer, states) is None
    layer.wkv.assert_not_called()
    layer.sequence_group.all_gather.assert_not_called()
