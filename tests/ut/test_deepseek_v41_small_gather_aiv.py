# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check opt-in communication scope without initializing an accelerator."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def policy(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/utils.py"
    tree = ast.parse(path.read_text())
    names = {"create_small_gather_hccl_options"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    config = SimpleNamespace(
        additional_config={"dsv41_small_gather_aiv": True},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            data_parallel_size=4,
            data_parallel_size_local=4,
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            enable_dbo=False,
            enable_eplb=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )
    runtime = ModuleType("vllm.config")
    runtime.get_current_vllm_config_or_none = lambda: config
    monkeypatch.setitem(sys.modules, "vllm.config", runtime)
    profile = SimpleNamespace(supports=Mock(return_value=True))
    namespace = {
        "_HCCL_PG_EXPANSION_AIV": 3,
        "torch_npu": SimpleNamespace(
            _C=SimpleNamespace(
                _distributed_c10d=SimpleNamespace(ProcessGroupHCCL=SimpleNamespace(Options=SimpleNamespace))
            )
        ),
        "create_hccl_pg_options": lambda name: SimpleNamespace(
            hccl_config={"hccl_buffer_size": 200, "group_name": name}
        ),
        "get_current_hardware_profile": lambda: profile,
        "HardwareCapability": SimpleNamespace(HCCL_AIV_TP_COMMUNICATION=object()),
        "logger": SimpleNamespace(info_once=Mock()),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, config, profile


def test_only_tp_gets_aiv(policy):
    namespace, _, _ = policy
    options = namespace["create_small_gather_hccl_options"]("tp")
    assert options.hccl_config == {
        "hccl_buffer_size": 200,
        "group_name": "tp_small_gather",
        "hccl_op_expansion_mode": 3,
    }


@pytest.mark.parametrize("group", ["mc2", "ep", "dp", "world", "dcp", "pcp", "pp", "anonymous"])
def test_other_groups_keep_options(policy, group):
    namespace, config, _ = policy
    config.additional_config["dsv41_small_gather_aiv"] = "invalid"
    assert namespace["create_small_gather_hccl_options"](group) is None


@pytest.mark.parametrize("additional", [None, {}, {"dsv41_small_gather_aiv": False}])
def test_default_is_unchanged(policy, additional):
    namespace, config, _ = policy
    config.additional_config = additional
    assert namespace["create_small_gather_hccl_options"]("tp") is None


@pytest.mark.parametrize("value", ["true", 1, 0, None, [], {}])
def test_non_boolean_is_rejected(policy, value):
    namespace, config, _ = policy
    config.additional_config["dsv41_small_gather_aiv"] = value
    with pytest.raises(ValueError, match="boolean"):
        namespace["create_small_gather_hccl_options"]("tp")


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 8),
        ("data_parallel_size", 2),
        ("data_parallel_size_local", 2),
        ("pipeline_parallel_size", 2),
        ("enable_expert_parallel", False),
        ("enable_dbo", True),
        ("enable_eplb", True),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
    ],
)
def test_unsupported_layout_rejected(policy, field, value):
    namespace, config, _ = policy
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError, match="DP4TP4"):
        namespace["create_small_gather_hccl_options"]("tp")


def test_other_model_rejected(policy):
    namespace, config, _ = policy
    config.model_config.hf_config.model_type = "deepseek_v4"
    with pytest.raises(ValueError, match="DP4TP4"):
        namespace["create_small_gather_hccl_options"]("tp")


def test_unsupported_hardware_rejected(policy):
    namespace, config, profile = policy
    profile.supports.return_value = False
    with pytest.raises(ValueError, match="DP4TP4"):
        namespace["create_small_gather_hccl_options"]("tp")


@pytest.fixture
def gather_path():
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/models/deepseek_v41/attention.py"
    tree = ast.parse(path.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Attention")
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_sequence_inputs")
    bf16 = object()
    full = object()
    context = SimpleNamespace(cudagraph_runtime_mode=full, is_draft_model=False)
    value = SimpleNamespace(shape=(6, 5120), dtype=bf16, device=SimpleNamespace(type="npu"))
    value.contiguous = lambda: value
    value.new_empty = lambda shape: SimpleNamespace(shape=shape)
    group = SimpleNamespace(
        world_size=4,
        rank_in_group=2,
        small_gather_device_group=object(),
        all_gather=Mock(return_value=SimpleNamespace(shape=(24, 5120))),
    )
    namespace = {
        "_SMALL_GATHER_TP_SIZE": 4,
        "_SMALL_GATHER_INPUT_SHAPE": (6, 5120),
        "torch": SimpleNamespace(bfloat16=bf16),
        "dist": SimpleNamespace(all_gather_into_tensor=Mock()),
        "CUDAGraphMode": SimpleNamespace(FULL=full),
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context,
        "logger": SimpleNamespace(info_once=Mock()),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace, SimpleNamespace(sequence_group=group, dsa_cp=True), value, context


def test_target_six_rows_uses_owned_group(gather_path):
    namespace, layer, value, _ = gather_path
    result, offset = namespace["_sequence_inputs"](layer, value, "uniform_decode")
    assert result.shape == (24, 5120) and offset == 12
    namespace["dist"].all_gather_into_tensor.assert_called_once_with(
        result, value, group=layer.sequence_group.small_gather_device_group
    )
    layer.sequence_group.all_gather.assert_not_called()


@pytest.mark.parametrize(
    "reason", ["twelve", "width", "dtype", "cpu", "tp8", "prefill", "draft", "eager", "no_context", "disabled"]
)
def test_all_other_gathers_keep_parent(gather_path, reason):
    namespace, layer, value, context = gather_path
    phase = "uniform_decode"
    if reason == "twelve":
        value.shape = (12, 5120)
    elif reason == "width":
        value.shape = (6, 512)
    elif reason == "dtype":
        value.dtype = object()
    elif reason == "cpu":
        value.device.type = "cpu"
    elif reason == "tp8":
        layer.sequence_group.world_size = 8
    elif reason == "prefill":
        phase = "prefill"
    elif reason == "draft":
        context.is_draft_model = True
    elif reason == "eager":
        context.cudagraph_runtime_mode = object()
    elif reason == "no_context":
        namespace["is_forward_context_available"] = lambda: False
    elif reason == "disabled":
        layer.sequence_group.small_gather_device_group = None
    namespace["_sequence_inputs"](layer, value, phase)
    namespace["dist"].all_gather_into_tensor.assert_not_called()
    layer.sequence_group.all_gather.assert_called_once_with(value, dim=0)
