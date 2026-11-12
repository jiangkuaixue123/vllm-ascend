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
    names = {"create_hccl_pg_options", "_deepseek_v41_tp_aiv_enabled"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    config = SimpleNamespace(
        additional_config={"dsv41_tp_aiv": True},
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
        "get_hccl_config_for_pg_options": lambda name: None if name == "mc2" else {"hccl_buffer_size": 200},
        "get_current_hardware_profile": lambda: profile,
        "HardwareCapability": SimpleNamespace(HCCL_AIV_TP_COMMUNICATION=object()),
        "logger": SimpleNamespace(info_once=Mock()),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, config, profile


def test_only_tp_gets_aiv(policy):
    namespace, _, _ = policy
    options = namespace["create_hccl_pg_options"]("tp")
    assert options.hccl_config == {"hccl_buffer_size": 200, "group_name": "tp", "hccl_op_expansion_mode": 3}


@pytest.mark.parametrize("group", ["mc2", "ep", "dp", "world", "dcp", "pcp", "pp", "anonymous"])
def test_other_groups_keep_options(policy, group):
    namespace, config, _ = policy
    config.additional_config["dsv41_tp_aiv"] = "invalid"
    options = namespace["create_hccl_pg_options"](group)
    assert "hccl_op_expansion_mode" not in options.hccl_config
    assert options.hccl_config["group_name"] == group
    if group == "mc2":
        assert options.hccl_config == {"group_name": "mc2"}


@pytest.mark.parametrize("additional", [None, {}, {"dsv41_tp_aiv": False}])
def test_default_is_unchanged(policy, additional):
    namespace, config, _ = policy
    config.additional_config = additional
    assert namespace["_deepseek_v41_tp_aiv_enabled"](config) is False
    assert "hccl_op_expansion_mode" not in namespace["create_hccl_pg_options"]("tp").hccl_config


@pytest.mark.parametrize("value", ["true", 1, 0, None, [], {}])
def test_non_boolean_is_rejected(policy, value):
    namespace, config, _ = policy
    config.additional_config["dsv41_tp_aiv"] = value
    with pytest.raises(ValueError, match="boolean"):
        namespace["_deepseek_v41_tp_aiv_enabled"](config)


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
    with pytest.raises(ValueError, match="layout"):
        namespace["_deepseek_v41_tp_aiv_enabled"](config)


def test_other_model_rejected(policy):
    namespace, config, _ = policy
    config.model_config.hf_config.model_type = "deepseek_v4"
    with pytest.raises(ValueError, match="layout"):
        namespace["_deepseek_v41_tp_aiv_enabled"](config)


def test_unsupported_hardware_rejected(policy):
    namespace, config, profile = policy
    profile.supports.return_value = False
    with pytest.raises(ValueError, match="layout"):
        namespace["_deepseek_v41_tp_aiv_enabled"](config)


def test_no_context_keeps_parent(policy):
    namespace, _, _ = policy
    assert namespace["_deepseek_v41_tp_aiv_enabled"](None) is False
