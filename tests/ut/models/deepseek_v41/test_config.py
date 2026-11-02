# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
from vllm.transformers_utils.config import get_config

from vllm_ascend.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
    register_config,
)


def test_v41_checkpoint_roundtrip_preserves_nested_metadata(tmp_path):
    """Loading a converted checkpoint must not drop its text/quantization fields."""
    rotation = {
        "value_projection_rotated": True,
        "value_basis": "quarot_global",
        "key_and_gate_basis": "original",
        "runtime_delta_rotation": False,
    }
    config = DeepseekV41Config(
        architectures=["DeepseekV41ForCausalLM"],
        dtype="bfloat16",
        quantization_config={"quant_method": "ascend"},
        engram_rotation_config=rotation,
        text_config={
            "index_topk": 256,
            "compress_ratios": [0, 2, 1],
            "kv_source_layer_ids": [1, 2],
            "checkpoint_extension": {"format_version": 1},
        },
        vision_config={"max_wh_ratio": 4.0},
    )
    config.save_pretrained(tmp_path)
    restored = DeepseekV41Config.from_pretrained(tmp_path)

    assert restored.model_type == "deepseek_v41"
    assert restored.architectures == ["DeepseekV41ForCausalLM"]
    assert restored.quantization_config == {"quant_method": "ascend"}
    assert isinstance(restored.get_text_config(), DeepseekV41TextConfig)
    assert restored.text_config.index_topk == 256
    assert restored.text_config.compress_ratios == [0, 2, 1]
    assert restored.text_config.kv_source_layer_ids == [1, 2]
    assert restored.engram_rotation_config == rotation
    assert restored.text_config.checkpoint_extension == {"format_version": 1}
    assert restored.vision_config.max_wh_ratio == 4.0


def test_v41_missing_fields_use_independent_reference_defaults():
    first = DeepseekV41Config(text_config={"hidden_size": 256})
    second = DeepseekV41Config()
    assert first.text_config.hidden_size == 256
    assert first.text_config.scoring_func == "sqrtsoftplus"
    assert first.text_config.rms_norm_eps == 1e-20
    assert second.text_config.hidden_size == 5120
    assert second.text_config.compress_ratios == [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
    first.text_config.engram_layer_ids.clear()
    first.text_config.rope_scaling["factor"] = 2
    assert second.text_config.engram_layer_ids == [1, 14]
    assert second.text_config.rope_scaling["factor"] == 16


def test_v41_subconfig_instances_and_empty_lists_are_preserved():
    text_config = DeepseekV41TextConfig(engram_layer_ids=[], engram_num_embeddings=[])
    vision_config = DeepseekV41VisionConfig()
    config = DeepseekV41Config(text_config=text_config, vision_config=vision_config)
    assert config.text_config is text_config
    assert config.vision_config is vision_config
    assert config.text_config.engram_layer_ids == []
    assert config.text_config.engram_num_embeddings == []


@pytest.mark.parametrize("name", ["text_config", "vision_config"])
def test_v41_invalid_subconfig_fails_at_load(name):
    with pytest.raises(TypeError, match=name):
        DeepseekV41Config(**{name: []})


@pytest.mark.parametrize("quant_method", ["fp8", "ascend"])
def test_get_config_loads_complete_v41_checkpoint(tmp_path, monkeypatch, quant_method):
    """Load the official nested schema without remote code or losing model fields.

    Fixture: deepseek-ai/DeepSeek-V4.1-Flash config.json at
    314113f8a7dadb388215396ba5fbee2c610a04d6 (MIT).
    """
    fixture = Path(__file__).parent / "fixtures" / "deepseek_v41_config.json"
    checkpoint = json.loads(fixture.read_text())
    if quant_method == "ascend":
        checkpoint["quantization_config"] = {"quant_method": "ascend"}
        checkpoint["engram_rotation_config"] = {
            "value_projection_rotated": True,
            "value_basis": "quarot_global",
            "key_and_gate_basis": "original",
            "runtime_delta_rotation": False,
        }
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", "0")
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    register_config()
    config = get_config(tmp_path, trust_remote_code=False)
    assert isinstance(config, DeepseekV41Config)
    assert config.architectures == ["DeepseekV41ForCausalLM"]
    assert config.quantization_config == checkpoint["quantization_config"]
    for name, value in checkpoint["text_config"].items():
        if name == "rope_scaling":
            # The framework adds rope_theta during standardization.
            for rope_key, rope_value in value.items():
                assert config.text_config.rope_scaling[rope_key] == rope_value
        else:
            assert getattr(config.get_text_config(), name) == value
    for name, value in checkpoint["vision_config"].items():
        assert getattr(config.vision_config, name) == value
    config.save_pretrained(tmp_path)
    restored = DeepseekV41Config.from_pretrained(tmp_path)
    assert restored.quantization_config == checkpoint["quantization_config"]
    if quant_method == "ascend":
        assert restored.engram_rotation_config == checkpoint["engram_rotation_config"]
