# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.models.deepseek_v41.attention import DeepseekV41RMSNorm
from vllm_ascend.models.deepseek_v41.core import make_identity_pre_mix
from vllm_ascend.models.deepseek_v41.flatten import (
    generate_forward,
    install_leaf_overrides,
    load_forward_source,
    specialization,
)
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Block, DeepseekV41ForCausalLM


class EffectfulSublayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, states, runtime=None, image_mask=None):
        self.calls.append(("forward", tuple(states.shape), image_mask is not None))
        return (states.float().sin() * 0.1).to(states.dtype)

    def forward_decode(self, states, active):
        self.calls.append(("decode", tuple(states.shape), None))
        return torch.where(active[:, None], states * 0.2, 0)


class Backbone(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12, hc_mult=4, rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20)
        self.layers = nn.ModuleList()
        for _ in range(3):
            block = object.__new__(DeepseekV41Block)
            nn.Module.__init__(block)
            block.config = self.config
            block.attn = EffectfulSublayer()
            block.ffn = EffectfulSublayer()
            block.attn_norm = DeepseekV41RMSNorm(12, self.config.rms_norm_eps).to(dtype)
            block.ffn_norm = DeepseekV41RMSNorm(12, self.config.rms_norm_eps).to(dtype)
            block.attn_norm.weight.data.fill_(1)
            block.ffn_norm.weight.data.fill_(1)
            for part in ("attn", "ffn"):
                for suffix, shape in (("fn", (24, 48)), ("base", (24,)), ("scale", (3,))):
                    block.register_parameter(f"hc_{part}_{suffix}", nn.Parameter(torch.randn(shape) * 0.1))
            self.layers.append(block)

    def forward(self, hidden_states, pre_mix, runtime, image_mask=None):
        for layer_id, layer in enumerate(self.layers):
            hidden_states, pre_mix = layer(hidden_states, pre_mix, runtime, image_mask=image_mask)
        return hidden_states, pre_mix


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["prefill", "decode", "image", "empty"])
def test_flattened_backbone_bitwise_equivalent_and_preserves_effects(dtype, mode):
    check_equivalence(dtype, mode, "cpu")


def check_equivalence(dtype, mode, device, graph=False):
    torch.manual_seed(17)
    model = Backbone(dtype).to(device)
    rows = 0 if mode == "empty" else 7
    states = torch.randn(rows, 4, 12, dtype=dtype, device=device)
    pre = make_identity_pre_mix(states)
    active = torch.arange(rows, device=device) % 2 == 0
    runtime = SimpleNamespace(decode=SimpleNamespace(active=active)) if mode == "decode" else None
    image_mask = active if mode == "image" else None
    original = model(states, pre, runtime, image_mask)
    effects = [(block.attn.calls[:], block.ffn.calls[:]) for block in model.layers]
    for block in model.layers:
        block.attn.calls.clear()
        block.ffn.calls.clear()
    source = generate_forward(Backbone.forward, DeepseekV41Block.forward, model.config, len(model.layers))
    namespace = {"torch": torch, "F": F}
    exec(compile(source, "<test-flat>", "exec"), namespace)
    actual = namespace["forward"](model, states, pre, runtime, image_mask)
    for expected, value in zip(original, actual):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert effects == [(block.attn.calls, block.ffn.calls) for block in model.layers]
    assert not any(isinstance(node, ast.For) for node in ast.walk(ast.parse(source)))
    assert "hc_mixes(" not in source and "hc_split_sinkhorn(" not in source
    assert "hc_pre(" not in source and "hc_post(" not in source
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            replayed = namespace["forward"](model, states, pre, runtime, image_mask)
        captured.replay()
        torch.npu.synchronize()
        for expected, value in zip(original, replayed):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["prefill", "decode", "image", "empty"])
def test_npu_flatten_eager_and_graph_equivalence(mode):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires an NPU")
    with torch.no_grad():
        check_equivalence(torch.bfloat16, mode, "npu:0", graph=mode == "decode")


@pytest.mark.parametrize("name", ["forward", "_forward_decode"])
def test_actual_model_source_expands_all_40_blocks(name):
    config = Backbone(torch.float32).config
    source = generate_forward(getattr(DeepseekV41ForCausalLM, name), DeepseekV41Block.forward, config, 40)
    compile(source, "<full-model-flat>", "exec")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.For) for node in ast.walk(tree))
    assert "self.layers[39]" in source
    assert "enumerate(self.layers)" not in source


def test_invalid_static_specialization_is_rejected():
    config = Backbone(torch.float32).config
    config.hc_sinkhorn_iters = 0
    with pytest.raises(ValueError, match="positive"):
        generate_forward(Backbone.forward, DeepseekV41Block.forward, config, 3)


def test_editable_bundle_requires_matching_specialization_and_recorded_source(tmp_path):
    config = Backbone(torch.float32).config
    config.num_hidden_layers = 3
    config.compress_ratios = [0] * 6
    source = generate_forward(Backbone.forward, DeepseekV41Block.forward, config, 3)
    path = tmp_path / "target_forward.py"
    path.write_text(source)
    manifest = {
        "specialization": specialization(config),
        "files": {path.name: {"sha256": hashlib.sha256(source.encode()).hexdigest()}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded, filename = load_forward_source(tmp_path, config, "forward")
    assert loaded == source and filename == str(path)
    path.write_text(source + "\n# new experiment\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_forward_source(tmp_path, config, "forward")
    config.hc_mult = 2
    with pytest.raises(ValueError, match="specialization"):
        load_forward_source(tmp_path, config, "forward")


def test_bundle_leaf_callbacks_are_instance_local_and_use_their_manifest(tmp_path):
    model = Backbone(torch.float32)
    model.config.num_hidden_layers = 3
    model.config.compress_ratios = [0] * 6
    other = Backbone(torch.float32)
    source = "def forward(self, states, runtime=None, image_mask=None):\n    return states + 7\n"
    (tmp_path / "attention.py").write_text(source)
    manifest = {
        "specialization": specialization(model.config),
        "files": {"attention.py": {"sha256": hashlib.sha256(source.encode()).hexdigest()}},
        "method_overrides": {"target": [{"path": "layers.*.attn", "file": "attention.py", "methods": ["forward"]}]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    install_leaf_overrides(model, tmp_path)
    states = torch.zeros(1, 12)
    for block in model.layers:
        torch.testing.assert_close(block.attn(states), states + 7)
    torch.testing.assert_close(other.layers[0].attn(states), states)
    (tmp_path / "attention.py").write_text(source + "# changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_leaf_overrides(model, tmp_path)
