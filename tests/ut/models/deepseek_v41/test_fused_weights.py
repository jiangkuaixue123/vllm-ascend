# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load real safetensors through the registered ModelSlim / RoutedExperts loader."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_ascend.models.deepseek_v41 import fused_weights
from vllm_ascend.models.deepseek_v41.model import read_parameter_slice
from vllm_ascend.quantization.configs.modelslim_config import (
    _make_modelslim_moe_weight_loader,
    get_packed_modules_mapping,
    get_quant_type_for_layer,
)
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod


def tiny_model(tmp_path, root, *, expert_map, moe_tp):
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=6, moe_intermediate_size=4)
    model.model_path = tmp_path
    blocks = nn.ModuleList([nn.Module()])
    setattr(model, root, blocks)
    blocks[0].ffn = nn.Module()
    blocks[0].ffn.experts = nn.Module()
    model.ordinary = nn.Linear(6, 2, bias=False)
    # Use the actual vLLM registered packing and TP loader, without launching
    # an NPU or a distributed group just to construct the expert placement.
    owner = object.__new__(fused_weights.RoutedExperts)
    nn.Module.__init__(owner)
    blocks[0].ffn.experts.routed_experts = owner
    owner.layer_name = f"{root}.0.ffn.experts"
    owner.global_num_experts = 4
    owner.local_num_experts = 4 if expert_map is None else sum(e >= 0 for e in expert_map)
    mapping = torch.tensor(expert_map) if expert_map is not None else None
    owner.expert_map_manager = SimpleNamespace(
        expert_map=mapping,
        map_global_to_local=lambda global_id: global_id if expert_map is None else expert_map[global_id],
    )
    owner.moe_config = SimpleNamespace(
        tp_rank=moe_tp - 1, tp_size=moe_tp, is_act_and_mul=True, moe_parallel_config=SimpleNamespace(tp_size=moe_tp)
    )
    owner.quant_config = None
    owner.quant_method = object.__new__(AscendFusedMoEMethod)
    owner.quant_method.quant_method = object.__new__(AscendW8A8DynamicFusedMoEMethod)
    owner.weight_loader = _make_modelslim_moe_weight_loader(owner.weight_loader)
    owner.quant_method.create_weights(
        owner, owner.local_num_experts, 6, 4 // moe_tp, torch.bfloat16, weight_loader=owner.weight_loader
    )
    return model, owner


def write_checkpoint(tmp_path, owner, *, malformed=None):
    parts = [{"ordinary.weight": torch.ones(2, 6)}, {}]
    per_expert = {}
    for expert in range(4):
        for i, projection in enumerate(("w1", "w3", "w2")):
            outs, ins = (6, 4) if projection == "w2" else (4, 6)
            weight = (torch.arange(outs * ins).reshape(outs, ins) + expert * 25 + i * 2).to(torch.int8)
            scale = torch.arange(outs).float().reshape(-1, 1) + expert * 32 + i * 4 + 0.25
            for field, value in (("weight", weight), ("weight_scale", scale), ("weight_offset", -scale - 0.5)):
                name = f"{owner.layer_name}.{expert}.{projection}.{field}"
                per_expert[name] = value
                if name != malformed:
                    parts[i % 2][name] = value
    index = {}
    for i, tensors in enumerate(parts):
        shard = f"part-{i}.safetensors"
        save_file(tensors, str(tmp_path / shard))
        index.update({name: shard for name in tensors})
    return index, per_expert


@pytest.mark.parametrize("root", ["layers", "mtp"])
@pytest.mark.parametrize("expert_map, moe_tp", [([-1, -1, 0, 1], 1), (None, 2)])
def test_modelslim_packs_each_source_once_with_ep_or_tp(tmp_path, root, expert_map, moe_tp):
    model, owner = tiny_model(tmp_path, root, expert_map=expert_map, moe_tp=moe_tp)
    index, sources = write_checkpoint(tmp_path, owner)
    pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
    if expert_map is not None:
        # Off-rank entries must not cause their shards to be opened.
        for name in sources:
            if int(name.split(".")[4]) < 2:
                index[name] = "off-rank-unavailable.safetensors"
    loaded = fused_weights.load_indexed_v41_weights(model, index, 7, 8, read_parameter_slice)
    assert loaded == set(pointers)
    assert pointers == {name: p.data_ptr() for name, p in model.named_parameters()}
    local_ids = [e for e in range(4) if expert_map is None or expert_map[e] >= 0]

    def source(expert, projection, field):
        data = sources[f"{root}.0.ffn.experts.{expert}.{projection}.{field}"]
        if moe_tp > 1 and (projection != "w2" or field == "weight"):
            axis = 1 if projection == "w2" else 0
            data = data.chunk(moe_tp, dim=axis)[-1]
        return data

    for local, expert in enumerate(local_ids):
        for field in ("weight", "weight_scale", "weight_offset"):
            w13 = getattr(owner, "w13_" + field)[local]
            w2 = getattr(owner, "w2_" + field)[local]
            expected13 = torch.cat((source(expert, "w1", field), source(expert, "w3", field)))
            torch.testing.assert_close(w13, expected13.to(w13.dtype), rtol=0, atol=0)
            torch.testing.assert_close(w2, source(expert, "w2", field).to(w2.dtype), rtol=0, atol=0)


def test_strict_coverage_and_shape_for_each_logical_part(tmp_path):
    model, owner = tiny_model(tmp_path, "mtp", expert_map=[0, 1, -1, -1], moe_tp=1)
    missing = "mtp.0.ffn.experts.1.w3.weight_offset"
    index, _ = write_checkpoint(tmp_path, owner, malformed=missing)
    with pytest.raises(KeyError, match="w3.weight_offset"):
        fused_weights.load_indexed_v41_weights(model, index, 0, 8, read_parameter_slice)
    index[missing] = "bad.safetensors"
    save_file({missing: torch.ones(1, 4)}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError, match="Invalid V4.1 expert tensor shape"):
        fused_weights.load_indexed_v41_weights(model, index, 0, 8, read_parameter_slice)


def test_native_postload_layout_preserves_gate_up_and_channel_order(tmp_path, monkeypatch):
    model, owner = tiny_model(tmp_path, "layers", expert_map=[-1, -1, 0, 1], moe_tp=1)
    index, _ = write_checkpoint(tmp_path, owner)
    fused_weights.load_indexed_v41_weights(model, index, 0, 8, read_parameter_slice)
    before = {name: p.detach().clone() for name, p in owner.named_parameters()}
    formats = []

    def format_cast(tensor, fmt):
        # CPU checks the logical layout; the hardware format conversion is
        # exercised when the production Ascend scheme processes NPU weights.
        formats.append((tuple(tensor.shape), fmt))
        return tensor

    monkeypatch.setattr(w8a8_dynamic.torch_npu, "npu_format_cast", format_cast)
    monkeypatch.setattr(w8a8_dynamic, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=0))
    monkeypatch.setattr(w8a8_dynamic, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(w8a8_dynamic, "use_cann_megamoe", lambda _: False)
    owner.quant_method.quant_method.use_expert_weight_list = False
    owner.quant_method.process_weights_after_loading(owner)
    assert formats == [((2, 6, 8), 29), ((2, 4, 6), 29)]
    for projection in ("w13", "w2"):
        name = projection + "_weight"
        torch.testing.assert_close(getattr(owner, name), before[name].transpose(1, 2))
        for suffix in ("_scale", "_offset"):
            torch.testing.assert_close(getattr(owner, name + suffix), before[name + suffix].squeeze(-1))
    torch.testing.assert_close(owner.w13_weight_scale_fp32, before["w13_weight_scale"].squeeze(-1).float())


@pytest.mark.parametrize("root", ["layers", "mtp"])
def test_quant_dispatch_resolves_named_v41_experts(root):
    prefix = f"{root}.0.ffn.experts"
    desc = {prefix + ".0." + proj + ".weight": "W8A8_DYNAMIC" for proj in ("w1", "w3", "w2")}
    mapping = get_packed_modules_mapping("deepseek_v41")
    assert get_quant_type_for_layer(desc, prefix, "moe", mapping) == "W8A8_DYNAMIC"
    desc[prefix + ".0.w3.weight"] = "FLOAT"
    with pytest.raises(ValueError):
        get_quant_type_for_layer(desc, prefix, "moe", mapping)


class SharedQuantConfig:
    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(w8a8_dynamic.AscendW8A8DynamicLinearMethod())


def shared_model_and_checkpoint(path, root):
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=6, moe_intermediate_size=4)
    model.model_path = path
    blocks = nn.ModuleList([nn.Module()])
    setattr(model, root, blocks)
    blocks[0].ffn = nn.Module()
    shared = blocks[0].ffn.shared_experts = nn.Module()
    prefix = f"{root}.0.ffn.shared_experts"
    shared.w13 = fused_weights.MergedColumnParallelLinear(
        6, [4, 4], bias=False, disable_tp=True, quant_config=SharedQuantConfig(), prefix=prefix + ".w13"
    )
    sources, index = {}, {}
    for i, projection in enumerate(("w1", "w3")):
        values = {
            f"{prefix}.{projection}.weight": torch.full((4, 6), i + 1, dtype=torch.int8),
            f"{prefix}.{projection}.weight_scale": torch.full((4, 1), i + 0.25),
            f"{prefix}.{projection}.weight_offset": torch.zeros(4, 1),
        }
        name = f"shared-{i}.safetensors"
        save_file(values, str(path / name))
        index.update({key: name for key in values})
        sources.update(values)
    return model, shared.w13, sources, index


@pytest.mark.parametrize("root", ["layers", "mtp"])
def test_merged_shared_loads_original_parts_and_channel_order(tmp_path, root):
    model, linear, sources, index = shared_model_and_checkpoint(tmp_path, root)
    loaded = fused_weights.load_indexed_v41_weights(model, index, 3, 4, read_parameter_slice)
    assert loaded == set(dict(model.named_parameters()))
    assert linear.tp_rank == 0 and linear.tp_size == 1
    for field in ("weight", "weight_scale", "weight_offset"):
        expected = torch.cat([sources[f"{root}.0.ffn.shared_experts.{p}.{field}"] for p in ("w1", "w3")])
        torch.testing.assert_close(getattr(linear, field), expected, rtol=0, atol=0, check_dtype=False)
    for model_type in ("deepseek_v41", "deepseek_v41_text"):
        desc = {f"{root}.0.ffn.shared_experts.{p}.weight": "W8A8_DYNAMIC" for p in ("w1", "w3")}
        mapping = get_packed_modules_mapping(model_type)
        assert get_quant_type_for_layer(desc, f"{root}.0.ffn.shared_experts.w13", "linear", mapping) == "W8A8_DYNAMIC"


@pytest.mark.parametrize("failure", ["missing", "shape", "dtype"])
def test_merged_shared_rejects_missing_or_invalid_source(tmp_path, failure):
    model, _, sources, index = shared_model_and_checkpoint(tmp_path, "mtp")
    key = "mtp.0.ffn.shared_experts.w3.weight"
    if failure == "missing":
        del index[key]
        error, match = KeyError, "w3.weight"
    else:
        value = sources[key][:1] if failure == "shape" else sources[key].float()
        save_file({key: value}, str(tmp_path / "invalid.safetensors"))
        index[key] = "invalid.safetensors"
        error, match = ValueError, f"shared projection {failure}"
    with pytest.raises(error, match=match):
        fused_weights.load_indexed_v41_weights(model, index, 3, 4, read_parameter_slice)
