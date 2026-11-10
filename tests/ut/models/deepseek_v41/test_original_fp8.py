# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original checkpoint contracts: raw FP4 bits, block scales and FP8 Engram rows."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

from tests.ut.models.deepseek_v41.test_engram import write_shard
from vllm_ascend.models.deepseek_v41.attention_ops import fp8_quant_dequant
from vllm_ascend.models.deepseek_v41.engram import IndexedMmapStore, MmapEngramEmbedding
from vllm_ascend.models.deepseek_v41.fused_experts import _grouped_projection
from vllm_ascend.models.deepseek_v41.fused_weights import IndexedPart, _load_expert_tensor
from vllm_ascend.models.deepseek_v41.model import read_parameter_slice
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDSDynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod


def test_original_fp8_engram_index_and_shards_preserve_fractional_values(tmp_path):
    prefix = "layers.1.engram.embed"
    raw = torch.tensor([0.5, -1.5, 3.0]).repeat_interleave(32).to(torch.float8_e4m3fn)
    write_shard(
        tmp_path / "table.safetensors",
        {
            prefix + ".weight": ("F8_E4M3", (3, 32), bytes(raw.view(torch.uint8).tolist())),
            prefix + ".scale": ("F8_E8M0", (3, 1), bytes([127, 128, 126])),
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {prefix + suffix: "table.safetensors" for suffix in (".weight", ".scale")}})
    )
    ids = torch.tensor([2, 0, 1, 2])
    with IndexedMmapStore(tmp_path) as store:
        parts = [MmapEngramEmbedding(store, prefix, tp_rank=i, tp_size=2)(ids, device="cpu") for i in range(2)]
    expected = torch.tensor([1.5, 0.5, -3.0, 1.5]).repeat_interleave(32).reshape(4, 32).bfloat16()
    torch.testing.assert_close(sum(parts), expected, rtol=0, atol=0)


@pytest.mark.parametrize("axis,kind", [(0, ColumnParallelLinear), (1, RowParallelLinear)])
def test_fp8_dense_projection_dequantizes_only_local_tp_blocks(tmp_path, axis, kind):
    weight = torch.arange(128 * 128).remainder(11).reshape(128, 128).to(torch.float8_e4m3fn)
    scale = torch.arange(121, 137).to(torch.uint8).reshape(4, 4).view(torch.float8_e8m0fnu)
    path = tmp_path / "dense.safetensors"
    save_file({"proj.weight": weight, "proj.scale": scale}, str(path))
    layer = object.__new__(kind)
    shape = [128, 128]
    shape[axis] //= 2
    parameter = torch.empty(shape, dtype=torch.bfloat16)
    with safe_open(str(path), framework="pt") as reader:
        read_parameter_slice(reader, "proj.weight", parameter, layer, 1, 2)
    full = weight.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    torch.testing.assert_close(parameter, full.chunk(2, dim=axis)[1].bfloat16(), rtol=0, atol=0)


def test_fp8_dense_projection_requires_paired_scale(tmp_path):
    path = tmp_path / "missing.safetensors"
    save_file({"proj.weight": torch.ones(32, 32).to(torch.float8_e4m3fn)}, str(path))
    with safe_open(str(path), framework="pt") as reader, pytest.raises(ValueError, match="Unpaired"):
        read_parameter_slice(reader, "proj.weight", torch.empty(32, 32), nn.Linear(32, 32), 0, 1)


def test_original_fp4_expert_keeps_packed_bits_and_group32_scales(tmp_path):
    key = "layers.0.ffn.experts.0.w1"
    packed = torch.arange(64 * 32).remainder(256).to(torch.uint8).reshape(64, 32)
    scale = torch.full((64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    path = tmp_path / "expert.safetensors"
    save_file({key + ".weight": packed.view(torch.int8), key + ".scale": scale}, str(path))
    received = []

    def loader(**kwargs):
        received.append(kwargs["loaded_weight"].clone())
        return True

    config = SimpleNamespace(hidden_size=64, moe_intermediate_size=64)
    with safe_open(str(path), framework="pt") as reader:
        for suffix, dtype in (("weight", torch.uint8), ("scale", torch.float8_e8m0fnu)):
            parameter = nn.Parameter(torch.empty(1, dtype=dtype), requires_grad=False)
            parameter.weight_loader = loader
            _load_expert_tensor(reader, IndexedPart(key + "." + suffix, "packed", "w1", 0), parameter, config)
    assert torch.equal(received[0], packed)
    assert torch.equal(received[1].view(torch.uint8), scale.view(torch.uint8))


@pytest.mark.parametrize("block", [32, 128])
def test_native_linear_scale_expansion_matches_checkpoint_tiles(block):
    method = object.__new__(AscendW8A8MXFP8DSDynamicLinearMethod)
    method.block_size, method.group_size = block, 32
    raw = torch.arange(121, 121 + (256 // block) ** 2).reshape(256 // block, -1).to(torch.uint8)
    layer = SimpleNamespace(
        prefix="layers.0.attn.wq_a",
        weight=nn.Parameter(torch.ones(256, 256, dtype=torch.float8_e4m3fn), requires_grad=False),
        weight_scale=nn.Parameter(raw.view(torch.float8_e8m0fnu), requires_grad=False),
    )
    method.process_weights_after_loading(layer)
    actual = layer.weight_scale.transpose(0, 1).reshape(256, 8)
    expected = raw.repeat_interleave(block, 0).repeat_interleave(block // 32, 1)
    assert torch.equal(actual, expected)


def test_original_fp4_grouped_gemm_on_npu():
    if not torch.npu.is_available():
        pytest.skip("Requires A5")
    if "950" not in torch.npu.get_device_name(0):
        pytest.skip("Requires native A5 MXFP4")
    torch.npu.set_device(0)
    packed = torch.arange(2 * 64 * 32).remainder(256).to(torch.uint8).reshape(2, 64, 32)
    scales = torch.arange(2 * 64 * 2).remainder(5).add(125).to(torch.uint8).reshape(2, 64, 2)
    layer = SimpleNamespace(
        w13_weight=nn.Parameter(packed.to("npu"), requires_grad=False),
        w2_weight=nn.Parameter(packed.to("npu"), requires_grad=False),
        w13_weight_scale=nn.Parameter(scales.view(torch.float8_e8m0fnu).to("npu"), requires_grad=False),
        w2_weight_scale=nn.Parameter(scales.view(torch.float8_e8m0fnu).to("npu"), requires_grad=False),
    )
    method = object.__new__(AscendW4A8MXFPDSDynamicFusedMoEMethod)
    method.process_weights_after_loading(layer)
    x = torch.randn(8, 64, generator=torch.Generator().manual_seed(9)).bfloat16().npu()
    q, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=1)
    result = _grouped_projection(
        q, layer.w13_weight, torch.tensor([4, 4], device="npu"), 1, layer.w13_weight_scale, scale, mxfp=True
    ).cpu()
    decoded_x = (
        q.cpu().float().reshape(8, 2, 32)
        * torch.exp2(scale.cpu().view(torch.uint8).float().reshape(8, 2) - 127)[..., None]
    )
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    weights = lut[codes] * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    expected = torch.cat([decoded_x.flatten(1)[i * 4 : (i + 1) * 4] @ weights[i].T for i in range(2)]).bfloat16()
    torch.testing.assert_close(result, expected, rtol=0.01, atol=0.125)


def test_original_fp8_activation_and_linear_on_npu():
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(0):
        pytest.skip("Requires native A5 MXFP8")
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(71)
    x = torch.randn(8, 256, generator=generator).bfloat16().npu()
    q, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=1)
    decoded = (
        q.cpu().float().reshape(8, 8, 32)
        * torch.exp2(scale.cpu().view(torch.uint8).float().reshape(8, 8) - 127)[..., None]
    )
    torch.testing.assert_close(decoded.flatten(1).bfloat16(), fp8_quant_dequant(x.cpu()), rtol=0, atol=0)
    weight = torch.randn(128, 256, generator=generator).to(torch.float8_e4m3fn)
    raw_scale = torch.arange(32).remainder(5).add(125).to(torch.uint8).reshape(4, 8)
    scale_value = raw_scale.view(torch.float8_e8m0fnu).float()
    layer = SimpleNamespace(
        prefix="layers.0.attn.wq_a",
        weight=nn.Parameter(weight.npu(), requires_grad=False),
        weight_scale=nn.Parameter(scale_value.npu(), requires_grad=False),
    )
    method = object.__new__(AscendW8A8MXFP8DSDynamicLinearMethod)
    method.block_size, method.group_size, method.dynamic_mx_quant_scale_alg = 32, 32, 1
    method.process_weights_after_loading(layer)
    result = method.apply(layer, x).cpu()
    expected = decoded.flatten(1) @ (weight.float() * scale_value.repeat_interleave(32, 0).repeat_interleave(32, 1)).T
    torch.testing.assert_close(result.float(), expected.bfloat16().float(), rtol=0.01, atol=0.125)
