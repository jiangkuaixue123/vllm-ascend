# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real shared expert weights through merged loading, activation and W2."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from torch import nn

from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig
from vllm_ascend.models.deepseek_v41.core import clipped_swiglu
from vllm_ascend.models.deepseek_v41.fused_weights import load_indexed_v41_weights
from vllm_ascend.models.deepseek_v41.model import read_parameter_slice
from vllm_ascend.models.deepseek_v41.moe import DeepseekV41MoEExpert
from vllm_ascend.ops.triton.deepseek_v41_shared_activation import shared_gate_up_activation
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod


def exact(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    mask = ~torch.isnan(expected)
    assert torch.equal(actual.view(torch.int16)[mask], expected.view(torch.int16)[mask])


@pytest.mark.parametrize("rows", [3, 6, 12, 513, 2040])
@pytest.mark.parametrize("hidden", [576, 2304])
@torch.inference_mode()
def test_activation_preserves_fp32_silu_and_bf16_boundary(rows, hidden):
    x = torch.randn(rows, 2 * hidden, device="npu", dtype=torch.bfloat16) * 20
    expected = clipped_swiglu(*x.chunk(2, dim=-1)).to(torch.bfloat16)
    actual = shared_gate_up_activation(x, 10.0)
    assert actual is not None
    exact(actual, expected)


@torch.inference_mode()
def test_activation_all_bf16_words_and_graph_updates():
    x = torch.arange(32 * 4608, dtype=torch.int32, device="npu").to(torch.int16).view(torch.bfloat16).view(32, 4608)
    exact(shared_gate_up_activation(x, 10.0), clipped_swiglu(*x.chunk(2, dim=-1)).bfloat16())
    x = torch.randn(6, 4608, dtype=torch.bfloat16, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        shared_gate_up_activation(x, 10.0)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = shared_gate_up_activation(x, 10.0)
    for step in range(4):
        x.copy_(torch.randn_like(x) * (step * 10 + 1))
        graph.replay()
        exact(output, clipped_swiglu(*x.chunk(2, dim=-1)).bfloat16())
    assert shared_gate_up_activation(x.float(), 10.0) is None
    assert shared_gate_up_activation(x, 0.0) is None


class NativeQuantConfig:
    def __init__(self, merged):
        self.quant_description = (
            {f"layers.3.ffn.shared_experts.{name}.weight": "W8A8_DYNAMIC" for name in ("w1", "w3")} if merged else {}
        )

    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(AscendW8A8DynamicLinearMethod())


def make_expert(path, merged, *, replicated=True):
    config = DeepseekV41TextConfig()
    model = nn.Module()
    model.model_path, model.config = path, config
    model.layers = nn.ModuleDict({"3": nn.Module()})
    model.layers["3"].ffn = nn.Module()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("npu"):
            expert = DeepseekV41MoEExpert(
                config,
                NativeQuantConfig(merged),
                "layers.3.ffn.shared_experts",
                shared=True,
                replicated_shared=replicated,
                sequence_parallel=True,
            )
    finally:
        torch.set_default_dtype(previous)
    model.layers["3"].ffn.shared_experts = expert
    index = json.loads((path / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    rank = 0 if replicated else torch.distributed.get_rank()
    tp = 1 if replicated else torch.distributed.get_world_size()
    loaded = load_indexed_v41_weights(model, index, rank, tp, read_parameter_slice)
    assert loaded == set(dict(model.named_parameters()))
    for layer in expert.children():
        layer.quant_method.process_weights_after_loading(layer)
    assert expert._merged_shared_gate_up is merged
    return expert.eval()


@pytest.fixture(scope="module")
def experts():
    with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
        patch.setattr("vllm_ascend.utils.get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
        patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
        patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
        patch.setattr("vllm_ascend.quantization.method_adapters.get_tensor_model_parallel_rank", lambda: 0)
        path = Path("/home/admin/model").resolve()
        yield make_expert(path, False), make_expert(path, True)


@pytest.mark.parametrize("rows", [3, 6, 12, 64, 513, 2040])
@torch.inference_mode()
def test_real_checkpoint_and_down_projection(experts, rows):
    old, new = experts
    x = torch.randn(rows, 5120, device="npu", dtype=torch.bfloat16)
    pair = new.gate_up_proj(x).chunk(2, dim=-1)
    exact(pair[0], old.w1(x))
    exact(pair[1], old.w3(x))
    exact(new(x), old(x))
    factors = torch.full((rows, 1), 0.3, dtype=torch.float32, device="npu")
    exact(new(x, factors), old(x, factors))
    if rows > 12:
        return
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        new(x)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = new(x)
    for step in range(4):
        x.copy_(torch.randn_like(x) * (step + 1))
        graph.replay()
        exact(captured, old(x))
