# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual checkpoint projections and full shared-expert graph comparisons."""

import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from safetensors import safe_open

from vllm_ascend.models.common.ops.shared_input_quant import shared_input_projections
from vllm_ascend.models.deepseek_v41.moe import DeepseekV41MoEExpert
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

HIDDEN = 5120
INTERMEDIATE = 2304
CALLS = 40
SAMPLES = 7


class NativeQuantConfig:
    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(w8a8_dynamic.AscendW8A8DynamicLinearMethod())


def exact(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.fixture(scope="module")
def experts():
    torch.npu.set_device(0)
    path = Path("/home/admin/model")
    index_file = path / "quant_model_weights.safetensors.index.json"
    if not index_file.exists():
        pytest.skip("actual checkpoint unavailable")
    index = json.loads(index_file.read_text())["weight_map"]
    config = SimpleNamespace(hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE, swiglu_limit=10.0)
    patch = pytest.MonkeyPatch()
    patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 3)
    patch.setattr(w8a8_dynamic, "maybe_trans_nz", lambda value: torch_npu.npu_format_cast(value, ACL_FORMAT_FRACTAL_NZ))
    old_dtype = torch.get_default_dtype()
    old_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
    torch.npu.config.allow_internal_format = True
    torch.set_default_dtype(torch.bfloat16)
    try:
        layers = []
        with torch.device("npu"):
            for reuse in (False, True):
                layers.append(
                    DeepseekV41MoEExpert(
                        config,
                        NativeQuantConfig(),
                        "layers.0.ffn.shared_experts",
                        shared=True,
                        replicated_shared=True,
                        sequence_parallel=True,
                        reuse_input_quant=reuse,
                    ).eval()
                )
        with torch.no_grad():
            for name, _ in layers[0].named_parameters():
                key = "layers.0.ffn.shared_experts." + name
                with safe_open(path / index[key], framework="pt", device="cpu") as reader:
                    value = reader.get_tensor(key)
                for layer in layers:
                    parameter = dict(layer.named_parameters())[name]
                    assert parameter.shape == value.shape
                    parameter.copy_(value)
            for layer in layers:
                for projection in (layer.w1, layer.w3, layer.w2):
                    projection.quant_method.process_weights_after_loading(projection)
                    assert torch_npu.get_npu_format(projection.weight) == ACL_FORMAT_FRACTAL_NZ
        yield layers
    finally:
        torch.set_default_dtype(old_dtype)
        torch.npu.config.allow_internal_format = old_format
        patch.undo()


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_actual_complete_expert_and_all_bf16_input_words(experts, rows):
    parent, candidate = experts
    for scale in (0.0, 0.01, 1.0, 64.0):
        states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16) * scale
        gate, up = shared_input_projections(candidate.w1, candidate.w3, states)
        exact(gate, parent.w1(states))
        exact(up, parent.w3(states))
        exact(candidate(states), parent(states))
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    for start in range(0, words.numel(), rows * HIDDEN):
        values = torch.zeros(rows * HIDDEN, dtype=torch.bfloat16)
        chunk = words[start : start + values.numel()]
        values[: chunk.numel()].copy_(chunk)
        states = values.reshape(rows, HIDDEN).npu()
        exact(candidate(states), parent(states))


@pytest.mark.parametrize("rows", [1, 3, 5, 20, 64])
@torch.inference_mode()
def test_other_rows_keep_original_projection_calls(experts, rows):
    parent, candidate = experts
    states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16)
    assert shared_input_projections(candidate.w1, candidate.w3, states) is None
    exact(candidate(states), parent(states))


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_refreshed_input_weights_and_paired_full_expert_chain(experts, rows):
    parent, candidate = experts
    states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16)
    graphs, outputs = [], []
    for layer in experts:
        layer(states)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            # Each call consumes the previous complete expert result.
            value = states
            for _ in range(CALLS):
                value = layer(value)
            outputs.append(value)
        graphs.append(graph)
    saved = [layer.w1.weight_scale.clone() for layer in experts]
    try:
        for scale in (0.0, 0.01, 1.0):
            states.copy_(torch.randn_like(states) * scale)
            for layer in experts:
                layer.w1.weight_scale.mul_(1.001)
            for graph in graphs:
                graph.replay()
            exact(outputs[1], outputs[0])
        samples = [[], []]
        for repeat in range(SAMPLES):
            for index in (repeat % 2, 1 - repeat % 2):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(begin.elapsed_time(end))
        print(
            {
                "rows": rows,
                "calls": CALLS,
                "parent_ms": statistics.median(samples[0]),
                "candidate_ms": statistics.median(samples[1]),
            }
        )
    finally:
        for layer, value in zip(experts, saved):
            layer.w1.weight_scale.copy_(value)
