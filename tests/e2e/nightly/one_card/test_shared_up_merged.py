# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load original W8A8 shards and compare complete shared experts on NPU."""

import json
import statistics
import time
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from safetensors.torch import save_file
from torch import nn

from vllm_ascend.models.deepseek_v41.fused_weights import load_indexed_v41_weights
from vllm_ascend.models.deepseek_v41.model import read_parameter_slice
from vllm_ascend.models.deepseek_v41.moe import DeepseekV41MoEExpert
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

HIDDEN = 5120
INTERMEDIATE = 2304
REPEATS = 5
TARGET_CALLS = 40
DRAFT_CALLS = 3


class NativeQuantConfig:
    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(w8a8_dynamic.AscendW8A8DynamicLinearMethod())


def make_model(path, root, merged):
    model = nn.Module()
    model.model_path = path
    model.config = SimpleNamespace(hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE, swiglu_limit=10.0)
    blocks = nn.ModuleList([nn.Module()])
    setattr(model, root, blocks)
    blocks[0].ffn = nn.Module()
    blocks[0].ffn.shared_experts = DeepseekV41MoEExpert(
        model.config,
        NativeQuantConfig(),
        f"{root}.0.ffn.shared_experts",
        shared=merged,
        replicated_shared=merged,
        sequence_parallel=merged,
    )
    return model.eval(), blocks[0].ffn.shared_experts


def exact(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@pytest.fixture(scope="module")
def experts(tmp_path_factory):
    torch.npu.set_device(0)
    torch.manual_seed(1707)
    path = tmp_path_factory.mktemp("merged_shared")
    patch = pytest.MonkeyPatch()
    patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 3)
    patch.setattr(w8a8_dynamic, "maybe_trans_nz", lambda value: torch_npu.npu_format_cast(value, ACL_FORMAT_FRACTAL_NZ))
    old_dtype = torch.get_default_dtype()
    old_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
    torch.npu.config.allow_internal_format = True
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("npu"):
            parent_model, parent = make_model(path, "layers", False)
            candidate_model, candidate = make_model(path, "layers", True)
        assert not parent._merged_up and candidate._merged_up
        assert candidate.w13.tp_size == 1 and candidate.w13.tp_rank == 0
        parts = [{}, {}]
        for i, projection in enumerate(("w1", "w3", "w2")):
            outputs, inputs = (HIDDEN, INTERMEDIATE) if projection == "w2" else (INTERMEDIATE, HIDDEN)
            prefix = f"layers.0.ffn.shared_experts.{projection}"
            parts[i % 2][prefix + ".weight"] = torch.randint(-8, 9, (outputs, inputs), dtype=torch.int8)
            parts[i % 2][prefix + ".weight_scale"] = torch.linspace(0.001, 0.006, outputs).reshape(-1, 1)
            parts[i % 2][prefix + ".weight_offset"] = torch.zeros(outputs, 1)
        index = {}
        for i, values in enumerate(parts):
            name = f"part-{i}.safetensors"
            save_file(values, str(path / name))
            index.update({key: name for key in values})
        for model in (parent_model, candidate_model):
            assert load_indexed_v41_weights(model, index, 3, 4, read_parameter_slice) == set(
                dict(model.named_parameters())
            )
        for field in ("weight", "weight_scale", "weight_offset"):
            expected = torch.cat([getattr(parent.w1, field), getattr(parent.w3, field)])
            torch.testing.assert_close(getattr(candidate.w13, field), expected, rtol=0, atol=0)
        for model in (parent_model, candidate_model):
            for module in model.modules():
                if isinstance(getattr(module, "quant_method", None), AscendLinearMethod):
                    module.quant_method.process_weights_after_loading(module)
                    assert torch_npu.get_npu_format(module.weight) == ACL_FORMAT_FRACTAL_NZ
        yield parent, candidate
    finally:
        torch.set_default_dtype(old_dtype)
        torch.npu.config.allow_internal_format = old_format
        patch.undo()


@pytest.mark.parametrize("rows", [1, 2, 5, 6, 12, 64, 512, 8192])
@torch.inference_mode()
def test_complete_shared_expert_and_projection_bits(experts, rows):
    parent, candidate = experts
    for scale in (1.0,) if rows == 8192 else (0.0, 0.01, 1.0, 64.0):
        states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16) * scale
        gate, up = candidate.w13(states).chunk(2, dim=-1)
        exact(gate, parent.w1(states))
        exact(up, parent.w3(states))
        exact(candidate(states), parent(states))
        assert candidate.shared_swiglu.gate_silu is not None


@torch.inference_mode()
def test_empty_native_matmuls_preserve_parent_rejection(experts):
    # The MoE caller skips shared experts for empty input. The underlying
    # native quantized matmul itself rejects zero rows on both paths.
    states = torch.empty(0, HIDDEN, device="npu", dtype=torch.bfloat16)
    for layer in experts:
        with pytest.raises(RuntimeError, match="npu_quant_matmul"):
            layer(states)


@pytest.mark.parametrize("rows", [2, 5, 6, 12])
@torch.inference_mode()
def test_complete_graph_replay_tracks_input_and_weight_updates(experts, rows):
    parent, candidate = experts
    states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16)
    graphs, outputs = [], []
    for layer in (parent, candidate):
        layer(states)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs.append(layer(states))
        graphs.append(graph)
    for scale in (1.0, 0.0, 4.0):
        states.copy_(torch.randn_like(states) * scale)
        for graph in graphs:
            graph.replay()
        exact(outputs[1], outputs[0])
    # Packed parameters remain the serving parameters, with no stale copied cache.
    parent.w1.weight_scale.mul_(1.01)
    candidate.w13.weight_scale[:INTERMEDIATE].mul_(1.01)
    for graph in graphs:
        graph.replay()
    exact(outputs[1], outputs[0])


def elapsed(call):
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    before = time.perf_counter_ns()
    call()
    host = (time.perf_counter_ns() - before) / 1e6
    end.record()
    end.synchronize()
    return host, start.elapsed_time(end)


@pytest.mark.parametrize("rows,graph_mode", [(2, False), (5, False), (6, True), (12, True)])
@torch.inference_mode()
def test_complete_shared_expert_timing(experts, rows, graph_mode):
    parent, candidate = experts
    states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16)
    calls = TARGET_CALLS if graph_mode else DRAFT_CALLS
    functions = []
    for layer in (parent, candidate):

        def run(layer=layer):
            for _ in range(calls):
                layer(states)

        run()
        if graph_mode:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run()
            functions.append(graph.replay)
        else:
            functions.append(run)
    samples = [[], []]
    for repeat in range(REPEATS):
        for index in (repeat % 2, 1 - repeat % 2):
            samples[index].append(elapsed(functions[index]))
    result = [list(map(statistics.median, zip(*sample))) for sample in samples]
    print(
        "SHARED_SWIGLU_TABLE_COMPONENT "
        + json.dumps(
            {
                "rows": rows,
                "graph": graph_mode,
                "calls": calls,
                "parent_host_ready_ms": result[0],
                "candidate_host_ready_ms": result[1],
                "scope": "complete shared expert, repeated same layer weights",
            }
        )
    )
