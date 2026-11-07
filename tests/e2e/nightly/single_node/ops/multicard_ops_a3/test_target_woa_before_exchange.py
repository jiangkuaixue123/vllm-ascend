# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real TP4 HCCL, native W8A8 and original checkpoint layout comparison."""

import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from safetensors.torch import save_file
from torch import nn
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear

from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Attention
from vllm_ascend.models.deepseek_v41.fused_weights import load_indexed_v41_weights
from vllm_ascend.models.deepseek_v41.model import read_parameter_slice
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

TP = 4
HEADS = 64
HEAD_DIM = 512
GROUPS = 8
RANK = 1024
HIDDEN = 5120
GROUP_WIDTH = HEADS * HEAD_DIM // GROUPS
TARGET_LAYERS = 40
REPEATS = 5


class NativeQuantConfig:
    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(w8a8_dynamic.AscendW8A8DynamicLinearMethod())


class TPGroup:
    device_group = None

    def reduce_scatter(self, value, dim):
        assert dim == 0 and value.dtype == torch.float32
        output = value.new_empty(value.shape[0] // TP, value.shape[1])
        dist.reduce_scatter_tensor(output, value.contiguous())
        return output


def make_model(path, projected):
    model = nn.Module()
    model.model_path = path
    model.config = SimpleNamespace(o_groups=GROUPS, o_lora_rank=RANK)
    model.layers = nn.ModuleList([nn.Module()])
    layer = DeepseekV41Attention.__new__(DeepseekV41Attention)
    nn.Module.__init__(layer)
    layer.config = model.config
    layer.layer_id = 0
    layer.tp, layer.dim = TP, HEAD_DIM
    layer.local_heads, layer.groups = HEADS // TP, GROUPS // TP
    layer.dsa_cp = True
    layer.project_before_exchange = projected
    layer.sequence_group = TPGroup()
    up = ReplicatedLinear if projected else ColumnParallelLinear
    options = {"disable_tp": True} if projected else {}
    layer.wo_a = up(
        GROUP_WIDTH,
        GROUPS * RANK,
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="layers.0.attn.wo_a",
        **options,
    )
    layer.wo_b = RowParallelLinear(
        GROUPS * RANK,
        HIDDEN,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=NativeQuantConfig(),
        reduce_results=False,
        prefix="layers.0.attn.wo_b",
    )
    model.layers[0].attn = layer
    return model.eval(), layer


def exact(actual, expected):
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


def elapsed(call):
    dist.barrier()
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    begin = time.perf_counter_ns()
    call()
    host = (time.perf_counter_ns() - begin) / 1e6
    end.record()
    end.synchronize()
    return host, start.elapsed_time(end)


def native_worker(rank, init_uri, path):
    """One process per die keeps all collective ordering identical on TP4."""
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", init_method=init_uri, rank=rank, world_size=TP)
    path = Path(path)
    old_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
    torch.npu.config.allow_internal_format = True
    try:
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: TP)
            patch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: rank)
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: rank)
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: TP)
            patch.setattr("vllm_ascend.quantization.method_adapters.get_tensor_model_parallel_rank", lambda: rank)
            patch.setattr(
                w8a8_dynamic, "maybe_trans_nz", lambda value: torch_npu.npu_format_cast(value, ACL_FORMAT_FRACTAL_NZ)
            )
            if rank == 0:
                generator = torch.Generator(device="cpu").manual_seed(1907)
                values = {
                    "layers.0.attn.wo_a.weight": (
                        torch.randn(GROUPS * RANK, GROUP_WIDTH, generator=generator, dtype=torch.float32) * 0.02
                    ).bfloat16(),
                    "layers.0.attn.wo_b.weight": torch.randint(
                        -8, 9, (HIDDEN, GROUPS * RANK), generator=generator, dtype=torch.int8
                    ),
                    "layers.0.attn.wo_b.weight_scale": torch.linspace(0.001, 0.006, HIDDEN).reshape(-1, 1),
                    "layers.0.attn.wo_b.weight_offset": torch.zeros(HIDDEN, 1),
                }
                save_file(values, str(path / "original.safetensors"))
                (path / "index.json").write_text(json.dumps(dict.fromkeys(values, "original.safetensors")))
                del values
            dist.barrier()
            with torch.device("npu"):
                parent_model, parent = make_model(path, False)
                candidate_model, candidate = make_model(path, True)
            assert parent.wo_a.weight.shape == (GROUPS * RANK // TP, GROUP_WIDTH)
            assert candidate.wo_a.weight.shape == (GROUPS * RANK, GROUP_WIDTH)
            index = json.loads((path / "index.json").read_text())
            for model in (parent_model, candidate_model):
                assert load_indexed_v41_weights(model, index, rank, TP, read_parameter_slice) == set(
                    dict(model.named_parameters())
                )
            exact(parent.wo_a.weight, candidate.wo_a.weight.chunk(TP, dim=0)[rank])
            for field in ("weight", "weight_scale", "weight_offset"):
                torch.testing.assert_close(getattr(parent.wo_b, field), getattr(candidate.wo_b, field), rtol=0, atol=0)
            for layer in (parent, candidate):
                layer.wo_b.quant_method.process_weights_after_loading(layer.wo_b)
                assert torch_npu.get_npu_format(layer.wo_b.weight) == ACL_FORMAT_FRACTAL_NZ
            torch.manual_seed(2000 + rank)
            for rows in (1, 6, 12, 64, 257):
                for scale in (0.0, 0.01, 1.0, 8.0):
                    states = torch.randn(rows, HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16) * scale
                    expected = parent._project_output(states.clone(), torch.bfloat16)
                    observed = candidate._project_output(states.clone(), torch.bfloat16)
                    exact(observed, expected)
            timings = []
            for rows in (6, 12):
                states = torch.randn(rows, HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
                graphs, outputs = [], []
                for layer in (parent, candidate):
                    layer._project_output(states, torch.bfloat16)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        outputs.append(layer._project_output(states, torch.bfloat16))
                    graphs.append(graph)
                for scale in (0.0, 1.0, 8.0):
                    states.copy_(torch.randn_like(states) * scale)
                    for graph in graphs:
                        graph.replay()
                    exact(outputs[1], outputs[0])
                parent.wo_a.weight.mul_(1.01)
                candidate.wo_a.weight.mul_(1.01)
                for graph in graphs:
                    graph.replay()
                exact(outputs[1], outputs[0])
                functions = []
                for layer in (parent, candidate):

                    def run(layer=layer, states=states):
                        for _ in range(TARGET_LAYERS):
                            layer._project_output(states, torch.bfloat16)

                    run()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        run()
                    functions.append(graph.replay)
                samples = [[], []]
                for repeat in range(REPEATS):
                    for i in (repeat % 2, 1 - repeat % 2):
                        samples[i].append(elapsed(functions[i]))
                medians = [list(map(statistics.median, zip(*sample))) for sample in samples]
                timings.append({"rows": rows, "calls": TARGET_LAYERS, "parent": medians[0], "candidate": medians[1]})
            (path / f"rank-{rank}.json").write_text(json.dumps(timings))
    finally:
        torch.npu.config.allow_internal_format = old_format
        dist.destroy_process_group()


def test_native_tp4_output_projection_and_graph_replay(tmp_path):
    mp.spawn(native_worker, args=((tmp_path / "store").as_uri(), str(tmp_path)), nprocs=TP, join=True)
    ranks = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(TP)]
    for case in range(len(ranks[0])):
        print(
            "TARGET_WOA_BEFORE_EXCHANGE_COMPONENT "
            + json.dumps(
                {
                    "rows": ranks[0][case]["rows"],
                    "calls": TARGET_LAYERS,
                    "parent_ready_ms": max(rank[case]["parent"][1] for rank in ranks),
                    "candidate_ready_ms": max(rank[case]["candidate"][1] for rank in ranks),
                    "per_rank": [rank[case] for rank in ranks],
                    "scope": "complete TP4 communication and output projection, repeated same layer weights",
                }
            )
        )
