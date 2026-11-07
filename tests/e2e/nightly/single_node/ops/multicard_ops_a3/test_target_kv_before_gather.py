# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native TP4 projection, rotary and paged-cache consumer equivalence."""

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
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41Cache,
    DeepseekV41DecodeMetadata,
    DeepseekV41RMSNorm,
)
from vllm_ascend.models.deepseek_v41.attention_ops import fp8_quant_dequant
from vllm_ascend.models.deepseek_v41.fused_weights import load_indexed_v41_weights
from vllm_ascend.models.deepseek_v41.model import read_parameter_slice
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

TP = 4
HIDDEN = 5120
HEAD_DIM = 512
ROPE_DIM = 64
BLOCK_SIZE = 128
BLOCKS_PER_ROW = 3
MAX_LENGTH = BLOCK_SIZE * BLOCKS_PER_ROW
NON_SOURCE_LAYERS = 36
REPEATS = 5


class NativeQuantConfig:
    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(w8a8_dynamic.AscendW8A8DynamicLinearMethod())


class TPGroup:
    world_size = TP

    @property
    def rank_in_group(self):
        return dist.get_rank()

    def all_gather(self, value, dim):
        assert dim == 0
        result = value.new_empty(value.shape[0] * TP, *value.shape[1:])
        dist.all_gather_into_tensor(result, value.contiguous())
        return result


def make_model(path, local_projection):
    model = nn.Module()
    model.model_path = path
    model.config = SimpleNamespace()
    model.layers = nn.ModuleList([nn.Module()])
    layer = DeepseekV41Attention.__new__(DeepseekV41Attention)
    nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.tp, layer.dim, layer.rope_dim = TP, HEAD_DIM, ROPE_DIM
    layer.ratio, layer.owns_kv = 2, False
    layer.dsa_cp = True
    layer.sequence_group = TPGroup()
    if not local_projection:
        layer._try_gather_projected_kv = lambda states: None
    layer.wkv = ReplicatedLinear(
        HIDDEN,
        HEAD_DIM,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=NativeQuantConfig(),
        prefix="layers.0.attn.wkv",
    )
    layer.kv_norm = DeepseekV41RMSNorm(HEAD_DIM, 1e-6)
    layer.register_buffer(
        "frequencies", 1.0 / (10000 ** (torch.arange(ROPE_DIM // 2, dtype=torch.float32) / (ROPE_DIM // 2)))
    )
    model.layers[0].attn = layer
    return model.eval(), layer


def exact(actual, expected):
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


def projection(layer, states):
    result = layer._try_gather_projected_kv(states)
    if result is None:
        gathered, _ = layer._sequence_inputs(states, "uniform_decode")
        result = layer.wkv(gathered)[0]
    return result


def make_cache(rows):
    cache = DeepseekV41Cache.__new__(DeepseekV41Cache)
    nn.Module.__init__(cache)
    cache.prefix = "kv-before-gather-test"
    cache.dim, cache.block_size, cache.source_ratio = HEAD_DIM, BLOCK_SIZE, 0
    cache.kv_cache = (
        torch.full((rows * BLOCKS_PER_ROW, BLOCK_SIZE, 1, HEAD_DIM), 1.25, device="npu", dtype=torch.bfloat16),
        torch.zeros(rows * BLOCKS_PER_ROW, BLOCK_SIZE, 1, 2, device="npu", dtype=torch.bfloat16),
    )
    return cache


def make_metadata(rows):
    index = torch.arange(rows, device="npu", dtype=torch.int64)
    pos = (index * 37) % MAX_LENGTH
    return DeepseekV41DecodeMetadata(
        seq_lens=pos + 1,
        slot_mapping=index * MAX_LENGTH + pos,
        block_table_tensor=(index[:, None] * BLOCKS_PER_ROW + torch.arange(BLOCKS_PER_ROW, device="npu")).int(),
        active=(index % 5 != 1),
        max_model_len=MAX_LENGTH,
        query_width=6,
        grouped_row_start=0,
    )


def consume(layer, cache, states, md):
    raw = layer.kv_norm(projection(layer, states))
    pos = md.seq_lens - 1
    layer.rotate(raw.unsqueeze(0), pos)
    values = fp8_quant_dequant(raw, 32)
    cache.decode_write(md, values, 0, 0, md.active)
    return values, cache.decode_read(md, pos[:, None], 0, 0, HEAD_DIM)


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
    """Load replicated native weights, compare cache consumers, then time graph replay on every rank."""
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
                generator = torch.Generator(device="cpu").manual_seed(3197)
                values = {
                    "layers.0.attn.wkv.weight": torch.randint(
                        -8, 9, (HEAD_DIM, HIDDEN), generator=generator, dtype=torch.int8
                    ),
                    "layers.0.attn.wkv.weight_scale": torch.linspace(0.001, 0.006, HEAD_DIM).reshape(-1, 1),
                    "layers.0.attn.wkv.weight_offset": torch.zeros(HEAD_DIM, 1),
                    "layers.0.attn.kv_norm.weight": torch.linspace(0.75, 1.25, HEAD_DIM),
                }
                save_file(values, str(path / "original.safetensors"))
                (path / "index.json").write_text(json.dumps(dict.fromkeys(values, "original.safetensors")))
            dist.barrier()
            with torch.device("npu"):
                parent_model, parent = make_model(path, False)
                candidate_model, candidate = make_model(path, True)
            index = json.loads((path / "index.json").read_text())
            for model in (parent_model, candidate_model):
                assert load_indexed_v41_weights(model, index, rank, TP, read_parameter_slice) == set(
                    dict(model.named_parameters())
                )
                model.layers[0].attn.wkv.quant_method.process_weights_after_loading(model.layers[0].attn.wkv)
            torch.manual_seed(4100 + rank)
            for rows in (1, 6, 12, 64, 257):
                for scale in (0.0, 0.01, 1.0, 8.0):
                    states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16) * scale
                    selected = candidate._try_gather_projected_kv(states)
                    assert (selected is not None) == (rows in (6, 12))
                    exact(projection(candidate, states), projection(parent, states))
            states = torch.randn(6, HIDDEN, device="npu", dtype=torch.bfloat16)
            for attr, value in (("owns_kv", True), ("dsa_cp", False), ("tp", 2)):
                previous = getattr(candidate, attr)
                setattr(candidate, attr, value)
                assert candidate._try_gather_projected_kv(states) is None
                setattr(candidate, attr, previous)
            scheme = candidate.wkv.quant_method.quant_method
            candidate.wkv.quant_method.quant_method = None
            assert candidate._try_gather_projected_kv(states) is None
            candidate.wkv.quant_method.quant_method = scheme
            assert candidate._try_gather_projected_kv(states.float()) is None
            assert (
                candidate._try_gather_projected_kv(
                    torch.empty(6, HIDDEN * 2, device="npu", dtype=torch.bfloat16)[:, ::2]
                )
                is None
            )
            with torch.inference_mode(False):
                assert candidate._try_gather_projected_kv(states) is None
            timings = []
            for rows in (6, 12):
                states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16)
                md = make_metadata(rows * TP)
                caches = [make_cache(rows * TP) for _ in range(2)]
                layers = (parent, candidate)
                graphs, outputs = [], []
                for layer, cache in zip(layers, caches):
                    consume(layer, cache, states, md)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        outputs.append(consume(layer, cache, states, md))
                    graphs.append(graph)
                for update in range(3):
                    states.copy_(torch.randn_like(states) * (0.01, 1.0, 8.0)[update])
                    index = torch.arange(rows * TP, device="npu", dtype=torch.int64)
                    pos = (index * 37 + update * 127) % MAX_LENGTH
                    md.seq_lens.copy_(pos + 1)
                    md.slot_mapping.copy_(index * MAX_LENGTH + pos)
                    md.active.copy_((index % 5 != 1) if update != 1 else torch.zeros_like(md.active))
                    before = [cache.field(0, 0, HEAD_DIM).clone() for cache in caches] if update == 1 else None
                    for graph in graphs:
                        graph.replay()
                    for observed, expected in zip(outputs[1], outputs[0]):
                        exact(observed, expected)
                    exact(caches[1].field(0, 0, HEAD_DIM), caches[0].field(0, 0, HEAD_DIM))
                    if before is not None:
                        for cache, expected in zip(caches, before):
                            exact(cache.field(0, 0, HEAD_DIM), expected)
                for layer in layers:
                    layer.wkv.weight.add_(1)
                for graph in graphs:
                    graph.replay()
                for observed, expected in zip(outputs[1], outputs[0]):
                    exact(observed, expected)
                functions = []
                for layer, cache in zip(layers, caches):

                    def run(layer=layer, cache=cache, states=states, md=md):
                        for _ in range(NON_SOURCE_LAYERS):
                            consume(layer, cache, states, md)

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
                timings.append(
                    {"rows": rows, "calls": NON_SOURCE_LAYERS, "parent": medians[0], "candidate": medians[1]}
                )
            (path / f"rank-{rank}.json").write_text(json.dumps(timings))
    finally:
        torch.npu.config.allow_internal_format = old_format
        dist.destroy_process_group()


def test_native_tp4_projected_kv_and_cache_graph_replay(tmp_path):
    mp.spawn(native_worker, args=((tmp_path / "store").as_uri(), str(tmp_path)), nprocs=TP, join=True)
    ranks = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(TP)]
    for case in range(len(ranks[0])):
        print(
            "TARGET_KV_BEFORE_GATHER_COMPONENT "
            + json.dumps(
                {
                    "rows": ranks[0][case]["rows"],
                    "calls": NON_SOURCE_LAYERS,
                    "parent_ready_ms": max(rank[case]["parent"][1] for rank in ranks),
                    "candidate_ready_ms": max(rank[case]["candidate"][1] for rank in ranks),
                    "per_rank": [rank[case] for rank in ranks],
                    "scope": "TP4 projection, normalization, rotary, quantization and paged-cache write/read",
                }
            )
        )
