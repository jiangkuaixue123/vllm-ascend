# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real TP4/TP8 query/rotary overlap through the raw-cache join and native consumers."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41Cache,
    DeepseekV41DecodeMetadata,
    DeepseekV41RMSNorm,
)
from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import enable_custom_op, npu_stream_switch, shared_experts_calculation_stream

SUPPORTED_TP = (4, 8)
HIDDEN = 5120
QUERY_RANK = 1280
HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
GROUPS = 8
OUTPUT_RANK = 1024
BLOCK_SIZE = 128
BLOCKS_PER_ROW = 3
MAX_LENGTH = BLOCK_SIZE * BLOCKS_PER_ROW
SELECTED_WIDTH = 640
CHECKPOINT_LAYER = 3
TIMED_CALLS = 40
TIMING_SAMPLES = 5
NUMERICAL_RTOL = 0.025
NUMERICAL_ATOL = 0.02


class NativeQuantConfig:
    def get_quant_method(self, layer, prefix):
        if prefix.endswith("wq_b"):
            return AscendLinearMethod(AscendW8A8DynamicLinearMethod())
        return AscendUnquantizedLinearMethod()


class TPGroup:
    @property
    def world_size(self):
        return dist.get_world_size()

    device_group = None

    @property
    def rank_in_group(self):
        return dist.get_rank()

    def all_gather(self, value, dim):
        assert dim == 0
        result = value.new_empty(value.shape[0] * dist.get_world_size(), *value.shape[1:])
        dist.all_gather_into_tensor(result, value.contiguous())
        return result

    def reduce_scatter(self, value, dim):
        assert dim == 0 and value.dtype == torch.float32
        result = value.new_empty(value.shape[0] // dist.get_world_size(), value.shape[1])
        dist.reduce_scatter_tensor(result, value.contiguous())
        return result


def checkpoint_tensor(path, index, suffix, selection=None):
    name = f"layers.{CHECKPOINT_LAYER}.attn.{suffix}"
    with safe_open(path / index[name], framework="pt", device="cpu") as source:
        value = source.get_tensor(name) if selection is None else source.get_slice(name)[selection]
    return value


def make_layer(path):
    """Load actual quantization types and original TP slices, without converting BF16 linears to W8A8."""
    index = json.loads((path / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    layer = DeepseekV41Attention.__new__(DeepseekV41Attention)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(o_lora_rank=OUTPUT_RANK)
    layer.layer_id = CHECKPOINT_LAYER
    layer.tp, layer.heads, layer.dim, layer.rope_dim = dist.get_world_size(), HEADS, HEAD_DIM, ROPE_DIM
    layer.local_heads, layer.groups = HEADS // dist.get_world_size(), GROUPS // dist.get_world_size()
    layer.ratio, layer.owns_kv, layer.dsa_cp = 0, False, True
    layer.sequence_group = TPGroup()
    layer.attention_kernel = DeepseekV41NativeAttention()
    layer.attention_kernel.sink_source_dtype = torch.bfloat16
    config = NativeQuantConfig()
    dimensions = {
        "wq_a": (HIDDEN, QUERY_RANK),
        "wq_b": (QUERY_RANK, HEADS * HEAD_DIM),
        "wkv": (HIDDEN, HEAD_DIM),
        "wo_a": (HEADS * HEAD_DIM // GROUPS, GROUPS * OUTPUT_RANK // dist.get_world_size()),
        "wo_b": (GROUPS * OUTPUT_RANK // dist.get_world_size(), HIDDEN),
    }
    rank = dist.get_rank()
    shard = GROUPS * OUTPUT_RANK // dist.get_world_size()
    for name, (input_size, output_size) in dimensions.items():
        with torch.device("npu"):
            linear = ReplicatedLinear(
                input_size,
                output_size,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=f"test.attn.{name}",
                disable_tp=True,
            )
        selection = None
        if name == "wo_a":
            selection = (slice(rank * shard, (rank + 1) * shard), slice(None))
        elif name == "wo_b":
            selection = (slice(None), slice(rank * shard, (rank + 1) * shard))
        weight = checkpoint_tensor(path, index, name + ".weight", selection)
        assert weight.dtype == (torch.int8 if name == "wq_b" else torch.bfloat16)
        assert weight.shape == linear.weight.shape
        linear.weight.copy_(weight)
        if name == "wq_b":
            for field in ("weight_scale", "weight_offset"):
                getattr(linear, field).copy_(checkpoint_tensor(path, index, name + "." + field))
        linear.quant_method.process_weights_after_loading(linear)
        setattr(layer, name, linear)
    for name, width in (("q_norm", QUERY_RANK), ("kv_norm", HEAD_DIM)):
        norm = DeepseekV41RMSNorm(width, 1e-6).to(device="npu", dtype=torch.bfloat16)
        norm.weight.copy_(checkpoint_tensor(path, index, name + ".weight"))
        setattr(layer, name, norm)
    sink = checkpoint_tensor(path, index, "attn_sink")
    assert sink.dtype == torch.bfloat16
    layer.attn_sink = nn.Parameter(sink.float().npu(), requires_grad=False)
    layer.register_buffer(
        "frequencies",
        (1.0 / (10000 ** (torch.arange(ROPE_DIM // 2, dtype=torch.float32) / (ROPE_DIM // 2)))).npu(),
    )
    layer._query_overlap_stream = shared_experts_calculation_stream()
    return layer.eval()


def make_metadata(rows):
    index = torch.arange(rows, device="npu", dtype=torch.int64)
    pos = (index * 37) % MAX_LENGTH
    return DeepseekV41DecodeMetadata(
        seq_lens=pos + 1,
        slot_mapping=index * MAX_LENGTH + pos,
        block_table_tensor=(index[:, None] * BLOCKS_PER_ROW + torch.arange(BLOCKS_PER_ROW, device="npu")).int(),
        active=index % 5 != 1,
        max_model_len=MAX_LENGTH,
        query_width=6,
        grouped_row_start=0,
    )


def make_cache(metadata):
    rows = metadata.active.shape[0]
    cache = DeepseekV41Cache.__new__(DeepseekV41Cache)
    nn.Module.__init__(cache)
    cache.prefix = "query-overlap-native"
    cache.dim, cache.block_size, cache.source_ratio = HEAD_DIM, BLOCK_SIZE, 0
    cache.kv_cache = (
        torch.full((rows * BLOCKS_PER_ROW, BLOCK_SIZE, 1, HEAD_DIM), 1.25, device="npu", dtype=torch.bfloat16),
        torch.zeros(rows * BLOCKS_PER_ROW, BLOCK_SIZE, 1, 2, device="npu", dtype=torch.bfloat16),
    )
    cache.metadata = lambda: SimpleNamespace(decode=metadata)
    return cache


def stage_cache(layer, query, metadata, runtime, extra):
    # Consume the updated original raw cache through the retained native FIA path.
    rows = query.shape[0]
    row = torch.arange(rows, device="npu", dtype=torch.int64) + dist.get_rank() * rows
    columns = metadata.seq_lens[:, None] - 1 - torch.arange(SELECTED_WIDTH, device="npu")
    valid = (columns >= 0) & metadata.active[:, None]
    slots = torch.where(valid, row[:, None] * MAX_LENGTH + columns, -1).int().unsqueeze(1)
    pages = layer.cache.kv_cache[0].reshape(rows, dist.get_world_size() * MAX_LENGTH, HEAD_DIM)
    return pages, slots, True


def exact(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


def capture(call):
    call()
    torch.npu.synchronize()
    dist.barrier()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = call()
    return graph, output


def time_complete(call):
    dist.barrier()
    torch.npu.synchronize()
    begin = time.perf_counter_ns()
    call()
    submitted = time.perf_counter_ns()
    torch.npu.synchronize()
    ended = time.perf_counter_ns()
    values = torch.tensor([(submitted - begin) / 1e6, (ended - begin) / 1e6], device="npu")
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return values.cpu().tolist()


@torch.inference_mode()
def run_case(layer, rows):
    """Exercise the actual decode method, cache/FIA/output consumers and both captured branches."""
    rank = dist.get_rank()
    torch.manual_seed(811 + rank)
    states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16)
    metadata = make_metadata(rows * dist.get_world_size())
    runtime = SimpleNamespace(decode=metadata)
    caches = [make_cache(metadata) for _ in range(2)]
    stream = layer._query_overlap_stream
    assert stream is not None
    pending = layer._start_decode_query(states, metadata)
    assert pending is not None
    torch.npu.current_stream().wait_event(pending[2])
    assert layer._start_decode_query(states.float(), metadata) is None
    assert layer._start_decode_query(states[:1], metadata) is None
    layer.dsa_cp = False
    assert layer._start_decode_query(states, metadata) is None
    layer.dsa_cp = True
    metadata.query_width = 1
    assert layer._start_decode_query(states, metadata) is None
    metadata.query_width = 6
    with npu_stream_switch(stream):
        assert layer._start_decode_query(states, metadata) is None
    with torch.inference_mode(False):
        assert layer._start_decode_query(states, metadata) is None
    layer._try_decode_slot_arena = lambda q, md, rt, extra: stage_cache(layer, q, md, rt, extra)
    parts = {}
    handles = []
    for name in ("q_norm", "wq_b", "kv_norm"):
        handles.append(
            getattr(layer, name).register_forward_hook(
                lambda module, args, output, name=name: parts.__setitem__(
                    name, output[0] if isinstance(output, tuple) else output
                )
            )
        )
    graphs, outputs, captured_parts, eager, eager_parts = [], [], [], [], []
    for index in range(2):
        layer.cache = caches[index]
        layer._query_overlap_stream = stream if index else None
        eager.append(layer._decode_forward(states, runtime).cpu())
        eager_parts.append({name: value.cpu() for name, value in parts.items()})
        graph, output = capture(lambda: layer._decode_forward(states, runtime))
        graphs.append(graph)
        outputs.append(output)
        captured_parts.append(dict(parts))
    torch.testing.assert_close(eager[1], eager[0], rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
    for name in eager_parts[0]:
        exact(eager_parts[1][name], eager_parts[0][name])
    for case in range(4):
        states.copy_(torch.randn_like(states) * (0.0, 0.01, 1.0, 8.0)[case])
        index = torch.arange(rows * dist.get_world_size(), device="npu", dtype=torch.int64)
        pos = (index * 37 + case * 127) % MAX_LENGTH
        metadata.seq_lens.copy_(pos + 1)
        metadata.slot_mapping.copy_(index * MAX_LENGTH + pos)
        metadata.active.copy_((index % 5 != 1) if case != 2 else torch.zeros_like(metadata.active))
        layer.wq_a.weight.mul_(1.001)
        layer.wq_b.weight_scale.mul_(1.001)
        layer.wkv.weight.mul_(1.001)
        before = [cache.field(0, 0, HEAD_DIM).cpu() for cache in caches] if case == 2 else None
        for graph in graphs:
            graph.replay()
            torch.npu.synchronize()
        for name in captured_parts[0]:
            exact(captured_parts[1][name].cpu(), captured_parts[0][name].cpu())
        exact(caches[1].field(0, 0, HEAD_DIM).cpu(), caches[0].field(0, 0, HEAD_DIM).cpu())
        torch.testing.assert_close(outputs[1].cpu(), outputs[0].cpu(), rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
        local_active = metadata.active[rank * rows : (rank + 1) * rows]
        assert torch.count_nonzero(outputs[1][~local_active].cpu()) == 0
        if before is not None:
            for cache, expected in zip(caches, before):
                exact(cache.field(0, 0, HEAD_DIM).cpu(), expected)
    for handle in handles:
        handle.remove()
    metadata.active.fill_(True)
    functions = []
    for index in range(2):
        layer.cache = caches[index]
        layer._query_overlap_stream = stream if index else None

        def complete():
            result = None
            for _ in range(TIMED_CALLS):
                result = layer._decode_forward(states, runtime)
            return result

        graph, _ = capture(complete)
        functions.append(graph.replay)
    samples = [[], []]
    for repeat in range(TIMING_SAMPLES):
        for index in (repeat % 2, 1 - repeat % 2):
            samples[index].append(time_complete(functions[index]))
    medians = [list(map(statistics.median, zip(*sample))) for sample in samples]
    layer._query_overlap_stream = stream
    if rank == 0:
        print(
            "TARGET_QUERY_OVERLAP_V2_COMPONENT "
            + json.dumps(
                {
                    "status": "PASS",
                    "rows": rows,
                    "world_size": dist.get_world_size(),
                    "calls": TIMED_CALLS,
                    "parent": medians[0],
                    "candidate": medians[1],
                    "metrics": ["host_ms", "ready_ms"],
                    "checkpoint_layer": CHECKPOINT_LAYER,
                    "scope": "decode Q/KV, gather, rotary, cache, FIA, all-to-all and WO/reduce-scatter",
                    "limitation": "controlled raw-cache slots; compressed/index branches covered by regression tests",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = True
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() in SUPPORTED_TP
        enable_custom_op()
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr("vllm_ascend.utils.get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", dist.get_rank)
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", dist.get_world_size)
            patch.setattr("vllm_ascend.quantization.method_adapters.get_tensor_model_parallel_rank", dist.get_rank)
            model = make_layer(args.checkpoint)
            for rows in (6, 12) if dist.get_world_size() == 4 else (6,):
                run_case(model, rows)
    finally:
        dist.destroy_process_group()
