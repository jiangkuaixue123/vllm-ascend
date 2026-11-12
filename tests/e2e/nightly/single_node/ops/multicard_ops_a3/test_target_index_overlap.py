# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual TP4 index preparation through compressed caches, selection and FIA."""

import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_target_query_overlap as native
import torch
import torch.distributed as dist
from test_tp_aiv_collectives import tp_aiv_options
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.models.deepseek_v41 import attention, attention_ops
from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig
from vllm_ascend.utils import enable_custom_op

TP = 4
MAX_MODEL_LEN = 153600
INDEX_HEADS = 32
INDEX_DIM = 128
TIMED_CALLS = 20
QUERY_WIDTH = 6
GROUP_POSITION_STEP = 37
REPLAY_POSITION_STEP = 127


def make_layer(path, layer_id):
    layer = native.make_layer(path, layer_id=layer_id)
    config = json.loads((path / "config.json").read_text())
    config = DeepseekV41TextConfig(**config.get("text_config", config))
    layer.config = config
    layer.ratio = config.compress_ratios[layer_id]
    layer.owns_kv = layer_id in config.kv_source_layer_ids
    layer.owns_index = True
    layer.uses_local_kv = True
    layer.index_heads = INDEX_HEADS
    layer.q_norm.eps = layer.kv_norm.eps = config.rms_norm_eps
    layer.frequencies.copy_(attention.rotary_frequencies(config, True))
    index = json.loads((path / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]

    def weight(suffix):
        return native.checkpoint_tensor(path, index, suffix, layer_id=layer_id)

    layer.indexer = nn.Module()
    for name, input_size, output_size in (
        ("wq_b", native.QUERY_RANK, INDEX_HEADS * INDEX_DIM),
        ("weights_proj", native.HIDDEN, INDEX_HEADS),
    ):
        with torch.device("npu"):
            linear = ReplicatedLinear(
                input_size,
                output_size,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=native.NativeQuantConfig(),
                prefix=f"test.indexer.{name}",
                disable_tp=True,
            )
        tensor = weight(f"indexer.{name}.weight")
        assert tensor.dtype == (torch.int8 if name == "wq_b" else torch.bfloat16)
        linear.weight.copy_(tensor)
        if name == "wq_b":
            for field in ("weight_scale", "weight_offset"):
                getattr(linear, field).copy_(weight(f"indexer.{name}.{field}"))
        linear.quant_method.process_weights_after_loading(linear)
        setattr(layer.indexer, name, linear)
    if layer.owns_kv:
        layer.compressor = attention.DeepseekV41Compressor(config, layer.ratio).npu()
        layer.compressor.norm.to(dtype=torch.bfloat16)
        layer.compressor.wkv.weight.copy_(weight("compressor.wkv.weight"))
        layer.compressor.norm.weight.copy_(weight("compressor.norm.weight"))
        if layer.ratio > 1:
            layer.compressor.wgate.weight.copy_(weight("compressor.wgate.weight"))
        layer.indexer.wk = nn.Linear(native.HEAD_DIM, INDEX_DIM, bias=False, dtype=torch.bfloat16).npu()
        layer.indexer.wk.weight.copy_(weight("indexer.wk.weight"))
        layer.indexer.k_norm = attention.DeepseekV41RMSNorm(INDEX_DIM, config.rms_norm_eps).to(
            device="npu", dtype=torch.bfloat16
        )
        layer.indexer.k_norm.weight.copy_(weight("indexer.k_norm.weight"))
    return layer.eval()


def make_metadata(rows):
    md = native.make_metadata(rows)
    row = torch.arange(rows, device="npu", dtype=torch.int64)
    group = row // QUERY_WIDTH
    table = torch.zeros(rows, MAX_MODEL_LEN // native.BLOCK_SIZE, device="npu", dtype=torch.int32)
    table[:, : native.BLOCKS_PER_ROW] = (
        group[:, None] * native.BLOCKS_PER_ROW + torch.arange(native.BLOCKS_PER_ROW, device="npu")
    ).int()
    md.block_table_tensor = table
    md.max_model_len = MAX_MODEL_LEN
    update_metadata(md, 0)
    return md


def update_metadata(md, case):
    # Each real DSpark request owns six consecutive proposal positions and one
    # shared page table. The retained slot arena relies on this grouped layout.
    row = torch.arange(md.active.numel(), device="npu", dtype=torch.int64)
    group = row // QUERY_WIDTH
    pos = (group * GROUP_POSITION_STEP + case * REPLAY_POSITION_STEP) % (native.MAX_LENGTH - QUERY_WIDTH)
    pos = pos + row % QUERY_WIDTH
    md.seq_lens.copy_(pos + 1)
    md.slot_mapping.copy_(group * native.MAX_LENGTH + pos)
    md.active.copy_((group % 3 != 1) if case != 2 else torch.zeros_like(md.active))


def make_cache(layer, metadata):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=native.BLOCK_SIZE),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache = attention.DeepseekV41Cache(
        layer.config, config, f"index-overlap-{layer.layer_id}", source_ratio=layer.ratio if layer.owns_kv else 0
    )
    blocks = metadata.active.numel() // QUERY_WIDTH * native.BLOCKS_PER_ROW
    cache.kv_cache = tuple(
        torch.zeros(blocks, native.BLOCK_SIZE, 1, width, device="npu", dtype=torch.bfloat16)
        for width in (cache.key_width, cache.value_width)
    )
    cache.field(0, 0, native.HEAD_DIM).fill_(0.125)
    if layer.owns_kv:
        cache.field(0, native.HEAD_DIM, native.HEAD_DIM).fill_(0.25)
        cache.field(1, native.HEAD_DIM, INDEX_DIM).fill_(0.125)
        if layer.ratio > 1:
            cache.field(1, 0, native.HEAD_DIM).fill_(0.125)
    cache.metadata = lambda: SimpleNamespace(decode=metadata)
    return cache


def exact(actual, expected):
    assert actual.dtype == expected.dtype
    if actual.is_floating_point():
        dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
        actual, expected = actual.contiguous().view(dtype), expected.contiguous().view(dtype)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def run_case(layers, rows, *, bind_quantizer=None):
    """Compare graph operands/state first, then time whole native attention chains."""
    torch.manual_seed(827 + dist.get_rank())
    states = torch.randn(rows, native.HIDDEN, device="npu", dtype=torch.bfloat16)
    md = make_metadata(rows * TP)
    caches = [[make_cache(layer, md) for layer in layers] for _ in range(2)]
    methods = [layer._prepare_overlapped_index for layer in layers]
    for layer in layers:
        layer.attention_kernel.pin_decode_plan(states.device, native.HEADS, rows, native.SELECTED_WIDTH)
    observed = {}
    original_scores = attention.paged_index_scores

    def scores(query, field, weights, *args):
        observed["query"] = query
        observed["weights"] = weights
        return original_scores(query, field, weights, *args)

    def bind(index):
        for layer, cache, method in zip(layers, caches[index], methods):
            layer.cache = cache
            layer._prepare_overlapped_index = method if index or bind_quantizer is not None else lambda *args: None
        if bind_quantizer is not None:
            bind_quantizer(index)

    def complete():
        runtime = SimpleNamespace(decode=md, slot_arena=attention.DecodeSlotArena())
        output = states
        parts = []
        for layer in layers:
            output = layer._decode_forward(output, runtime)
            parts.append(
                (
                    observed.get("query"),
                    observed.get("weights"),
                    runtime.decode_topk,
                    getattr(runtime, "decode_candidate_blocks", None),
                )
            )
        return output, parts

    graphs, outputs = [], []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(attention, "paged_index_scores", scores)
        for index in range(2):
            bind(index)
            graph, output = native.capture(complete)
            graphs.append(graph)
            outputs.append(output)
        for case, scale in enumerate((0.0, 0.01, 1.0, 8.0)):
            states.copy_(torch.randn_like(states) * scale)
            update_metadata(md, case)
            before = [[tuple(bank.clone() for bank in cache.kv_cache) for cache in group] for group in caches]
            for graph in graphs:
                graph.replay()
                torch.npu.synchronize()
            if not all(torch.isfinite(output[0]).all().item() for output in outputs):
                print(
                    "INDEX_NONFINITE_DIAG "
                    + json.dumps(
                        {
                            "rank": dist.get_rank(),
                            "case": case,
                            "output_nonfinite": [int((~torch.isfinite(output[0])).sum()) for output in outputs],
                            "query_nonfinite": [int((~torch.isfinite(output[1][0][0])).sum()) for output in outputs],
                            "weight_nonfinite": [int((~torch.isfinite(output[1][0][1])).sum()) for output in outputs],
                            "cache_nonfinite": [
                                [int((~torch.isfinite(bank)).sum()) for cache in group for bank in cache.kv_cache]
                                for group in caches
                            ],
                        }
                    ),
                    flush=True,
                )
            torch.testing.assert_close(
                outputs[1][0].cpu(), outputs[0][0].cpu(), rtol=native.NUMERICAL_RTOL, atol=native.NUMERICAL_ATOL
            )
            if bind_quantizer is not None:
                exact(outputs[1][0], outputs[0][0])
            for candidate, parent in zip(outputs[1][1], outputs[0][1]):
                for actual, expected in zip(candidate, parent):
                    if expected is not None:
                        exact(actual, expected)
            for left, right in zip(caches[0], caches[1]):
                for parent, candidate in zip(left.kv_cache, right.kv_cache):
                    exact(candidate, parent)
            if case == 2:
                for group, previous in zip(caches, before):
                    for cache, banks in zip(group, previous):
                        for actual, expected in zip(cache.kv_cache, banks):
                            exact(actual, expected)
            local_active = md.active[dist.get_rank() * rows : (dist.get_rank() + 1) * rows]
            assert torch.count_nonzero(outputs[1][0][~local_active].cpu()) == 0
    md.active.fill_(True)
    functions = []
    for index in range(2):
        bind(index)

        def timed():
            result = None
            for _ in range(TIMED_CALLS):
                result = complete()[0]
            return result

        graph, _ = native.capture(timed)
        functions.append(graph.replay)
    samples = [[], []]
    for repeat in range(native.TIMING_SAMPLES):
        for index in (repeat % 2, 1 - repeat % 2):
            samples[index].append(native.time_complete(functions[index]))
    medians = [list(map(statistics.median, zip(*sample))) for sample in samples]
    bind(1)
    if dist.get_rank() == 0:
        print(
            "TARGET_INDEX_OVERLAP_COMPONENT "
            + json.dumps(
                {
                    "status": "PASS",
                    "rows": rows,
                    "layers": [layer.layer_id for layer in layers],
                    "world_size": TP,
                    "calls": TIMED_CALLS,
                    "parent": medians[0],
                    "candidate": medians[1],
                    "metrics": ["host_ms", "ready_ms"],
                    "scope": "full native compressed attention, index/candidate chain, cache, FIA and WO",
                    "optimization": "fp4_decode" if bind_quantizer is not None else "index_overlap",
                    "limitation": "full static 153600 capacity; controlled live history up to384 tokens",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fp4-decode", action="store_true")
    parser.add_argument("--tp-aiv", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = True
    dist.init_process_group("hccl", pg_options=tp_aiv_options() if args.tp_aiv else None)
    try:
        assert dist.get_world_size() == TP
        assert enable_custom_op()
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr("vllm_ascend.utils.get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", dist.get_rank)
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: TP)
            patch.setattr("vllm_ascend.quantization.method_adapters.get_tensor_model_parallel_rank", dist.get_rank)
            patch.setattr(attention, "is_forward_context_available", lambda: True)
            patch.setattr(
                attention,
                "get_forward_context",
                lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL, is_draft_model=False),
            )
            chains = [[make_layer(args.checkpoint, layer_id) for layer_id in ids] for ids in ((2,), (20, 24))]
            # Match service startup: all native metadata is initialized before
            # capturing or replaying any attention graph on the worker.
            for models in chains:
                for model in models:
                    for rows in (6, 12):
                        model.attention_kernel.pin_decode_plan(
                            model.attn_sink.device, native.HEADS, rows, native.SELECTED_WIDTH
                        )
            original_quantizer = attention.fp4_quant_dequant
            used_formats = set()

            def bind_quantizer(index):
                def quantize(*args, **kwargs):
                    kwargs["decode_fusion"] = bool(index) and kwargs.get("decode_fusion", False)
                    return original_quantizer(*args, **kwargs)

                patch.setattr(attention, "fp4_quant_dequant", quantize)

            if args.fp4_decode:
                for scale, name in (("e8m0", "_try_fp4_e8m0_aiv"), ("e4m3", "_try_fp4_e4m3_aiv")):
                    original_dispatch = getattr(attention_ops, name)

                    def observed_dispatch(*args, dispatch=original_dispatch, scale=scale, **kwargs):
                        output = dispatch(*args, **kwargs)
                        if output is not None:
                            used_formats.add(scale)
                        return output

                    patch.setattr(attention_ops, name, observed_dispatch)
            for models in chains:
                for rows in (6, 12):
                    used_formats.clear()
                    run_case(models, rows, bind_quantizer=bind_quantizer if args.fp4_decode else None)
                    if args.fp4_decode:
                        assert used_formats == {"e8m0", "e4m3"}
    finally:
        dist.destroy_process_group()
