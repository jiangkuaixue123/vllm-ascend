# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual TP4 HC pre, norm, compressed attention, state and HC post consumers."""

import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_target_index_overlap as index_native
import test_target_query_overlap as native
import torch
import torch.distributed as dist
import vllm.forward_context as forward_context
from safetensors import safe_open
from vllm.config import CUDAGraphMode

from vllm_ascend.models.common.ops.hc_attention_overlap import try_start_hc_attention
from vllm_ascend.models.common.ops.hc_post import hc_post_fused
from vllm_ascend.models.common.ops.hc_pre import hc_pre_delayed
from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.utils import enable_custom_op

TP = 4
CALLS = 20


def make_block(path, layer_id):
    attn = index_native.make_layer(path, layer_id)
    index = json.loads((path / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]

    def weight(suffix, dtype):
        key = f"layers.{layer_id}.{suffix}"
        with safe_open(path / index[key], framework="pt", device="cpu") as source:
            return source.get_tensor(key).to(device="npu", dtype=dtype)

    norm = attention.DeepseekV41RMSNorm(native.HIDDEN, attn.config.rms_norm_eps).to(device="npu", dtype=torch.bfloat16)
    norm.weight.copy_(weight("attn_norm.weight", torch.bfloat16))
    return SimpleNamespace(
        attn=attn,
        config=attn.config,
        attn_norm=norm,
        hc_attn_fn=weight("hc_attn_fn", torch.float32),
        hc_attn_scale=weight("hc_attn_scale", torch.float32),
        hc_attn_base=weight("hc_attn_base", torch.float32),
    )


@torch.inference_mode()
def run_case(blocks, rows):
    """Compare changing-input graphs and all cache words before paired chain timing."""
    torch.manual_seed(946 + dist.get_rank())
    states = torch.randn(rows, 4, native.HIDDEN, device="npu", dtype=torch.bfloat16)
    previous = torch.ones(rows, 4, device="npu", dtype=torch.float32)
    md = index_native.make_metadata(rows * TP)
    caches = [[index_native.make_cache(block.attn, md) for block in blocks] for _ in range(2)]
    local_mask = md.active[dist.get_rank() * rows : (dist.get_rank() + 1) * rows]

    def bind(index):
        for block, cache in zip(blocks, caches[index]):
            block.attn.cache = cache

    def complete(overlap):
        runtime = SimpleNamespace(decode=md, slot_arena=attention.DecodeSlotArena())
        hidden, pre = states, previous
        intermediates = []
        for block in blocks:
            if overlap:
                pending = try_start_hc_attention(block, hidden, pre, runtime)
                assert pending is not None
                mixed, outputs, ready = pending
                _, pre, post, comb = outputs
            else:
                mixed, pre, post, comb = hc_pre_delayed(
                    hidden,
                    block.hc_attn_fn,
                    block.hc_attn_scale,
                    block.hc_attn_base,
                    pre,
                    block.config.rms_norm_eps,
                    block.config.hc_eps,
                    block.config.hc_sinkhorn_iters,
                )
            attended = block.attn._decode_forward(block.attn_norm(mixed), runtime, defer_output_mask=True)
            if overlap:
                torch.npu.current_stream().wait_event(ready)
            hidden = hc_post_fused(attended, hidden, post, comb, local_mask)
            assert hidden is not None
            intermediates.append((mixed, pre, post, comb, hidden, runtime.decode_topk))
        return hidden, intermediates

    graphs, outputs = [], []
    for index in range(2):
        bind(index)
        graph, output = native.capture(lambda index=index: complete(bool(index)))
        graphs.append(graph)
        outputs.append(output)
    for case, scale in enumerate((0.0, 0.01, 1.0, 8.0)):
        states.copy_(torch.randn_like(states) * scale)
        previous.uniform_(-1, 1)
        index_native.update_metadata(md, case)
        before = [[tuple(bank.clone() for bank in cache.kv_cache) for cache in group] for group in caches]
        for graph in graphs:
            graph.replay()
            torch.npu.synchronize()
        index_native.exact(outputs[1][0], outputs[0][0])
        for left, right in zip(outputs[1][1], outputs[0][1]):
            for actual, expected in zip(left, right):
                if expected is not None:
                    index_native.exact(actual, expected)
        for left, right in zip(caches[1], caches[0]):
            for actual, expected in zip(left.kv_cache, right.kv_cache):
                index_native.exact(actual, expected)
        if case == 2:
            for group, prior in zip(caches, before):
                for cache, banks in zip(group, prior):
                    for actual, expected in zip(cache.kv_cache, banks):
                        index_native.exact(actual, expected)
    md.active.fill_(True)
    functions = []
    for index in range(2):
        bind(index)

        def timed(index=index):
            for _ in range(CALLS):
                complete(bool(index))

        graph, _ = native.capture(timed)
        functions.append(graph.replay)
    samples = [[], []]
    for repeat in range(native.TIMING_SAMPLES):
        for index in (repeat % 2, 1 - repeat % 2):
            samples[index].append(native.time_complete(functions[index]))
    medians = [list(map(statistics.median, zip(*sample))) for sample in samples]
    if dist.get_rank() == 0:
        print(
            "HC_ATTN_OVERLAP_COMPONENT "
            + json.dumps(
                {
                    "status": "PASS",
                    "rows": rows,
                    "layers": [block.attn.layer_id for block in blocks],
                    "calls": CALLS,
                    "parent": medians[0],
                    "candidate": medians[1],
                    "metrics": ["host_ms", "ready_ms"],
                    "scope": "real checkpoint HC pre/norm/attention/HC post, all output/index/cache words",
                    "limitation": "attention HC chain only; controlled live history up to384 tokens",
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
        assert dist.get_world_size() == TP
        assert enable_custom_op()
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr("vllm_ascend.utils.get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", dist.get_rank)
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: TP)
            patch.setattr("vllm_ascend.quantization.method_adapters.get_tensor_model_parallel_rank", dist.get_rank)
            for module in (attention, forward_context):
                patch.setattr(module, "is_forward_context_available", lambda: True)
                patch.setattr(
                    module,
                    "get_forward_context",
                    lambda: SimpleNamespace(
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        is_draft_model=False,
                    ),
                )
            chains = [[make_block(args.checkpoint, layer_id) for layer_id in ids] for ids in ((2,), (20, 24))]
            for blocks in chains:
                for block in blocks:
                    for rows in (6, 12):
                        block.attn.attention_kernel.pin_decode_plan(
                            block.attn.attn_sink.device,
                            native.HEADS,
                            rows,
                            native.SELECTED_WIDTH,
                        )
            for blocks in chains:
                for rows in (6, 12):
                    run_case(blocks, rows)
    finally:
        dist.destroy_process_group()
