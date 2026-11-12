# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact target FP4 prefill through actual TP4 checkpoint attention and caches."""

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
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import attention, attention_ops
from vllm_ascend.utils import enable_custom_op

TP = 4
MAX_ROWS = 8192
BLOCKS = MAX_ROWS // native.BLOCK_SIZE
TIMING_SAMPLES = 5


def make_cache(layer):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=native.BLOCK_SIZE),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache = attention.DeepseekV41Cache(
        layer.config, config, f"fp4-prefill-{layer.layer_id}", source_ratio=layer.ratio if layer.owns_kv else 0
    )
    cache.kv_cache = tuple(
        torch.zeros(BLOCKS, native.BLOCK_SIZE, 1, width, device="npu", dtype=torch.bfloat16)
        for width in (cache.key_width, cache.value_width)
    )
    return cache


@torch.inference_mode()
def run_case(layers, ends, *, narrow_index=False):
    """Compare one optimization at a time over identical chunks, caches and weights."""
    caches = [[make_cache(layer) for layer in layers] for _ in range(2)]
    table = torch.arange(BLOCKS - 1, -1, -1, device="npu", dtype=torch.int32).unsqueeze(0)
    original_quantize = attention.fp4_quant_dequant
    original_narrow_guard = attention._target_prefill_index_narrow
    used = set()
    narrowed = []
    original_eligible = attention._use_prefill_narrow_columns
    original_dispatch = {scale: getattr(attention_ops, f"_try_fp4_{scale}_aiv") for scale in ("e8m0", "e4m3")}
    with pytest.MonkeyPatch.context() as patch:
        if narrow_index:

            def observed_eligible(scores, k):
                result = original_eligible(scores, k)
                if result:
                    narrowed.append((tuple(scores.shape), k))
                return result

            patch.setattr(attention, "_use_prefill_narrow_columns", observed_eligible)
        for scale, dispatch in original_dispatch.items():

            def observed(*args, dispatch=dispatch, scale=scale, **kwargs):
                output = dispatch(*args, **kwargs)
                if output is not None and kwargs.get("prefill"):
                    used.add(scale)
                return output

            patch.setattr(attention_ops, f"_try_fp4_{scale}_aiv", observed)

        begin = 0
        for end in ends:
            rows = (end - begin + TP - 1) // TP
            torch.manual_seed(20261113 + begin + dist.get_rank())
            states = torch.randn(rows, native.HIDDEN, device="npu", dtype=torch.bfloat16)
            positions = torch.arange(begin, end, device="npu")
            slots = table[0][positions // native.BLOCK_SIZE] * native.BLOCK_SIZE + positions % native.BLOCK_SIZE
            spans = [attention.RequestSpan(0, end - begin, end)]
            metadata = attention.DeepseekV41Metadata(spans, table, slots, end - begin)

            def bind(index, metadata=metadata):
                for layer, cache in zip(layers, caches[index]):
                    layer.cache = cache
                    cache.metadata = lambda: metadata

                def quantize(*args, **kwargs):
                    kwargs["prefill_fusion"] = (narrow_index or bool(index)) and kwargs.get("prefill_fusion", False)
                    kwargs["decode_fusion"] = False
                    return original_quantize(*args, **kwargs)

                patch.setattr(attention, "fp4_quant_dequant", quantize)
                if narrow_index:
                    patch.setattr(
                        attention, "_target_prefill_index_narrow", original_narrow_guard if index else lambda tp: False
                    )

            def complete(spans=spans, states=states):
                runtime = attention.DeepseekV41Runtime(spans)
                output = states
                intermediates = []
                for layer in layers:
                    output = layer(output, runtime)
                    shared = runtime.shared[0]
                    intermediates.append((shared.compressed, shared.index_keys, shared.topk, shared.candidates))
                return output, intermediates

            outputs = []
            for index in range(2):
                bind(index)
                outputs.append(complete())
                torch.npu.synchronize()
            assert used == {"e8m0", "e4m3"}
            assert not narrow_index or narrowed
            for reference, candidate in zip(outputs[0], outputs[1]):
                if isinstance(reference, torch.Tensor):
                    index_native.exact(candidate, reference)
                else:
                    for previous, current in zip(reference, candidate):
                        for expected, actual in zip(previous, current):
                            if expected is None:
                                assert actual is None
                            else:
                                index_native.exact(actual, expected)
            for parent_cache, candidate_cache in zip(caches[0], caches[1]):
                for expected, actual in zip(parent_cache.kv_cache, candidate_cache.kv_cache):
                    index_native.exact(actual, expected)
            if ends == (MAX_ROWS,):
                samples = [[], []]
                for _ in range(TIMING_SAMPLES):
                    for index in range(2):
                        bind(index)
                        samples[index].append(native.time_complete(complete))
                medians = [
                    [statistics.median(sample[column] for sample in group) for column in (0, 1)] for group in samples
                ]
                if dist.get_rank() == 0:
                    print(
                        "TARGET_FP4_PREFILL_COMPONENT "
                        + json.dumps(
                            {
                                "status": "PASS",
                                "optimization": "prefill_index_narrow" if narrow_index else "fp4_prefill",
                                "layers": [layer.layer_id for layer in layers],
                                "tokens": end,
                                "parent": medians[0],
                                "candidate": medians[1],
                                "metrics": ["host_ms", "ready_ms"],
                                "scope": "complete TP4 8K prefill, native attention, index reuse, full cache and WO",
                            }
                        ),
                        flush=True,
                    )
            begin = end
    if dist.get_rank() == 0:
        print(
            "PREFILL_EXACT_PASS " + json.dumps({"layers": [layer.layer_id for layer in layers], "ends": ends}),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--narrow-index", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = True
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == TP and enable_custom_op()
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr("vllm_ascend.utils.get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", dist.get_rank)
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: TP)
            patch.setattr("vllm_ascend.quantization.method_adapters.get_tensor_model_parallel_rank", dist.get_rank)
            patch.setattr(attention, "is_forward_context_available", lambda: True)
            patch.setattr(
                attention,
                "get_forward_context",
                lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, is_draft_model=False),
            )
            for ids in ((2,), (20, 24)):
                layers = [index_native.make_layer(args.checkpoint, layer_id) for layer_id in ids]
                for ends in ((127, 260, 517), (MAX_ROWS,)):
                    run_case(layers, ends, narrow_index=args.narrow_index)
    finally:
        dist.destroy_process_group()
