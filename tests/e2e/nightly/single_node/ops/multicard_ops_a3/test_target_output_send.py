# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native TP4 output exchange/projection and full attention cache replay."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_target_index_overlap as index_native
import test_target_query_overlap as native
import torch
import torch.distributed as dist
import vllm.forward_context as forward_context
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.utils import enable_custom_op

TP = 4


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
        assert dist.get_world_size() == TP and enable_custom_op()
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr("vllm_ascend.utils.get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", dist.get_rank)
            patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: TP)
            patch.setattr("vllm_ascend.quantization.method_adapters.get_tensor_model_parallel_rank", dist.get_rank)
            context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL, is_draft_model=False)
            for module in (attention, forward_context):
                patch.setattr(module, "is_forward_context_available", lambda: True)
                patch.setattr(module, "get_forward_context", lambda: context)
            chains = [
                [index_native.make_layer(args.checkpoint, layer_id) for layer_id in ids] for ids in ((2,), (20, 24))
            ]
            for models in chains:
                for model in models:
                    for rows in (6, 12):
                        model.attention_kernel.pin_decode_plan(
                            model.attn_sink.device, native.HEADS, rows, native.SELECTED_WIDTH
                        )
            original = attention.try_target_output_send
            hits = set()

            def observed(model, output, positions):
                send = original(model, output, positions)
                if send is not None:
                    hits.add((model.layer_id, output.shape[0]))
                return send

            def bind(index):
                patch.setattr(attention, "try_target_output_send", observed if index else lambda *args: None)

            for models in chains:
                for rows in (6, 12):
                    hits.clear()
                    # Supplying this callback retains identical index/FP4 paths
                    # in both graphs and enables the existing bitwise comparison.
                    index_native.run_case(models, rows, bind_quantizer=bind)
                    assert hits == {(model.layer_id, rows) for model in models}
                    if dist.get_rank() == 0:
                        print("TARGET_OUTPUT_SEND_FULL_PASS", [model.layer_id for model in models], rows, flush=True)
    finally:
        dist.destroy_process_group()
