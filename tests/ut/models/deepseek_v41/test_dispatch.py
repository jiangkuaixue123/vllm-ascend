# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

import vllm_ascend.ops  # noqa: F401 -- initialize the plugin ops before importing its runner
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize(
    ("v41", "computed", "encoder_reqs", "force", "expect_full"),
    [
        pytest.param(True, 128, 0, None, False, id="text-or-cached-image-tail"),
        pytest.param(True, 128, 1, None, False, id="first-encoding-image-tail"),
        pytest.param(True, 134, 0, None, True, id="genuine-q6-decode"),
        pytest.param(True, 0, 0, True, True, id="explicit-capture"),
        pytest.param(False, 128, 0, None, True, id="unrelated-model-existing-dispatch"),
    ],
)
def test_six_scheduled_tokens_use_full_only_after_v41_prompt(v41, computed, encoder_reqs, force, expect_full):
    # Use the real dispatcher with TP8 physical capture sizes. No device or
    # distributed context is needed to distinguish prompt progress from shape.
    graph = CUDAGraphMode.FULL_DECODE_ONLY
    compilation = CompilationConfig(
        cudagraph_mode=graph, cudagraph_capture_sizes=[24, 48], max_cudagraph_capture_size=48
    )
    config = SimpleNamespace(
        compilation_config=compilation,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(tensor_parallel_size=8, data_parallel_size=1, use_sequence_parallel_moe=True),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
        lora_config=None,
        num_speculative_tokens=5,
    )
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = config
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.model = SimpleNamespace(requires_image_token_mask=v41)
    runner.dcp_size = 1
    runner.uniform_decode_query_len = 6
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([computed]),
        num_prompt_tokens=np.array([134]),
        lora_id_to_lora_request={},
    )
    runner.cudagraph_dispatcher = CudagraphDispatcher(config)
    runner.cudagraph_dispatcher.initialize_cudagraph_keys(graph, 6)
    mode, descriptor, *_ = runner._determine_batch_execution_and_padding(
        num_tokens=6,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([6]),
        max_num_scheduled_tokens=6,
        use_cascade_attn=False,
        force_uniform_decode=force,
        num_encoder_reqs=encoder_reqs,
    )
    assert mode == (CUDAGraphMode.FULL if expect_full else CUDAGraphMode.NONE)
    assert descriptor.num_tokens == (24 if expect_full else 8)
