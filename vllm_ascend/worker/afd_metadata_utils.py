# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import ParallelConfig
from vllm.forward_context import DPMetadata


def make_uniform_dp_metadata(
    parallel_config: ParallelConfig,
    num_tokens: int,
) -> DPMetadata:
    """Build DPMetadata when every DP rank has the same token count."""
    assert parallel_config.data_parallel_size > 1
    num_tokens_across_dp_cpu = torch.full(
        (parallel_config.data_parallel_size,),
        num_tokens,
        device="cpu",
        dtype=torch.int32,
    )
    return DPMetadata(
        num_tokens_across_dp_cpu[parallel_config.data_parallel_rank],
        num_tokens_across_dp_cpu,
    )
