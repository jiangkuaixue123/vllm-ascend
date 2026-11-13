# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gather and normalize the unchanged target router's six selected scores."""

import torch
from vllm import envs
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

TARGET_EXPERTS = 384
TARGET_TOP_K = 6
MAX_TARGET_ROWS = 12
ROUTE_EPS = 1e-20
ROUTE_SCALE = 1.5

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _selected_route_weights(
        scores, indices, output, EXPERTS: tl.constexpr, TOP_K: tl.constexpr, EPS: tl.constexpr, SCALE: tl.constexpr
    ):
        row = tl.program_id(0)
        lanes = tl.arange(0, 8)
        selected = tl.load(indices + row * TOP_K + lanes, lanes < TOP_K, other=0)
        values = tl.load(scores + row * EXPERTS + selected, lanes < TOP_K, other=0.0)
        total = tl.sum(values, 0) + EPS
        weights = (values / total) * SCALE
        tl.store(output + row * TOP_K + lanes, weights, lanes < TOP_K)

    def selected_route_weights(scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if (
            scores.ndim != 2
            or not 0 < scores.shape[0] <= MAX_TARGET_ROWS
            or scores.shape[1] != TARGET_EXPERTS
            or indices.shape != (scores.shape[0], TARGET_TOP_K)
            or scores.dtype != torch.float32
            or indices.dtype != torch.int64
            or scores.device.type != "npu"
            or scores.device != indices.device
            or not scores.is_contiguous()
            or not indices.is_contiguous()
            or not torch.is_inference_mode_enabled()
        ):
            raise ValueError("Expected inference FP32 target scores and contiguous INT64 top-six indices")
        output = torch.empty(indices.shape, device=scores.device, dtype=torch.float32)
        _selected_route_weights[(scores.shape[0],)](
            scores,
            indices,
            output,
            EXPERTS=TARGET_EXPERTS,
            TOP_K=TARGET_TOP_K,
            EPS=ROUTE_EPS,
            SCALE=ROUTE_SCALE,
            enable_fp_fusion=False,
        )
        logger.info_once("V4.1 target route weights fused gather/normalize executed", scope="process")
        return output

else:
    selected_route_weights = None
