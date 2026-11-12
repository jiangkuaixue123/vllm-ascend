# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start target HC coefficients while the main stream prepares attention."""

import os

import torch

from vllm_ascend.models.common.ops.hc_pre import hc_pre_delayed

_TP = 4
_ROWS = (6, 12)
_WIDTH = 4
_HIDDEN = 5120
_ND_FORMAT = 2


def try_start_hc_attention(layer, x, previous, runtime):
    stream = getattr(layer.attn, "_query_overlap_stream", None)
    if (
        getattr(runtime, "decode", None) is None
        or layer.attn.tp != _TP
        or stream is None
        or x.device.type != "npu"
        or x.dtype != torch.bfloat16
        or x.ndim != 3
        or x.shape[0] not in _ROWS
        or x.shape[1:] != (_WIDTH, _HIDDEN)
        or previous.shape != x.shape[:2]
        or previous.dtype != torch.float32
        or x.requires_grad
        or previous.requires_grad
        or not torch.is_inference_mode_enabled()
        or torch.compiler.is_compiling()
    ):
        return None
    from torch_npu import get_npu_format
    from vllm import envs
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.logger import logger
    from vllm.triton_utils import HAS_TRITON

    from vllm_ascend.utils import npu_stream_switch

    if (
        not HAS_TRITON
        or envs.VLLM_BATCH_INVARIANT
        or not is_forward_context_available()
        or getattr(get_forward_context(), "is_draft_model", None) is not False
    ):
        return None
    inputs = (x, previous, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
    if any(
        value.device != x.device
        or value.layout != torch.strided
        or not value.is_contiguous()
        or value.is_neg()
        or value.is_conj()
        or get_npu_format(value) != _ND_FORMAT
        for value in inputs
    ):
        return None
    main = torch.npu.current_stream(x.device)
    if main == stream:
        return None
    from vllm_ascend.ops.triton.deepseek_v41_hc_input import hc_input4

    ready = main.record_event()
    for value in inputs:
        value.record_stream(stream)
    with npu_stream_switch(stream):
        stream.wait_event(ready)
        outputs = hc_pre_delayed(
            x,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            previous,
            layer.config.rms_norm_eps,
            layer.config.hc_eps,
            layer.config.hc_sinkhorn_iters,
        )
        if outputs is None:
            return None
        complete = stream.record_event()
    # Keep all native outputs alive in the returned tuple through the join.
    for value in outputs:
        value.record_stream(main)
    attention_input = hc_input4(x, previous)
    logger.info_once("TARGET_HC_ATTN_OVERLAP_RUNTIME pid=%s", os.getpid(), scope="process")
    return attention_input, outputs, complete
