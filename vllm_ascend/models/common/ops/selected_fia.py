# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use native sink-aware FIA for an already selected target decode prefix."""

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

TARGET_ROWS = (6, 12)
TARGET_HEADS = 64
LATENT_DIM = 512
SELECTED_WIDTH = 640


def try_selected_fia(q, kv, sink, slots, scale, *, sink_source_dtype=None):
    """Consume compact-prefix physical slots; retain duplicates and their order."""
    if (
        q.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or q.ndim != 4
        or q.shape[0] not in TARGET_ROWS
        or q.shape[1:] != (1, TARGET_HEADS, LATENT_DIM)
        or kv.ndim != 3
        or kv.shape[0] != q.shape[0]
        or kv.shape[1] == 0
        or kv.shape[2] != LATENT_DIM
        or slots.shape != (q.shape[0], 1, SELECTED_WIDTH)
        or slots.dtype != torch.int32
        or q.dtype != torch.bfloat16
        or kv.dtype != q.dtype
        or sink.shape != (TARGET_HEADS,)
        or sink.dtype != torch.float32
        or any(
            tensor.device != q.device or not tensor.is_contiguous() or tensor.is_neg() or tensor.is_conj()
            for tensor in (q, kv, sink, slots)
        )
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    # Isolate the optional native backend import from CPU/reference execution.
    import torch_npu

    rows = q.shape[0]
    flat_slots = slots.reshape(rows, SELECTED_WIDTH)
    valid = flat_slots >= 0
    selected = kv.reshape(-1, LATENT_DIM).index_select(0, flat_slots.clamp_min(0).flatten())
    selected = selected.view(rows, 1, SELECTED_WIDTH, LATENT_DIM)
    native_sink = sink_source_dtype == torch.bfloat16
    options = {"learnable_sink": sink.to(q.dtype)} if native_sink else {"return_softmax_lse": True}
    output, lse = torch_npu.npu_fused_infer_attention_score_v2(
        q.transpose(1, 2),
        selected,
        selected,
        atten_mask=(~valid).view(rows, 1, 1, SELECTED_WIDTH),
        **options,
        num_query_heads=TARGET_HEADS,
        num_key_value_heads=1,
        input_layout="BNSD",
        sparse_mode=0,
        softmax_scale=scale,
    )
    if not native_sink:
        # Original FP32 parameters retain the full-precision normalization.
        factor = torch.sigmoid(lse.reshape(rows, TARGET_HEADS, 1, 1) - sink.view(1, TARGET_HEADS, 1, 1))
        output = (output.float() * factor).to(q.dtype)
    return output.transpose(1, 2).masked_fill((~valid[:, 0]).view(rows, 1, 1, 1), 0)
