# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small decode HC residual mixing with the reference FP32 evaluation order."""

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_DECODE_ROWS = (3, 6, 12)
_HC_WIDTH = 4
_HIDDEN_SIZE = 5120
_ND_FORMAT = 2


def hc_post_fused(sublayer_output, residual, post, comb):
    """Return the fused decode result, or None for unsupported metadata."""
    if (
        sublayer_output.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or sublayer_output.ndim != 2
        or sublayer_output.shape[0] not in _DECODE_ROWS
        or sublayer_output.shape[1] != _HIDDEN_SIZE
        or residual.shape != (sublayer_output.shape[0], _HC_WIDTH, _HIDDEN_SIZE)
        or post.shape != (sublayer_output.shape[0], _HC_WIDTH)
        or comb.shape != (sublayer_output.shape[0], _HC_WIDTH, _HC_WIDTH)
        or sublayer_output.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or post.dtype != torch.float32
        or comb.dtype != torch.float32
        or any(
            tensor.device != sublayer_output.device or not tensor.is_contiguous() or tensor.is_neg() or tensor.is_conj()
            for tensor in (sublayer_output, residual, post, comb)
        )
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    from torch_npu import get_npu_format

    if any(get_npu_format(tensor) != _ND_FORMAT for tensor in (sublayer_output, residual, post, comb)):
        return None
    # Discover Triton only in supported NPU inference workers.
    from vllm_ascend.ops.triton.hc_post import hc_post4

    if hc_post4 is None:
        return None
    return hc_post4(sublayer_output, residual, post, comb)
