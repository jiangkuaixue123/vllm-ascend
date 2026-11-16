# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native delayed HC gate mixing, with an explicit inference-only contract."""

import math

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_HC_WIDTH = 4
_HIDDEN_SIZE = 5120
_MIX_WIDTH = 24
_DECODE_ROWS = (3, 6, 12)
_SINKHORN_ROUNDS = 20
_HC_EPS = 1e-6
_ND_FORMAT = 2
_MIN_FP32_POSITIVE = 2.0**-149
_MAX_FP32_FINITE = (2.0 - 2.0**-23) * 2.0**127


def _supports_delayed(x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters):
    if (
        x.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or x.ndim != 3
        or x.shape[0] not in _DECODE_ROWS
        or x.shape[1:] != (_HC_WIDTH, _HIDDEN_SIZE)
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.is_neg()
        or x.is_conj()
        or fn.shape != (_MIX_WIDTH, _HC_WIDTH * _HIDDEN_SIZE)
        or scale.shape != (3,)
        or base.shape != (_MIX_WIDTH,)
        or previous_pre.shape != x.shape[:-1]
        or any(
            t.dtype != torch.float32 or t.device != x.device or not t.is_contiguous() or t.is_neg() or t.is_conj()
            for t in (fn, scale, base, previous_pre)
        )
        or sinkhorn_iters != _SINKHORN_ROUNDS
        or hc_eps != _HC_EPS
        or not math.isfinite(norm_eps)
        or not _MIN_FP32_POSITIVE <= norm_eps <= _MAX_FP32_FINITE
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return False
    # Only NPU workers reaching the qualified path need torch-npu metadata.
    from torch_npu import get_npu_format

    return all(get_npu_format(t) == _ND_FORMAT for t in (x, fn, scale, base, previous_pre))


def hc_pre_delayed(x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters):
    """Return the qualified native delayed gate path, otherwise None."""
    if not hasattr(torch.ops._C_ascend, "npu_hc_pre_delayed") or not _supports_delayed(
        x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters
    ):
        return None
    y, next_pre, post, comb, _projection, _rstd = torch.ops._C_ascend.npu_hc_pre_delayed(
        x, fn, scale, base, previous_pre, norm_eps
    )
    return y, next_pre, post, comb


def hc_post_pre_delayed(
    x, fn, scale, base, previous_pre, sublayer, post, comb, row_mask, input_mask, norm_eps, hc_eps, sinkhorn_iters
):
    """Fuse the materialized attention residual and immediate FFN producer."""
    if (
        not hasattr(torch.ops._C_ascend, "npu_hc_post_pre_delayed")
        or row_mask is None
        or not _supports_delayed(x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters)
        or sublayer.shape != (x.shape[0], _HIDDEN_SIZE)
        or sublayer.dtype != torch.bfloat16
        or post.shape != previous_pre.shape
        or comb.shape != (x.shape[0], _HC_WIDTH, _HC_WIDTH)
        or post.dtype != torch.float32
        or comb.dtype != torch.float32
    ):
        return None
    # A masked residual row is zero regardless of its sublayer input.
    if input_mask is None:
        input_mask = row_mask
    if any(mask.shape != x.shape[:1] or mask.dtype != torch.bool for mask in (row_mask, input_mask)):
        return None
    from torch_npu import get_npu_format

    if any(
        t.device != x.device or not t.is_contiguous() or t.is_neg() or t.is_conj() or get_npu_format(t) != _ND_FORMAT
        for t in (sublayer, post, comb, row_mask, input_mask)
    ):
        return None
    residual, y, next_pre, next_post, next_comb, _projection, _rstd, squares = (
        torch.ops._C_ascend.npu_hc_post_pre_delayed(
            x, fn, scale, base, previous_pre, sublayer, post, comb, row_mask, input_mask, norm_eps
        )
    )
    return residual, y, next_pre, next_post, next_comb, squares
