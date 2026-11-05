# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native delayed HC gate mixing, with an explicit inference-only contract."""

import math

import torch
import torch.nn.functional as F

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_HC_WIDTH = 4
_HIDDEN_SIZE = 5120
_MIX_WIDTH = 24
_DECODE_ROWS = (6, 12)
_DRAFT_BLOCK_ROWS = 5
_DRAFT_MAX_REQUESTS = 8
_DRAFT_ROWS = tuple(_DRAFT_BLOCK_ROWS * requests for requests in range(1, _DRAFT_MAX_REQUESTS + 1))
_SINKHORN_ROUNDS = 20
_HC_EPS = 1e-6
_ND_FORMAT = 2
_MIN_FP32_POSITIVE = 2.0**-149
_MAX_FP32_FINITE = (2.0 - 2.0**-23) * 2.0**127


def _supports_hc_pre(x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters, rows):
    if (
        x.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or x.ndim != 3
        or x.shape[0] not in rows
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
        or not hasattr(torch.ops._C_ascend, "npu_hc_pre_delayed")
    ):
        return False
    # Only NPU workers reaching the qualified path need torch-npu metadata.
    from torch_npu import get_npu_format

    return all(get_npu_format(t) == _ND_FORMAT for t in (x, fn, scale, base, previous_pre))


def hc_pre_delayed(x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters):
    """Return y/next_pre/post/comb on the qualified native path, otherwise None.

    y uses the previous sublayer's FP32 gate and rounds to BF16 before the
    caller invokes its existing RMSNorm module. The native call always also
    materializes FP32 projection/rstd diagnostics; their cost is intentional.
    """
    if not _supports_hc_pre(x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters, _DECODE_ROWS):
        return None
    y, next_pre, post, comb, _projection, _rstd = torch.ops._C_ascend.npu_hc_pre_delayed(
        x, fn, scale, base, previous_pre, norm_eps
    )
    return y, next_pre, post, comb


def hc_pre_draft(x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters):
    """Pack draft rows into qualified native HC batches without splitting reductions.

    HC projection and normalization are independent for each row. Zero padding
    only adds discarded rows; each live row retains all four residual copies
    and the full hidden dimension. The caller still applies its existing norm
    after the BF16 pre-mix output.
    """
    if not _supports_hc_pre(x, fn, scale, base, previous_pre, norm_eps, hc_eps, sinkhorn_iters, _DRAFT_ROWS):
        return None
    chunks = []
    for begin in range(0, x.shape[0], _DECODE_ROWS[-1]):
        states = x[begin : begin + _DECODE_ROWS[-1]]
        previous = previous_pre[begin : begin + _DECODE_ROWS[-1]]
        live = states.shape[0]
        bucket = _DECODE_ROWS[0] if live <= _DECODE_ROWS[0] else _DECODE_ROWS[-1]
        if live < bucket:
            states = F.pad(states, (0, 0, 0, 0, 0, bucket - live))
            previous = F.pad(previous, (0, 0, 0, bucket - live))
        outputs = torch.ops._C_ascend.npu_hc_pre_delayed(states, fn, scale, base, previous, norm_eps)
        chunks.append(tuple(output[:live] for output in outputs[:4]))
    if len(chunks) == 1:
        return chunks[0]
    return tuple(torch.cat(parts, dim=0) for parts in zip(*chunks))
