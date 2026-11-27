# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import logger
from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer, HYV4HCPreLayer
from vllm.models.hy_v4.nvidia.triton_ihc import _ihc_pre_stage1, _ihc_pre_stage2
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

from .attention import get_flash_attention_metadata

_IHC_PRE_SHAPE = (4, 3072)
_IHC_PRE_DECODE_MAX_ROWS = 16
_IHC_PRE_TP1_DECODE_MAX_ROWS = 64
_IHC_PRE_BLOCK = 1024
# Eight FP32 gate sums and an RMS sum per split, padded to 64 bytes.
_IHC_PRE_PARTIAL_STRIDE = 16


@triton.jit
def _ihc_post_kernel(
    hidden,
    residual,
    gates,
    output,
    HIDDEN_SIZE: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN_SIZE
    lane = token * CHANNELS + channel
    x = tl.load(hidden + token * HIDDEN_SIZE + col, mask, 0.0).to(tl.float32)
    r = tl.load(residual + lane * HIDDEN_SIZE + col, mask, 0.0).to(tl.float32)
    p = tl.load(gates + lane)
    tl.store(output + lane * HIDDEN_SIZE + col, r + p * x, mask)


def fused_ihc_post(hidden, residual, gates):
    tokens, channels, hidden_size = residual.shape
    output = torch.empty_like(residual)
    block_size = 1024
    _ihc_post_kernel[(tokens, channels, triton.cdiv(hidden_size, block_size))](
        hidden,
        residual,
        gates,
        output,
        hidden_size,
        channels,
        block_size,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return output


def _inverse_rms(x: torch.Tensor, eps: float, row_offset: int = 0) -> torch.Tensor:
    squares = x.square()
    if x.shape[0] <= 1:
        return (squares.mean(-1, keepdim=True) + eps).rsqrt()
    batch = get_flash_attention_metadata()
    if batch is None or not any(batch.precise_decode_reqs):
        return (squares.mean(-1, keepdim=True) + eps).rsqrt()

    # CANN's multi-row ReduceMean can use a different reduction order than
    # the single-row decode. Reduce decode rows separately and keep the
    # original batch reduction for any prefill rows.
    row_end = row_offset + x.shape[0]
    has_padding = row_end > batch.query_start_loc[-1]
    default = None if all(batch.precise_decode_reqs) and not has_padding else squares.mean(-1, keepdim=True)
    means = []
    for start, stop, precise in zip(batch.query_start_loc, batch.query_start_loc[1:], batch.precise_decode_reqs):
        start = max(0, start - row_offset)
        stop = min(x.shape[0], stop - row_offset)
        if start >= stop:
            continue
        if precise:
            means.extend(row.mean(-1, keepdim=True) for row in squares[start:stop].split(1, dim=0))
        else:
            means.append(default[start:stop])
    if has_padding:
        means.append(default[max(0, batch.query_start_loc[-1] - row_offset) :])
    return (torch.cat(means, dim=0) + eps).rsqrt()


class FlashHCPreLayer(HYV4HCPreLayer):
    sequence_parallel_input = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config()
        self.fuse_decode_pre = (
            HAS_TRITON
            and config.additional_config.get("hy4flash_fuse_ihc_pre", False)
            and get_ascend_device_type() is AscendDeviceType.A5
        )
        transfer = config.kv_transfer_config
        use_wide_decode_pre = (
            config.additional_config.get("hy4flash_fuse_tp1_ihc_pre", False)
            and config.parallel_config.tensor_parallel_size == 1
            and transfer is not None
            and transfer.kv_role == "kv_consumer"
        )
        self.fused_decode_max_rows = _IHC_PRE_TP1_DECODE_MAX_ROWS if use_wide_decode_pre else _IHC_PRE_DECODE_MAX_ROWS

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            self.fuse_decode_pre
            and x.device.type == "npu"
            and x.dtype == torch.bfloat16
            and self.hpc_op is None
            and x.is_contiguous()
            and x.shape[1:] == _IHC_PRE_SHAPE
            and 0 < x.shape[0] <= self.fused_decode_max_rows
            and self.hc_fn.weight.dtype == self.hc_base.dtype == self.hc_scale.dtype == torch.float32
            and self.hc_fn.weight.shape == (2 * self.hc_mult, self.hc_mult * self.hidden_dim)
            and self.hc_fn.weight.is_contiguous()
            and self.hc_base.is_contiguous()
            and self.hc_scale.is_contiguous()
        ):
            return super().forward(x)

        rows, channels, hidden = x.shape
        logger.info_once("HYV4 Flash uses fused iHC pre-processing for %d NPU token rows.", rows)
        flat_width = channels * hidden
        splits = triton.cdiv(flat_width, _IHC_PRE_BLOCK)
        partial = x.new_empty((rows, splits, _IHC_PRE_PARTIAL_STRIDE), dtype=torch.float32)
        output = x.new_empty((rows, hidden))
        post = partial.new_empty((rows, channels))
        _ihc_pre_stage1[(rows, splits)](
            x,
            self.hc_fn.weight,
            partial,
            flat_width,
            channels,
            channels,
            splits,
            _IHC_PRE_BLOCK,
            _IHC_PRE_PARTIAL_STRIDE,
            num_warps=1,
            enable_fp_fusion=False,
        )
        _ihc_pre_stage2[(rows, triton.cdiv(hidden, _IHC_PRE_BLOCK))](
            x,
            partial,
            self.hc_scale,
            self.hc_base,
            output,
            post,
            hidden,
            flat_width,
            channels,
            channels,
            splits,
            _IHC_PRE_PARTIAL_STRIDE,
            _IHC_PRE_BLOCK,
            self.magnitude,
            self.layernorm_epsilon,
            self.hc_eps,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return output, post

    def _inverse_rms(self, x: torch.Tensor) -> torch.Tensor:
        row_offset = get_tensor_model_parallel_rank() * x.shape[0] if self.sequence_parallel_input else 0
        return _inverse_rms(x, self.layernorm_epsilon, row_offset)


class FlashHCLayer(HYV4HCLayer):
    pre_layer_cls = FlashHCPreLayer

    def post(self, output_with_bias, residual, post_gates):
        if (
            HAS_TRITON
            and self.enable_ihc
            and output_with_bias.device.type == "npu"
            and output_with_bias.dtype == residual.dtype == torch.bfloat16
            and post_gates is not None
            and post_gates.dtype == torch.float32
            and output_with_bias.shape[0] > 0
            and output_with_bias.is_contiguous()
            and residual.is_contiguous()
            and post_gates.is_contiguous()
        ):
            return fused_ihc_post(output_with_bias, residual, post_gates)
        return super().post(output_with_bias, residual, post_gates)


class FlashHCHeadLayer(HYV4HCHeadLayer):
    def _inverse_rms(self, x: torch.Tensor) -> torch.Tensor:
        return _inverse_rms(x, self.config.rms_norm_eps)
