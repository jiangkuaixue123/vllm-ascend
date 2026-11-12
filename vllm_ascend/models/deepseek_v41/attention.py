# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged V4.1 latent attention with eager prefill and uniform graph decode.

Paged raw and compressed keys retain the official BF16 quantize/dequantize
results. The pre-normalization compressor inputs and gates retain FP32.
The A3 BF16 shared-KV operator consumes the selected latent vectors. Decode
replay updates positions, slots and live-row masks at fixed addresses.
"""

import math
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.models.common.ops.rms_rsqrt_fused import try_rms_rsqrt_fused
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_custom_op,
    enable_dsa_cp,
    enable_sp,
    get_ascend_device_type,
    npu_stream_switch,
    shared_experts_calculation_stream,
)

from .attention_ops import (
    apply_rotary_emb_real,
    fp4_quant_dequant,
    fp8_quant_dequant,
)
from .core import rms_norm
from .native_attention import DeepseekV41NativeAttention

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm_ascend.ops.triton.deepseek_v41_paged_index_score import paged_index_scores, supports_paged_index_scores
    from vllm_ascend.ops.triton.deepseek_v41_paged_read import read_strided_cache_field
    from vllm_ascend.ops.triton.deepseek_v41_selected_kv import try_prepare_selected_kv
    from vllm_ascend.ops.triton.deepseek_v41_slot_arena import stage as stage_decode_arena
else:
    paged_index_scores = None
    read_strided_cache_field = None
    try_prepare_selected_kv = None
    stage_decode_arena = None

# Bound the per-head score tensor for eager prefill; cached candidates span the whole request.
_INDEX_SCORE_TILE_BYTES = 32 * 1024 * 1024
# At TP4 this bounds each WO quantizer's temporary FP32 tensor to 8 MiB.
_OUTPUT_QDQ_TILE_ROWS = 256
# A FULL graph uses a static context width even when only a few pages are live.
# Tile the key axis; candidate-block and top-k selection still see all scores.
_DECODE_INDEX_TILE_KEYS = 8192
_INDEX_OVERLAP_LAYOUT = (32, 128)
_INDEX_OVERLAP_TP = 4
_INDEX_OVERLAP_ROWS = (6, 12)
_DECODE_INDEX_SORT_ROWS = (3, 6, 12)
_DECODE_INDEX_SORT_K = 512
_DECODE_INDEX_SORT_MAX_WIDTH = 1 << 24
_DECODE_INDEX_TIE_WIDTHS = (76800, 153600)
# Head widths and rotary tail supported by the V4.1 fused path.
_PARTIAL_ROPE_HEAD_DIMS = (128, 512)
_PARTIAL_ROPE_DIM = 64
_ARENA_BLOCK_SIZE = 128
_ARENA_LEGACY_LENGTH = 16384
_ARENA_LONG_LENGTH = 153600
_QUERY_OVERLAP_ROWS = (3, 6, 12)
_QUERY_OVERLAP_LAYOUT = (5120, 1280, 64, 512)
_QUERY_OVERLAP_TP_SIZES = (4, 8)
_QUERY_OVERLAP_QUERY_WIDTH = 6


def _quantize_projection_input(output):
    """Consume the attention output; E4M3 quantization groups only its last dimension."""
    if output.shape[0] <= _OUTPUT_QDQ_TILE_ROWS:
        return fp8_quant_dequant(output, 32)
    for tile in output.split(_OUTPUT_QDQ_TILE_ROWS):
        fp8_quant_dequant(tile, 32, inplace=True)
    return output


@dataclass(frozen=True)
class RequestSpan:
    begin: int
    end: int
    seq_len: int

    @property
    def start_pos(self) -> int:
        return self.seq_len - (self.end - self.begin)


@dataclass
class DeepseekV41DecodeMetadata:
    """Fixed-address graph inputs, filled by the metadata builder before replay."""

    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table_tensor: torch.Tensor
    active: torch.Tensor
    max_model_len: int

    query_width: int = 1
    # Only the uniform builder populates this witness, including for an idle capture.
    grouped_row_start: int | None = None

    def shard(self, start, rows):
        local = slice(start, start + rows)
        return DeepseekV41DecodeMetadata(
            self.seq_lens[local],
            self.slot_mapping[local],
            self.block_table_tensor[local],
            self.active[local],
            self.max_model_len,
            self.query_width,
            self.grouped_row_start + start if self.grouped_row_start is not None else None,
        )


@dataclass
class DeepseekV41Metadata:
    # Replay reads the refreshed sequence, page and slot tensors directly.
    # This backend does not record mutable FIA task handles.
    requires_ascend_graph_task_update = False

    requests: list[RequestSpan]
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    decode: DeepseekV41DecodeMetadata | None = None


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    query_width = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.decode_buffers = {}
        self.capture_sizes = frozenset(vllm_config.compilation_config.cudagraph_capture_sizes or ())
        self.decode_graph_enabled = not vllm_config.model_config.enforce_eager
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = kv_cache_spec.block_size
        self.query_width = 1 + vllm_config.num_speculative_tokens
        context = vllm_config.compilation_config.static_forward_context
        self.native_layers = [context[name] for name in layer_names]
        self.native_decode_plans = {}

    def _update_decode(self, common, capture=False, bounds=None):
        if not self.decode_graph_enabled and not capture:
            return None
        rows = getattr(common, "num_input_tokens", None) or common.num_actual_tokens
        width_q = self.query_width
        if (
            rows not in self.capture_sizes
            or rows % width_q
            or common.num_reqs > rows // width_q
            or common.max_query_len != width_q
        ):
            if capture:
                raise ValueError("V4.1 graph capture requires a configured uniform decode bucket")
            return None
        if common.max_seq_len > self.max_model_len:
            raise ValueError("V4.1 decode context exceeds configured max_model_len")
        device = common.seq_lens.device
        width = math.ceil(self.max_model_len / self.block_size)
        if rows not in self.decode_buffers:
            if device.type == "npu" and self.native_layers:
                if not enable_custom_op():
                    raise RuntimeError("V4.1 graph decode requires the Ascend shared-KV custom operator")
                for layer in self.native_layers:
                    if layer.native_layout is not None:
                        kernel, heads, tp, selected = layer.native_layout
                        kernel.pin_decode_plan(device, heads, rows // tp, selected, self.native_decode_plans)
            self.decode_buffers[rows] = DeepseekV41DecodeMetadata(
                torch.zeros(rows, dtype=torch.int64, device=device),
                torch.full((rows,), -1, dtype=torch.int64, device=device),
                torch.zeros(rows, width, dtype=torch.int32, device=device),
                torch.zeros(rows, dtype=torch.bool, device=device),
                self.max_model_len,
                width_q,
                0 if width_q == 6 and rows in (24, 48) else None,
            )
        decode = self.decode_buffers[rows]
        decode.block_table_tensor.zero_()
        # The Ascend runner may pad both num_reqs and num_input_tokens while
        # num_actual_tokens remains the count of scheduled one-token rows.
        live = min(common.num_actual_tokens, common.num_reqs * width_q)
        if width_q > 1 and not capture:
            if live % width_q or any(end - begin != width_q for begin, end in zip(bounds, bounds[1:]) if begin < live):
                return None
        decode.seq_lens[live:].zero_()
        decode.slot_mapping[live:].fill_(-1)
        if capture or not live:
            decode.active.zero_()
            decode.seq_lens.zero_()
            decode.slot_mapping.fill_(-1)
            return decode
        src = common.block_table_tensor
        columns = min(width, src.shape[1])
        if columns * self.block_size < common.max_seq_len:
            raise ValueError("V4.1 block table does not span the decode context")
        if width_q == 1:
            decode.block_table_tensor[:live, :columns].copy_(src[:live, :columns])
            decode.seq_lens[:live].copy_(common.seq_lens[:live])
        else:
            # Keep one physical page table per query token. Earlier proposals
            # must neither attend to later proposals nor index their compressed KV.
            token = torch.arange(live, device=device)
            request = token // width_q
            decode.block_table_tensor[:live, :columns].copy_(src[:, :columns].index_select(0, request))
            decode.seq_lens[:live].copy_(common.seq_lens.index_select(0, request) + token % width_q + 1 - width_q)
        decode.slot_mapping[:live].copy_(common.slot_mapping[:live])
        decode.active.copy_((decode.seq_lens > 0) & (decode.slot_mapping >= 0))
        return decode

    def build_for_cudagraph_capture(self, common_attn_metadata):
        common = common_attn_metadata
        decode = self._update_decode(common, capture=True)
        return DeepseekV41Metadata([], common.block_table_tensor, common.slot_mapping, 0, decode)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common: CommonAttentionMetadata = common_attn_metadata
        bounds = common.query_start_loc_cpu[: common.num_reqs + 1].tolist()
        decode = self._update_decode(common, bounds=bounds)
        # Under async rejection only the device has the accepted boundary.
        lengths_cpu = getattr(common, "seq_lens_cpu_upper_bound", None) if self.query_width == 1 else None
        if lengths_cpu is None:
            lengths_cpu = common.seq_lens.cpu()
        lengths = lengths_cpu[: common.num_reqs].tolist()
        actual = common.num_actual_tokens
        requests = [
            RequestSpan(begin, min(end, actual), length)
            for begin, end, length in zip(bounds, bounds[1:], lengths)
            if begin < actual and begin < end
        ]
        return DeepseekV41Metadata(
            requests,
            common.block_table_tensor,
            common.slot_mapping,
            actual,
            decode,
        )


class DeepseekV41Backend(AttentionBackend):
    @staticmethod
    def get_name():
        return "DEEPSEEK_V41_REFERENCE"

    @staticmethod
    def get_impl_cls():
        # Projection and cache operations are composed by the model directly.
        return DeepseekV41Attention

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


# The largest validated dense temporary is 379.25 MiB. Bound the allocation
# separately for each field width and element size before recording a graph.
_MAX_DECODE_SCATTER_BYTES = 384 * 1024 * 1024


def _supports_decode_scatter(page, slots, updates, md, source_ratio):
    if not (
        page.device.type == "npu"
        and source_ratio in (1, 2)
        and page.ndim == 2
        and page.shape[1] in (128, 512)
        and page.stride() == (1024, 1)
        and not page.is_contiguous()
        and page.dtype in (torch.bfloat16, torch.float32)
        and not page.requires_grad
        and page.shape[0] > 0
        and page.numel() * page.element_size() <= _MAX_DECODE_SCATTER_BYTES
        and page.storage_offset() + (page.shape[0] - 1) * page.stride(0) + page.shape[1] - 1
        <= torch.iinfo(torch.int32).max
        and isinstance(md, DeepseekV41DecodeMetadata)
        and md.query_width == 6
        and md.grouped_row_start == 0
        and slots.device == updates.device == page.device
        and slots.dtype == torch.int64
        and slots.ndim == 1
        and slots.numel() in (6, 24, 48)
        and slots.is_contiguous()
        and updates.dtype == page.dtype
        and updates.shape == (slots.numel(), page.shape[1])
        and updates.is_contiguous()
        and not updates.requires_grad
        and torch.is_inference_mode_enabled()
        and not envs.VLLM_BATCH_INVARIANT
        and not torch.compiler.is_compiling()
        and is_forward_context_available()
        and (context := get_forward_context()).cudagraph_runtime_mode == CUDAGraphMode.FULL
        and getattr(context, "is_draft_model", True) is False
        and get_ascend_device_type() == AscendDeviceType.A3
        and enable_custom_op()
        and hasattr(torch.ops._C_ascend, "npu_scatter_nd_update_v2")
    ):
        return False
    import torch_npu

    return all(int(torch_npu.get_npu_format(t)) == int(torch_npu.Format.ND) for t in (page, slots, updates))


def _scatter_decode_field(page, slots, updates):
    # Keep the original floating-point index_copy_ shape and dtype so its stock
    # kernel selects the same NaN treatment. Only rows it overwrote are read.
    dense = torch.empty(page.shape, device=page.device, dtype=page.dtype)
    dense.index_copy_(0, slots, updates)
    integer = torch.int32 if page.dtype == torch.float32 else torch.int16
    written = dense.view(integer).index_select(0, slots).view(page.dtype)
    torch.ops._C_ascend.npu_scatter_nd_update_v2(page, slots.int()[:, None], written)


class DeepseekV41Cache(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, config, vllm_config, prefix, *, source_ratio=0, token_history=False):
        super().__init__()
        self.prefix = prefix
        self.dim = config.head_dim
        self.index_dim = config.index_head_dim
        self.source_ratio = source_ratio
        self.token_history = token_history
        # Both banks are allocated by the ordinary vLLM page manager. Use
        # power-of-two bank widths so Ascend's K/V split factors are exact.
        # Every layer retains its BF16 raw latent at every physical slot.
        key_words = self.dim + (self.index_dim + 2 * token_history if source_ratio > 1 else 0)
        self.key_width = 1 << (key_words - 1).bit_length()
        if source_ratio > 1:
            self.value_width = 4 * self.dim  # Two FP32 vectors per slot.
        elif source_ratio == 1:
            self.value_width = 1 << (self.dim + self.index_dim + 2 * token_history - 1).bit_length()
        else:
            self.value_width = 2  # FP32 token history; the runner requires a positive V width.
        self.block_size = vllm_config.cache_config.block_size
        self.kv_cache = None
        self.native_layout = None
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate cache layer {prefix}")
        context[prefix] = self

    def get_attn_backend(self):
        return DeepseekV41Backend

    def get_kv_cache_spec(self, vllm_config):
        return FullAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.key_width,
            head_size_v=self.value_width,
            dtype=torch.bfloat16,
        )

    def metadata(self):
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return None
        return metadata[self.prefix]

    def pages(self):
        if self.kv_cache is None:
            raise RuntimeError("V4.1 cache was not bound by the model runner")
        key, value = self.kv_cache
        return key.reshape(-1, key.shape[-1]), value.reshape(-1, value.shape[-1])

    def field(self, bank, offset, width):
        """Address a logical state field within the physical pages for this layer."""
        key, value = self.pages()
        if bank == 0 and 0 <= offset < self.dim and offset + width <= self.dim:
            return key[:, offset : offset + width]
        if bank == 0 and offset == self.dim and self.source_ratio:
            values = value.view(torch.float32) if self.source_ratio > 1 else value
            return values[:, :width]
        if bank == 1 and offset == 0 and self.source_ratio > 1:
            return value.view(torch.float32)[:, self.dim : self.dim + width]
        if bank == 1 and offset == self.dim and self.source_ratio:
            keys = key if self.source_ratio > 1 else value
            return keys[:, self.dim : self.dim + width]
        if bank == 1 and offset == self.dim + self.index_dim and width == 1 and self.token_history:
            if self.source_ratio:
                storage = key if self.source_ratio > 1 else value
                return storage[:, offset : offset + 2].view(torch.float32)
            return value[:, :2].view(torch.float32)
        raise ValueError(f"Unallocated V4.1 cache field: bank={bank}, offset={offset}, width={width}")

    def slots(self, metadata, request_idx, positions):
        blocks = metadata.block_table_tensor[request_idx]
        return blocks[positions // self.block_size].long() * self.block_size + (positions % self.block_size)

    def read(self, metadata, request_idx, positions, bank, offset, width):
        slots = self.slots(metadata, request_idx, positions)
        return self.field(bank, offset, width)[slots]

    def write_slots(self, slots, values, bank, offset):
        page = self.field(bank, offset, values.shape[-1])
        page.index_copy_(0, slots.long(), values.to(page.dtype))

    def write(self, metadata, values, bank, offset):
        slots = metadata.slot_mapping[: values.shape[0]].long()
        self.write_slots(slots, values, bank, offset)

    def decode_read(self, md, positions, bank, offset, width):
        """Read fixed-shape logical positions, zeroing dead rows and future slots."""
        pos = positions.clamp(0, md.max_model_len - 1)
        blocks = md.block_table_tensor.gather(1, pos // self.block_size).long()
        slots = blocks * self.block_size + pos % self.block_size
        pages = self.field(bank, offset, width)
        valid = md.active[:, None] & (positions >= 0) & (positions < md.seq_lens[:, None])
        valid = valid & (slots >= 0) & (slots < pages.shape[0])
        # On the A3, index_select on a narrowed field first materializes all physical cache rows.
        if (
            read_strided_cache_field is not None
            and pages.device.type == "npu"
            and width in (128, 512)
            and pages.dtype in (torch.bfloat16, torch.float32)
            and pages.stride(1) == 1
            and not pages.is_contiguous()
            and not pages.requires_grad
            and enable_custom_op()
        ):
            return read_strided_cache_field(pages, slots, valid)
        safe = torch.where(valid, slots, 0)
        values = pages.index_select(0, safe.flatten())
        values = values.reshape(*positions.shape, width)
        return torch.where(valid[..., None], values, 0)

    def decode_write(self, md, values, bank, offset, enabled):
        """Preserve cache bits for inactive rows, including a live write to slot zero."""
        page = self.field(bank, offset, values.shape[-1])
        input_dtype = values.dtype
        values = values.to(page.dtype)
        valid = enabled & md.active & (md.slot_mapping >= 0) & (md.slot_mapping < page.shape[0])
        slots = torch.where(valid, md.slot_mapping, 0)
        if md.query_width > 1:
            # A verify bucket can contain 48 rows. All live slots are unique.
            # For the one shared destination of masked rows, select its final
            # bits before the write, so every alias writes the same value.
            zero_is_live = valid & (slots == 0)
            zero_row = zero_is_live.long().argmax().reshape(1)
            zero_value = torch.where(zero_is_live.any(), values.index_select(0, zero_row), page[:1])
            updates = torch.where(slots[:, None] == 0, zero_value, values)
            if _supports_decode_scatter(page, slots, updates, md, self.source_ratio):
                _scatter_decode_field(page, slots, updates)
                if torch.npu.is_current_stream_capturing():
                    logger.info(
                        "V4.1 stock cache scatter capture cache=%s ratio=%d bank=%d offset=%d rows=%d "
                        "physical_rows=%d width=%d dtype=%s input_dtype=%s storage_offset=%d row_stride=%d",
                        self.prefix,
                        self.source_ratio,
                        bank,
                        offset,
                        slots.numel(),
                        page.shape[0],
                        page.shape[1],
                        page.dtype,
                        input_dtype,
                        page.storage_offset(),
                        page.stride(0),
                    )
            else:
                page.index_copy_(0, slots, updates)
            return
        # Unroll the fixed bucket on the capture stream. A masked row that
        # aliases slot zero reads its latest value, after any earlier live row.
        # A vectorized index_copy_ with duplicate destinations is undefined.
        for row in range(values.shape[0]):
            slot = slots[row : row + 1]
            old = page.index_select(0, slot)
            value = torch.where(valid[row : row + 1, None], values[row : row + 1], old)
            page.index_copy_(0, slot, value)


class DeepseekV41RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self._fused_witness_pending = False

    def forward(self, x):
        fused = try_rms_rsqrt_fused(x, self.weight, self.eps)
        if fused is not None:
            if self._fused_witness_pending:
                logger.info("RMS_RSQRT_FUSED_RUNTIME pid=%s rows=%s", os.getpid(), x.shape[0])
                self._fused_witness_pending = False
            return fused
        return rms_norm(x, self.weight, self.eps)


def rotary_frequencies(config, compressed):
    dim = config.qk_rope_head_dim
    base = config.compress_rope_theta if compressed else config.rope_theta
    # Compute constants on CPU exactly once, independent of default device.
    axis = torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")
    frequencies = 1 / (base ** (axis / dim))
    scaling = config.rope_scaling or {}
    original = scaling.get("original_max_position_embeddings", 0) if compressed else 0
    if original:

        def corrected(rotations):
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected(scaling.get("beta_fast", 32))), 0)
        high = min(math.ceil(corrected(scaling.get("beta_slow", 1))), dim - 1)
        ramp = ((axis / 2 - low) / max(high - low, 1e-3)).clamp(0, 1)
        frequencies = frequencies / scaling["factor"] * ramp + frequencies * (1 - ramp)
    return frequencies


def _use_float32_tie_columns(scores, k):
    return (
        scores.device.type == "npu"
        and scores.ndim == 2
        and scores.shape[0] in _DECODE_INDEX_SORT_ROWS
        and scores.shape[-1] in _DECODE_INDEX_TIE_WIDTHS
        and k == _DECODE_INDEX_SORT_K
        and torch.is_inference_mode_enabled()
    )


def topk_columns(scores, k):
    """Keep the FP32 top-k, preferring the lower column only at a tied cutoff."""
    columns = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int32)
    if k == 0 or k == scores.shape[-1]:
        return columns[:k].long().expand(*scores.shape[:-1], k)
    top = scores.topk(k, dim=-1)
    cutoff = top.values[..., -1:]
    if _use_float32_tie_columns(scores, k):
        # Both column IDs and the out-of-range sentinel are exact FP32 integers.
        # Keep the original score TopK and its NaN/non-cutoff selections intact.
        columns = columns.float()
        logger.info_once("INDEX_TIE_FP32_RUNTIME pid=%d", os.getpid(), scope="process")
    # topk may choose different equal items at different padded widths.
    # Retain all strictly higher scores; fill only the tied part by column.
    tied = torch.where(scores == cutoff, columns, scores.shape[-1])
    first = tied.topk(k, dim=-1, largest=False).values
    at_cutoff = top.values == cutoff
    ordinal = at_cutoff.long().cumsum(-1) - 1
    return torch.where(at_cutoff, first.gather(-1, ordinal.clamp_min(0)).long(), top.indices)


def sorted_topk_columns(scores, k):
    indices = topk_columns(scores, k)
    if (
        scores.device.type == "npu"
        and scores.ndim == 2
        and scores.shape[0] in _DECODE_INDEX_SORT_ROWS
        and k == _DECODE_INDEX_SORT_K
        and k <= scores.shape[-1] <= _DECODE_INDEX_SORT_MAX_WIDTH
        and torch.is_inference_mode_enabled()
    ):
        # TopK indices are nonnegative and below the score width, so the
        # FP32 conversion is exact within this bound. Return int64 for page lookup.
        result = indices.float().sort(-1).values.long()
        logger.info_once("INDEX_SORT_NARROW_RUNTIME pid=%d", os.getpid(), scope="process")
        return result
    return indices.sort(-1).values


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size, *, per_token=True):
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)
    last = (compress_lens - 1) // block_size
    blocks = torch.arange(scores.shape[-1], device=logits.device)
    scores = scores.masked_fill(blocks == last, torch.inf)
    top = topk_columns(scores, min(topk_blocks, scores.shape[-1]))
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top, scores.gather(-1, top) > -torch.inf)
    if per_token:
        return keep.repeat_interleave(block_size, -1)[..., :width]
    return keep


def mask_candidate_columns_(scores, keep, block_size, source_width):
    """Broadcast one decision per block, including a partial final block."""
    if source_width != scores.shape[-1]:
        # Preserve the original dense broadcasting contract for unequal compression widths.
        scores.masked_fill_(~keep.repeat_interleave(block_size, -1)[..., :source_width], -torch.inf)
        return
    blocks, tail = divmod(scores.shape[-1], block_size)
    end = blocks * block_size
    if blocks:
        scores[..., :end].unflatten(-1, (blocks, block_size)).masked_fill_(~keep[..., :blocks, None], -torch.inf)
    if tail:
        scores[..., end:].masked_fill_(~keep[..., blocks : blocks + 1], -torch.inf)


@dataclass
class SharedRequestAttention:
    compressed: torch.Tensor | None = None
    index_keys: torch.Tensor | None = None
    topk: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


@dataclass
class DecodeSlotArena:
    """Owned by one model forward; graph capture retains its activation addresses."""

    values: torch.Tensor | None = None
    stream: object | None = None
    grouped_row_start: int | None = None
    source_generation: int = 0
    index_generation: int = 0
    staged_generation: tuple[int, int, int] | None = None

    def values_for_shard(self, q, stream, grouped_row_start):
        shape = (q.shape[0] * 640, 512)
        if self.values is None:
            self.values = q.new_empty(shape)
            self.stream = stream
            self.grouped_row_start = grouped_row_start
        elif (
            self.stream != stream
            or self.grouped_row_start != grouped_row_start
            or self.values.shape != shape
            or self.values.device != q.device
        ):
            return None
        return self.values


def _disjoint_cache_fields(raw, compressed):
    # data_ptr and strides are host-side storage metadata. A conservative bound also
    # rejects views whose address envelopes overlap only in unused columns.
    def interval(field):
        begin = field.data_ptr()
        return begin, begin + ((field.shape[0] - 1) * field.stride(0) + field.shape[1]) * field.element_size()

    raw_start, raw_end = interval(raw)
    comp_start, comp_end = interval(compressed)
    return raw_end <= comp_start or comp_end <= raw_start


def _supports_arena_metadata(md, device, rows):
    if not (
        isinstance(md, DeepseekV41DecodeMetadata)
        and rows in (3, 6, 12)
        and type(md.max_model_len) is int
        and md.max_model_len in (_ARENA_LEGACY_LENGTH, _ARENA_LONG_LENGTH)
    ):
        return False
    pages = md.max_model_len // _ARENA_BLOCK_SIZE
    table = md.block_table_tensor
    return (
        md.query_width == 6
        and type(md.grouped_row_start) is int
        and md.grouped_row_start in range(0, 24 if rows == 3 else 48, rows)
        and md.seq_lens.device == md.active.device == table.device == device
        and md.seq_lens.dtype == torch.int64
        and md.active.dtype == torch.bool
        and md.seq_lens.shape == md.active.shape == (rows,)
        and md.seq_lens.stride() == md.active.stride() == (1,)
        and table.dtype == torch.int32
        and table.ndim == 2
        and table.shape[0] == rows
        and table.shape[1] >= pages
        and table.stride(1) == 1
        # The gather kernel indexes the table with a signed 32-bit row offset.
        # Accept padded long tables, but only when all reachable columns fit.
        and (md.max_model_len == _ARENA_LEGACY_LENGTH or pages <= table.stride(0) <= (2**31 - pages) // (rows - 1))
    )


def _supports_arena_pair(md, src, device, rows, max_model_len):
    return (
        _supports_arena_metadata(md, device, rows)
        and _supports_arena_metadata(src, device, rows)
        and md.max_model_len == src.max_model_len == max_model_len
        and md.grouped_row_start == src.grouped_row_start
    )


@dataclass
class DeepseekV41Runtime:
    requests: list[RequestSpan]
    shared: list[SharedRequestAttention] = field(default_factory=list)
    decode: DeepseekV41DecodeMetadata | None = None
    source_cache: DeepseekV41Cache | None = None
    source_decode: DeepseekV41DecodeMetadata | None = None
    decode_topk: torch.Tensor | None = None
    decode_candidate_blocks: torch.Tensor | None = None
    decode_candidate_width: int = 0
    slot_arena: DecodeSlotArena | None = None

    @classmethod
    def for_decode(cls, metadata: DeepseekV41Metadata):
        if metadata.decode is None:
            raise ValueError("Missing V4.1 graph decode metadata")
        arena = DecodeSlotArena() if getattr(metadata.decode, "grouped_row_start", None) == 0 else None
        return cls([], decode=metadata.decode, slot_arena=arena)

    def __post_init__(self):
        if not self.shared:
            self.shared = [SharedRequestAttention() for _ in self.requests]


class DeepseekV41Compressor(nn.Module):
    def __init__(self, config, ratio):
        super().__init__()
        self.norm = DeepseekV41RMSNorm(config.head_dim, config.rms_norm_eps)
        dtype = torch.float32 if ratio > 1 else torch.bfloat16
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False, dtype=dtype)
        self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False, dtype=dtype) if ratio > 1 else None


class DeepseekV41Indexer(nn.Module):
    def __init__(self, config, owns_k, quant_config, prefix, dsa_cp=False):
        super().__init__()
        projection = ReplicatedLinear if dsa_cp else ColumnParallelLinear
        self.wq_b = projection(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = projection(
            config.hidden_size,
            config.index_n_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=f"{prefix}.weights_proj",
        )
        if owns_k:
            self.wk = nn.Linear(config.head_dim, config.index_head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(config.index_head_dim, config.rms_norm_eps)


class DeepseekV41Attention(nn.Module):
    @staticmethod
    def update_graph_params(*args, **kwargs):
        # The Ascend wrapper calls this hook after replay. These torch operators
        # use only fixed-address input tensors and have no mutable kernel plans.
        return None

    def __init__(self, config, vllm_config, layer_id, prefix):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp or config.o_groups % tp:
            raise ValueError("V4.1 attention heads and output groups must divide TP")
        self.config = config
        self.layer_id = layer_id
        self.ratio = config.compress_ratios[layer_id]
        self.sequence_group = get_tp_group() if tp > 1 and enable_sp(vllm_config) else None
        self.dsa_cp = self.sequence_group is not None and enable_dsa_cp()
        self.local_heads = config.num_attention_heads // tp
        self.heads = config.num_attention_heads if self.dsa_cp else self.local_heads
        self.groups = config.o_groups // tp
        self.index_heads = config.index_n_heads if self.dsa_cp else config.index_n_heads // tp
        self.tp = tp
        self.dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.attention_kernel = DeepseekV41NativeAttention()
        self.uses_local_kv = getattr(vllm_config, "kv_transfer_config", None) is None
        self.owns_kv = layer_id in config.kv_source_layer_ids
        self.owns_index = layer_id in config.index_source_layer_ids
        quant_config = vllm_config.quant_config
        self.attn_sink = nn.Parameter(torch.empty(self.heads, dtype=torch.float32))
        self.wq_a = ReplicatedLinear(
            config.hidden_size,
            config.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_a",
        )
        self.q_norm = DeepseekV41RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        q_projection = ReplicatedLinear if self.dsa_cp else ColumnParallelLinear
        self.wq_b = q_projection(
            config.q_lora_rank,
            config.num_attention_heads * self.dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wkv = ReplicatedLinear(
            config.hidden_size,
            self.dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
        )
        self.kv_norm = DeepseekV41RMSNorm(self.dim, config.rms_norm_eps)
        self._query_overlap_stream = None
        if (
            self.dsa_cp
            and tp in _QUERY_OVERLAP_TP_SIZES
            and (config.hidden_size, config.q_lora_rank, self.heads, self.dim) == _QUERY_OVERLAP_LAYOUT
            and self.wq_a.weight.device.type == "npu"
            and get_ascend_device_type() == AscendDeviceType.A3
        ):
            # Reuse the existing process stream, initialized before graph capture.
            self._query_overlap_stream = shared_experts_calculation_stream()
        self.wo_a = ColumnParallelLinear(
            config.num_attention_heads * self.dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_b = RowParallelLinear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.wo_b",
        )
        self.compressor = DeepseekV41Compressor(config, self.ratio) if self.owns_kv else None
        self.indexer = (
            DeepseekV41Indexer(config, self.owns_kv, quant_config, f"{prefix}.indexer", self.dsa_cp)
            if self.owns_index
            else None
        )
        self.cache = DeepseekV41Cache(
            config,
            vllm_config,
            f"{prefix}.cache",
            source_ratio=self.ratio if self.owns_kv else 0,
            token_history=layer_id == 0,
        )
        self.register_buffer(
            "frequencies", rotary_frequencies(config, self.ratio > 0).to(self.attn_sink.device), persistent=False
        )
        if getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", None):
            capacity = vllm_config.model_config.max_model_len
            compressed = min(config.index_topk, capacity // self.ratio) if self.ratio else 0
            selected_width = min(config.sliding_window, capacity) + compressed
            self.cache.native_layout = (self.attention_kernel, self.heads, tp if self.dsa_cp else 1, selected_width)

    def rotate(self, x, positions, inverse=False):
        phases = positions.float().unsqueeze(-1) * self.frequencies
        if (
            x.device.type == "npu"
            and x.dtype == torch.bfloat16
            and x.ndim in (3, 4)
            and x.shape[0] == 1
            and x.numel() > 0
            and x.shape[-1] in _PARTIAL_ROPE_HEAD_DIMS
            and self.rope_dim == _PARTIAL_ROPE_DIM
            and phases.shape == (x.shape[1], self.rope_dim // 2)
            and phases.dtype == torch.float32
            and x.is_contiguous()
            and torch.is_inference_mode_enabled()
            and enable_custom_op()
            and hasattr(torch.ops._C_ascend, "inplace_partial_rotary_mul")
        ):
            # The interleaved kernel consumes one FP32 sine/cosine per channel.
            # Adjacent channels in a rotary pair use the same phase.
            phases = torch.stack((phases, phases), dim=-1).view(x.shape[1], 1, 1, self.rope_dim)
            sine = phases.sin()
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                x.view(x.shape[1], 1, -1, x.shape[-1]),
                phases.cos(),
                -sine if inverse else sine,
                rotary_mode="interleave",
                partial_slice=[x.shape[-1] - self.rope_dim, x.shape[-1]],
            )
            logger.info_once("V4.1 stock partial rotary head_dim=%d inverse=%s", x.shape[-1], inverse, scope="process")
            return x
        apply_rotary_emb_real(x[..., -self.rope_dim :], phases.cos(), phases.sin(), inverse=inverse)
        return x

    def _compressed(self, x, positions, metadata, request_idx, request, shared):
        if not self.owns_kv:
            if shared.compressed is None or shared.index_keys is None:
                raise RuntimeError("Compressed attention source did not execute")
            return
        ratio, dim = self.ratio, self.dim
        compressor = self.compressor
        raw = compressor.wkv(x.float() if ratio > 1 else x)
        scores = compressor.wgate(x.float()) if ratio > 1 else None
        completed = positions[(positions + 1) % ratio == 0]
        if metadata is None:
            group_kv = raw[: completed.numel() * ratio]
            group_scores = scores[: completed.numel() * ratio] if scores is not None else None
        else:
            slots = self.cache.slots(metadata, request_idx, positions)
            self.cache.write_slots(slots, raw, 0, dim)
            if scores is not None:
                self.cache.write_slots(slots, scores, 1, 0)
            group_positions = (completed.unsqueeze(-1) - torch.arange(ratio - 1, -1, -1, device=x.device)).flatten()
            group_kv = self.cache.read(metadata, request_idx, group_positions, 0, dim, dim)
            group_scores = (
                self.cache.read(metadata, request_idx, group_positions, 1, 0, dim) if scores is not None else None
            )
        if completed.numel():
            if ratio > 1:
                latent = (
                    (group_kv.unflatten(0, (-1, ratio)) * group_scores.unflatten(0, (-1, ratio)).softmax(1))
                    .sum(1)
                    .to(x.dtype)
                )
            else:
                latent = group_kv.to(x.dtype)
            latent = compressor.norm(latent)
            keys = self.indexer.k_norm(self.indexer.wk(latent))
            self.rotate(keys.unsqueeze(0), completed + 1 - ratio)
            keys = fp4_quant_dequant(keys, 32, scale_format="e8m0")
            self.rotate(latent.unsqueeze(0), completed + 1 - ratio)
            latent = fp4_quant_dequant(latent, 16, scale_format="e4m3")
            if metadata is not None:
                slots = self.cache.slots(metadata, request_idx, completed)
                self.cache.write_slots(slots, latent, 0, dim)
                self.cache.write_slots(slots, keys, 1, dim)
        else:
            latent = x.new_empty((0, dim))
            keys = x.new_empty((0, self.config.index_head_dim))
        if metadata is None:
            shared.compressed, shared.index_keys = latent, keys
        else:
            ends = torch.arange(ratio - 1, request.seq_len, ratio, device=x.device)
            shared.compressed = self.cache.read(metadata, request_idx, ends, 0, dim, dim).to(x.dtype)
            shared.index_keys = self.cache.read(metadata, request_idx, ends, 1, dim, self.config.index_head_dim).to(
                x.dtype
            )

    def _index(self, x, qr, positions, shared):
        if not self.owns_index:
            if shared.topk is None:
                raise RuntimeError("Attention index source did not execute")
            return shared.topk
        count = shared.compressed.shape[0]
        if count == 0:
            shared.topk = torch.empty((x.shape[0], 0), device=x.device, dtype=torch.long)
            return shared.topk
        indexer = self.indexer
        q = indexer.wq_b(qr)[0].view(-1, self.index_heads, self.config.index_head_dim)
        weights = indexer.weights_proj(x)[0] * (self.config.index_head_dim**-0.5 * self.config.index_n_heads**-0.5)
        row_bytes = self.index_heads * max(
            count * max(q.element_size(), weights.element_size()), self.config.index_head_dim * 4
        )
        tile_rows = max(1, _INDEX_SCORE_TILE_BYTES // row_bytes)
        k = min(self.config.index_topk, count)
        topk = torch.empty((x.shape[0], k), device=x.device, dtype=torch.long)
        candidate_layer = self.config.candidate_source_layer_id
        if self.layer_id == candidate_layer:
            shared.candidates = torch.empty((x.shape[0], count), device=x.device, dtype=torch.bool)
        elif 0 <= candidate_layer < self.layer_id and shared.candidates is None:
            raise RuntimeError("Candidate source did not execute")
        columns = torch.arange(count, device=x.device)
        for start in range(0, x.shape[0], tile_rows):
            rows = slice(start, start + tile_rows)
            query = q[rows]
            self.rotate(query.unsqueeze(0), positions[rows])
            query = fp4_quant_dequant(query, 32, scale_format="e8m0")
            per_head = torch.einsum("shd,td->sht", query, shared.index_keys)
            per_head.relu_()
            per_head = per_head.to(torch.promote_types(per_head.dtype, weights.dtype))
            scores = per_head.mul_(weights[rows].unsqueeze(-1)).sum(1)
            del per_head
            if self.tp > 1 and not self.dsa_cp:
                scores = tensor_model_parallel_all_reduce(scores)
            lengths = ((positions[rows] + 1) // self.ratio).unsqueeze(-1)
            scores.masked_fill_(columns >= lengths, -torch.inf)
            if self.layer_id == candidate_layer:
                shared.candidates[rows] = select_candidate_blocks(
                    scores, lengths, self.config.candidate_topk_blocks, self.config.candidate_block_size
                )
            elif 0 <= candidate_layer < self.layer_id:
                scores.masked_fill_(~shared.candidates[rows], -torch.inf)
            indices = topk_columns(scores, k).sort(-1).values
            topk[rows] = torch.where(indices < lengths, indices, -1)
        shared.topk = topk
        return shared.topk

    def _decode_compressed(self, x, pos, md, runtime):
        if not self.owns_kv:
            return
        ratio, dim = self.ratio, self.dim
        raw = self.compressor.wkv(x.float() if ratio > 1 else x)
        complete = ((pos + 1) % ratio) == 0
        self.cache.decode_write(md, raw, 0, dim, md.active)
        if ratio > 1:
            score = self.compressor.wgate(x.float())
            self.cache.decode_write(md, score, 1, 0, md.active)
            relative = torch.arange(ratio - 1, -1, -1, device=x.device)
            group_pos = pos[:, None] - relative
            values = self.cache.decode_read(md, group_pos, 0, dim, dim)
            scores = self.cache.decode_read(md, group_pos, 1, 0, dim)
            raw = (values * scores.softmax(1)).sum(1).to(x.dtype)
        latent = self.compressor.norm(raw)
        group_start = (pos + 1 - ratio).clamp_min(0)
        keys = self.indexer.k_norm(self.indexer.wk(latent))
        self.rotate(keys.unsqueeze(0), group_start)
        self.rotate(latent.unsqueeze(0), group_start)
        keys = fp4_quant_dequant(keys, 32, scale_format="e8m0")
        latent = fp4_quant_dequant(latent, 16, scale_format="e4m3")
        self.cache.decode_write(md, latent, 0, dim, complete)
        self.cache.decode_write(md, keys, 1, dim, complete)
        runtime.source_cache = self.cache
        runtime.source_decode = md
        if (arena := getattr(runtime, "slot_arena", None)) is not None:
            # The compressed source was written by this layer, including the current proposals.
            arena.source_generation += 1

    def _decode_index(self, x, qr, pos, md, runtime, prepared=None):
        if not self.owns_index:
            if runtime.decode_topk is None:
                raise RuntimeError("Missing V4.1 decode index source")
            return
        capacity = md.max_model_len // self.ratio
        if not capacity:
            runtime.decode_topk = torch.empty((x.shape[0], 0), dtype=torch.int64, device=x.device)
            return
        groups = torch.arange(capacity, device=x.device).unsqueeze(0)
        if prepared is None:
            indexer = self.indexer
            q = indexer.wq_b(qr)[0].view(-1, self.index_heads, self.config.index_head_dim)
            self.rotate(q.unsqueeze(0), pos)
            q = fp4_quant_dequant(q, 32, scale_format="e8m0")
            weights = indexer.weights_proj(x)[0] * (self.config.index_head_dim**-0.5 * self.config.index_n_heads**-0.5)
        else:
            q, weights = prepared
        scores = None
        if (
            q.device.type == "npu"
            and self.tp in _QUERY_OVERLAP_TP_SIZES
            and self.dsa_cp
            and torch.is_inference_mode_enabled()
            and paged_index_scores is not None
            and get_ascend_device_type() == AscendDeviceType.A3
            and enable_custom_op()
        ):
            cache = runtime.source_cache
            field = cache.field(1, self.dim, self.config.index_head_dim)
            if supports_paged_index_scores(q, field, weights, runtime.source_decode, md, self.ratio, cache.block_size):
                scores = paged_index_scores(q, field, weights, runtime.source_decode, md, self.ratio, cache.block_size)
                logger.info_once(
                    "V4.1 paged decode index scores rows=%d capacity=%d ratio=%d",
                    q.shape[0],
                    capacity,
                    self.ratio,
                    scope="process",
                )
        if scores is None:
            ends = ((groups + 1) * self.ratio - 1).expand(x.shape[0], -1)
            parts = []
            for tile in ends.split(_DECODE_INDEX_TILE_KEYS, dim=-1):
                keys = runtime.source_cache.decode_read(
                    runtime.source_decode, tile, 1, self.dim, self.config.index_head_dim
                )
                per_head = torch.einsum("bhd,btd->bht", q, keys.to(x.dtype))
                parts.append((per_head.relu_() * weights.unsqueeze(-1)).sum(1))
            scores = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
            if self.tp > 1 and not self.dsa_cp:
                scores = tensor_model_parallel_all_reduce(scores)
        lengths = torch.where(md.active, md.seq_lens // self.ratio, 0)[:, None]
        scores.masked_fill_(groups >= lengths, -torch.inf)
        source = self.config.candidate_source_layer_id
        if self.layer_id == source:
            runtime.decode_candidate_width = capacity
            runtime.decode_candidate_blocks = select_candidate_blocks(
                scores,
                lengths,
                self.config.candidate_topk_blocks,
                self.config.candidate_block_size,
                per_token=False,
            )
            if scores.device.type == "npu":
                logger.info_once(
                    "V4.1 decode candidate mask rows=%d columns=%d blocks=%d",
                    scores.shape[0],
                    capacity,
                    runtime.decode_candidate_blocks.shape[-1],
                    scope="process",
                )
        elif 0 <= source < self.layer_id:
            mask_candidate_columns_(
                scores,
                runtime.decode_candidate_blocks,
                self.config.candidate_block_size,
                runtime.decode_candidate_width,
            )
        indices = sorted_topk_columns(scores, min(self.config.index_topk, capacity))
        runtime.decode_topk = torch.where(indices < lengths, indices, -1)
        if (arena := getattr(runtime, "slot_arena", None)) is not None:
            # Ascending nonnegative indices followed by -1. Generic external IDs get no witness.
            arena.index_generation += 1

    def _sequence_inputs(self, hidden_states, phase):
        if self.sequence_group is None:
            return hidden_states, 0
        group = self.sequence_group
        gathered = group.all_gather(hidden_states.contiguous(), dim=0)
        local_rows = hidden_states.shape[0]
        logger.info_once(
            "V4.1 sequence attention layout=%s phase=%s tp_rank=%d local_rows=%d global_rows=%d",
            "dsa_cp" if self.dsa_cp else "flashcomm1",
            phase,
            group.rank_in_group,
            local_rows,
            gathered.shape[0],
            scope="process",
        )
        return gathered, group.rank_in_group * local_rows

    def _try_decode_slot_arena(self, q, md, runtime, extra):
        arena = getattr(runtime, "slot_arena", None)
        if arena is None:
            return None
        previous = arena.staged_generation
        # Any fallback layer may write a shared source; the next hit must refresh it.
        arena.staged_generation = None
        rows = q.shape[0]
        if not (
            q.device.type == "npu"
            and extra == 512
            and rows in (3, 6, 12)
            and q.shape == (rows, 64, 512)
            and q.dtype == torch.bfloat16
            and self.tp in _QUERY_OVERLAP_TP_SIZES
            and self.dsa_cp
            and self.uses_local_kv
            and self.config.sliding_window == 128
            and self.ratio in (1, 2)
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and stage_decode_arena is not None
            and get_ascend_device_type() == AscendDeviceType.A3
            and enable_custom_op()
            and isinstance(self.attention_kernel, DeepseekV41NativeAttention)
            and (q.device, 64, rows, 640) in self.attention_kernel.decode_plans
            and runtime.decode.active.shape == (self.tp * rows,)
            and arena.source_generation > 0
            and arena.index_generation > 0
        ):
            return None
        src = runtime.source_decode
        topk = runtime.decode_topk
        if not (
            _supports_arena_pair(md, src, q.device, rows, runtime.decode.max_model_len)
            and topk.device == q.device
            and topk.dtype == torch.int64
            and topk.shape == (rows, 512)
            and topk.stride(1) == 1
            and self.cache.block_size == runtime.source_cache.block_size == _ARENA_BLOCK_SIZE
            and runtime.source_cache.source_ratio == self.ratio
        ):
            return None
        raw = self.cache.field(0, 0, 512)
        comp = runtime.source_cache.field(0, 512, 512)
        if not (
            raw.device == comp.device == q.device
            and raw.dtype == torch.bfloat16
            and comp.dtype == (torch.float32 if self.ratio == 2 else torch.bfloat16)
            and raw.ndim == comp.ndim == 2
            and raw.shape[1] == comp.shape[1] == 512
            and raw.shape[0] > 0
            and comp.shape[0] > 0
            and raw.stride() in ((512, 1), (1024, 1))
            and comp.stride() == (1024, 1)
            and _disjoint_cache_fields(raw, comp)
        ):
            return None
        stream = torch.npu.current_stream(q.device)
        values = arena.values_for_shard(q, stream, md.grouped_row_start)
        if values is None:
            return None
        generation = (arena.source_generation, arena.index_generation, self.ratio)
        refresh = previous != generation
        ids = stage_decode_arena(values, raw, comp, md, src, topk, self.ratio, refresh)
        arena.staged_generation = generation
        return values.view(rows, 640, 512), ids, refresh

    def _prepare_decode_kv(self, x, md, runtime, extra):
        window = self.config.sliding_window
        if (
            extra
            and try_prepare_selected_kv is not None
            and x.device.type == "npu"
            and x.dtype == torch.bfloat16
            and not torch.is_grad_enabled()
            and enable_custom_op()
        ):
            source = runtime.source_cache
            prepared = try_prepare_selected_kv(
                self.cache.field(0, 0, self.dim),
                source.field(0, self.dim, self.dim),
                md,
                runtime.source_decode,
                runtime.decode_topk,
                window,
                self.ratio,
                self.cache.block_size,
                source.block_size,
            )
            if prepared is not None:
                logger.info_once(
                    "V4.1 joint selected KV rows=%d ratio=%d extra=%d", x.shape[0], self.ratio, extra, scope="process"
                )
                return prepared
        columns = torch.arange(min(window, md.max_model_len) + extra, device=x.device)[None, :]
        width = md.seq_lens.clamp(0, window)[:, None]
        raw_pos = (md.seq_lens - window).clamp_min(0)[:, None] + columns
        in_window = (columns < width) & md.active[:, None]
        selected = self.cache.decode_read(md, raw_pos, 0, 0, self.dim)
        valid = in_window
        if extra:
            # Preserve eager's exact slot order: compressed items immediately
            # follow the *live* window even when it is shorter than 128. Only
            # trailing invalid slots pad the fixed-size 64-slot online softmax.
            ordinal = columns - width
            indices = runtime.decode_topk.gather(1, ordinal.clamp(0, extra - 1))
            compressed = (ordinal >= 0) & (ordinal < extra) & (indices >= 0)
            comp = runtime.source_cache.decode_read(
                runtime.source_decode,
                (indices + 1) * self.ratio - 1,
                0,
                self.dim,
                self.dim,
            )
            selected = torch.where(compressed[..., None], comp, selected)
            valid = valid | (compressed & md.active[:, None])
        ids = torch.where(valid, columns, -1).to(torch.int32).unsqueeze(1)
        return selected, ids

    def _prepare_overlapped_index(self, states, qr, pos, metadata):
        # Called inside the qualified query stream. Cache state is first needed
        # by scoring, so only these input/weight-dependent operations move.
        if not (
            self.tp == _INDEX_OVERLAP_TP
            and states.shape[0] in _INDEX_OVERLAP_ROWS
            and self.ratio in (1, 2)
            and self.owns_index
            and metadata.max_model_len // self.ratio > 0
            and (self.index_heads, self.config.index_head_dim) == _INDEX_OVERLAP_LAYOUT
            and isinstance(self.indexer.wq_b.quant_method, AscendLinearMethod)
            and isinstance(self.indexer.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod)
            and isinstance(self.indexer.weights_proj.quant_method, AscendUnquantizedLinearMethod)
            and self.indexer.weights_proj.weight.dtype == torch.bfloat16
        ):
            return None
        q = self.indexer.wq_b(qr)[0].view(-1, self.index_heads, self.config.index_head_dim)
        self.rotate(q.unsqueeze(0), pos)
        q = fp4_quant_dequant(q, 32, scale_format="e8m0")
        weights = self.indexer.weights_proj(states)[0] * (
            self.config.index_head_dim**-0.5 * self.config.index_n_heads**-0.5
        )
        return q, weights

    def _start_decode_query(self, states, metadata):
        stream = getattr(self, "_query_overlap_stream", None)
        if stream is None or metadata is None:
            return None
        hidden, q_rank, heads, dim = _QUERY_OVERLAP_LAYOUT
        tp = self.tp
        if not (
            self.dsa_cp
            and tp in _QUERY_OVERLAP_TP_SIZES
            and self.sequence_group is not None
            and self.sequence_group.world_size == tp
            and (self.heads, self.dim) == (heads, dim)
            and states.device.type == "npu"
            and states.dtype == torch.bfloat16
            and states.ndim == 2
            and states.shape[0] in _QUERY_OVERLAP_ROWS
            and states.shape[1] == hidden
            and states.is_contiguous()
            and metadata.query_width == _QUERY_OVERLAP_QUERY_WIDTH
            and metadata.active.shape[0] == tp * states.shape[0]
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and get_ascend_device_type() == AscendDeviceType.A3
            and (self.wq_a.input_size, self.wq_a.output_size) == (hidden, q_rank)
            and (self.wq_b.input_size, self.wq_b.output_size) == (q_rank, heads * dim)
            and isinstance(self.wq_a.quant_method, AscendUnquantizedLinearMethod)
            and isinstance(self.wkv.quant_method, AscendUnquantizedLinearMethod)
            and self.wq_a.weight.dtype == self.wkv.weight.dtype == torch.bfloat16
            and isinstance(self.wq_b.quant_method, AscendLinearMethod)
            and isinstance(self.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod)
        ):
            return None
        main = torch.npu.current_stream(states.device)
        if main == stream:
            return None
        full_pos = (metadata.seq_lens - 1).clamp_min(0)
        offset = self.sequence_group.rank_in_group * states.shape[0]
        pos = full_pos[offset : offset + states.shape[0]]
        inputs_ready = main.record_event()
        states.record_stream(stream)
        full_pos.record_stream(stream)
        with npu_stream_switch(stream):
            stream.wait_event(inputs_ready)
            qr = self.q_norm(self.wq_a(states)[0])
            q = self.wq_b(qr)[0].view(states.shape[0], self.heads, self.dim)
            self.rotate(q.unsqueeze(0), pos)
            prepared_index = self._prepare_overlapped_index(states, qr, pos, metadata)
            complete = stream.record_event()
        return qr, q, complete, full_pos, prepared_index

    def _decode_forward(self, local_states, runtime, *, defer_output_mask=False):
        full_md = self.cache.metadata().decode
        pending_query = self._start_decode_query(local_states, full_md)
        gathered, offset = self._sequence_inputs(local_states, "uniform_decode")
        if full_md is None or gathered.shape[0] != full_md.active.shape[0]:
            raise ValueError("V4.1 graph token rows must match the decode metadata bucket")
        local_md = full_md.shard(offset, local_states.shape[0]) if self.sequence_group is not None else full_md
        x, md = (local_states, local_md) if self.dsa_cp else (gathered, full_md)
        full_pos = (full_md.seq_lens - 1).clamp_min(0) if pending_query is None else pending_query[3]
        pos = full_pos[offset : offset + x.shape[0]] if self.dsa_cp else full_pos
        if pending_query is None:
            qr = self.q_norm(self.wq_a(x)[0])
            q = self.wq_b(qr)[0].view(x.shape[0], self.heads, self.dim)
            raw = self.kv_norm(self.wkv(gathered)[0])
            self.rotate(q.unsqueeze(0), pos)
        else:
            raw = self.kv_norm(self.wkv(gathered)[0])
        self.rotate(raw.unsqueeze(0), full_pos)
        self.cache.decode_write(full_md, fp8_quant_dequant(raw, 32), 0, 0, full_md.active)
        extra = 0
        if self.ratio:
            self._decode_compressed(gathered, full_pos, full_md, runtime)
            if self.owns_kv:
                runtime.source_decode = md
        prepared_index = None
        if pending_query is not None:
            # Cache preparation does not consume Q. Join at its first consumer.
            qr, q, complete, _, prepared_index = pending_query
            main = torch.npu.current_stream(local_states.device)
            main.wait_event(complete)
            qr.record_stream(main)
            q.record_stream(main)
            if prepared_index is not None:
                for tensor in prepared_index:
                    tensor.record_stream(main)
                logger.info_once(
                    "TARGET_INDEX_OVERLAP_RUNTIME pid=%d rows=%d layer=%d",
                    os.getpid(),
                    local_states.shape[0],
                    self.layer_id,
                    scope="process",
                )
            logger.info_once(
                "TARGET_QUERY_OVERLAP_V2_RUNTIME pid=%d rows=%d layer=%d",
                os.getpid(),
                local_states.shape[0],
                self.layer_id,
                scope="process",
            )
        if self.ratio:
            self._decode_index(x, qr, pos, md, runtime, prepared_index)
            extra = runtime.decode_topk.shape[-1]
        staged = self._try_decode_slot_arena(q, md, runtime, extra)
        if staged is None:
            selected, ids = self._prepare_decode_kv(x, md, runtime, extra)
            attended = self.attention_kernel(
                q.unsqueeze(1),
                selected.to(x.dtype),
                self.attn_sink,
                ids,
                self.dim**-0.5,
                mode="DSA target indexed decode" if self.ratio else "SWA target indexed decode",
                uniform_decode=True,
            )[:, 0]
        else:
            pages, slots, refresh = staged
            attended = self.attention_kernel.decode_with_physical_slots(
                q.unsqueeze(1), pages, self.attn_sink, slots, self.dim**-0.5
            )[:, 0]
            logger.info_once(
                f"V4.1 native slot arena layer={self.layer_id} rows={q.shape[0]} "
                f"ratio={self.ratio} refresh={int(refresh)}",
                scope="process",
            )
        self.rotate(attended.unsqueeze(0), pos, inverse=True)
        output = self._project_output(attended, x.dtype)
        if defer_output_mask:
            return output
        return torch.where(local_md.active[:, None], output, 0)

    def forward(self, hidden_states, runtime, *, defer_output_mask=False):
        if runtime.decode is not None:
            return self._decode_forward(hidden_states, runtime, defer_output_mask=defer_output_mask)
        metadata = self.cache.metadata()
        gathered, offset = self._sequence_inputs(hidden_states, "eager")
        queries_input = hidden_states if self.dsa_cp else gathered
        start = offset if self.dsa_cp else 0
        qr = self.q_norm(self.wq_a(queries_input)[0])
        queries = self.wq_b(qr)[0].view(-1, self.heads, self.dim)
        raw_kv = self.kv_norm(self.wkv(gathered)[0])
        output = torch.zeros_like(queries)
        for request_idx, request in enumerate(runtime.requests):
            begin, end = request.begin, request.end
            if begin == end:
                continue
            positions = torch.arange(request.start_pos, request.seq_len, device=hidden_states.device)
            kv = raw_kv[begin:end]
            self.rotate(kv.unsqueeze(0), positions)
            kv = fp8_quant_dequant(kv, 32)
            first = max(0, request.start_pos - self.config.sliding_window + 1)
            if metadata is not None:
                slots = self.cache.slots(metadata, request_idx, positions)
                self.cache.write_slots(slots, kv, 0, 0)
                history = torch.arange(first, request.seq_len, device=hidden_states.device)
                kv = self.cache.read(metadata, request_idx, history, 0, 0, self.dim).to(kv.dtype)
            if self.ratio:
                shared = runtime.shared[request_idx]
                self._compressed(
                    gathered[begin:end],
                    positions,
                    metadata,
                    request_idx,
                    request,
                    shared,
                )
            # Every CP rank writes the same complete KV cache, even when this
            # request has no queries in the rank's contiguous token shard.
            lo, hi = max(begin, start), min(end, start + queries_input.shape[0])
            if lo >= hi:
                continue
            rows = slice(lo - start, hi - start)
            positions = positions[lo - begin : hi - begin]
            q = queries[rows]
            self.rotate(q.unsqueeze(0), positions)
            width = min(self.config.sliding_window, request.seq_len)
            window = (positions - self.config.sliding_window + 1).clamp_min(0)
            window = window.unsqueeze(-1) + torch.arange(width, device=hidden_states.device)
            indices = torch.where(window <= positions.unsqueeze(-1), window - first, -1)
            if self.ratio:
                compressed_indices = self._index(queries_input[rows], qr[rows], positions, shared)
                indices = torch.cat(
                    (
                        indices,
                        torch.where(
                            compressed_indices >= 0,
                            compressed_indices + kv.shape[0],
                            -1,
                        ),
                    ),
                    -1,
                )
                kv = torch.cat((kv, shared.compressed), 0)
            else:
                # The implicit SWA operator aligns Q with the suffix of KV.
                # A CP shard can end before the last token of this request.
                kv = kv[: kv.shape[0] - (end - hi)]
            attended = self.attention_kernel(
                q.unsqueeze(0),
                kv.unsqueeze(0),
                self.attn_sink,
                indices.int().unsqueeze(0),
                self.dim**-0.5,
                mode="DSA target indexed prefill" if self.ratio else "SWA target causal prefill",
                swa_window=None if self.ratio else self.config.sliding_window,
            )[0]
            self.rotate(attended.unsqueeze(0), positions, inverse=True)
            output[rows] = attended
        return self._project_output(output, hidden_states.dtype)

    def _project_output(self, output, dtype):
        if self.dsa_cp:
            # Exchange local tokens and complete heads for complete tokens and
            # local heads. Each receiver owns the matching wo_a / wo_b slice.
            send = output.reshape(-1, self.tp, self.local_heads, self.dim).transpose(0, 1).contiguous()
            send = send.reshape(-1, self.local_heads, self.dim)
            output = torch.empty_like(send)
            dist.all_to_all_single(output, send, group=self.sequence_group.device_group)
        output = output.view(-1, self.groups, self.local_heads * self.dim // self.groups)
        output = _quantize_projection_input(output)
        projection = self.wo_a.weight.view(self.groups, self.config.o_lora_rank, -1)
        output = torch.einsum("sgd,grd->sgr", output, projection).flatten(1)
        if self.sequence_group is not None:
            # The input is already the local WO column slice. Keep the BF16
            # linear partial and perform precisely one FP32 TP reduce-scatter.
            output = self.wo_b.quant_method.apply(self.wo_b, output, bias=None)
            output = self.sequence_group.reduce_scatter(output.float(), dim=0)
        else:
            output = self.wo_b(output)[0].float()
            if self.tp > 1:
                output = tensor_model_parallel_all_reduce(output)
        return output.to(dtype)
