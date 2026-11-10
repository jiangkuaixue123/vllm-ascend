# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute V4.1's selected latent attention on Ascend native operators."""

import importlib
from functools import lru_cache

import torch
import torch.nn.functional as F
from vllm.logger import logger

from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

from .attention_ops import sparse_attn

PAGE_SIZE = 16
SPARSE_ALIGNMENT = 128
MAX_SPARSE_WIDTH = 2048
LATENT_DIM = 512


@lru_cache(maxsize=1)
def _enable_a5_attention_ops():
    # Generic custom ops remain disabled on A5. Load only the extension's
    # registered CANN wrappers; A2/A3-only Python meta kernels are unavailable.
    bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    return True


def enable_native_attention():
    return _enable_a5_attention_ops() if is_950() else enable_custom_op()


def aligned_length(length, alignment):
    return (length + alignment - 1) // alignment * alignment


def physical_indices(indices, page_stride, kv_length, sparse_width):
    """Encode slots into the packed BF16 pages. Keep duplicates, shift holes to the tail."""
    batch, sequence, _ = indices.shape
    valid = (indices >= 0) & (indices < kv_length)
    offsets = torch.arange(batch, device=indices.device, dtype=torch.int32).view(-1, 1, 1) * page_stride
    slots = torch.where(valid, indices.int() + offsets, -1)
    # A3 stops scanning a row at its first negative entry. Send every invalid
    # entry to a discarded scratch column; valid entries have unique destinations.
    destinations = torch.where(valid, valid.int().cumsum(-1) - 1, sparse_width).long()
    packed = torch.full((batch, sequence, sparse_width + 1), -1, dtype=torch.int32, device=indices.device)
    packed.scatter_(-1, destinations, slots)
    return packed[..., :sparse_width].reshape(batch * sequence, 1, sparse_width).contiguous()


class DeepseekV41NativeAttention:
    def __init__(self):
        # The CANN metadata is immutable for this geometry. Cache no batch data,
        # request pages or token positions. Bound memory for variable eager spans.
        self.plan = lru_cache(maxsize=32)(self._eager_plan)
        self.decode_plans = {}
        self.verified_eager_templates = set()

    def pin_decode_plan(self, device, heads, rows, selected_width, shared=None):
        """Hold every address used by the recorded graph for the model's lifetime."""
        if device.type == "npu" and is_950():
            # A5 SFA has no AICPU metadata plan; ACLGraph owns its workspaces.
            return
        width = aligned_length(selected_width, SPARSE_ALIGNMENT)
        key = (device, heads, rows, width)
        if key not in self.decode_plans:
            if shared is not None and key in shared:
                self.decode_plans[key] = shared[key]
                return
            plan = self._verified_plan("decode", device, heads, rows, 1, width, width, width - 1)
            self.decode_plans[key] = plan
            if shared is not None:
                shared[key] = plan

    def _eager_plan(self, device, heads, rows, q_length, length, topk, left):
        indexed = bool(topk)
        if indexed not in self.verified_eager_templates:
            plan = self._verified_plan("eager", device, heads, rows, q_length, length, topk, left)
            self.verified_eager_templates.add(indexed)
            return plan
        return self._plan(device, heads, rows, q_length, length, topk, left)

    def _verified_plan(self, phase, device, heads, rows, q_length, length, topk, left):
        args = (device, heads, rows, q_length, length, topk, left)
        if device.type != "npu":
            return self._plan(*args)
        logger.info_once("V4.1 native metadata begin phase=%s geometry=%s", phase, args, scope="process")
        plan = self._plan(*args)
        # AICPU errors surface at a later synchronization. Check graph plans
        # once at construction and each eager template once before first use.
        torch.npu.current_stream(device).synchronize()
        logger.info_once("V4.1 native metadata ready phase=%s geometry=%s", phase, args, scope="process")
        return plan

    @staticmethod
    def _plan(device, heads, batch, q_length, kv_length, topk, window_left):
        # The AICPU consumes these tiny constants immediately. Finish their
        # host-to-device copies before submitting the metadata task.
        cu = (torch.arange(batch + 1, dtype=torch.int32, device="cpu") * q_length).to(device)
        lengths = torch.full((batch,), kv_length, dtype=torch.int32, device="cpu").to(device)
        common = dict(
            cu_seqlens_q=cu,
            seqused_kv=lengths,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=window_left,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
            heads,
            1,
            LATENT_DIM,
            batch_size=batch,
            max_seqlen_q=q_length,
            max_seqlen_kv=kv_length,
            ori_topk=topk,
            cmp_topk=0,
            has_ori_kv=True,
            has_cmp_kv=False,
            device=str(device),
            **common,
        )
        blocks_per_row = aligned_length(kv_length, PAGE_SIZE) // PAGE_SIZE
        if topk:
            # Explicit slots address the PA_ND buffer directly. The required
            # table is never dereferenced by this template.
            table = torch.zeros((batch, blocks_per_row), dtype=torch.int32, device=device)
        else:
            table = torch.arange(batch * blocks_per_row, dtype=torch.int32, device=device).reshape(batch, -1)
        return metadata, table, common

    def __call__(self, q, kv, sink, indices, scale, *, mode, swa_window=None, uniform_decode=False):
        if q.device.type == "cpu":
            return sparse_attn(q, kv, sink, indices, scale)
        if q.device.type != "npu" or not enable_native_attention():
            raise RuntimeError("V4.1 attention requires Ascend's npu_sparse_attn_sharedkv custom operator")
        return self._native(q, kv, sink, indices, scale, mode, swa_window, uniform_decode)

    def _native(self, q, kv, sink, indices, scale, mode, swa_window, uniform_decode=False):
        batch, sequence, heads, dim = q.shape
        if dim != LATENT_DIM or heads % 4 or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
            raise ValueError(
                "A3 shared-KV attention requires BF16 Q/KV, head dimension 512 and a multiple of four heads"
            )
        if sink.shape != (heads,) or sink.dtype != torch.float32:
            raise ValueError("A3 shared-KV attention requires a per-head FP32 sink")
        length = kv.shape[1]
        if kv.shape != (batch, length, dim) or indices.shape[:2] != (batch, sequence):
            raise ValueError("V4.1 Q, KV and index rows must agree")
        if q.device.type == "npu" and is_950():
            return self._native_a5(q, kv, sink, indices, scale, mode)
        stride = aligned_length(length, PAGE_SIZE)
        padded = F.pad(kv, (0, 0, 0, stride - length)) if stride != length else kv
        pages = padded.contiguous().view(-1, PAGE_SIZE, 1, dim)
        slots = None
        if swa_window is not None:
            # The contiguous query rows are the suffix of KV, even on a CP rank.
            metadata, table, common = self.plan(q.device, heads, batch, sequence, length, 0, swa_window - 1)
        else:
            # The A3 compressed template only has 4x/128x cache compression.
            # V4.1's 1x/2x compressed items use the same latent as key and value.
            # Encode the whole joint softmax in explicit slots of the original-KV
            # template; never reinterpret a 1x or 2x cache as a 4x cache.
            sparse_width = aligned_length(indices.shape[-1], SPARSE_ALIGNMENT)
            if not 0 < sparse_width <= MAX_SPARSE_WIDTH:
                raise ValueError("A3 sparse attention requires between 1 and 2048 selected slots")
            slots = physical_indices(indices, stride, length, sparse_width)
            # One native sequence per query: each local CP query can have a
            # different index list, including the noncausal DSpark block.
            if uniform_decode:
                key = (q.device, heads, batch * sequence, sparse_width)
                if key not in self.decode_plans:
                    raise RuntimeError(f"V4.1 native decode bucket was not prepared before graph capture: {key}")
                metadata, table, common = self.decode_plans[key]
            else:
                metadata, table, common = self.plan(
                    q.device, heads, batch * sequence, 1, sparse_width, sparse_width, sparse_width - 1
                )
        result = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q.reshape(-1, heads, dim).contiguous(),
            ori_kv=pages,
            ori_sparse_indices=slots,
            ori_block_table=table,
            sinks=sink,
            metadata=metadata,
            softmax_scale=scale,
            **common,
        )[0].reshape_as(q)
        logger.info_once("V4.1 native %s npu_sparse_attn_sharedkv heads=%d", mode, heads, scope="process")
        if slots is not None:
            # Rows masked by the scheduler have no valid slots. Keep their
            # result zero irrespective of how the kernel initializes empty rows.
            result = result.masked_fill((slots[:, 0, 0] < 0).reshape(batch, sequence, 1, 1), 0)
        return result

    @staticmethod
    def _native_a5(q, kv, sink, indices, scale, mode):
        """Use A5 BF16 SFA with explicit slots and merge the zero-value sink.

        V4.1's 512-dimensional latent already contains RoPE. The separate
        64-dimensional RoPE arguments required by SFA therefore contain zeros.
        Valid slots form a prefix, preserving duplicates and arbitrary holes.
        """
        batch, sequence, heads, dim = q.shape
        length = kv.shape[1]
        width = aligned_length(indices.shape[-1], SPARSE_ALIGNMENT)
        if not 0 < width <= MAX_SPARSE_WIDTH:
            raise ValueError("A5 sparse attention requires between 1 and 2048 selected slots")
        slots = physical_indices(indices, 0, length, width).reshape(batch, sequence, 1, width)
        # SFA caps the selected count at KV length, so reserve capacity even
        # when a short sequence selects the same key more than once.
        capacity = max(length, width)
        keys = F.pad(kv, (0, 0, 0, capacity - length)).unsqueeze(2).contiguous()
        output, maximum, total = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=q.contiguous(),
            key=keys,
            value=keys,
            sparse_indices=slots,
            scale_value=scale,
            query_rope=q.new_zeros(batch, sequence, heads, 64),
            key_rope=kv.new_zeros(batch, capacity, 1, 64),
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
        )
        lse = (maximum.float() + total.float().log()).reshape(batch, sequence, heads)
        # The sink contributes exp(sink) to the denominator and zero to V.
        factor = torch.sigmoid(lse - sink.view(1, 1, heads))
        result = (output.float() * factor.unsqueeze(-1)).to(q.dtype)
        result.masked_fill_((slots[..., 0] < 0).unsqueeze(-1), 0)
        logger.info_once("V4.1 native %s A5 sparse_flash_attention heads=%d", mode, heads, scope="process")
        return result
