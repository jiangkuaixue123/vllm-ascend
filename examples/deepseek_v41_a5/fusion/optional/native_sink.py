# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_epilogues import attention_sink


def _native_a5(self, q, kv, sink, indices, scale, mode):
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
    result = attention_sink(output, maximum, total, sink, slots)
    logger.info_once("V4.1 native %s A5 sparse_flash_attention heads=%d", mode, heads, scope="process")
    return result
