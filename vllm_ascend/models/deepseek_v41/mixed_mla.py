# SPDX-License-Identifier: Apache-2.0
"""Feed the existing 950DT CANN MixedQuantSparseFlashMla from the model's canonical KV rows."""

from functools import cache
from importlib import import_module

import torch

from .compact_kv import unpack_fp4, unpack_window

HEAD_DIM = 512
ROPE_DIM = 64
NATIVE_PAGE_SIZE = 128
NATIVE_ROW_BYTES = 584
QUERY_TILE = 16


@cache
def _operators(device):
    if not torch.npu.get_device_name(device).startswith("Ascend950DT"):
        return None
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        metadata, attention = namespace.mixed_quant_sparse_flash_mla_metadata, namespace.mixed_quant_sparse_flash_mla
    except (ImportError, AttributeError):
        return None
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "kv_compress_epilog"):
        return None
    from .compact_kv import store_mixed_kv

    return metadata, attention, store_mixed_kv


def supports_mixed_mla(q, rope_dim):
    return (
        q.device.type == "npu"
        and q.dtype == torch.bfloat16
        and q.shape[-1] == HEAD_DIM
        and 1 <= q.shape[-2] <= 128
        and rope_dim == ROPE_DIM
        and _operators(q.device) is not None
    )


def _native_pages(bank, ids, lut, *, compressed, store):
    """Map each query's actual physical KV rows into a bounded private native page table."""
    batch, keys = ids.shape
    blocks = (keys + NATIVE_PAGE_SIZE - 1) // NATIVE_PAGE_SIZE
    numbers = torch.arange(batch * blocks, device=ids.device, dtype=torch.int32).reshape(batch, blocks)
    cache = torch.empty(
        (batch * blocks, NATIVE_PAGE_SIZE, 1, NATIVE_ROW_BYTES), device=ids.device, dtype=torch.float8_e4m3fn
    )
    valid = ids >= 0
    values = bank[ids.clamp_min(0)]
    if lut is not None:
        values = unpack_fp4(values, HEAD_DIM, 16, True, lut) if compressed else unpack_window(values, HEAD_DIM)
    ranks = valid.to(torch.int32).cumsum(dim=1, dtype=torch.int32) - 1
    lengths = valid.sum(1, dtype=torch.int32)
    local = torch.where(valid, ranks, -1)
    if not compressed:
        # An inactive graph lane attends a single all-zero KV so its output stays zero.
        first = (lengths[:, None] == 0) & (torch.arange(keys, device=ids.device) == 0)
        local = torch.where(first, 0, local)
        values = torch.where(first[..., None], 0, values)
        lengths = lengths.clamp_min(1)
    slots = torch.where(local >= 0, numbers[:, :1] * NATIVE_PAGE_SIZE + local, -1)
    store(cache, values.reshape(-1, HEAD_DIM).contiguous(), slots.reshape(-1).contiguous())
    return cache, numbers, lengths


def _attend_tile(q, window_bank, window_ids, compressed_bank, compressed_ids, sink, scale, lut, operators):
    metadata, attention, store = operators
    batch, heads, _ = q.shape
    raw, raw_table, raw_length = _native_pages(window_bank, window_ids, lut, compressed=False, store=store)
    q_ends = torch.arange(batch + 1, device=q.device, dtype=torch.int32)
    config = dict(
        cu_seqlens_q=q_ends,
        seqused_ori_kv=raw_length,
        quant_mode=2,
        rope_head_dim=ROPE_DIM,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=window_ids.size(1) - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    main_inputs = dict(ori_kv=raw, ori_block_table=raw_table)
    k = 0 if compressed_ids is None else compressed_ids.size(1)
    extra_meta = {}
    if k:
        cmp, cmp_table, cmp_length = _native_pages(compressed_bank, compressed_ids, lut, compressed=True, store=store)
        # Causality and the compressor ratio have already been applied by the target's selector.
        # Local IDs name exactly the selected rows in this query's native scratch pages.
        config["seqused_cmp_kv"] = cmp_length
        config["cmp_topk_length"] = cmp_length[:, None]
        column = torch.arange(k, device=q.device, dtype=torch.int32)
        local = torch.where(column < cmp_length[:, None], column, -1)
        main_inputs.update(cmp_kv=cmp, cmp_block_table=cmp_table, cmp_sparse_indices=local.unsqueeze(1))
        extra_meta["max_seqlen_cmp_kv"] = k
    plan = metadata(
        num_heads_q=heads,
        num_heads_kv=1,
        head_dim=HEAD_DIM,
        batch_size=batch,
        max_seqlen_q=1,
        max_seqlen_ori_kv=window_ids.size(1),
        ori_topk=0,
        cmp_topk=k,
        has_ori_kv=True,
        has_cmp_kv=bool(k),
        **extra_meta,
        **config,
    )
    output, _ = attention(
        q.contiguous(),
        sinks=sink,
        metadata=plan,
        softmax_scale=scale,
        topk_value_mode=1,
        return_softmax_lse=False,
        **main_inputs,
        **config,
    )
    return output


def mixed_mla(q, window, raw_ids, compressed, compressed_ids, sink, scale, lut=None):
    """Consume canonical BF16 or packed rows, with a bounded number of native scratch pages per query tile."""
    ops = _operators(q.device)
    outputs = []
    for begin in range(0, len(q), QUERY_TILE):
        rows = slice(begin, begin + QUERY_TILE)
        cmp_ids = None if compressed_ids is None else compressed_ids[rows]
        outputs.append(_attend_tile(q[rows], window, raw_ids[rows], compressed, cmp_ids, sink, scale, lut, ops))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def mixed_mla_from_bf16(q, kv, ids, window_columns, sink, scale):
    """The serial HF-style target stores already dequantized values for both KV streams."""
    batch, seq, heads, dim = q.shape
    base = torch.arange(batch, device=ids.device, dtype=ids.dtype)[:, None, None] * kv.size(1)
    physical = torch.where(ids >= 0, ids + base, -1).reshape(batch * seq, -1)
    bank = kv.flatten(0, 1)
    out = mixed_mla(
        q.reshape(batch * seq, heads, dim),
        bank,
        physical[:, :window_columns],
        bank,
        physical[:, window_columns:],
        sink,
        scale,
    )
    return out.view_as(q)
