# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill attention using the existing Ascend C8 sparse attention operator."""

import torch

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES

C8_QUERY_DIM = 576
C8_ROPE_DIM = 64
MIN_FUSED_PREFILL_QUERIES = 128
WINDOW_QUERY_BATCH = 4096
INDEX_SCORE_QUERY_BATCH = 256


def prefill_window_indices(logical, offsets, window):
    """Build causal sliding-window rows with invalid entries at the end."""
    first = (logical - window + 1).clamp_min(0)
    candidates = first[:, None] + offsets
    return candidates.masked_fill_(candidates > logical[:, None], -1)


def apply_attention_sink(context, maximum, total, sink):
    """Extend the native softmax denominator with a zero-valued learned sink."""
    log_partition = maximum + total.log()
    retained_mass = (log_partition - sink).sigmoid()
    return (context.float() * retained_mass.unsqueeze(-1)).to(context.dtype)


def attend_packed_c8(query, packed_kv, selected, sink, scale):
    """Use logical candidate rows in a contiguous request-local C8 view."""
    kv = packed_kv.view(1, -1, 1, C8_ROW_BYTES).view(torch.float8_e4m3fn)
    context, maximum, total = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
        query.contiguous().unsqueeze(0),
        kv,
        kv,
        selected.to(torch.int32).unsqueeze(0).unsqueeze(2).contiguous(),
        scale,
        key_quant_mode=2,
        value_quant_mode=2,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=SFA_QSFA_TILE_SIZE,
        rope_head_dim=C8_ROPE_DIM,
        return_softmax_lse=True,
    )
    return apply_attention_sink(context[0], maximum[0, 0], total[0, 0], sink)
