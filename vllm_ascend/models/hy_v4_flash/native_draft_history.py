# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES

from .cache_pages import PAGE_TOKENS, compact_physical_c8_indices, gather_page_field
from .prefill_c8 import attend_packed_c8


def prepare_native_draft_history(cache, table, draft, history, positions):
    """Stage the C8 history and private draft rows once per request."""
    batch, width = positions.shape
    target = history.target_rows
    local_history = (target >= 0) & (target < positions[:, :1])
    gather_rows = target.masked_fill(~local_history, 0)
    pages = table.gather(1, gather_rows // PAGE_TOKENS)
    packed_history = gather_page_field(cache, pages, gather_rows, slice(0, C8_ROW_BYTES))
    packed = torch.cat((packed_history, draft), dim=1)
    offsets = torch.arange(packed.shape[1], device=cache.device, dtype=torch.int32)
    private = offsets[-width:]
    visible = torch.cat(
        (local_history[:, None].expand(-1, width, -1), (private <= private[:, None]).expand(batch, -1, -1)),
        dim=-1,
    )
    valid = visible & history.active[:, None, None]
    selected = torch.where(valid, offsets, -1)
    return packed, selected


def attend_native_draft_history(query, cache, table, draft, history, positions, sink, scale):
    packed, selected = prepare_native_draft_history(cache, table, draft, history, positions)
    # Each request owns one contiguous interval in the staged C8 buffer.
    requests, rows, _ = packed.shape
    pages = torch.arange(requests, device=cache.device, dtype=torch.int32).view(-1, 1)
    physical, counts = compact_physical_c8_indices(pages, selected, page_tokens=rows, physical_rows=rows)
    return attend_packed_c8(
        query,
        packed.flatten(0, 1),
        physical,
        sink,
        scale,
        decode_rows=(positions, counts),
    )
