# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.models.hy_v4_flash.reference import INDEX_DIM
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.sparse_index_remap import _sort

from .cache_pages import PAGE_BYTES, POOL_RATIO, POOL_START
from .index_pool_cache import pool_key_addresses

_QUERIES = 6
_GROUP_TILE = 128
_SCORE_COPY_TILE = 512
_MAX_EXACT_FP32_GROUP = 1 << 24


@triton.jit
def _score_query_block(
    query,
    head_weight,
    visible,
    scores,
    key,
    group,
    QUERY_START: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    GROUP_TILE: tl.constexpr,
):
    qid = QUERY_START + tl.arange(0, QUERY_BLOCK)
    q_head = QUERY_START * HEADS + tl.arange(0, QUERY_BLOCK * HEADS)
    lane = tl.arange(0, WIDTH)
    # IndexPool encodes E4M3 values with power-of-two row scales. The
    # reconstructed query values, like the pooled keys, fit in BF16.
    q = tl.load(query + q_head[:, None] * WIDTH + lane[None, :]).to(tl.bfloat16)
    weight = tl.load(head_weight + q_head)
    qk = tl.dot(q, tl.trans(key), out_dtype=tl.float32)
    weighted = tl.maximum(qk, 0.0) * weight[:, None]
    score = tl.sum(weighted.reshape((QUERY_BLOCK, HEADS, GROUP_TILE)), axis=1)
    end_group = tl.load(visible + qid) // RATIO
    recall = (end_group[:, None] > BUDGET) & (group[None, :] < end_group[:, None])
    tl.store(
        scores + qid[:, None] * MAX_GROUPS + group[None, :],
        tl.where(recall, score, float("-inf")),
        group[None, :] < MAX_GROUPS,
    )


@triton.jit
def _score_paged_pool(
    query,
    head_weight,
    cache,
    block_table,
    visible,
    scores,
    MAX_GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    QUERIES: tl.constexpr,
    QUERY_TILE: tl.constexpr,
):
    start = tl.program_id(0) * GROUP_TILE
    group = start + tl.arange(0, GROUP_TILE)
    qid = tl.arange(0, QUERY_TILE)
    end_group = tl.load(visible + qid, qid < QUERIES, -1) // RATIO
    recall = (end_group > BUDGET) & (qid < QUERIES)
    max_visible = tl.max(tl.where(recall, end_group, 0), axis=0)
    if start < max_visible:
        lane = tl.arange(0, WIDTH)
        live = (group < max_visible) & (group < MAX_GROUPS)
        address = pool_key_addresses(block_table, group, lane, live, ROW_WIDTH, PAGE_SIZE, RATIO, WIDTH // RATIO)
        key = tl.load(cache + address, live[:, None], 0)
        # Sharing one key tile and using 4+2 queries keeps a 128-group tile in UB.
        _score_query_block(
            query, head_weight, visible, scores, key, group, 0, 4, MAX_GROUPS, WIDTH, HEADS, RATIO, BUDGET, GROUP_TILE
        )
        _score_query_block(
            query, head_weight, visible, scores, key, group, 4, 2, MAX_GROUPS, WIDTH, HEADS, RATIO, BUDGET, GROUP_TILE
        )
    else:
        tl.store(
            scores + qid[:, None] * MAX_GROUPS + group[None, :],
            float("-inf"),
            (qid[:, None] < QUERIES) & (group[None, :] < MAX_GROUPS),
        )


def _score_groups(query, weights, cache, table, visible, page_size, max_groups, ratio, budget):
    cache = cache.view(torch.bfloat16)
    scores = torch.empty((_QUERIES, max_groups), dtype=torch.float32, device=query.device)
    _score_paged_pool[(triton.cdiv(max_groups, _GROUP_TILE),)](
        query,
        weights,
        cache,
        table,
        visible,
        scores,
        max_groups,
        INDEX_DIM,
        query.shape[1],
        cache.shape[-1],
        page_size,
        ratio,
        budget,
        _GROUP_TILE,
        _QUERIES,
        triton.next_power_of_2(_QUERIES),
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return scores


def supports_sharded_recall(page_size, max_groups, ratio, budget):
    # Align each rank's first group with a page. The native vector sort uses FP32 group IDs.
    return (
        _GROUP_TILE * ratio % page_size == 0
        and max_groups < _MAX_EXACT_FP32_GROUP
        and budget + 1 == triton.next_power_of_2(budget)
    )


@triton.jit
def _join_score_shards(
    gathered, output, QUERIES: tl.constexpr, GROUPS: tl.constexpr, SHARD: tl.constexpr, TILE: tl.constexpr
):
    query = tl.program_id(0)
    group = tl.program_id(1) * TILE + tl.arange(0, TILE)
    rank = group // SHARD
    address = (rank * QUERIES + query) * SHARD + group % SHARD
    score = tl.load(gathered + address, group < GROUPS, -float("inf"))
    tl.store(output + query * GROUPS + group, score, group < GROUPS)


@triton.jit
def _expand_groups(
    values,
    groups,
    visible,
    rows,
    BUDGET: tl.constexpr,
    RATIO: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
):
    query = tl.program_id(0)
    offset = tl.arange(0, GROUP_TILE)
    token = tl.arange(0, GROUP_TILE * RATIO)
    end = tl.load(visible + query).to(tl.int32)
    # For an inactive row the final dense comparison alone yields -1 at every column.
    cutoff = tl.maximum(end, 0) // RATIO
    if cutoff > BUDGET:
        value = tl.load(values + query * BUDGET + offset, offset < BUDGET, -float("inf"))
        group = tl.load(groups + query * BUDGET + offset, offset < BUDGET, 0).to(tl.int32)
        ordered = _sort(tl.where(value == -float("inf"), MAX_GROUPS, group).to(tl.float32), descending=False)
        group = tl.gather(ordered, token // RATIO, 0).to(tl.int32)
        pos = tl.where(token < BUDGET * RATIO, group * RATIO + token % RATIO, cutoff * RATIO + token % RATIO)
        live = tl.where(token < BUDGET * RATIO, group < cutoff, pos <= end)
    else:
        pos = token
        live = token <= end
    tl.store(rows + query * GROUP_TILE * RATIO + token, tl.where(live, pos, -1))


def _finish_recall(scores, visible, ratio, budget):
    # Global top-k receives the original shape and logical order, including ties.
    max_groups = scores.shape[-1]
    values, group_ids = scores.topk(budget, dim=-1)
    group_tile = budget + 1
    rows = torch.empty((_QUERIES, group_tile * ratio), dtype=torch.int64, device=scores.device)
    _expand_groups[(_QUERIES,)](values, group_ids, visible, rows, budget, ratio, group_tile, max_groups, num_warps=1)
    return rows


def select_sharded_paged_pool(query, weights, cache, table, visible, page_size, max_groups, ratio, budget):
    """Score each page-aligned history shard locally and restore the global top-k input order."""
    tp = get_tp_group()
    shard = triton.cdiv(triton.cdiv(max_groups, _GROUP_TILE), tp.world_size) * _GROUP_TILE
    first = tp.rank_in_group * shard
    local_table = table[first * ratio // page_size :]
    local_visible = torch.where(visible // ratio > budget, visible - first * ratio, -1)
    local = _score_groups(query, weights, cache, local_table, local_visible, page_size, shard, ratio, 0)
    gathered = local.new_empty((tp.world_size * _QUERIES, shard))
    dist.all_gather_into_tensor(gathered, local, group=tp.device_group)
    scores = local.new_empty((_QUERIES, max_groups))
    _join_score_shards[(_QUERIES, triton.cdiv(max_groups, _SCORE_COPY_TILE))](
        gathered, scores, _QUERIES, max_groups, shard, _SCORE_COPY_TILE, num_warps=1
    )
    return _finish_recall(scores, visible, ratio, budget)


@triton.jit
def _visible_pool_table(
    table,
    visible,
    pages,
    PAGE_COUNT: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    QUERIES: tl.constexpr,
    GROUPS_PER_PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    TILE: tl.constexpr,
):
    page = tl.program_id(0) * TILE + tl.arange(0, TILE)
    q = tl.arange(0, triton.next_power_of_2(QUERIES))
    cutoff = tl.max(tl.load(visible + q, q < QUERIES, -1), 0) // RATIO
    live = (page < TABLE_SIZE) & (cutoff > BUDGET) & (page * GROUPS_PER_PAGE < cutoff)
    physical = tl.load(table + page, live, 0)
    tl.store(pages + page, physical, page < PAGE_COUNT)


@triton.jit
def _copy_pool_pages(
    cache,
    table,
    keys,
    PAGE_ELEMENTS: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT_PAGE_BYTES: tl.constexpr = PAGE_BYTES,
    LAYOUT_POOL_START: tl.constexpr = POOL_START,
):
    page = tl.program_id(0)
    lane = tl.program_id(1) * TILE + tl.arange(0, TILE)
    physical = tl.load(table + page).to(tl.int64)
    values = tl.load(cache + physical * (LAYOUT_PAGE_BYTES // 2) + LAYOUT_POOL_START // 2 + lane)
    tl.store(keys + page * PAGE_ELEMENTS + lane, values)


@triton.jit
def _weight_page_scores(
    matrix,
    weights,
    visible,
    scores,
    GROUPS: tl.constexpr,
    HEADS: tl.constexpr,
    BUDGET: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT_POOL_RATIO: tl.constexpr = POOL_RATIO,
):
    query = tl.program_id(0)
    head = tl.arange(0, HEADS)
    group = tl.program_id(1) * TILE + tl.arange(0, TILE)
    product = tl.load(matrix + (query * HEADS + head[:, None]) * GROUPS + group[None, :], group[None, :] < GROUPS, 0.0)
    weight = tl.load(weights + query * HEADS + head)
    score = tl.sum(tl.maximum(product, 0.0) * weight[:, None], axis=0)
    cutoff = tl.load(visible + query) // LAYOUT_POOL_RATIO
    tl.store(
        scores + query * GROUPS + group,
        tl.where((cutoff > BUDGET) & (group < cutoff), score, -float("inf")),
        group < GROUPS,
    )


def select_native_page_pool(
    query, weights, cache, table, visible, page_size, max_groups, ratio, budget, *, page_tables=None
):
    """Read contiguous per-page keys and score all logical groups using the original global top-k."""
    queries, heads, head_dim = query.shape
    groups_per_page = page_size // ratio
    page_count = triton.cdiv(max_groups, groups_per_page)
    # VerificationLayout already identifies one metadata group and one request.
    # Its block table and the visible positions are shared by that group's layers.
    table_key = (page_count, groups_per_page, ratio, budget)
    pages = page_tables.get(table_key) if page_tables is not None else None
    if pages is None:
        pages = torch.empty((page_count,), device=table.device, dtype=table.dtype)
        # Redirect unused entries before the bulk copy. Each physical page is in
        # bounds and its keys form a contiguous plane. Score masking handles the
        # unfilled tail of the last visible page and the reserved zero page.
        _visible_pool_table[(triton.cdiv(page_count, _GROUP_TILE),)](
            table,
            visible,
            pages,
            page_count,
            table.numel(),
            queries,
            groups_per_page,
            ratio,
            budget,
            _GROUP_TILE,
            num_warps=1,
        )
        if page_tables is not None:
            page_tables[table_key] = pages
    keys = torch.empty((page_count * groups_per_page, head_dim), device=query.device, dtype=torch.bfloat16)
    page_elements = groups_per_page * head_dim
    _copy_pool_pages[(page_count, page_elements // _SCORE_COPY_TILE)](
        cache.view(torch.bfloat16), pages, keys, page_elements, _SCORE_COPY_TILE, num_warps=1
    )
    keys = keys[:max_groups]
    matrix = torch.mm(query.bfloat16().view(queries * heads, head_dim), keys.t(), out_dtype=torch.float32)
    scores = torch.empty((queries, max_groups), device=query.device, dtype=torch.float32)
    _weight_page_scores[(queries, triton.cdiv(max_groups, _GROUP_TILE))](
        matrix,
        weights,
        visible,
        scores,
        max_groups,
        heads,
        budget,
        _GROUP_TILE,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return _finish_recall(scores, visible, ratio, budget)


def select_paged_pool(query, weights, cache, table, visible, page_size, max_groups, ratio, budget):
    """Score six verification queries against the BF16 keys in paged IndexPool."""
    scores = _score_groups(query, weights, cache, table, visible, page_size, max_groups, ratio, budget)
    cutoff = visible // ratio
    offsets = torch.arange(ratio, device=visible.device)
    values, selected = scores.topk(budget, dim=-1)
    selected = selected.masked_fill(values == -torch.inf, max_groups).sort(dim=-1).values.unsqueeze(-1)
    history = selected * ratio + offsets
    history = history.masked_fill(selected >= cutoff[:, None, None], -1).flatten(1)
    tail = cutoff[:, None] * ratio + offsets
    return torch.cat((history, tail.masked_fill(tail > visible[:, None], -1)), dim=-1)
