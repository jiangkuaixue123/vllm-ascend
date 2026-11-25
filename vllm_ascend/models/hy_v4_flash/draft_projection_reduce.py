# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce native small BF16 reductions after a copy-only collective."""

import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.logger import logger
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.linear import AscendRowParallelLinear
from vllm_ascend.ops.linear_op import CustomRowParallelOp


@triton.jit
def _finish_small_reduce(terms, routes, result, WIDTH: tl.constexpr, TILE: tl.constexpr):
    block = tl.program_id(0)
    col = block * TILE + tl.arange(0, TILE)
    first = tl.load(routes + block * 8).to(tl.int32)
    value = tl.load(terms + first * WIDTH + col).to(tl.float32)
    for step in tl.static_range(1, 8):
        peer = tl.load(routes + block * 8 + step).to(tl.int32)
        term = tl.load(terms + peer * WIDTH + col).to(tl.float32)
        value = (value + term).to(tl.bfloat16).to(tl.float32)
    tl.store(result + col, value)


def ordered_sum(source, group, routes, tile):
    size = source.numel()
    incoming = source.new_empty((8 * size,))
    dist.all_gather_into_tensor(incoming, source.reshape(-1), group=group)
    out = torch.empty_like(source)
    _finish_small_reduce[(size // tile,)](incoming, routes, out, size, tile, num_warps=1, enable_fp_fusion=False)
    return out


_WIDTH = 3072
_SUM_TILE = 256
_RANK_ORDER = tuple(range(8))
_MAX_ORDERED_ROWS = 192


class DraftSum:
    def __init__(self, copy_group, routes, tile):
        self.group = copy_group
        self.routes = routes
        self.tile = tile

    def reduce(self, operand):
        return ordered_sum(operand, self.group, self.routes, self.tile)


def check_plan(rows, group, tp):
    rank = tp.rank_in_group
    tile = _SUM_TILE
    route = torch.tensor(_RANK_ORDER, device=tp.device, dtype=torch.int32)
    route = route.expand(rows * _WIDTH // tile, -1).contiguous()
    reducer = DraftSum(group, route, tile)
    # Compare against the same vLLM communicator used by the ordinary projection.
    generator = torch.Generator(device="cpu").manual_seed(23017 + 103 * rank)
    values = torch.randn(rows, _WIDTH, device="cpu", generator=generator)
    values *= 2.0 ** ((torch.arange(_WIDTH, device="cpu") + 3 * rank) % 17 - 8)
    partial = values.to(device=tp.device, dtype=torch.bfloat16)
    want = tp.all_reduce(partial.clone())
    got = reducer.reduce(partial)
    different = torch.any(got.view(torch.int16) != want.view(torch.int16)).int()
    dist.all_reduce(different, op=dist.ReduceOp.MAX, group=tp.device_group)
    return None if bool(different.cpu()) else reducer


class DraftRowParallelOp(CustomRowParallelOp):
    def __init__(self, layer, reducers):
        super().__init__(layer)
        self.reducers = reducers
        self.update_attrs()

    def apply_impl(self, operand):
        # The installer accepts only unbiassed, pre-sharded row projections.
        partial = self.quant_method.apply(self.layer, operand, None)
        if (
            partial.ndim == 2
            and partial.shape[1] == _WIDTH
            and partial.dtype == torch.bfloat16
            and partial.is_contiguous()
        ):
            reducer = self.reducers.get(partial.shape[0])
            if reducer is not None:
                logger.info_once(
                    "HYV4 Flash uses checked BF16 draft projection reductions for %d rows.", partial.shape[0]
                )
                return reducer.reduce(partial), None
        return self.comm_group.all_reduce(partial), None


def ordinary_projection(layer):
    return (
        isinstance(layer, AscendRowParallelLinear)
        and layer.custom_op is None
        and layer.tp_size == 8
        and layer.input_is_parallel
        and layer.reduce_results
        and layer.bias is None
    )


def configure_draft_sums(draft, group, config, query_width):
    tp = get_tp_group()
    limit = min(config.scheduler_config.max_num_batched_tokens, _MAX_ORDERED_ROWS)
    batches = {1, *(config.compilation_config.decode_cudagraph_capture_request_sizes or ())}
    correction_rows = {batch for batch in batches if 0 < batch <= limit}
    verifier_rows = {batch * query_width for batch in batches if 0 < batch * query_width <= limit}
    # A native reduction can change its arithmetic order at a message-size boundary.
    # Validate each distinct geometry once and share its reduction plan across layers.
    plans = {}
    for rows in sorted(correction_rows | verifier_rows):
        reducer = check_plan(rows, group, tp)
        if reducer is not None:
            plans[rows] = reducer
    projections = [p for layer in draft.layers for p in (layer.self_attn.o_proj, layer.mlp.down_proj)]
    for sizes, layers in ((correction_rows, [draft.hidden_correction.down_proj]), (verifier_rows, projections)):
        reducers = {rows: plans[rows] for rows in sizes if rows in plans}
        if reducers:
            for layer in layers:
                if ordinary_projection(layer):
                    layer.custom_op = DraftRowParallelOp(layer, reducers)
