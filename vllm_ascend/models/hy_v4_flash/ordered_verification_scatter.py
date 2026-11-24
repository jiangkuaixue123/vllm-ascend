# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exchange verification rows before reproducing the original BF16 rank order."""

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

_VERIFY_ROWS = 6
_WIDTH = 3072
_TP = 8
_TILE = 512


@triton.jit
def _rank_sum(
    operands,
    output,
    RANK: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    WORLD: tl.constexpr,
    EXCHANGED: tl.constexpr,
    TILE: tl.constexpr,
):
    lane = tl.program_id(0) * TILE + tl.arange(0, TILE)
    if RANK >= ROWS:
        tl.store(output + lane, tl.full((TILE,), 0, tl.bfloat16))
    else:
        stride = WIDTH if EXCHANGED else ROWS * WIDTH
        start = 0 if EXCHANGED else RANK * WIDTH
        y = tl.load(operands + RANK * stride + start + lane).to(tl.float32)
        for peer in tl.static_range(WORLD):
            if peer != RANK:
                x = tl.load(operands + peer * stride + start + lane).to(tl.float32)
                y = (y + x).to(tl.bfloat16).to(tl.float32)
        tl.store(output + lane, y)


def rank_ordered_scatter(x, group, rank):
    rows, width = x.shape
    incoming = x.new_empty((_TP * rows, width))
    dist.all_gather_into_tensor(incoming, x, group=group)
    out = x.new_empty((1, width))
    _rank_sum[(width // _TILE,)](
        incoming, out, rank, rows, width, _TP, False, _TILE, num_warps=1, enable_fp_fusion=False
    )
    return out


def supports_ordered_scatter(x):
    return x.shape == (_VERIFY_ROWS, _WIDTH) and x.dtype == torch.bfloat16 and x.is_contiguous()


@torch.inference_mode()
def matches_native_scatter(tp, copy_group):
    # The HCCL FP reduction order is topology-dependent. Validate it on all
    # destinations once, with the same padded geometry and the installed TP group.
    rng = torch.Generator(device="cpu").manual_seed(1841 + tp.rank_in_group)
    probe = torch.randn(_TP, _WIDTH, generator=rng, device="cpu")
    probe *= 2.0 ** ((torch.arange(_WIDTH, device="cpu")[None, :] + tp.rank_in_group * 3) % 17 - 8)
    source = probe.to(device=tp.device, dtype=torch.bfloat16)
    expected = source.new_empty((1, _WIDTH))
    dist.reduce_scatter_tensor(expected, source, group=tp.device_group)
    actual = rank_ordered_scatter(source, copy_group, tp.rank_in_group)
    different = torch.any(actual.view(torch.int16) != expected.view(torch.int16)).int()
    dist.all_reduce(different, op=dist.ReduceOp.MAX, group=tp.device_group)
    return not bool(different.cpu())
