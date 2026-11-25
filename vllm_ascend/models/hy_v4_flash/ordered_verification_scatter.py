# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exchange verification rows before reproducing the original BF16 rank order."""

import torch
import torch.distributed as dist
from torch.nn import functional as F
from vllm.triton_utils import tl, triton

_VERIFY_ROWS = 6
_WIDTH = 3072
_TP = 8
_TILE = 512


@triton.jit
def _finish_scatter(operands, output, RANK: tl.constexpr, WORLD: tl.constexpr, SIZE: tl.constexpr, TILE: tl.constexpr):
    lane = tl.program_id(0) * TILE + tl.arange(0, TILE)
    value = tl.load(operands + RANK * SIZE + lane).to(tl.float32)
    for peer in tl.static_range(WORLD):
        if peer != RANK:
            term = tl.load(operands + peer * SIZE + lane).to(tl.float32)
            value = (value + term).to(tl.bfloat16).to(tl.float32)
    tl.store(output + lane, value)


def rank_ordered_scatter(x, group, rank):
    rows, width = x.shape
    missing = -rows % _TP
    padded = F.pad(x, (0, 0, 0, missing)) if missing else x
    incoming = torch.empty_like(padded)
    dist.all_to_all_single(incoming, padded, group=group)
    output = x.new_empty((padded.shape[0] // _TP, width))
    _finish_scatter[(output.numel() // _TILE,)](
        incoming, output, rank, _TP, output.numel(), _TILE, num_warps=1, enable_fp_fusion=False
    )
    return output


def supports_ordered_scatter(x, checked_shards):
    return (
        x.ndim == 2
        and x.shape[0] > 0
        and (x.shape[0] + _TP - 1) // _TP in checked_shards
        and x.shape[1] == _WIDTH
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def verification_sizes(config):
    # Six rows form the original path even in an eager deployment. Graphs may
    # also contain one query per request or several DFly verification queries.
    sizes = {_VERIFY_ROWS}
    requests = config.compilation_config.decode_cudagraph_capture_request_sizes or ()
    spec = config.speculative_config
    widths = {1, spec.num_speculative_tokens + 1} if spec is not None else {1}
    limit = config.scheduler_config.max_num_batched_tokens
    for batch in requests:
        sizes.update(batch * width for width in widths if 0 < batch * width <= limit)
    return sorted(sizes)


@torch.inference_mode()
def matches_native_scatter(tp, copy_group, local_rows):
    # The HCCL FP reduction order is topology-dependent. Validate it on all
    # destinations once, with the same padded geometry and the installed TP group.
    rng = torch.Generator(device="cpu").manual_seed(1841 + tp.rank_in_group)
    probe = torch.randn(_TP * local_rows, _WIDTH, generator=rng, device="cpu")
    probe *= 2.0 ** ((torch.arange(_WIDTH, device="cpu")[None, :] + tp.rank_in_group * 3) % 17 - 8)
    source = probe.to(device=tp.device, dtype=torch.bfloat16)
    expected = source.new_empty((local_rows, _WIDTH))
    dist.reduce_scatter_tensor(expected, source, group=tp.device_group)
    actual = rank_ordered_scatter(source, copy_group, tp.rank_in_group)
    different = torch.any(actual.view(torch.int16) != expected.view(torch.int16)).int()
    dist.all_reduce(different, op=dist.ReduceOp.MAX, group=tp.device_group)
    return not bool(different.cpu())


def checked_verification_shards(config, tp, copy_group):
    sizes = verification_sizes(config)
    # The HCCL algorithm sees the padded payload. Test each distinct native
    # geometry once for the process group and share the result between layers.
    padded = {(rows + _TP - 1) // _TP for rows in sizes}
    return frozenset(n for n in sorted(padded) if matches_native_scatter(tp, copy_group, n))
