# SPDX-License-Identifier: Apache-2.0
"""CPU replay of the three V4.1 dispatch metadata contracts, using the installed types."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import weighted_experts as w


@pytest.mark.parametrize("topk", [3, 6])
def test_allgather_mask_and_inverse(topk):
    n = 17 * topk
    original = torch.arange(n).float().add_(0.3571).div_(n).reshape(17, topk)
    original[0] = 0
    inverse = torch.randperm(n, generator=torch.Generator().manual_seed(18))
    signs = torch.where(original.reshape(-1) == 0, -1, 1)
    metadata = w.MoEAllGatherCombineMetadata(original, (inverse * signs).int(), torch.Size([17, 5120]))
    aligned, combine = w._route_weights(
        SimpleNamespace(combine_metadata=metadata, hidden_states=torch.empty(n, 1)), None
    )
    assert torch.equal(aligned[inverse], original.flatten())
    assert not torch.equal(aligned, aligned.bfloat16().float())
    assert torch.equal(combine.topk_weights.float(), original.ne(0).float())
    assert metadata.topk_weights is original


@pytest.mark.parametrize("topk", [3, 6])
def test_uneven_alltoall_preserves_original_fp32(topk, monkeypatch):
    # Independent value-level oracle: each assignment records its source rank,
    # token and lane. Use 8 uneven source batches and shuffled expert IDs.
    gen = torch.Generator().manual_seed(104)
    world, per_rank = 8, 16 if topk == 3 else 48
    data, source_sorted, source_rev = [], [], []
    for src in range(world):
        experts = torch.randperm((4 + src) * topk, generator=gen) % (world * per_rank)
        experts = (experts * 71 + 51 * src) % (world * per_rank)
        weights = torch.rand((4 + src, topk), generator=gen)
        rows = list(zip(experts.tolist(), weights.flatten().tolist()))
        perm = sorted(range(len(rows)), key=lambda i: rows[i][0])
        data.append((rows, weights))
        source_sorted.append([rows[i] for i in perm])
        source_rev.append(torch.argsort(torch.tensor(perm)))

    for rank in range(world):
        rows, weights = data[rank]
        send = [sum(e // per_rank == dst for e, _ in rows) for dst in range(world)]
        peer_rows = [[(e, val) for e, val in peers if e // per_rank == rank] for peers in source_sorted]
        recv = list(map(len, peer_rows))
        wire = [pair for batch in peer_rows for pair in batch]
        expert_order = sorted(range(len(wire)), key=lambda i: wire[i][0] % per_rank)
        inverse = torch.argsort(torch.tensor(expert_order, dtype=torch.long))
        metadata = w.MoEAllToAllCombineMetadata(
            send, recv, weights, source_rev[rank], inverse, (4 + rank, 5120), (4 + rank, 5120)
        )
        work = Mock()

        received = torch.tensor([v for _, v in wire], dtype=torch.float32)
        exchange = Mock(return_value=(None, received, work))
        monkeypatch.setattr(w, "async_all_to_all", exchange)
        aligned, combine = w._route_weights(SimpleNamespace(combine_metadata=metadata), SimpleNamespace(ep_group=rank))
        exchange.assert_called_once()
        dispatched, output_splits, input_splits, group = exchange.call_args.args
        assert (input_splits, output_splits, group) == (send, recv, rank)
        assert dispatched.dtype == torch.float32
        assert dispatched.tolist() == [v for _, v in source_sorted[rank]]
        assert aligned.tolist() == [wire[i][1] for i in expert_order]
        assert torch.equal(combine.topk_weights.float(), torch.ones_like(weights))
        work.wait.assert_called_once_with()


def test_mc2_consumes_expanded_scales_once():
    topk = torch.tensor([[0.345678, 0.654322]], dtype=torch.float32)
    expanded = torch.tensor([0.654322, 0.345678, 0, 0], dtype=torch.float32)
    metadata = w.MoEMC2CombineMetadata(torch.tensor([[4, 1]]), topk, None, None, None, None, expanded, None)
    d = SimpleNamespace(combine_metadata=metadata, hidden_states=torch.empty(4, 5120))
    aligned, combine = w._route_weights(d, None)
    assert torch.equal(aligned, expanded)
    assert torch.equal(combine.topk_weights, torch.ones_like(topk))
    assert torch.equal(combine.expand_scales, torch.ones_like(expanded))
    with pytest.raises(ValueError):
        w._route_weights(SimpleNamespace(combine_metadata=metadata, hidden_states=torch.empty(3, 1)), None)
