# SPDX-License-Identifier: Apache-2.0
"""CPU replay of the three V4.1 dispatch metadata contracts, using the installed types."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import weighted_experts as w
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput, MoETokenDispatchOutput


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


def test_v41_mc2_transports_bf16_and_ignores_unused_scale(monkeypatch):
    x = torch.randn(4, 128).bfloat16()
    weights = torch.tensor([[0.345678, 0.654322]]).repeat(4, 1)
    quant = MoEQuantParams(quant_type=w.QuantType.W4A8MXFP, mxfp=MoEMxfpParams(act_quant_type=torch.float8_e4m3fn))
    inputs = MoETokenDispatchInput(x, weights, torch.tensor([[1, 2]]).repeat(4, 1), None, quant)
    counts = torch.tensor([4])
    metadata = object()
    expanded = x.flip(0)
    unused_scale = torch.full((4,), float("nan"))
    returned = MoETokenDispatchOutput(expanded, counts, 0, metadata, unused_scale)
    dispatcher = object.__new__(w.TokenDispatcherWithMC2)
    dispatch = Mock(return_value=returned)
    monkeypatch.setattr(dispatcher, "token_dispatch", dispatch)
    codes = torch.empty_like(expanded, dtype=torch.float8_e4m3fn)
    scales = torch.ones((4, 2, 2), dtype=torch.uint8)
    encode = Mock(return_value=(codes, scales))
    monkeypatch.setattr(w, "_quantize", encode)
    actual = w._dispatch_for_w1(dispatcher, inputs)
    exchange = dispatch.call_args.args[0]
    assert exchange.hidden_states is x and exchange.topk_weights is weights and exchange.topk_ids is inputs.topk_ids
    assert (
        not exchange.quant.dispatch_with_quant and exchange.quant.comm_quant_mode == 0 and exchange.quant.mxfp is None
    )
    assert inputs.quant is quant and quant.dispatch_with_quant
    encode.assert_called_once_with(expanded)
    assert actual.hidden_states is codes and actual.dynamic_scale is scales
    assert actual.combine_metadata is metadata and actual.group_list is counts and actual.group_list_type == 0

    other = SimpleNamespace(token_dispatch=Mock(return_value=returned))
    assert w._dispatch_for_w1(other, inputs) is returned
    other.token_dispatch.assert_called_once_with(inputs)
    encode.assert_called_once()


@pytest.mark.parametrize("rows", [6, 910, 911])
def test_mx_floor_preserves_codes_and_native_geometry(rows, monkeypatch):
    width = 2304
    values = torch.tensor([0.0, -0.0, 2**-28, -(2**-28), 2**-22, 2**-12, 1.0, -3.0])
    x = values.repeat(rows, width // values.numel()).bfloat16()
    source = x.clone()
    exponents = torch.tensor([0, 104, 105, 106, 254, 255], dtype=torch.uint8)
    scale = exponents.repeat(rows, width // 32 // exponents.numel()).reshape(rows, width // 64, 2)
    codes = torch.arange(256, dtype=torch.uint8).repeat(rows, width // 256)
    floored = (x.float() * (2**22)).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    expected = torch.where(scale.reshape(rows, -1).repeat_interleave(32, dim=-1) < 105, floored, codes)
    native = Mock(return_value=(codes.clone().view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)))
    monkeypatch.setattr(w.torch_npu, "npu_dynamic_mx_quant", native)
    helper = Mock(wraps=w._floor_codes)
    monkeypatch.setattr(w, "_floor_codes", helper)
    q, s = w._quantize(x)
    native.assert_called_once()
    assert native.call_args.args[0] is x
    assert native.call_args.kwargs == {"dst_type": torch.float8_e4m3fn, "scale_alg": 1}
    assert [item.args[0].shape[0] for item in helper.call_args_list] == ([910, 1] if rows == 911 else [rows])
    assert torch.equal(q.view(torch.uint8), expected) and torch.equal(s.view(torch.uint8), scale.clamp_min(105))
    assert q.dtype == torch.float8_e4m3fn and s.shape == scale.shape and s.dtype == torch.float8_e8m0fnu
    assert torch.equal(x.view(torch.int16), source.view(torch.int16))
