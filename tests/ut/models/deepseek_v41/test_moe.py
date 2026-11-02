# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v41 import moe
from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig


def test_ep16_owns_each_of_384_experts_exactly_once():
    partitions = [moe.local_expert_ids(384, 16, rank) for rank in range(16)]
    assert all(len(partition) == 24 for partition in partitions)
    assert [expert for partition in partitions for expert in partition] == list(range(384))


@pytest.mark.parametrize("num_experts,ep_size,rank", [(3, 2, 0), (4, 2, 2), (0, 1, 0)])
def test_invalid_expert_ownership_is_rejected(num_experts, ep_size, rank):
    with pytest.raises(ValueError):
        moe.local_expert_ids(num_experts, ep_size, rank)


class GatherGroup:
    def __init__(self, chunks, rank):
        self.chunks = chunks
        self.world_size = len(chunks)
        self.rank_in_group = rank
        self.calls = 0

    def all_gather(self, local_padded, dim):
        assert dim == 0
        self.calls += 1
        rows = max(chunk.shape[0] for chunk in self.chunks)
        padded = []
        for chunk in self.chunks:
            part = chunk.new_zeros(rows, chunk.shape[1])
            part[: chunk.shape[0]] = chunk
            padded.append(part)
        torch.testing.assert_close(local_padded, padded[self.rank_in_group])
        return torch.cat(padded)


class ReduceGroup:
    def __init__(self, rank=0, size=1, expected=None, result=None):
        self.rank_in_group = rank
        self.world_size = size
        self.expected = expected
        self.result = result
        self.calls = 0

    def all_reduce(self, contribution):
        self.calls += 1
        assert contribution.dtype == torch.float32
        if self.expected is not None:
            torch.testing.assert_close(contribution, self.expected)
        return contribution.clone() if self.result is None else self.result.clone()


@pytest.mark.parametrize("counts", [(3, 1), (0, 2), (2, 0), (0, 0)])
def test_dp_padding_keeps_rank_order_and_handles_idle_rank(counts):
    chunks = [torch.arange(count, dtype=torch.float32).unsqueeze(-1) + 100 * rank for rank, count in enumerate(counts)]
    for rank, chunk in enumerate(chunks):
        dp_group = GatherGroup(chunks, rank)
        communication = FixedCountsCommunication(dp_group, ReduceGroup())
        packed, actual_counts = communication.gather(chunk)
        torch.testing.assert_close(packed, torch.cat(chunks))
        torch.testing.assert_close(moe.select_dp_tokens(packed, actual_counts, rank), chunk)
        assert dp_group.calls == (1 if max(counts) else 0)


def test_gather_rejects_disagreement_with_collected_local_token_count():
    chunks = [torch.zeros(4, 2), torch.zeros(2, 2)]
    communication = FixedCountsCommunication(GatherGroup(chunks, 0), ReduceGroup())
    with pytest.raises(ValueError, match="replicated across TP"):
        communication.gather(chunks[0][:2])


@pytest.mark.parametrize("local_count", [1, 0])
def test_dp_collects_actual_sizes_on_cpu_even_for_idle_rank(monkeypatch, local_count):
    chunks = [
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[100.0]])[:local_count],
    ]
    dp_group = GatherGroup(chunks, 1)
    dp_group.cpu_group = object()
    communication = moe.DeepseekV41MoECommunication(dp_group, ReduceGroup())
    counts = torch.tensor([3, local_count], device="cpu")
    calls = []

    def collect_sizes(sizes, local_size, group):
        calls.append(group)
        assert group is dp_group.cpu_group
        assert local_size.device.type == "cpu"
        torch.testing.assert_close(local_size, counts[1:])
        for rank, size in enumerate(sizes):
            size.copy_(counts[rank : rank + 1])

    monkeypatch.setattr(moe.dist, "all_gather", collect_sizes)
    packed, actual_counts = communication.gather(chunks[1])
    assert actual_counts == (3, local_count)
    torch.testing.assert_close(packed, torch.cat(chunks))
    assert len(calls) == 1


class ScaleExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, states, weights):
        self.calls += 1
        return self.scale * states * weights


def test_disjoint_ep_contributions_match_manual_two_dp_batch():
    states = torch.tensor([[1.0], [10.0], [100.0], [1000.0]])
    indices = torch.tensor([[0, 2], [3, 1], [2, 0], [1, 3]])
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.9, 0.1], [0.8, 0.2]])
    partials = []
    for rank in range(4):
        experts = [ScaleExpert(rank + 1) if expert == rank else None for expert in range(4)]
        partials.append(
            moe.local_expert_contributions(states, weights, indices, experts, moe.local_expert_ids(4, 4, rank))
        )
        assert experts[rank].calls == 1
    reduced = torch.stack(partials).sum(0)
    torch.testing.assert_close(reduced, torch.tensor([[2.5], [28.0], [280.0], [2400.0]]))
    torch.testing.assert_close(moe.select_dp_tokens(reduced, (3, 1), 1), reduced[3:])


class TinyLinear(nn.Module):
    def __init__(self, input_size, output_size, **options):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(output_size, input_size))
        self.options = options
        self.seen = None

    def forward(self, states):
        self.seen = states.clone()
        return states @ self.weight.to(states.dtype).T


class TinyColumnLinear(TinyLinear):
    pass


class TinyRowLinear(TinyLinear):
    pass


class QuantizedDownLinear(TinyLinear):
    def forward(self, states):
        self.seen = states.clone()
        if self.options["prefix"].endswith(".w2"):
            states = states.round()
        return states @ self.weight.T


@pytest.fixture
def tiny_linears(monkeypatch):
    monkeypatch.setattr(moe, "ReplicatedLinear", TinyLinear)
    monkeypatch.setattr(moe, "ColumnParallelLinear", TinyColumnLinear)
    monkeypatch.setattr(moe, "RowParallelLinear", TinyRowLinear)
    monkeypatch.setattr(moe, "tensor_model_parallel_all_reduce", lambda values: values)


def tiny_config():
    return DeepseekV41TextConfig(
        hidden_size=1,
        moe_intermediate_size=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )


def test_routing_weight_reaches_down_activation_quantization(monkeypatch):
    monkeypatch.setattr(moe, "ReplicatedLinear", QuantizedDownLinear)
    expert = moe.DeepseekV41MoEExpert(tiny_config(), None, "layers.0.ffn.experts.0")
    states = torch.ones(1, 1)
    actual = expert(states, torch.tensor([[0.5]]))
    expected_down_input = 0.5 / (1 + torch.exp(torch.tensor(-1.0)))
    torch.testing.assert_close(expert.w2.seen, expected_down_input.reshape(1, 1))
    # quantize(0.5 * silu(1)) = 0; post-down weighting would give 0.5.
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    assert not torch.equal(actual, torch.full_like(actual, 0.5))


def test_shared_down_reduces_in_fp32_before_restoring_dtype(tiny_linears, monkeypatch):
    contributions = []

    def reduce_fp32(values):
        contributions.append(values)
        return values + 0.001

    monkeypatch.setattr(moe, "tensor_model_parallel_all_reduce", reduce_fp32)
    expert = moe.DeepseekV41MoEExpert(tiny_config(), None, "layers.0.ffn.shared_experts", shared=True)
    actual = expert(torch.ones(1, 1, dtype=torch.bfloat16))
    assert len(contributions) == 1
    assert contributions[0].dtype == torch.float32
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, (contributions[0] + 0.001).bfloat16())


class FixedCountsCommunication(moe.DeepseekV41MoECommunication):
    def _token_counts(self, local_count):
        return tuple(chunk.shape[0] for chunk in self.dp_group.chunks)


@pytest.mark.parametrize("counts", [(2, 1), (2, 0), (0, 1), (0, 0)])
def test_moe_sums_ep_once_then_selects_own_dp_and_adds_shared(tiny_linears, counts):
    chunks = [
        torch.tensor([[0.5], [1.5]])[: counts[0]],
        torch.tensor([[3.0]])[: counts[1]],
    ]
    packed = torch.cat(chunks)
    swiglu = packed.square() / (1 + torch.exp(-packed))
    # Uniform unbiased scores select experts 0 and 3 via bias: each weight=.75.
    routed = 3.75 * swiglu
    for rank in range(4):
        dp_rank = rank // 2
        dp_group = GatherGroup(chunks, dp_rank)
        local_scale = {0: 0.75, 3: 3.0}.get(rank, 0)
        ep_group = ReduceGroup(rank, 4, expected=local_scale * swiglu, result=routed)
        communication = FixedCountsCommunication(dp_group, ep_group)
        layer = moe.DeepseekV41MoE(tiny_config(), None, "layers.0.ffn", communication=communication)
        with torch.no_grad():
            layer.gate.weight.zero_()
            layer.gate.bias.copy_(torch.tensor([2.0, 0.0, 0.0, 1.0]))
            layer.experts[rank].w2.weight.fill_(rank + 1)
        actual = layer(chunks[dp_rank])
        expected = moe.select_dp_tokens(4.75 * swiglu, counts, dp_rank)
        torch.testing.assert_close(actual, expected)
        assert ep_group.calls == (1 if max(counts) else 0)
        assert dp_group.calls == (1 if max(counts) else 0)
        if counts[dp_rank] == 0:
            assert layer.shared_experts.w1.seen is None


def test_checkpoint_names_are_global_and_only_owned_experts_allocate(tiny_linears):
    communication = moe.DeepseekV41MoECommunication(GatherGroup([torch.zeros(1, 1)], 0), ReduceGroup(rank=1, size=2))
    quant_config = object()
    layer = moe.DeepseekV41MoE(tiny_config(), quant_config, "layers.0.ffn", communication=communication)
    assert layer.experts[0] is None and layer.experts[1] is None
    assert layer.n_local_experts == 2
    names = set(dict(layer.named_parameters()))
    assert names == {
        "gate.weight",
        "gate.bias",
        "experts.2.w1.weight",
        "experts.2.w2.weight",
        "experts.2.w3.weight",
        "experts.3.w1.weight",
        "experts.3.w2.weight",
        "experts.3.w3.weight",
        "shared_experts.w1.weight",
        "shared_experts.w2.weight",
        "shared_experts.w3.weight",
    }
    assert layer.gate.weight.dtype == layer.gate.bias.dtype == torch.float32
    assert layer.experts[2].w2.options["disable_tp"]
    assert layer.experts[2].w2.options["quant_config"] is quant_config
    assert isinstance(layer.shared_experts.w1, TinyColumnLinear)
    assert isinstance(layer.shared_experts.w2, TinyRowLinear)
    assert not layer.shared_experts.w2.options["reduce_results"]
    assert layer.shared_experts.w2.options["input_is_parallel"]


@pytest.mark.parametrize("second_dp_active", [True, False])
def test_decode_keeps_dp_bucket_ownership_when_slots_reorder_or_go_idle(tiny_linears, second_dp_active):
    masks = [
        torch.tensor([True, False, True]),
        torch.tensor([False, second_dp_active, False]),
    ]
    inputs = [
        torch.tensor([[0.5], [float("nan")], [1.5]]),
        torch.tensor([[0.0], [3.0], [0.0]]),
    ]
    padded = [torch.where(mask[:, None], states, 0.0) for states, mask in zip(inputs, masks)]
    gathered = torch.cat(padded)
    activation = gathered.square() / (1 + (-gathered).exp())
    for rank in range(4):
        dp_rank = rank // 2
        group = GatherGroup(padded, dp_rank)
        local_factor = {0: 0.75, 3: 3.0}.get(rank, 0.0)
        ep = ReduceGroup(rank, 4, expected=activation * local_factor, result=activation * 3.75)
        communication = moe.DeepseekV41MoECommunication(group, ep)
        layer = moe.DeepseekV41MoE(tiny_config(), None, "ffn", communication=communication)
        with torch.no_grad():
            layer.gate.weight.zero_()
            layer.gate.bias.copy_(torch.tensor([2.0, 0.0, 0.0, 1.0]))
            layer.experts[rank].w2.weight.fill_(rank + 1)
        result = layer.forward_decode(inputs[dp_rank], masks[dp_rank])
        expected = (4.75 * activation).narrow(0, dp_rank * 3, 3)
        torch.testing.assert_close(result, expected)
        assert group.calls == ep.calls == 1
        assert layer.experts[rank].w2.seen.shape == (6, 1)
        assert layer.shared_experts.w2.seen.shape == (3, 1)


def test_decode_changes_routes_without_changing_expert_input_shape(tiny_linears):
    slots = 3
    communication = moe.DeepseekV41MoECommunication(GatherGroup([torch.zeros(slots, 1)], 0), ReduceGroup())
    layer = moe.DeepseekV41MoE(tiny_config(), None, "ffn", communication=communication)
    with torch.no_grad():
        layer.gate.weight.copy_(torch.tensor([[2.0], [-2.0], [0.4], [-0.4]]))
        layer.gate.bias.zero_()
        for index, expert in enumerate(layer.experts):
            expert.w2.weight.fill_(index + 1)
    for inputs, mask in (
        ([1.0, -1.0, 0.0], [True, True, False]),
        ([-3.0, 0.0, 2.0], [True, False, True]),
    ):
        states = torch.tensor(inputs).reshape(slots, 1)
        valid = torch.tensor(mask)
        eager = layer(states[valid])
        captured_shape = layer.forward_decode(states, valid)
        torch.testing.assert_close(captured_shape[valid], eager)
        assert (captured_shape[~valid] == 0).all()
        assert all(expert.w2.seen.shape == (slots, 1) for expert in layer.experts)
