# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from vllm_ascend.models.deepseek_v41 import moe
from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig
from vllm_ascend.models.deepseek_v41.model import read_parameter_slice


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
@pytest.mark.parametrize("defer_mask", [False, True])
def test_decode_keeps_dp_bucket_ownership_when_slots_reorder_or_go_idle(tiny_linears, second_dp_active, defer_mask):
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
        result = layer.forward_decode(inputs[dp_rank], masks[dp_rank], defer_output_mask=defer_mask)
        if defer_mask:
            result = torch.where(masks[dp_rank][:, None], result, 0)
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


class ExpectedGroup:
    """Check the exact collective inputs and return the dense-reference result."""

    def __init__(self, rank, size, operations):
        self.rank_in_group, self.world_size = rank, size
        self.operations = list(operations)

    def run(self, op, values):
        expected_op, expected, result = self.operations.pop(0)
        assert op == expected_op
        torch.testing.assert_close(values, expected)
        return result.clone()

    def all_reduce(self, values):
        return self.run("sum", values)

    def all_gather(self, values, dim):
        assert dim == 0
        return self.run("gather", values)

    def reduce_scatter(self, values, dim):
        assert dim == 0 and values.dtype == torch.float32
        return self.run("scatter", values)


def pad_rows(states, size):
    return torch.cat((states, states.new_zeros(size - states.shape[0], states.shape[1])))


def silu_times_up(x):
    return torch.nn.functional.silu(x) * x


@pytest.mark.parametrize(
    "sequence_parallel,replicated_shared", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("decode,idle_second", [(False, False), (False, True), (True, False), (True, True)])
def test_parallel_modes_match_dense_reference_on_tp8_dp2_ep16(
    monkeypatch, sequence_parallel, replicated_shared, decode, idle_second
):
    tp_size, ep_size = 8, 16

    class Column(TinyLinear):
        def __init__(self, inputs, outputs, **kwargs):
            super().__init__(inputs, outputs // tp_size, **kwargs)

    class Row(TinyLinear):
        def __init__(self, inputs, outputs, **kwargs):
            super().__init__(inputs // tp_size, outputs, **kwargs)

    monkeypatch.setattr(moe, "ReplicatedLinear", TinyLinear)
    monkeypatch.setattr(moe, "ColumnParallelLinear", Column)
    monkeypatch.setattr(moe, "RowParallelLinear", Row)
    monkeypatch.setattr(moe, "tensor_model_parallel_all_reduce", lambda x: x * tp_size)
    config = moe.DeepseekV41TextConfig(
        hidden_size=1, moe_intermediate_size=tp_size, n_routed_experts=ep_size, num_experts_per_tok=2
    )
    sizes = (8, 8) if decode else (11, 0 if idle_second else 3)
    raw = [(torch.arange(n).float().unsqueeze(1) + 1 + 2 * d) / 8 for d, n in enumerate(sizes)]
    masks = [torch.ones(n, dtype=torch.bool) for n in sizes]
    if decode:
        masks[0][1::2] = False
        masks[1][::2] = False
        if idle_second:
            masks[1].fill_(False)
    full = [torch.where(mask[:, None], x, 0.0) for x, mask in zip(raw, masks)]
    widths = [(n + tp_size - 1) // tp_size for n in sizes]
    padded = [pad_rows(x, w * tp_size) for x, w in zip(full, widths)]
    shards = [padded[d].narrow(0, rank * widths[d], widths[d]) for d in range(2) for rank in range(tp_size)]
    ep_width = max(widths)
    gathered = torch.cat([pad_rows(x, ep_width) for x in shards]) if sequence_parallel else torch.cat(full)
    activations = silu_times_up(gathered)
    # Bias selects experts 0 and 15 at weight .75 each. W2 multiplies expert i
    # by i+1, and every full expert has eight identical intermediate channels.
    routed = (1 + ep_size) * 0.75 * tp_size * activations
    for ep_rank in range(ep_size):
        dp_rank, tp_rank = divmod(ep_rank, tp_size)
        local = shards[ep_rank] if sequence_parallel else full[dp_rank]
        factor = 0.75 * tp_size * (ep_rank + 1) if ep_rank in (0, ep_size - 1) else 0
        expected_partial = factor * activations
        if sequence_parallel:
            dp_chunks = [shards[d * tp_size + tp_rank] for d in range(2)]
            ep_ops = [
                ("gather", pad_rows(local, ep_width), gathered),
                ("scatter", expected_partial, routed.narrow(0, ep_rank * ep_width, ep_width)),
            ]
        else:
            dp_chunks = full
            ep_ops = [("sum", expected_partial, routed)]
        tp_ops = []
        if local.shape[0] and sequence_parallel and not replicated_shared:
            tp_ops = [
                ("gather", local, padded[dp_rank]),
                ("scatter", silu_times_up(padded[dp_rank]), tp_size * silu_times_up(local)),
            ]
        elif local.shape[0] and replicated_shared and not sequence_parallel:
            tp_ops = [("gather", tp_size * silu_times_up(shards[ep_rank]), tp_size * silu_times_up(padded[dp_rank]))]
        tp = ExpectedGroup(tp_rank, tp_size, tp_ops)
        ep = ExpectedGroup(ep_rank, ep_size, ep_ops)
        comm = FixedCountsCommunication(
            GatherGroup(dp_chunks, dp_rank),
            ep,
            tp,
            sequence_parallel=sequence_parallel,
            replicated_shared=replicated_shared,
        )
        if decode:

            def reject_host_count(count):
                raise AssertionError("Decode replay performed a dynamic CPU size collective")

            comm._token_counts = reject_host_count
        layer = moe.DeepseekV41MoE(config, None, "layers.0.ffn", communication=comm)
        with torch.no_grad():
            layer.gate.weight.zero_()
            layer.gate.bias.zero_()
            layer.gate.bias[0], layer.gate.bias[-1] = 2.0, 1.0
            layer.experts[ep_rank].w2.weight.fill_(ep_rank + 1)
        shared_width = tp_size if replicated_shared else 1
        assert layer.shared_experts.w1.weight.shape == (shared_width, 1)
        assert layer.shared_experts.w2.weight.shape == (1, shared_width)
        if decode:
            mask = (
                masks[dp_rank].narrow(0, tp_rank * widths[dp_rank], widths[dp_rank])
                if sequence_parallel
                else masks[dp_rank]
            )
            inputs = local.masked_fill(~mask[:, None], float("nan"))
            actual = layer.forward_decode(inputs, mask)
        else:
            actual = layer(local)
        expected_output = ((1 + ep_size) * 0.75 + 1) * tp_size * silu_times_up(local)
        torch.testing.assert_close(actual, expected_output)
        assert not tp.operations and not ep.operations


def test_shared_dp_loads_the_complete_w8a8_matrix_and_channel_companions_on_every_tp_rank(tmp_path, monkeypatch):
    class QuantReplica(nn.Module):
        def __init__(self, inputs, outputs, *, disable_tp, **kwargs):
            super().__init__()
            assert disable_tp is True
            self.weight = nn.Parameter(torch.empty(outputs, inputs, dtype=torch.int8), requires_grad=False)
            self.weight_scale = nn.Parameter(torch.empty(outputs, 1, dtype=torch.bfloat16), requires_grad=False)
            self.weight_offset = nn.Parameter(torch.empty(outputs, 1, dtype=torch.bfloat16), requires_grad=False)

    monkeypatch.setattr(moe, "ReplicatedLinear", QuantReplica)
    config = DeepseekV41TextConfig(hidden_size=4, moe_intermediate_size=8)
    prefix = "layers.0.ffn.shared_experts"
    tensors = {}
    for name, (rows, columns) in {"w1": (8, 4), "w2": (4, 8), "w3": (8, 4)}.items():
        tensors[f"{prefix}.{name}.weight"] = torch.arange(rows * columns, dtype=torch.int8).reshape(rows, columns)
        tensors[f"{prefix}.{name}.weight_scale"] = torch.arange(1, rows + 1, dtype=torch.bfloat16).reshape(rows, 1)
        tensors[f"{prefix}.{name}.weight_offset"] = torch.zeros(rows, 1, dtype=torch.bfloat16)
    path = tmp_path / "shared.safetensors"
    save_file(tensors, str(path))
    with torch.no_grad(), safe_open(str(path), framework="pt", device="cpu") as reader:
        for tp_rank in range(8):
            expert = moe.DeepseekV41MoEExpert(config, None, prefix, shared=True, replicated_shared=True)
            for name, param in expert.named_parameters():
                module = getattr(expert, name.split(".")[0])
                key = prefix + "." + name
                read_parameter_slice(reader, key, param, module, tp_rank, 8)
                assert torch.equal(param, tensors[key])


def test_native_runner_owns_routing_and_receives_local_rows(tiny_linears, monkeypatch):
    from vllm_ascend.models.deepseek_v41 import fused_moe

    parallel = SimpleNamespace(
        enable_eplb=False,
        enable_elastic_ep=False,
        expert_placement_strategy="linear",
        enable_dbo=False,
        prefill_context_parallel_size=1,
        tensor_parallel_size=8,
        data_parallel_size=2,
        enable_expert_parallel=True,
        use_sequence_parallel_moe=True,
    )
    group = SimpleNamespace(world_size=16, rank_in_group=0)
    monkeypatch.setattr(moe, "get_current_vllm_config", lambda: SimpleNamespace(parallel_config=parallel))
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=0))
    monkeypatch.setattr(moe, "get_ep_group", lambda: group)
    monkeypatch.setattr(moe, "get_tp_group", lambda: SimpleNamespace(world_size=8, rank_in_group=0))
    monkeypatch.setattr(moe, "get_ascend_config", lambda: SimpleNamespace(enable_shared_expert_dp=True))
    received = []

    class Recorder(nn.Module):
        def forward(self, *, hidden_states, router_logits):
            received.append((hidden_states.clone(), router_logits.clone()))
            return hidden_states + 2

    def factory(**kwargs):
        assert kwargs["runner_cls"] is fused_moe.DeepseekV41FusedMoERunner
        assert kwargs["is_sequence_parallel"] and kwargs["n_shared_experts"] == 0
        assert kwargs["ckpt_names"] == ("w1", "w2", "w3")
        return Recorder()

    monkeypatch.setattr(fused_moe, "FusedMoEFactory", factory)
    config = tiny_config()
    config.n_routed_experts = 32
    layer = moe.DeepseekV41MoE(config, None, "layers.0.ffn")
    states = torch.ones(2, config.hidden_size)
    with torch.no_grad():
        layer.gate.weight.fill_(1)
        actual = layer._routed_output(states, decode=True)
        empty = layer._routed_output(states[:0], decode=False)
    torch.testing.assert_close(actual, states + 2)
    assert empty.shape == (0, config.hidden_size)
    assert [s.shape[0] for s, _ in received] == [2, 0]
    assert [logits.shape for _, logits in received] == [(2, 32), (0, 32)]


def test_fused_runner_restores_outer_comm_on_failure(monkeypatch):
    from vllm_ascend.models.deepseek_v41 import fused_moe

    outer = object()
    forward = SimpleNamespace(moe_comm_type="outer", moe_comm_method=outer, padded_num_tokens=0)
    monkeypatch.setattr(fused_moe, "_EXTRA_CTX", forward)
    monkeypatch.setattr(fused_moe, "is_forward_context_available", lambda: False)
    runner = SimpleNamespace(v41_comm=object())

    def fail(self, *args):
        assert forward.moe_comm_type == fused_moe.MoECommType.ALLGATHER
        assert forward.moe_comm_method is runner.v41_comm
        raise RuntimeError("test dispatch failed")

    monkeypatch.setattr(fused_moe.AscendMoERunner, "forward", fail)
    real = object.__new__(fused_moe.DeepseekV41FusedMoERunner)
    nn.Module.__init__(real)
    real.v41_comm = runner.v41_comm
    real.v41_fused_mc2 = None
    real.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=1))
    with pytest.raises(RuntimeError, match="test dispatch failed"):
        real.forward(torch.empty(0))
    assert forward.moe_comm_method is outer and forward.moe_comm_type == "outer"


def test_modality_after_ep_gather_changes_selection_but_not_routing_scores():
    config = tiny_config()
    config.num_experts_per_tok = 1
    gate = moe.DeepseekV41Gate(config, vision_routing=True)
    with torch.no_grad():
        gate.weight.fill_(1)
        gate.bias.zero_()
        gate.bias[0] = 3
        gate.bias_vl.zero_()
        gate.bias_vl[1] = 5
    inputs = torch.ones(2, config.hidden_size)
    mask = torch.tensor([False, True])
    logits = gate.routing_logits(inputs, decode=False, image_mask=mask)
    weights, indices = gate.select_experts(inputs, logits.flip(0), 1, False)
    assert indices.flatten().tolist() == [1, 0]
    torch.testing.assert_close(weights[0], weights[1])
    assert gate.routing_logits(inputs, decode=True, image_mask=None).shape[-1] == config.n_routed_experts
    assert gate.routing_logits(inputs[:0], decode=False, image_mask=None).shape == (0, config.n_routed_experts + 1)
