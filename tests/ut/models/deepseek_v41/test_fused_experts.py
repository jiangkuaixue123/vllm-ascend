# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import fused_experts as fused
from vllm_ascend.ops import register_custom_ops as custom_ops
from vllm_ascend.ops.fused_moe import moe_comm_method, token_dispatcher
from vllm_ascend.ops.fused_moe.dataclass import moe_mlp
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput


def quantize(x, dst_type=torch.int8):
    assert dst_type == torch.int8
    scale = x.float().abs().amax(-1) / 127
    scale = torch.where(scale == 0, 1.0, scale)
    return (x.float() / scale[:, None]).round().to(torch.int8), scale


def grouped_cpu(x, weight, group_list, group_list_type, group_type, split_item, output_dtype, **options):
    assert (group_list_type, group_type, split_item) == (1, 0, 2)
    w = weight[0]
    y = torch.full((len(x[0]), w.shape[-1]), float("nan"), dtype=output_dtype)
    start = 0
    for expert, count in enumerate(group_list.tolist()):
        stop = start + count
        product = x[0][start:stop].float() @ w[expert].float()
        if "scale" in options:
            product *= options["scale"][0][expert].float()
            product *= options["per_token_scale"][0][start:stop, None].float()
        y[start:stop] = product
        start = stop
    return [y]


@pytest.mark.parametrize("top_k,quant", [(3, False), (6, False), (3, True), (6, True)])
@pytest.mark.parametrize("idle", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_grouped_allgather_keeps_fp32_route_before_down(monkeypatch, top_k, quant, idle, chunked):
    # Two of eight experts are local. Remote pairs deliberately alias valid
    # destinations using the signed native index encoding; padded GMM rows are NaN.
    experts, hidden, intermediate = 8, 4, 6
    generator = torch.Generator().manual_seed(17)
    qtype = fused.QuantType.W8A8 if quant else fused.QuantType.NONE
    comm = object.__new__(fused.DeepseekV41AllGatherComm)
    comm.moe_config = SimpleNamespace(
        experts_per_token=top_k,
        num_experts=experts,
        num_local_experts=2,
        swiglu_limit=2.0,
        activation="silu",
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    comm.token_dispatcher = comm._get_token_dispatcher()
    if chunked:
        monkeypatch.setattr(fused, "MAX_EAGER_ROUTED_TOKENS", 2)
    monkeypatch.setattr(moe_comm_method, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=comm))
    monkeypatch.setattr(fused, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=comm))
    monkeypatch.setattr(token_dispatcher, "get_ep_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(moe_mlp, "enable_fusion_gmmswigluquant", lambda: False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: Mock())
    monkeypatch.setattr(fused.torch_npu, "npu_dynamic_quant", quantize)
    monkeypatch.setattr(fused.torch_npu, "npu_grouped_matmul", grouped_cpu)
    states = torch.randn(5, hidden, generator=generator).bfloat16()
    if idle:
        states.zero_()
    ids = torch.argsort(torch.randn(5, experts, generator=generator), dim=-1)[:, :top_k].int()
    route = torch.rand(5, top_k, generator=generator)
    owned = torch.full((experts,), -1, dtype=torch.int32)
    owned[2:4] = torch.tensor([0, 1], dtype=torch.int32)
    dtype = torch.int8 if quant else torch.bfloat16
    w1 = torch.randint(-6, 7, (2, hidden, 2 * intermediate), generator=generator).to(dtype)
    w2 = torch.randint(-6, 7, (2, intermediate, hidden), generator=generator).to(dtype)
    s1 = torch.rand(2, 2 * intermediate, generator=generator)
    s2 = torch.rand(2, hidden, generator=generator).bfloat16()

    dispatch_ids = []
    prepared = quant and chunked

    def dispatch(x, expert_ids, *, active_num, active_expert_range, quant_mode, scale, **kwargs):
        dispatch_ids.append(expert_ids)
        assert active_num == expert_ids.numel() and active_expert_range == [2, 4]
        assert quant_mode == (1 if quant and not prepared else -1)
        flat = expert_ids.flatten()
        belonging = [torch.where(flat == i)[0] for i in (2, 3)]
        original = torch.cat(belonging)
        mapping = torch.full((active_num,), -1, dtype=torch.int32)
        mapping[original] = torch.arange(len(original), dtype=torch.int32)
        sorted_x = torch.zeros(active_num, hidden, dtype=x.dtype)
        sorted_x[: len(original)] = x[original // top_k]
        if prepared:
            assert x.dtype == torch.int8 and scale.shape == (x.shape[0],)
            sorted_scale = scale.new_zeros(active_num)
            sorted_scale[: len(original)] = scale[original // top_k]
            scale = sorted_scale
        elif quant:
            sorted_x, scale = quantize(sorted_x)
        return sorted_x, mapping, torch.tensor([len(index) for index in belonging]), scale

    def combine(permuted_tokens, sorted_indices, probs):
        assert permuted_tokens.dtype == probs.dtype == torch.float32
        torch.testing.assert_close(probs, (owned[dispatch_ids[-1].long()] >= 0).float())
        return (permuted_tokens[sorted_indices.abs().long()].view(-1, top_k, hidden) * probs[:, :, None]).sum(1)

    monkeypatch.setattr(token_dispatcher.DeviceOperator, "npu_moe_init_routing", dispatch)
    monkeypatch.setattr(token_dispatcher.DeviceOperator, "npu_moe_token_unpermute", combine)
    prepared_states, token_scale = quantize(states) if prepared else (states, None)
    inputs = build_fused_experts_input(
        hidden_states=prepared_states,
        pertoken_scale=token_scale,
        topk_weights=route,
        topk_ids=ids,
        w1=w1,
        w2=w2,
        w1_scale=s1 if quant else None,
        w2_scale=s2 if quant else None,
        dynamic_eplb=False,
        quant_type=qtype,
        expert_map=owned,
    )
    result = comm.fused_experts(inputs)
    output = result.routed_out
    assert [len(batch) for batch in dispatch_ids] == ([2, 2, 1] if chunked else [5])
    torch.testing.assert_close(torch.cat(dispatch_ids), ids)
    torch.testing.assert_close(result.expert_tokens, torch.stack([(ids == expert).sum() for expert in (2, 3)]))

    def project(x, w, s):
        if quant:
            q, scale = quantize(x)
            return ((q.float() @ w.float()) * s.float() * scale[:, None]).bfloat16()
        return (x.float() @ w.float()).bfloat16()

    expected = torch.zeros_like(states, dtype=torch.float32)
    for row in range(len(states)):
        for slot, expert in enumerate(ids[row].tolist()):
            if expert not in (2, 3):
                continue
            idx = expert - 2
            gate, up = project(states[row : row + 1], w1[idx], s1[idx]).float().chunk(2, -1)
            act = torch.nn.functional.silu(gate.clamp(max=2)) * up.clamp(-2, 2)
            value = project((act * route[row, slot]).bfloat16(), w2[idx], s2[idx])
            expected[row] += value[0].float()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_precomputed_router_preserves_factors_and_indices():
    weights = torch.tensor([[0.124578, 0.333456, 0.675518]], dtype=torch.float32)
    ids = torch.tensor([[7, 3, 0]], dtype=torch.int32)
    router = Mock()
    router._select_experts.return_value = (weights, ids)
    states = torch.ones(1, 4, dtype=torch.int8)
    out, selected = fused.DeepseekV41RoutedExperts._select_experts(
        SimpleNamespace(router=router), states, states, False
    )
    assert out is weights and selected is ids


@pytest.mark.parametrize("sizes, rank", [(None, 11), ([5] * 8 + [3] * 8, 10), ([5] * 8 + [0] * 8, 9)])
def test_v41_draft_scatter_returns_local_rows_even_with_multimodal_wrapper(monkeypatch, sizes, rank):
    # The entire TP8 × DP2 rank set is in the expert group. Inputs contain
    # identical rank-local partials here, so a true sum multiplies each by 16.
    counts = sizes if sizes is not None else [5] * 16
    data = torch.arange(sum(counts), dtype=torch.float32).view(-1, 1)
    group = SimpleNamespace(rank_in_group=rank, world_size=16)
    max_rows = max(counts)
    parts = data.split(counts)
    padded = torch.stack([torch.nn.functional.pad(rows, (0, 0, 0, max_rows - len(rows))) for rows in parts])

    def reduce_scatter(x, dim):
        assert dim == 0
        torch.testing.assert_close(x, padded.view(-1, 1))
        return 16 * x.view(16, max_rows, 1)[rank]

    group.reduce_scatter = reduce_scatter
    metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: sizes) if sizes is not None else None
    context = SimpleNamespace(dp_metadata=metadata)
    monkeypatch.setattr(fused, "get_forward_context", lambda: context)
    monkeypatch.setattr(custom_ops, "get_ep_group", lambda: group)
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(custom_ops, "_EXTRA_CTX", SimpleNamespace(is_draft_model=True))
    monkeypatch.setattr(custom_ops, "is_vl_model", lambda: True)
    monkeypatch.setattr(custom_ops, "tensor_model_parallel_all_reduce", lambda value: value * 8)
    assert custom_ops._maybe_pad_and_reduce_impl(data).shape == data.shape
    stage = object.__new__(fused.DeepseekV41PrepareAndFinalize)
    stage.moe_config = SimpleNamespace(is_sequence_parallel=True, pcp_size=1)
    result = stage.finalize(data, reduce_results=True, padded_hidden_states_shape=parts[rank].shape)
    torch.testing.assert_close(result, parts[rank] * 16)


@pytest.mark.parametrize(
    "physical, actual, rank",
    [
        ([3] * 16, [1] * 8 + [3] * 8, 2),
        ([3] * 16, [0] * 8 + [3] * 8, 2),
        ([3] * 8 + [2] * 8, [2] * 8 + [1] * 8, 12),
        ([3] * 8 + [0] * 8, [1] * 8 + [0] * 8, 12),
    ],
)
@pytest.mark.parametrize("compact", [False, True])
def test_ep_prepare_and_finalize_trim_physical_rows_to_the_input_shard(monkeypatch, physical, actual, rank, compact):
    width = max(physical)
    parts = [torch.arange(n, dtype=torch.float32).view(-1, 1) + (r + 1) * 10 for r, n in enumerate(actual)]
    wire = torch.stack([torch.nn.functional.pad(rows, (0, 0, 0, width - len(rows))) for rows in parts])
    metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: physical)
    context = SimpleNamespace(dp_metadata=metadata, ep_actual_local_sizes=actual if compact else None)
    group = SimpleNamespace(world_size=16, rank_in_group=rank)

    def all_gather(x, dim):
        assert dim == 0
        torch.testing.assert_close(x, wire[rank])
        return wire.flatten(0, 1)

    def reduce_scatter(x, dim):
        assert dim == 0
        torch.testing.assert_close(x, wire.flatten(0, 1))
        return wire[rank] * 16

    group.all_gather, group.reduce_scatter = all_gather, reduce_scatter
    monkeypatch.setattr(fused, "get_forward_context", lambda: context)
    monkeypatch.setattr(fused, "get_ep_group", lambda: group)
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(custom_ops, "get_ep_group", lambda: group)
    monkeypatch.setattr(
        torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", custom_ops._maybe_all_gather_and_maybe_unpad_impl
    )
    stage = object.__new__(fused.DeepseekV41PrepareAndFinalize)
    stage.moe_config = SimpleNamespace(is_sequence_parallel=True, pcp_size=1)
    prepared = stage.prepare(parts[rank], parts[rank])
    assert prepared.padded_hidden_states_shape == parts[rank].shape
    packed = torch.cat([wire[r, :size] for r, size in enumerate(actual if compact else physical)])
    torch.testing.assert_close(prepared.hidden_states, packed)
    result = stage.finalize(prepared.hidden_states, True, prepared.padded_hidden_states_shape)
    torch.testing.assert_close(result, parts[rank] * 16)
    oversized = torch.zeros(physical[rank] + 1, 1)
    with pytest.raises(ValueError, match="exceed"):
        stage.prepare(oversized, oversized)


@pytest.mark.parametrize(
    "mode,draft,profile", [("NONE", False, False), ("FULL", False, False), ("NONE", True, False), ("NONE", False, True)]
)
def test_actual_ep_layout_is_only_installed_for_target_eager(monkeypatch, mode, draft, profile):
    context = SimpleNamespace(
        dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [8] * 16),
        num_actual_tokens_across_dp=torch.tensor([5, 0, 32, 9]),
        ep_actual_local_sizes=None,
        cudagraph_runtime_mode=getattr(fused.CUDAGraphMode, mode),
        is_draft_model=draft,
        in_profile_run=profile,
    )
    monkeypatch.setattr(fused, "get_forward_context", lambda: context)
    monkeypatch.setattr(fused, "get_ep_group", lambda: SimpleNamespace(world_size=16, rank_in_group=0))
    monkeypatch.setattr(fused, "get_tensor_model_parallel_world_size", lambda: 4)
    prepared = MoEPrepareOutput(
        hidden_states=torch.zeros(2, 1),
        router_logits=torch.zeros(2, 1),
        mc2_mask=None,
        padded_hidden_states_shape=None,
        pertoken_scale=None,
    )
    monkeypatch.setattr(fused.PrepareAndFinalizeWithAllGather, "_prepare_with_ep_group", lambda *args: prepared)
    stage = object.__new__(fused.DeepseekV41PrepareAndFinalize)
    stage._prepare_with_ep_group(torch.zeros(2, 1), torch.zeros(2, 1))
    expected = [2] * 4 + [0] * 4 + [8] * 4 + [3] * 4 if mode == "NONE" and not draft and not profile else None
    assert context.ep_actual_local_sizes == expected
    if expected is not None:
        with pytest.raises(ValueError, match="disagree"):
            stage._prepare_with_ep_group(torch.zeros(3, 1), torch.zeros(3, 1))


@pytest.mark.parametrize("full", [False, True])
def test_large_expert_batch_keeps_routing_rows_and_full_capture_path(monkeypatch, full):
    rows, hidden = 5, 4
    states = torch.arange(rows * hidden).float().reshape(rows, hidden)
    ids = torch.arange(rows).int().remainder(2)[:, None]
    route = torch.linspace(0.25, 0.75, rows)[:, None]
    scales = torch.arange(1, rows + 1).float()
    mask = torch.tensor([True, False, True, True, False])
    weights = torch.zeros(1)
    inputs = build_fused_experts_input(
        hidden_states=states,
        topk_weights=route,
        topk_ids=ids,
        w1=weights,
        w2=weights,
        quant_type=fused.QuantType.NONE,
        dynamic_eplb=False,
        pertoken_scale=scales,
        mc2_mask=mask,
    )
    comm = object.__new__(fused.DeepseekV41AllGatherComm)
    comm.moe_config = SimpleNamespace(num_local_experts=2)
    chunks = []

    def local_dispatch(self, batch):
        chunks.append(batch)
        return moe_comm_method.FusedExpertsResult(
            routed_out=batch.hidden_states * batch.topk_weights,
            expert_tokens=torch.bincount(batch.topk_ids.flatten().long(), minlength=2),
            before_dispatch_evt=len(chunks),
            before_combine_evt=len(chunks),
        )

    monkeypatch.setattr(fused, "MAX_EAGER_ROUTED_TOKENS", 2)
    monkeypatch.setattr(fused, "is_forward_context_available", lambda: True)
    mode = fused.CUDAGraphMode.FULL if full else fused.CUDAGraphMode.NONE
    monkeypatch.setattr(fused, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    monkeypatch.setattr(fused.AllGatherCommImpl, "fused_experts", local_dispatch)
    result = comm.fused_experts(inputs)
    assert [len(c.hidden_states) for c in chunks] == ([5] if full else [2, 2, 1])
    torch.testing.assert_close(torch.cat([c.routing.mc2_mask for c in chunks]), mask)
    torch.testing.assert_close(torch.cat([c.routing.pertoken_scale for c in chunks]), scales)
    torch.testing.assert_close(result.routed_out, states * route)
    torch.testing.assert_close(result.expert_tokens, torch.tensor([3, 2]))
    assert result.before_dispatch_evt == 1 and result.before_combine_evt == len(chunks)


@pytest.mark.parametrize("top_k", [3, 6])
@pytest.mark.parametrize("chunked", [False, True])
def test_w8a8_grouped_capture_replay_on_npu(monkeypatch, top_k, chunked):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    if chunked:
        monkeypatch.setattr(fused, "MAX_EAGER_ROUTED_TOKENS", 3)

    gen = torch.Generator().manual_seed(137)
    rows, hidden, intermediate, experts, local = 8, 128, 128, 8, 4
    states = torch.randn(rows, hidden, generator=gen).bfloat16().npu()
    ids = torch.argsort(torch.randn(rows, experts, generator=gen), -1)[:, :top_k].to(torch.int32).contiguous().npu()
    route = torch.rand(rows, top_k, generator=gen).npu()
    owned = torch.tensor([-1] * 4 + list(range(local)), dtype=torch.int32).npu()
    w1 = torch.randint(-16, 17, (local, hidden, 2 * intermediate), generator=gen, dtype=torch.int8).npu()
    w2 = torch.randint(-16, 17, (local, intermediate, hidden), generator=gen, dtype=torch.int8).npu()
    s1 = (torch.rand(local, 2 * intermediate, generator=gen) * 0.04 + 0.005).bfloat16().npu()
    s2 = (torch.rand(local, hidden, generator=gen) * 0.04 + 0.005).bfloat16().npu()
    comm = object.__new__(fused.DeepseekV41AllGatherComm)
    comm.moe_config = SimpleNamespace(
        experts_per_token=top_k,
        num_experts=experts,
        num_local_experts=local,
        swiglu_limit=2.0,
        activation="silu",
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    comm.token_dispatcher = comm._get_token_dispatcher()
    monkeypatch.setattr(moe_comm_method, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=comm))
    monkeypatch.setattr(fused, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=comm))
    monkeypatch.setattr(token_dispatcher, "get_ep_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(moe_mlp, "enable_fusion_gmmswigluquant", lambda: False)
    inputs = build_fused_experts_input(
        hidden_states=states,
        topk_weights=route,
        topk_ids=ids,
        expert_map=owned,
        w1=fused.torch_npu.npu_format_cast(w1, ACL_FORMAT_FRACTAL_NZ),
        w2=fused.torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ),
        w1_scale=s1,
        w2_scale=s2,
        dynamic_eplb=False,
        quant_type=fused.QuantType.W8A8,
    )

    def linear(x, w, scale):
        q, token_scale = fused.torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        return fused.torch_npu.npu_quant_matmul(q, w, scale, pertoken_scale=token_scale, output_dtype=torch.bfloat16)

    def expected():
        answer = torch.zeros_like(states, dtype=torch.float32)
        for e in range(local):
            gate, up = linear(states, w1[e], s1[e]).float().chunk(2, -1)
            matches = ids == (e + local)
            coefficient = torch.where(matches, route, 0).sum(1, keepdim=True)
            act = torch.nn.functional.silu(gate.clamp(max=2)) * up.clamp(-2, 2)
            value = linear((act * coefficient).bfloat16(), w2[e], s2[e])
            answer += torch.where(matches.any(1, keepdim=True), value.float(), 0)
        return answer

    def execute():
        batch = inputs
        if chunked:
            # The production SP prepare has already quantized gathered tokens.
            prepared, scale = fused.torch_npu.npu_dynamic_quant(states, dst_type=torch.int8)
            batch = replace(inputs, hidden_states=prepared, routing=replace(inputs.routing, pertoken_scale=scale))
        return comm.fused_experts(batch).routed_out

    with torch.inference_mode():
        eager = execute()
        torch.npu.synchronize()
        torch.testing.assert_close(eager.cpu(), expected().cpu(), rtol=0, atol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            observed = execute()
        ids.copy_(torch.roll(ids, 3, dims=0))
        route.mul_(0.73)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(observed.cpu(), expected().cpu(), rtol=0, atol=0)
        previous_ids = ids.clone()
        ids.zero_()  # All routes are owned by a different EP rank.
        graph.replay()
        torch.npu.synchronize()
        assert torch.count_nonzero(observed.cpu()) == 0
        ids.copy_(previous_ids)
        states.zero_()
        graph.replay()
        torch.npu.synchronize()
        assert torch.count_nonzero(observed.cpu()) == 0


@pytest.mark.parametrize(
    "rows,placement",
    [
        (8192, "balanced"),
        (8193, "balanced"),
        (8193, "local"),
        (8193, "remote"),
        (4097, "local"),
        (4097, "remote"),
        (8652, "compact"),
        (16648, "compact"),
    ],
)
def test_prefill_chain_and_batch_match_original_target_width_on_npu(monkeypatch, rows, placement):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op

    enable_custom_op()

    hidden, intermediate, experts, local, top_k = 5120, 2304, 384, 24, 6
    gen = torch.Generator().manual_seed(317)
    states = torch.randn(rows, hidden, generator=gen).bfloat16().npu()
    ids = (torch.arange(rows)[:, None] * top_k + torch.arange(top_k)) % experts
    if placement == "local":
        ids.remainder_(local)
    elif placement == "remote":
        ids = ids.remainder(experts - local) + local
    ids = ids.to(torch.int32).npu()
    route = torch.rand(rows, top_k, generator=gen).npu()
    owned = torch.tensor([*range(local), *([-1] * (experts - local))], dtype=torch.int32).npu()
    w1 = torch.randint(-8, 9, (local, hidden, 2 * intermediate), generator=gen, dtype=torch.int8).npu()
    w2 = torch.randint(-8, 9, (local, intermediate, hidden), generator=gen, dtype=torch.int8).npu()
    s1 = (torch.rand(local, 2 * intermediate, generator=gen) * 0.01 + 0.001).bfloat16().npu()
    s2 = (torch.rand(local, hidden, generator=gen) * 0.01 + 0.001).bfloat16().npu()
    comm = object.__new__(fused.DeepseekV41AllGatherComm)
    comm.moe_config = SimpleNamespace(
        experts_per_token=top_k,
        num_experts=experts,
        num_local_experts=local,
        swiglu_limit=10.0,
        activation="silu",
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    comm.token_dispatcher = comm._get_token_dispatcher()
    monkeypatch.setattr(moe_comm_method, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=comm))
    monkeypatch.setattr(fused, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=comm))
    monkeypatch.setattr(token_dispatcher, "get_ep_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(moe_mlp, "enable_fusion_gmmswigluquant", lambda: False)
    monkeypatch.setattr(fused, "is_forward_context_available", lambda: False)
    quantized, token_scale = fused.torch_npu.npu_dynamic_quant(states, dst_type=torch.int8)
    inputs = build_fused_experts_input(
        hidden_states=quantized,
        pertoken_scale=token_scale,
        topk_weights=route,
        topk_ids=ids,
        expert_map=owned,
        w1=fused.torch_npu.npu_format_cast(w1, ACL_FORMAT_FRACTAL_NZ),
        w2=fused.torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ),
        w1_scale=s1,
        w2_scale=s2,
        dynamic_eplb=False,
        quant_type=fused.QuantType.W8A8,
    )
    del w1, w2
    reference_inputs, kept_rows = inputs, None
    if placement == "compact":
        sizes = [41] * 8 + [2040] * 4 + [41] * 4 if rows == 8652 else ([41] * 4 + [2040] * 4) * 2
        kept_rows = torch.cat([torch.arange(n) + rank * 2040 for rank, n in enumerate(sizes)]).npu()
        padded_states = quantized.new_zeros((32640, hidden))
        padded_states[kept_rows] = quantized
        padded_scale = token_scale.new_zeros((32640,))
        padded_scale[kept_rows] = token_scale
        padded_ids = torch.arange(top_k, dtype=ids.dtype, device=ids.device).expand(32640, -1).clone()
        padded_ids[kept_rows] = ids
        padded_weights = route.new_ones((32640, top_k))
        padded_weights[kept_rows] = route
        reference_inputs = replace(
            inputs,
            hidden_states=padded_states,
            topk_ids=padded_ids,
            topk_weights=padded_weights,
            routing=replace(inputs.routing, pertoken_scale=padded_scale),
        )
    with torch.inference_mode():
        with monkeypatch.context() as reference:
            reference.setattr(fused, "prefill_fusion_available", lambda states: False)
            old = comm.fused_experts(reference_inputs)
            expected = (old.routed_out if kept_rows is None else old.routed_out[kept_rows]).cpu()
            counts = (
                old.expert_tokens.cpu()
                if kept_rows is None
                else torch.bincount(ids.cpu().flatten().long(), minlength=experts)[:local]
            )
            del old
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        resident = torch.npu.memory_allocated()
        requantized_rows = []
        original_requant = fused.prefill_route_quantized_activation

        def record_requant(*args, **kwargs):
            output = original_requant(*args, **kwargs)
            if output is not None:
                requantized_rows.append(args[0].shape[0])
            return output

        monkeypatch.setattr(fused, "prefill_route_quantized_activation", record_requant)
        result = comm.fused_experts(inputs)
        if fused.prefill_native_requant_available(inputs.hidden_states):
            assert requantized_rows, "The full W2 consumer must execute the native requant path"
        assert all(
            event is not None
            for event in (result.before_dispatch_evt, result.before_gmm2_evt, result.before_combine_evt)
        )
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated() - resident
        print(f"PREFILL_WORKSPACE rows={rows} placement={placement} peak_bytes={peak}")
        torch.testing.assert_close(result.routed_out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(result.expert_tokens.cpu(), counts, rtol=0, atol=0)
        assert torch.isfinite(result.routed_out).all()
        if placement == "remote":
            assert torch.count_nonzero(result.routed_out) == 0
