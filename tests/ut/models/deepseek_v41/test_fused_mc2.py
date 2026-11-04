# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import fused_experts as fused
from vllm_ascend.models.deepseek_v41 import fused_moe as runner
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input


@pytest.mark.parametrize("padded,tp", [(24, 4), (48, 4), (24, 8), (48, 8), (256, 4)])
def test_target_and_draft_decode_buckets_select_fused_mc2(padded, tp):
    assert (
        runner.select_v41_comm_type(1, runner.MoECommType.FUSED_MC2, padded, tp, True) == runner.MoECommType.FUSED_MC2
    )


def test_large_prefill_preserves_bounded_grouped_path():
    assert runner.select_v41_comm_type(1, runner.MoECommType.FUSED_MC2, 8160, 4, False) == runner.MoECommType.ALLGATHER
    assert runner.select_v41_comm_type(0, None, 0, 0, True) == runner.MoECommType.ALLGATHER


@pytest.mark.parametrize("padded,tp,kind", [(260, 4, "FUSED_MC2"), (24, 4, "MC2"), (48, 4, "ALLGATHER")])
def test_full_decode_never_silently_disables_fused_mc2(padded, tp, kind):
    with pytest.raises(ValueError, match="cannot fall back"):
        runner.select_v41_comm_type(1, getattr(runner.MoECommType, kind), padded, tp, True)


@pytest.mark.parametrize("padded,tp", [(0, 4), (24, 0), (25, 4)])
def test_fused_mc2_rejects_nonuniform_padding(padded, tp):
    with pytest.raises(ValueError, match="TP-aligned"):
        runner.select_v41_comm_type(1, runner.MoECommType.FUSED_MC2, padded, tp, False)


@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
@pytest.mark.parametrize("rows", [6, 12])
def test_fused_mc2_passes_unrounded_routes_mask_and_layer_local_capacity(monkeypatch, experts, top_k, rows):
    hidden, intermediate, ep = 64, 32, 16
    states = torch.ones(rows, hidden, dtype=torch.bfloat16)
    route = torch.linspace(0.124578, 0.675518, rows * top_k).reshape(rows, top_k)
    mask = torch.arange(rows).remainder(2).bool()
    w1 = torch.ones(experts // ep, hidden, 2 * intermediate, dtype=torch.int8)
    w2 = torch.ones(experts // ep, intermediate, hidden, dtype=torch.int8)
    s1, s2 = torch.ones(experts // ep, 2 * intermediate), torch.ones(experts // ep, hidden)
    inputs = build_fused_experts_input(
        hidden_states=states,
        topk_weights=route,
        topk_ids=torch.zeros(rows, top_k, dtype=torch.int64),
        w1=w1,
        w2=w2,
        w1_scale=s1,
        w2_scale=s2,
        quant_type=fused.QuantType.W8A8,
        dynamic_eplb=False,
        mc2_mask=mask,
    )
    comm = object.__new__(fused.DeepseekV41FusedMC2Comm)
    comm.moe_config = SimpleNamespace(
        num_experts=experts,
        num_local_experts=experts // ep,
        experts_per_token=top_k,
        ep_size=ep,
        ep_group=SimpleNamespace(rank_in_group=11),
    )
    comm.token_dispatcher = SimpleNamespace(moe_all_to_all_group_name="ep16")
    comm.swiglu_limit = 10.0
    comm.expert_token_nums = torch.zeros(experts // ep, dtype=torch.int32)
    monkeypatch.setattr(fused, "get_ascend_config", lambda: SimpleNamespace(mega_moe_max_tokens=65536))
    operator = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "dispatch_ffn_combine", operator, raising=False)
    logger = Mock()
    monkeypatch.setattr(fused, "logger", logger)
    result = comm.fused_experts(inputs)
    call = operator.call_args.kwargs
    assert call["route_before_down"] is True
    assert call["expert_idx"].dtype == torch.int32
    torch.testing.assert_close(call["expert_idx"].long(), inputs.topk_ids)
    assert call["probs"] is route and call["x_active_mask"] is mask
    assert call["scale1"][0] is s1 and call["scale2"][0] is s2
    assert call["weight1"][0] is w1 and call["weight2"][0] is w2
    assert call["max_output_size"] == rows * ep * top_k
    assert call["out"] is result.routed_out and result.routed_out.dtype == torch.float32
    assert result.expert_tokens is comm.expert_token_nums
    assert logger.info_once.call_args.args[1:] == (experts, top_k, 11)
    monkeypatch.setattr(fused, "get_ascend_config", lambda: SimpleNamespace(mega_moe_max_tokens=rows * ep * top_k - 1))
    with pytest.raises(ValueError, match="capacity"):
        comm.fused_experts(inputs)
    assert operator.call_count == 1


@pytest.mark.parametrize("raise_forward", [False, True])
def test_layer_local_communication_restores_target_context_after_draft(monkeypatch, raise_forward):
    instance = object.__new__(runner.DeepseekV41FusedMoERunner)
    torch.nn.Module.__init__(instance)
    instance.v41_comm = object()
    instance.v41_fused_mc2 = object()
    instance.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=4))
    previous = object()
    context = SimpleNamespace(
        moe_comm_type=runner.MoECommType.FUSED_MC2, moe_comm_method=previous, padded_num_tokens=24
    )
    monkeypatch.setattr(runner, "_EXTRA_CTX", context)
    monkeypatch.setattr(runner, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        runner, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=runner.CUDAGraphMode.FULL)
    )

    def forward(self):
        assert context.moe_comm_method is instance.v41_fused_mc2
        if raise_forward:
            raise RuntimeError("test draft failure")
        return "result"

    monkeypatch.setattr(runner.AscendMoERunner, "forward", forward)
    if raise_forward:
        with pytest.raises(RuntimeError, match="test draft"):
            instance.forward()
    else:
        assert instance.forward() == "result"
    assert context.moe_comm_method is previous
    assert context.moe_comm_type == runner.MoECommType.FUSED_MC2


@pytest.mark.parametrize("has_new_operator", [True, False])
def test_target_and_draft_own_separate_fused_mc2_layouts(monkeypatch, has_new_operator):
    def base_init(self, config):
        torch.nn.Module.__init__(self)
        self.moe_config = config
        self.quant_type = fused.QuantType.W8A8

    monkeypatch.setattr(runner.AscendMoERunner, "__init__", base_init)
    monkeypatch.setattr(runner, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    monkeypatch.setattr(runner, "DeepseekV41AllGatherComm", lambda config: SimpleNamespace(moe_config=config))
    monkeypatch.setattr(runner, "DeepseekV41FusedMC2Comm", lambda config: SimpleNamespace(moe_config=config))
    arguments = [SimpleNamespace(name="route_before_down")] if has_new_operator else []
    operator = SimpleNamespace(default=SimpleNamespace(_schema=SimpleNamespace(arguments=arguments)))
    monkeypatch.setattr(torch.ops._C_ascend, "dispatch_ffn_combine", operator, raising=False)
    target_config = SimpleNamespace(ep_size=16, num_experts=384, num_local_experts=24, experts_per_token=6)
    draft_config = SimpleNamespace(ep_size=16, num_experts=128, num_local_experts=8, experts_per_token=3)
    if not has_new_operator:
        with pytest.raises(RuntimeError, match="Rebuild"):
            runner.DeepseekV41FusedMoERunner(target_config)
        return
    target = runner.DeepseekV41FusedMoERunner(target_config)
    draft = runner.DeepseekV41FusedMoERunner(draft_config)
    assert target.v41_fused_mc2 is not draft.v41_fused_mc2
    assert target.v41_fused_mc2.moe_config is target_config
    assert draft.v41_fused_mc2.moe_config is draft_config
