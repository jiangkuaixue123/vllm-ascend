# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.models.deepseek_v41.fused_experts import dispatched_scales
from vllm_ascend.ops.triton.deepseek_v41_prefill_experts import prefill_dispatched_scales


@pytest.mark.parametrize(
    "rows,placement",
    [
        (129, "balanced"),
        (2048, "balanced"),
        (4096, "balanced"),
        (4097, "local"),
        (4097, "remote"),
        (264, "local"),
        (264, "remote"),
        (129, "nomap"),
        (513, "rank15"),
        (129, "repeated"),
    ],
)
def test_native_prefill_route_ownership_and_graph_refresh(rows, placement):
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    k, experts, local = 6, 384, 24
    if placement == "nomap":
        experts = local
    ids = ((torch.arange(rows)[:, None] * k + torch.arange(k)) % experts).int().npu()
    if placement == "local":
        ids.remainder_(local)
    elif placement == "remote":
        ids.remainder_(experts - local).add_(local)
    elif placement == "repeated":
        ids[:, 1].copy_(ids[:, 0])
    states = torch.zeros((rows, 5120), dtype=torch.int8, device="npu")
    weights = torch.rand((rows, k), device="npu")
    weights[::3, 0] = -0.0
    expert_map = (
        None
        if placement == "nomap"
        else torch.tensor([*range(local), *([-1] * (experts - local))], dtype=torch.int32, device="npu")
    )
    first = experts - local if placement == "rank15" else 0
    if first:
        expert_map.fill_(-1)
        expert_map[first:].copy_(torch.arange(local, dtype=torch.int32, device="npu"))

    def route():
        expanded, row_ids, _, _ = torch_npu.npu_moe_init_routing_v2(
            states,
            ids,
            active_num=rows * k,
            expert_num=experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first, first + local],
            quant_mode=-1,
        )
        return expanded.shape[0], row_ids

    def execute():
        num_rows, row_ids = route()
        return prefill_dispatched_scales(row_ids, weights, ids, expert_map, num_rows)

    def check(actual):
        num_rows, row_ids = route()
        owned = torch.ones_like(ids, dtype=torch.bool) if expert_map is None else expert_map[ids.long()] >= 0
        local_rows = row_ids.abs().flatten()[owned.flatten()].cpu()
        assert local_rows.numel() == local_rows.unique().numel()
        expected = dispatched_scales(row_ids, weights, owned, num_rows)
        torch.testing.assert_close(actual[0].cpu().view(torch.int32), expected.cpu().view(torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(actual[1].cpu(), owned.float().cpu(), rtol=0, atol=0)

    with torch.inference_mode():
        check(execute())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = execute()
        for step in range(8):
            weights.mul_(0.97)
            weights[step % rows, step % k] = 0.0
            ids.copy_(torch.roll(ids, 1, dims=1))
            graph.replay()
            check(actual)
