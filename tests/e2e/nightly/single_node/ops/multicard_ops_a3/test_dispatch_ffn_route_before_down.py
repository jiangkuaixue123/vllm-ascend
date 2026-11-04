# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed numerical and replay regression for the opt-in W8A8 epilogue.

Run directly with torchrun --nproc-per-node=16 <this file> for the campaign
EP16 gate, or with pytest for a smaller two-device CI regression.
"""

import argparse
import json
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op

# Frozen before the first NPU execution, matching the campaign L0 gate.
NUMERICAL_ATOL = 0.001
NUMERICAL_RTOL = 0.005


def _all_gather(tensor):
    tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors)


def _reference(states, ids, routes, active, w1, w2, s1, s2, limit):
    rank, world = dist.get_rank(), dist.get_world_size()
    local_experts, rows = w1.shape[0], states.shape[0]
    states, ids, routes, active = (_all_gather(value) for value in (states, ids, routes, active))
    q1, a1 = torch_npu.npu_dynamic_quant(states, dst_type=torch.int8)
    expected = torch.zeros_like(states, dtype=torch.float32)
    counts = []
    for expert in range(local_experts):
        gate_up = torch_npu.npu_quant_matmul(q1, w1[expert], s1[expert], pertoken_scale=a1, output_dtype=torch.bfloat16)
        gate, up = gate_up.float().chunk(2, dim=-1)
        selected = ids == (rank * local_experts + expert)
        counts.append((selected & active[:, None]).sum().to(torch.int32))
        factors = torch.where(selected, routes, 0.0).sum(-1, keepdim=True)
        activation = (torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit) * factors).bfloat16()
        q2, a2 = torch_npu.npu_dynamic_quant(activation, dst_type=torch.int8)
        result = torch_npu.npu_quant_matmul(q2, w2[expert], s2[expert], pertoken_scale=a2, output_dtype=torch.bfloat16)
        expected.add_(torch.where(selected.any(-1, keepdim=True) & active[:, None], result.float(), 0.0))
    dist.all_reduce(expected)
    assert expected.shape[0] == rows * world
    return expected[rank * rows : (rank + 1) * rows], torch.stack(counts)


@torch.inference_mode()
def _case(local_experts, top_k, rows, hidden, intermediate):
    rank, world = dist.get_rank(), dist.get_world_size()
    generator = torch.Generator().manual_seed(191 + rank)
    states = torch.randn(rows, hidden, generator=generator).bfloat16().npu()
    # Distinct routes cover both local and remote ownership on every rank.
    route_ids = torch.argsort(torch.randn(rows, world * local_experts, generator=generator), -1)[:, :top_k]
    ids = route_ids.to(torch.int32).contiguous().npu()
    routes = (torch.rand(rows, top_k, generator=generator) * 0.9 + 0.01).npu()
    routes[0, 0] = 0
    active = torch.ones(rows, dtype=torch.bool, device="npu")
    active[-1] = False
    w1 = torch.randint(-16, 17, (local_experts, hidden, 2 * intermediate), generator=generator, dtype=torch.int8).npu()
    w2 = torch.randint(-16, 17, (local_experts, intermediate, hidden), generator=generator, dtype=torch.int8).npu()
    s1 = (torch.rand(local_experts, 2 * intermediate, generator=generator) * 0.03 + 0.005).bfloat16().npu()
    s2 = (torch.rand(local_experts, hidden, generator=generator) * 0.03 + 0.005).bfloat16().npu()
    packed1 = torch_npu.npu_format_cast(w1, ACL_FORMAT_FRACTAL_NZ)
    packed2 = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
    scale1, scale2 = s1.float(), s2.float()
    output = torch.empty_like(states, dtype=torch.float32)
    counts = torch.zeros(local_experts, dtype=torch.int32, device="npu")
    group = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    def execute():
        torch.ops._C_ascend.dispatch_ffn_combine(
            states,
            [packed1],
            [packed2],
            ids,
            [scale1],
            [scale2],
            [],
            [],
            routes,
            group,
            rows * world * top_k,
            output,
            counts,
            x_active_mask=active,
            swiglu_limit=10.0,
            route_before_down=True,
        )

    expected, expected_counts = _reference(states, ids, routes, active, w1, w2, s1, s2, 10.0)
    execute()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
    torch.testing.assert_close(counts.cpu(), expected_counts.cpu(), rtol=0, atol=0)
    assert torch.count_nonzero(output[~active].cpu()) == 0
    ids.copy_(route_ids.to(device="npu", dtype=torch.int32))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        execute()
    for iteration in range(3):
        ids.copy_(route_ids.roll(iteration + 1, 0).to(device="npu", dtype=torch.int32))
        routes.mul_(0.73)
        if iteration == 1 and rank % 2:
            active.zero_()  # Idle DP-equivalent participant still enters every collective.
        if iteration == 2:
            states.zero_()
            active.fill_(True)
        expected, expected_counts = _reference(states, ids, routes, active, w1, w2, s1, s2, 10.0)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
        torch.testing.assert_close(counts.cpu(), expected_counts.cpu(), rtol=0, atol=0)
        assert torch.count_nonzero(output[~active].cpu()) == 0
        if iteration == 2:
            assert torch.count_nonzero(output.cpu()) == 0
    dist.barrier()
    if rank == 0:
        print(
            "FUSED_MC2_ROUTE_L0 "
            + json.dumps(
                {
                    "status": "PASS",
                    "world_size": world,
                    "experts": local_experts * world,
                    "top_k": top_k,
                    "rows": rows,
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "eager": True,
                    "replays": 3,
                    "rtol": NUMERICAL_RTOL,
                    "atol": NUMERICAL_ATOL,
                }
            ),
            flush=True,
        )


def _run(full_shapes=False):
    enable_custom_op()
    torch.npu.config.allow_internal_format = True
    hidden, intermediate = (5120, 2304) if full_shapes else (256, 128)
    for experts, top_k in ((24, 6), (8, 3)):
        for rows in (6, 12):
            _case(experts, top_k, rows, hidden, intermediate)


def _worker(rank, port):
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        _run()
    finally:
        dist.destroy_process_group()


def test_route_before_down_eager_and_replay():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-shapes", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        _run(args.full_shapes)
    finally:
        dist.destroy_process_group()
