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
    """Use the model's grouped W8A8 arithmetic with independent CPU routing."""
    rank, world = dist.get_rank(), dist.get_world_size()
    local_experts, rows = w1.shape[0], states.shape[0]
    states, ids, routes, active = (_all_gather(value) for value in (states, ids, routes, active))
    groups = [[] for _ in range(local_experts)]
    lower = rank * local_experts
    for token, (token_ids, enabled) in enumerate(zip(ids.cpu().tolist(), active.cpu().tolist())):
        if enabled:
            for slot, expert in enumerate(token_ids):
                if lower <= expert < lower + local_experts:
                    groups[expert - lower].append((token, slot))
    counts = torch.tensor([len(group) for group in groups], dtype=torch.int64, device=states.device)
    pairs = [pair for group in groups for pair in group]
    expected = torch.zeros_like(states, dtype=torch.float32)

    def grouped(values, weights, scale, token_scale):
        return torch_npu.npu_grouped_matmul(
            x=[values],
            weight=[weights],
            scale=[scale],
            per_token_scale=[token_scale.reshape(-1)],
            group_list=counts,
            group_list_type=1,
            group_type=0,
            split_item=2,
            output_dtype=torch.bfloat16,
        )[0]

    if pairs:
        token_index = torch.tensor([pair[0] for pair in pairs], device=states.device)
        slot_index = torch.tensor([pair[1] for pair in pairs], device=states.device)
        q1, a1 = torch_npu.npu_dynamic_quant(states, dst_type=torch.int8)
        gate_up = grouped(q1[token_index], w1, s1, a1[token_index])
        gate, up = gate_up.float().chunk(2, dim=-1)
        factors = routes[token_index, slot_index, None]
        activation = (torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit) * factors).bfloat16()
        q2, a2 = torch_npu.npu_dynamic_quant(activation, dst_type=torch.int8)
        result = grouped(q2, w2, s2, a2).float()
        expected.index_add_(0, token_index, torch.where(factors != 0, result, 0.0))
    dist.all_reduce(expected)
    assert expected.shape[0] == rows * world
    return expected[rank * rows : (rank + 1) * rows], counts.to(torch.int32)


@torch.inference_mode()
def _case(local_experts, top_k, rows, hidden, intermediate, *, concentrated=False, measure_replays=0):
    rank, world = dist.get_rank(), dist.get_world_size()
    generator = torch.Generator().manual_seed(191 + rank)
    states = torch.randn(rows, hidden, generator=generator).bfloat16().npu()
    original_states = states.clone() if measure_replays else None
    # Distinct routes cover both local and remote ownership on every rank.
    route_ids = torch.argsort(torch.randn(rows, world * local_experts, generator=generator), -1)[:, :top_k]
    if concentrated:
        # At EP16, 6/12 local rows send 96/192 rows to each selected expert.
        # This crosses the decode kernel's M=64 tile boundary, including a tail.
        route_ids = torch.arange(top_k).expand(rows, -1).contiguous()
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
    if measure_replays:
        # Restore nonzero inputs after the replay correctness cases. All ranks
        # participate, and the slowest rank defines this collective-chain time.
        states.copy_(original_states)
        ids.copy_(route_ids.to(device="npu", dtype=torch.int32))
        active.fill_(True)
        graph.replay()
        graph.replay()
        torch.npu.synchronize()
        dist.barrier()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(measure_replays):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed = torch.tensor([start.elapsed_time(end)], dtype=torch.float32, device=states.device)
        rank_times = _all_gather(elapsed).cpu().tolist()
        if rank == 0:
            print(
                "FUSED_MC2_ROUTE_BENCH "
                + json.dumps(
                    {
                        "experts": local_experts * world,
                        "top_k": top_k,
                        "rows": rows,
                        "concentrated": concentrated,
                        "replays": measure_replays,
                        "slowest_rank_ms": max(rank_times),
                        "rank_ms": rank_times,
                    }
                ),
                flush=True,
            )
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
                    "concentrated": concentrated,
                    "eager": True,
                    "replays": 3,
                    "rtol": NUMERICAL_RTOL,
                    "atol": NUMERICAL_ATOL,
                }
            ),
            flush=True,
        )


def _run(full_shapes=False, measure_replays=0):
    enable_custom_op()
    torch.npu.config.allow_internal_format = True
    hidden, intermediate = (5120, 2304) if full_shapes else (256, 128)
    for experts, top_k in ((24, 6), (8, 3)):
        for rows in (6, 12):
            _case(experts, top_k, rows, hidden, intermediate, measure_replays=measure_replays)
    for rows in (6, 12):
        _case(24, 6, rows, hidden, intermediate, concentrated=True, measure_replays=measure_replays)


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
    parser.add_argument("--measure-replays", type=int, default=0)
    args = parser.parse_args()
    if args.measure_replays < 0:
        parser.error("--measure-replays must be nonnegative")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        _run(args.full_shapes, args.measure_replays)
    finally:
        dist.destroy_process_group()
