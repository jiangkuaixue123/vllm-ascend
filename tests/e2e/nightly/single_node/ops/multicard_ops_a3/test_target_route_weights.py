# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run with torchrun --nproc-per-node=16 for target router/MC2 integration."""

import json
import os

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.models.deepseek_v41.core import route_sqrtsoftplus
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op

NUMERICAL_ATOL = 0.001
NUMERICAL_RTOL = 0.005


@torch.inference_mode()
def run_case(rows):
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 16
    generator = torch.Generator().manual_seed(719 + rank)
    states = torch.randn(rows, 5120, generator=generator).bfloat16().npu()
    logits = torch.randn(rows, 384, generator=generator).npu()
    bias = torch.randn(384, generator=generator).npu()
    active = torch.ones(rows, device="npu", dtype=torch.bool)
    packed = []
    for shape in ((24, 5120, 4608), (24, 2304, 5120)):
        weight = torch.randint(-16, 17, shape, generator=generator, dtype=torch.int8).npu()
        packed.append(torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ))
    scale1 = (torch.rand(24, 4608, generator=generator) * 0.03 + 0.005).bfloat16().npu().float()
    scale2 = (torch.rand(24, 5120, generator=generator) * 0.03 + 0.005).bfloat16().npu().float()
    group = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    def execute(fused):
        routes, selected = route_sqrtsoftplus(logits, bias, 6, fuse_weights=fused)
        ids = selected.to(torch.int32)
        output = torch.empty_like(states, dtype=torch.float32)
        counts = torch.zeros(24, device="npu", dtype=torch.int32)
        torch.ops._C_ascend.dispatch_ffn_combine(
            states,
            [packed[0]],
            [packed[1]],
            ids,
            [scale1],
            [scale2],
            [],
            [],
            routes,
            group,
            rows * world * 6,
            output,
            counts,
            x_active_mask=active,
            swiglu_limit=10.0,
            route_before_down=True,
        )
        return output, counts, routes, selected

    execute(False)
    execute(True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = execute(True)
    for case in ("ordinary", "concentrated", "mixed_idle", "all_idle", "last_rank", "zero"):
        logits.copy_(torch.randn(rows, 384, generator=generator))
        active.fill_(True)
        if case == "concentrated":
            logits[:, :6].add_(50)
        elif case == "mixed_idle" and rank % 2 or case == "all_idle":
            active.zero_()
        elif case == "last_rank":
            active.fill_(rank == world - 1)
        elif case == "zero":
            states.zero_()
        expected = tuple(value.clone() for value in execute(False))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual[0], expected[0], rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2])
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
        if case in ("zero", "all_idle"):
            assert torch.count_nonzero(actual[0]).cpu() == 0
        if rank == 0:
            print("TARGET_ROUTE_MC2 " + json.dumps({"rows": rows, "case": case, "status": "PASS"}), flush=True)
    states.copy_(torch.randn(rows, 5120, generator=generator).bfloat16())
    active.fill_(True)
    times = {}
    for fused in (False, True):
        chain = torch.npu.NPUGraph()
        with torch.npu.graph(chain, capture_error_mode="thread_local", auto_dispatch_capture=True):
            execute(fused)
        chain.replay()
        torch.npu.synchronize()
        dist.barrier()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(20):
            chain.replay()
        end.record()
        end.synchronize()
        elapsed = torch.tensor([begin.elapsed_time(end)], device="npu")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        times["candidate" if fused else "parent"] = elapsed.cpu().tolist()[0]
    if rank == 0:
        print("TARGET_ROUTE_MC2_BENCH " + json.dumps({"rows": rows, "calls": 20, "ms": times}), flush=True)


if __name__ == "__main__":
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        enable_custom_op()
        torch.npu.config.allow_internal_format = True
        for row_count in (6, 12):
            run_case(row_count)
    finally:
        dist.destroy_process_group()
