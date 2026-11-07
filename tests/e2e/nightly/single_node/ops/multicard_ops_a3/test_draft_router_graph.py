# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native EP16 complete routing plus expert execution, eager versus local graph."""

import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from tests.e2e.nightly.one_card.test_draft_router_graph import make_gate
from tests.e2e.nightly.single_node.ops.multicard_ops_a3.test_dispatch_ffn_route_before_down import (
    NUMERICAL_ATOL,
    NUMERICAL_RTOL,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op

TP_SIZE = 4
EP_SIZE = 16
LOCAL_EXPERTS = 8
TOP_K = 3
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 2304
TIMED_CALLS = 6
TIMING_SAMPLES = 5


@torch.inference_mode()
def run_case(query_tokens):
    """Compare identical routing inputs, then time the complete distributed calls."""
    rank = dist.get_rank()
    rows = (query_tokens + TP_SIZE - 1) // TP_SIZE
    generator = torch.Generator().manual_seed(217 + rank)
    states = torch.randn(rows, HIDDEN_SIZE, generator=generator).bfloat16().npu()
    w1 = torch.randint(
        -16, 17, (LOCAL_EXPERTS, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), generator=generator, dtype=torch.int8
    ).npu()
    w2 = torch.randint(
        -16, 17, (LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE), generator=generator, dtype=torch.int8
    ).npu()
    scale1 = (torch.rand(LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, generator=generator) * 0.03 + 0.005).npu()
    scale2 = (torch.rand(LOCAL_EXPERTS, HIDDEN_SIZE, generator=generator) * 0.03 + 0.005).npu()
    packed1, packed2 = (torch_npu.npu_format_cast(value, ACL_FORMAT_FRACTAL_NZ) for value in (w1, w2))
    output = torch.empty_like(states, dtype=torch.float32)
    counts = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device="npu")
    valid = torch.arange(rows * TP_SIZE, device="npu")[rank % TP_SIZE * rows : (rank % TP_SIZE + 1) * rows]
    active = valid < query_tokens
    group = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    gate = make_gate(f"test.draft.{query_tokens}")
    graph = gate._draft_router_graph

    def execute(use_graph, fixed_logits=None):
        gate._draft_router_graph = graph if use_graph else None
        logits = gate.routing_logits(states, decode=False, image_mask=None) if fixed_logits is None else fixed_logits
        routes, indices = gate.select_experts(states, logits, TOP_K, True)
        torch.ops._C_ascend.dispatch_ffn_combine(
            states,
            [packed1],
            [packed2],
            indices.to(torch.int32),
            [scale1],
            [scale2],
            [],
            [],
            routes,
            group,
            rows * EP_SIZE * TOP_K,
            output,
            counts,
            x_active_mask=active,
            swiglu_limit=10.0,
            route_before_down=True,
        )
        return logits, routes, indices

    for active_dp in (4, 1, 0, 4):
        active.copy_((valid < query_tokens) & (rank // TP_SIZE < active_dp))
        states.mul_(0.9)
        gate.bias.add_(torch.linspace(-0.01, 0.01, 128, device="npu"))
        # Compare the changed stage with identical inputs. Independent FP32
        # gate GEMMs can differ by a few ulps before routing is even called.
        fixed_logits = gate.routing_logits(states, decode=False, image_mask=None)
        repeated_logits = gate.routing_logits(states, decode=False, image_mask=None)
        if active_dp == 4 and not graph.entries:
            print(
                "DSPARK_ROUTER_PARENT_GEMM_REPEAT "
                + json.dumps(
                    {
                        "rank": rank,
                        "different_elements": int(torch.count_nonzero(fixed_logits != repeated_logits).cpu()),
                        "maximum_absolute_difference": float((fixed_logits - repeated_logits).abs().max().cpu()),
                    }
                ),
                flush=True,
            )
        parent_routing = [value.cpu().clone() for value in execute(False, fixed_logits)]
        expected, expected_counts = output.cpu(), counts.cpu()
        output.fill_(float("nan"))
        candidate_routing = [value.cpu().clone() for value in execute(True, fixed_logits)]
        for name, actual, reference in zip(("logits", "weights", "indices"), candidate_routing, parent_routing):
            if not torch.equal(actual, reference):
                print(
                    "DSPARK_ROUTER_GRAPH_ROUTE_DIFFERENCE "
                    + json.dumps(
                        {
                            "rank": rank,
                            "active_dp": active_dp,
                            "part": name,
                            "different_elements": int(torch.count_nonzero(actual != reference)),
                            "maximum_absolute_difference": float((actual - reference).abs().max()),
                        }
                    ),
                    flush=True,
                )
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(output.cpu(), expected, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
        torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
        assert torch.count_nonzero(output[~active].cpu()) == 0
        assert rows in graph.entries and len(graph.entries) == 1

    timings = {}
    for use_graph in (False, True):
        execute(use_graph)
        torch.npu.synchronize()
        samples = []
        for _ in range(TIMING_SAMPLES):
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(TIMED_CALLS):
                execute(use_graph)
            submitted = time.perf_counter()
            torch.npu.synchronize()
            ready = time.perf_counter()
            elapsed = torch.tensor(
                [(submitted - start) * 1000, (ready - start) * 1000], dtype=torch.float32, device="npu"
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.cpu().tolist())
        timings["candidate" if use_graph else "parent"] = [statistics.median(x[i] for x in samples) for i in range(2)]
    dist.barrier()
    if rank == 0:
        print(
            "DSPARK_ROUTER_GRAPH_COMPONENT "
            + json.dumps(
                {
                    "status": "PASS",
                    "world_size": EP_SIZE,
                    "query_tokens": query_tokens,
                    "local_rows": rows,
                    "native_cases": 4,
                    "calls": TIMED_CALLS,
                    "metrics": ["host_ms", "ready_ms"],
                    "timings": timings,
                    "scope": "gate matrix multiplication, selection, native EP16 fused experts; shared expert excluded",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == EP_SIZE
        enable_custom_op()
        torch.npu.config.allow_internal_format = True
        for query_tokens in (5, 20):
            run_case(query_tokens)
    finally:
        dist.destroy_process_group()
