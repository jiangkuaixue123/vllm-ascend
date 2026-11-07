# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torchrun --nproc-per-node=16 for the TP4/DP4 draft mask gate."""

import json
import os
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from tests.e2e.nightly.single_node.ops.multicard_ops_a3.test_dispatch_ffn_route_before_down import (
    NUMERICAL_ATOL,
    NUMERICAL_RTOL,
    _reference,
)
from vllm_ascend.ops.fused_moe import prepare_finalize
from vllm_ascend.spec_decode.dspark_proposer import AscendDeepseekV41DSparkProposer
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
    rank = dist.get_rank()
    rows = (query_tokens + TP_SIZE - 1) // TP_SIZE
    generator = torch.Generator().manual_seed(217 + rank)
    states = torch.randn(rows, HIDDEN_SIZE, generator=generator).bfloat16().npu()
    ids = torch.argsort(torch.randn(rows, EP_SIZE * LOCAL_EXPERTS, generator=generator), -1)[:, :TOP_K]
    ids = ids.to(device="npu", dtype=torch.int32).contiguous()
    source_ids = ids.clone()
    routes = (torch.rand(rows, TOP_K, generator=generator) * 0.9 + 0.01).npu()
    source_routes = routes.clone()
    source_states = states.cpu()
    w1 = torch.randint(
        -16, 17, (LOCAL_EXPERTS, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), generator=generator, dtype=torch.int8
    ).npu()
    w2 = torch.randint(
        -16, 17, (LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE), generator=generator, dtype=torch.int8
    ).npu()
    s1 = (torch.rand(LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, generator=generator) * 0.03 + 0.005).bfloat16().npu()
    s2 = (torch.rand(LOCAL_EXPERTS, HIDDEN_SIZE, generator=generator) * 0.03 + 0.005).bfloat16().npu()
    packed1, packed2 = (torch_npu.npu_format_cast(value, ACL_FORMAT_FRACTAL_NZ) for value in (w1, w2))
    scale1, scale2 = s1.float(), s2.float()
    output = torch.empty_like(states, dtype=torch.float32)
    counts = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device="npu")
    active = torch.empty(rows, dtype=torch.bool, device="npu")
    group = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    proposer = object.__new__(AscendDeepseekV41DSparkProposer)
    preparer = object.__new__(prepare_finalize.PrepareAndFinalizeWithMC2)
    preparer.tp_size, preparer.tp_rank = TP_SIZE, rank % TP_SIZE

    def set_active(active_dp):
        # Use the production hook, then the real SP prepare path. Only idle
        # DP masks differ; communication slots remain identical on all ranks.
        actual = query_tokens if rank // TP_SIZE < active_dp else proposer._dummy_num_actual_tokens(query_tokens, False)
        mask = torch.arange(rows * TP_SIZE, device="npu") < actual
        context = SimpleNamespace(mc2_mask=mask, padded_num_tokens=rows * TP_SIZE)
        with patch.object(prepare_finalize, "_EXTRA_CTX", context):
            prepared = preparer.prepare(states, routes, replace_allreduce=True)
        assert prepared.hidden_states is states and prepared.router_logits is routes
        assert prepared.mc2_mask.shape == active.shape
        active.copy_(prepared.mc2_mask)
        # The router produces fresh IDs and weights on every real forward.
        # MC2 overwrites inactive route IDs, so refresh both for each batch.
        ids.copy_(source_ids)
        routes.copy_(source_routes)
        expected_mask = torch.arange(rows * TP_SIZE)[rank % TP_SIZE * rows : (rank % TP_SIZE + 1) * rows] < actual
        torch.testing.assert_close(active.cpu(), expected_mask, rtol=0, atol=0)

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
            rows * EP_SIZE * TOP_K,
            output,
            counts,
            x_active_mask=active,
            swiglu_limit=10.0,
            route_before_down=True,
        )

    def verify():
        torch.testing.assert_close(states.cpu(), source_states, rtol=0, atol=0)
        torch.testing.assert_close(ids[active].cpu(), source_ids[active].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(routes[active].cpu(), source_routes[active].cpu(), rtol=0, atol=0)
        expected, expected_counts = _reference(states, source_ids, source_routes, active, w1, w2, s1, s2, 10.0)
        torch.testing.assert_close(counts.cpu(), expected_counts.cpu(), rtol=0, atol=0)
        assert torch.count_nonzero(output[~active].cpu()) == 0
        return expected

    set_active(4)
    execute()
    torch.npu.synchronize()
    expected = verify()
    # This change only alters token eligibility, not expert arithmetic. Use
    # the unchanged native parent as the numerical reference below. Keep
    # independent-formula discrepancies visible: small-row GMM tiling can
    # differ from the reference's individual per-expert GEMMs even before
    # any idle masks are applied. Do not relax the comparison tolerance.
    formula_mismatches = (~torch.isclose(output, expected, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)).sum().int()
    dist.all_reduce(formula_mismatches)
    formula_mismatches = int(formula_mismatches.cpu())
    parent = output.clone()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        execute()
    for active_dp in (1, 0, 4):
        set_active(active_dp)
        saved_mask = active.cpu().clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(active.cpu(), saved_mask, rtol=0, atol=0)
        if rank == 0:
            print("DSPARK_IDLE_MASK_TRANSITION " + str(active_dp), flush=True)
        if rank in (0, 4):
            print(
                "DSPARK_IDLE_MASK_INPUT "
                + json.dumps(
                    {
                        "rank": rank,
                        "active_dp": active_dp,
                        "changed_route_elements": int(torch.count_nonzero(routes != source_routes).cpu()),
                        "changed_route_ids": int(torch.count_nonzero(ids != source_ids).cpu()),
                    }
                ),
                flush=True,
            )
        verify()
        if rank // TP_SIZE < active_dp:
            torch.testing.assert_close(output.cpu(), parent.cpu(), rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)

    timings = {}
    for active_dp in (4, 1):
        set_active(active_dp)
        execute()
        torch.npu.synchronize()
        samples = []
        for _ in range(TIMING_SAMPLES):
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(TIMED_CALLS):
                execute()
            torch.npu.synchronize()
            elapsed = torch.tensor([(time.perf_counter() - start) * 1000], dtype=torch.float32, device="npu")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.cpu().item())
        timings[str(active_dp)] = statistics.median(samples)
    dist.barrier()
    if rank == 0:
        print(
            "DSPARK_IDLE_MASK_COMPONENT "
            + json.dumps(
                {
                    "status": "PASS",
                    "world_size": EP_SIZE,
                    "query_tokens": query_tokens,
                    "local_expert_rows": rows,
                    "active_dp_sequence": [4, 1, 0, 4],
                    "native_cases": 4,
                    "numerical_reference": "unchanged native parent active-token outputs",
                    "parent_formula_mismatches": formula_mismatches,
                    "calls": TIMED_CALLS,
                    "parent_ready_ms": timings["4"],
                    "candidate_ready_ms": timings["1"],
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
