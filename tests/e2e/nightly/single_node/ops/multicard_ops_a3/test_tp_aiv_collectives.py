# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 AIV rank routing and reductions in eager execution and graph replay."""

import json
import os
import statistics
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.utils import create_hccl_pg_options

TP = 4
HIDDEN = 5120
TIMED_REPLAYS = 20
TIMING_SAMPLES = 5


def tp_aiv_options():
    config = SimpleNamespace(
        additional_config={"dsv41_tp_aiv": True},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=TP,
            data_parallel_size=TP,
            data_parallel_size_local=TP,
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            enable_dbo=False,
            enable_eplb=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("vllm.config.get_current_vllm_config", lambda: config)
        patch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
        return create_hccl_pg_options("tp")


def payload(rows, rank, variation, dtype):
    values = torch.arange(TP * rows * HIDDEN).reshape(TP * rows, HIDDEN)
    values = ((values + variation * 7) % 13) - 6 + rank * 2
    return (values if variation != 1 else torch.zeros_like(values)).to(dtype)


def collectives(value, group):
    rows = value.shape[0] // TP
    gathered = value.new_empty(TP * rows, HIDDEN)
    exchanged = torch.empty_like(value)
    reduced = value.new_empty(rows, HIDDEN)
    all_reduced = value[:rows].clone()
    dist.all_gather_into_tensor(gathered, value[:rows], group=group)
    dist.all_to_all_single(exchanged, value, group=group)
    dist.reduce_scatter_tensor(reduced, value, group=group)
    dist.all_reduce(all_reduced, group=group)
    return gathered, exchanged, reduced, all_reduced


def expected_outputs(rows, rank, variation, dtype):
    inputs = [payload(rows, source, variation, dtype) for source in range(TP)]
    total = torch.stack(inputs).sum(0).to(dtype)
    return (
        torch.cat([value[:rows] for value in inputs]),
        torch.cat([value[rank * rows : (rank + 1) * rows] for value in inputs]),
        total[rank * rows : (rank + 1) * rows],
        total[:rows],
    )


def exact(outputs, expected, context=""):
    for operation, actual, reference in zip(
        ("gather", "all-to-all", "reduce-scatter", "all-reduce"), outputs, expected
    ):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0, msg=f"{context} operation={operation}")


@torch.inference_mode()
def run_case(rows, dtype, aiv_group):
    rank = dist.get_rank()
    value = payload(rows, rank, 0, dtype).npu()
    graphs, outputs = [], []
    for group in (None, aiv_group):
        exact(collectives(value, group), expected_outputs(rows, rank, 0, dtype))
        torch.npu.synchronize()
        dist.barrier()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = collectives(value, group)
        graphs.append(graph)
        outputs.append(result)
    for variation in (0, 1, 2):
        value.copy_(payload(rows, rank, variation, dtype))
        expected = expected_outputs(rows, rank, variation, dtype)
        for index, (graph, result) in enumerate(zip(graphs, outputs)):
            graph.replay()
            torch.npu.synchronize()
            exact(result, expected, f"rows={rows} dtype={dtype} variation={variation} candidate={bool(index)}")
    samples = [[], []]
    for _ in range(TIMING_SAMPLES):
        for index, graph in enumerate(graphs):
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            for _ in range(TIMED_REPLAYS):
                graph.replay()
            torch.npu.synchronize()
            elapsed = torch.tensor((time.perf_counter_ns() - start) / 1e6, device="npu")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples[index].append(elapsed.item())
    if rank == 0:
        print(
            "TP_AIV_COMPONENT "
            + json.dumps(
                {
                    "status": "PASS",
                    "rows": rows,
                    "dtype": str(dtype),
                    "replays": TIMED_REPLAYS,
                    "parent_ms": statistics.median(samples[0]),
                    "candidate_ms": statistics.median(samples[1]),
                    "scope": "all-gather/all-to-all/reduce-scatter/all-reduce; exact CPU oracle and graph updates",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    parent = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    parent.hccl_config = {"hccl_op_expansion_mode": 2}
    dist.init_process_group("hccl", pg_options=parent)
    try:
        assert dist.get_world_size() == TP
        candidate = dist.new_group(list(range(TP)), backend="hccl", pg_options=tp_aiv_options())
        try:
            for local_rows in (6, 12):
                for data_type in (torch.bfloat16, torch.float32):
                    run_case(local_rows, data_type, candidate)
        finally:
            dist.destroy_process_group(candidate)
    finally:
        dist.destroy_process_group()
