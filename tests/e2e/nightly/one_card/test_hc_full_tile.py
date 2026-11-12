# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record parent HC graphs, then compare an isolated candidate operator package."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_hc_pre_delayed import make_inputs
from vllm_ascend.utils import enable_custom_op

CALLS = 40
SAMPLES = 7
CASES = 8


def exact(actual, expected):
    word = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    actual = actual.cpu().view(word)
    expected = expected.view(word)
    assert torch.equal(actual, expected), f"{int((actual != expected).sum())} differing words"


@torch.inference_mode()
def run(reference, record):
    """Refresh every graph operand; compare all six outputs before timing the consumer."""
    saved = {} if record else torch.load(reference, weights_only=True)
    native = torch.ops._C_ascend.npu_hc_pre_delayed
    for rows in (3, 6, 12):
        args = list(make_inputs(rows))
        if record:
            saved[rows] = {"parameters": [value.cpu() for value in args[1:4]], "cases": []}
        else:
            for actual, expected in zip(args[1:4], saved[rows]["parameters"]):
                actual.copy_(expected)
        native(*args, 1e-20)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = native(*args, 1e-20)
        for case in range(CASES):
            if record:
                args[0].normal_().mul_(2.0 ** (case - 4))
                args[4].uniform_(-2, 2)
                if case in (0, 1):
                    args[0].fill_(0.0 if case == 0 else -0.0)
                    args[4].fill_(1)
                elif case == 2:
                    args[0][:, 1].copy_(-args[0][:, 0])
                    args[0][:, 3].copy_(-args[0][:, 2])
                    args[4].fill_(1)
                elif case == 3:
                    args[0][:, :, ::97] = float("inf")
                    args[0][:, :, 1::97] = float("nan")
            else:
                args[0].copy_(saved[rows]["cases"][case]["x"])
                args[4].copy_(saved[rows]["cases"][case]["previous"])
            graph.replay()
            torch.npu.synchronize()
            if record:
                saved[rows]["cases"].append(
                    {
                        "x": args[0].cpu(),
                        "previous": args[4].cpu(),
                        "outputs": [value.cpu() for value in outputs],
                    }
                )
            else:
                for actual, expected in zip(outputs, saved[rows]["cases"][case]["outputs"]):
                    exact(actual, expected)
        print(
            f"HC_FULL_TILE_ACCURACY rows={rows} cases={CASES} status={'PARENT_RECORDED' if record else 'PASS'}",
            flush=True,
        )
        weight = torch.ones(1280, 5120, device="npu", dtype=torch.bfloat16) * 0.01
        timed = torch.npu.NPUGraph()
        with torch.npu.graph(timed):
            for _ in range(CALLS):
                values = native(*args, 1e-20)
                torch.nn.functional.linear(values[0], weight)
        samples = []
        for _ in range(SAMPLES):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            timed.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        print(
            "HC_FULL_TILE_COMPONENT "
            + json.dumps(
                {
                    "rows": rows,
                    "calls": CALLS,
                    "median_ms": statistics.median(samples),
                    "mode": "parent" if record else "candidate",
                    "scope": "HC plus BF16 projection",
                }
            ),
            flush=True,
        )
    if record:
        torch.save(saved, reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    options = parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    assert enable_custom_op()
    run(options.reference, options.record)
