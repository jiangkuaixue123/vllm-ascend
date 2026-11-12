# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualified arithmetic boundaries and paired dependent HC chains on A3."""

import statistics

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.one_card import test_hc_post_fused as base
from vllm_ascend.ops.triton import hc_post as kernels

CALLS = 80
SAMPLES = 7
PARENT_BLOCK = 1024


def parent(sub, residual, post, comb, mask):
    output = torch.empty_like(residual)
    kernels._hc_post4_kernel[(sub.shape[0] * base.WIDTH, base.HIDDEN // PARENT_BLOCK)](
        sub,
        residual,
        post,
        comb,
        output,
        mask,
        MASK_SUBLAYER=True,
        HIDDEN=base.HIDDEN,
        WIDTH=base.WIDTH,
        BLOCK=PARENT_BLOCK,
        enable_fp_fusion=False,
    )
    return output


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_reused_residual_keeps_qualified_arithmetic_boundaries(monkeypatch, rows):
    torch.npu.set_device(0)
    mask = torch.ones(rows, device="npu", dtype=torch.bool)
    monkeypatch.setattr(base, "hc_post_fused", lambda *args: kernels.hc_post4(*args, mask))
    base.test_reference_order_and_finite_boundaries(rows)


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_changing_inputs_masks_and_paired_complete_chain_time(rows):
    torch.npu.set_device(0)
    sub, residual, post, comb = base.inputs(rows)
    comb.fill_(0.25)
    mask = torch.ones(rows, device="npu", dtype=torch.bool)
    graphs, outputs = [], []
    for function in (parent, kernels.hc_post4):
        function(sub, residual, post, comb, mask)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = residual
            for _ in range(CALLS):
                result = function(sub, result, post, comb, mask)
        graphs.append(graph)
        outputs.append(result)
    for case in range(8):
        sub.normal_()
        residual.normal_()
        post.uniform_(-1, 1)
        comb.uniform_(0.20, 0.30)
        mask.copy_((torch.arange(rows, device="npu") + case) % 3 != 0)
        if case == 0:
            mask.zero_()
            sub[:, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0], device="npu")
        for graph in graphs:
            graph.replay()
            torch.npu.synchronize()
        base.exact_output(outputs[1], outputs[0])
    mask.fill_(True)
    samples = [[], []]
    for repeat in range(SAMPLES):
        for index in (repeat % 2, 1 - repeat % 2):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
    print(
        {
            "rows": rows,
            "calls": CALLS,
            "parent_ms": statistics.median(samples[0]),
            "candidate_ms": statistics.median(samples[1]),
        }
    )
