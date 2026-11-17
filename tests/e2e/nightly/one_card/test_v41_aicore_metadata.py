# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original selection oracles, consumer layouts, and changing graph inputs."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.attention import select_candidate_blocks
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.fixture(autouse=True)
def require_target_device():
    assert get_ascend_device_type() == AscendDeviceType.A3


def block_reference(logits, lengths):
    scores = logits.unflatten(-1, (-1, 8)).amax(-1)
    columns = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int32)
    scores = scores.masked_fill(columns == (lengths - 1) // 8, torch.inf)
    top = scores.topk(2048, dim=-1)
    cutoff = top.values[..., -1:]
    tied = torch.where(scores == cutoff, columns, scores.shape[-1])
    first = tied.topk(2048, dim=-1, largest=False).values
    equal = top.values == cutoff
    ordinal = equal.long().cumsum(-1) - 1
    ids = torch.where(equal, first.gather(-1, ordinal.clamp_min(0)).long(), top.indices)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, scores.gather(-1, ids) > -torch.inf)


@pytest.mark.parametrize("rows,width", [(3, 76800), (6, 76800), (12, 76800), (3, 153600), (6, 153600), (12, 153600)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_candidate_blocks_keep_native_selection(rows, width, dtype):
    torch.npu.set_device(0)
    torch.manual_seed(3112)
    lengths = torch.arange(rows, device="npu").remainder(3).mul(width // 2).unsqueeze(-1)
    for mode in ("random", "tied", "zero", "masked", "infinite", "few_nan", "many_nan"):
        source = torch.randn(rows, width, device="npu", dtype=dtype)
        if mode == "tied":
            source.round_()
        elif mode == "zero":
            source.zero_()
            source[:, ::2] = -0.0
        elif mode == "masked":
            source.fill_(-torch.inf)
        elif mode == "infinite":
            source[:, ::3] = torch.inf
            source[:, ::5] = -torch.inf
        elif mode.endswith("nan"):
            source[:, : (24 if mode == "few_nan" else 24000)] = torch.nan
        saved = source.view(torch.int16).clone()
        expected = block_reference(source, lengths)
        actual = select_candidate_blocks(source, lengths, 2048, 8, per_token=False)
        assert torch.equal(actual, expected), mode
        assert torch.equal(source.view(torch.int16), saved)
        if mode == "tied":
            dense = select_candidate_blocks(source, lengths, 2048, 8)
            assert torch.equal(dense, expected.repeat_interleave(8, -1))


@torch.inference_mode()
def test_candidate_mask_changed_graph_inputs():
    torch.npu.set_device(0)
    source = torch.zeros((6, 153600), dtype=torch.bfloat16, device="npu")
    lengths = torch.full((6, 1), 8192, device="npu", dtype=torch.int64)

    def run():
        return select_candidate_blocks(source, lengths, 2048, 8, per_token=False)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        result = run()
    address = result.data_ptr()
    for step in range(64):
        fresh = torch.randn_like(source).round()
        fresh[:, step * 251 :] = -torch.inf
        if step % 4 == 0:
            fresh.zero_()
        source.copy_(fresh)
        lengths.fill_(step * 251)
        expected = block_reference(source, lengths)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(result, expected), step
        assert result.data_ptr() == address
