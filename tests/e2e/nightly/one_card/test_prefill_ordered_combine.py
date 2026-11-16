# SPDX-License-Identifier: Apache-2.0
"""The expert return keeps native slot order and never reads zero-weight rows."""

import pytest
import torch

import vllm_ascend  # noqa: F401
from vllm_ascend.ops.triton.deepseek_v41_prefill_experts import _prefill_combine, prefill_combine_output
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True)
def native_binding():
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "prefill_ordered_combine")


def reference(x, factor, ids, owned, out, mapping):
    _prefill_combine[(ids.shape[0], 1)](
        x,
        factor,
        ids,
        owned,
        out,
        ids if mapping is None else mapping,
        M=ids.shape[0],
        EXPANDED=x.shape[0],
        H=5120,
        K=6,
        C=8192,
        MAPPED=mapping is not None,
        enable_fp_fusion=False,
    )


def guarded_reference(x, factor, ids, owned, out, mapping):
    values, scales, indices, weights = x.cpu().float(), factor.cpu(), ids.cpu().long().abs(), owned.cpu()
    total = torch.zeros(ids.shape[0], 5120)
    for slot in range(6):
        row = indices[:, slot]
        active = (row < len(values)) & (weights[:, slot] != 0)
        safe = row.clamp(max=len(values) - 1)
        value = torch.where((active & (scales[safe] != 0))[:, None], values[safe], 0)
        value = value * weights[:, slot, None]
        total = torch.where(active[:, None], value if slot == 0 else total + value, total)
    expected = out.cpu()
    expected[:] = total if mapping is None else expected
    if mapping is not None:
        expected[mapping.cpu().long()] = total
    return expected


def equal_bits(actual, expected):
    a, b = actual.cpu(), expected.cpu()
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(a.view(torch.int32)[~torch.isnan(a)], b.view(torch.int32)[~torch.isnan(b)])


@pytest.mark.parametrize("rows,mapped", [(1, False), (17, True), (128, False), (4096, True)])
def test_native_ordered_combine_matches_existing_consumer(rows, mapped):
    gen = torch.Generator().manual_seed(709)
    x = torch.randn(rows * 6, 5120, generator=gen).bfloat16().npu()
    factor = torch.ones(rows * 6, device="npu")
    ids = torch.arange(rows * 6, dtype=torch.int32).view(rows, 6).npu()
    ids[:, 1].neg_()
    owned = (torch.rand(rows, 6, generator=gen) > 0.6).float().npu()
    owned[:, 0] = 1
    factor[::7] = 0
    x[::7] = float("nan")
    x[1::17] = -0.0
    ids[::9, 2] = rows * 6 + 13
    ids[::13, 3] = -2147483648
    mapping = (torch.arange(rows - 1, -1, -1, device="npu", dtype=torch.int32) * 2) if mapped else None
    out = torch.full((rows * (2 if mapped else 1), 5120), 123.0, device="npu")
    expected = guarded_reference(x, factor, ids, owned, out, mapping)
    with torch.inference_mode():
        if rows >= 128:
            prefill_combine_output(x, factor, ids.flatten(), owned, out=out, output_rows=mapping)
        else:
            torch.ops._C_ascend.prefill_ordered_combine(x, factor, ids, owned, out, mapping)
    equal_bits(out, expected)


def test_native_ordered_combine_replays_dynamic_metadata():
    rows = 33
    x = torch.randn(rows * 6, 5120, dtype=torch.bfloat16, device="npu")
    factor = torch.ones(rows * 6, device="npu")
    ids = torch.arange(rows * 6, dtype=torch.int32, device="npu").view(rows, 6)
    owned = torch.ones(rows, 6, device="npu")
    mapping = torch.arange(rows, dtype=torch.int32, device="npu")
    out = torch.empty(rows, 5120, device="npu")
    with torch.inference_mode():
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                torch.ops._C_ascend.prefill_ordered_combine(x, factor, ids, owned, out, mapping)
        torch.npu.current_stream().wait_stream(stream)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            torch.ops._C_ascend.prefill_ordered_combine(x, factor, ids, owned, out, mapping)
        for i in range(12):
            owned.fill_(1 if i % 3 else 0)
            factor.fill_(1 if i % 4 else 0)
            x.mul_(-0.5)
            mapping.copy_(torch.arange(rows, dtype=torch.int32, device="npu").roll(i))
            expected = torch.empty_like(out)
            reference(x, factor, ids, owned, expected, mapping)
            graph.replay()
            equal_bits(out, expected)


def test_native_ordered_combine_empty_and_shape_guard():
    x = torch.empty(0, 5120, dtype=torch.bfloat16, device="npu")
    factor = torch.empty(0, device="npu")
    ids = torch.empty(0, 6, dtype=torch.int32, device="npu")
    owned = torch.empty(0, 6, device="npu")
    out = torch.empty(0, 5120, device="npu")
    torch.ops._C_ascend.prefill_ordered_combine(x, factor, ids, owned, out, None)
    with pytest.raises(RuntimeError, match="top-six"):
        torch.ops._C_ascend.prefill_ordered_combine(x, factor, ids[:, :5], owned[:, :5], out, None)
