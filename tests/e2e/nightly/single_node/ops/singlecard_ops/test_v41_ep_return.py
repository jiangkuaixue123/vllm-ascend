# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final EP destinations preserve combine bits, guards and forward ownership."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import fused_experts as model
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoEAllGatherCombineMetadata
from vllm_ascend.ops.triton.deepseek_v41_prefill_experts import prefill_combine_output, prefill_zero_ep_padding
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("rows", [1, 17, 264])
@pytest.mark.parametrize("native_tail", [False, True])
@torch.inference_mode()
def test_mapped_combine_preserves_bits_and_other_rows(rows, native_tail):
    torch.npu.set_device(0)
    assert enable_custom_op()
    init_device_properties_triton()
    values = torch.randn(rows * 6, 5120, device="npu", dtype=torch.bfloat16)
    factors = torch.ones(rows * 6, device="npu")
    ids = torch.randperm(rows * 6, device="npu").int()
    owned = torch.rand(rows, 6, device="npu")
    owned[::3, 1::2] = 0
    destinations = torch.arange(rows, device="npu", dtype=torch.int32) * 2 + 1
    metadata = MoEAllGatherCombineMetadata(owned, ids, (rows, 5120))
    dispatcher = object.__new__(model.DeepseekV41Dispatcher)

    for step in range(3):
        output = torch.full((rows * 2 + 3, 5120), 123.0, device="npu")
        if native_tail:
            expected_rows = dispatcher.token_combine(values.float(), metadata)
            actual = dispatcher.token_combine(values.float(), metadata, out=output, output_rows=destinations)
        else:
            expected_rows = prefill_combine_output(values, factors, ids, owned)
            actual = prefill_combine_output(values, factors, ids, owned, out=output, output_rows=destinations)
        expected = torch.full_like(output, 123.0)
        expected.index_copy_(0, destinations.long(), expected_rows)
        assert actual is output
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), step
        values.neg_()
        ids.copy_(ids.roll(3))


def test_forward_layout_cache_reuses_and_invalidates(monkeypatch):
    sizes = [2] * 16
    context = SimpleNamespace(
        cudagraph_runtime_mode=model.CUDAGraphMode.NONE,
        is_draft_model=False,
        in_profile_run=False,
        dp_metadata=object(),
        ep_actual_local_sizes=sizes,
    )
    monkeypatch.setattr(model, "prefill_fusion_available", lambda _: True)
    monkeypatch.setattr(model, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(model, "get_forward_context", lambda: context)
    monkeypatch.setattr(model, "get_ep_group", lambda: SimpleNamespace(world_size=16))
    monkeypatch.setattr(model, "_get_ep_token_layout", lambda *_: (context.ep_actual_local_sizes, 4))
    states = torch.empty(sum(sizes), 5120)
    mapping, capacity, device_sizes = model.prefill_ep_return_layout(states)
    assert capacity == 64
    assert mapping.tolist() == [i * 4 + j for i in range(16) for j in range(2)]
    assert model.prefill_ep_return_layout(states)[0] is mapping
    assert model.prefill_ep_return_layout(states)[2] is device_sizes
    context.ep_actual_local_sizes = [0, 4] + sizes[2:]
    refreshed = model.prefill_ep_return_layout(states)[0]
    assert refreshed is not mapping and refreshed[:4].tolist() == [4, 5, 6, 7]
    context.is_draft_model = True
    assert model.prefill_ep_return_layout(states) is None
    context.is_draft_model = False
    context.cudagraph_runtime_mode = model.CUDAGraphMode.FULL
    assert model.prefill_ep_return_layout(states) is None


@pytest.mark.parametrize("capacity", [17, 2040])
@torch.inference_mode()
def test_padding_only_preserves_owned_rows_on_replay(capacity):
    torch.npu.set_device(0)
    sizes = torch.tensor([0, 1, capacity, capacity // 2] * 4, device="npu", dtype=torch.int32)
    output = torch.full((16 * capacity, 5120), -7.0, device="npu")
    prefill_zero_ep_padding(output, sizes, capacity)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        prefill_zero_ep_padding(output, sizes, capacity)
    for step in range(3):
        host_sizes = [0, 1, capacity, capacity // 2] * 4
        host_sizes = host_sizes[step:] + host_sizes[:step]
        sizes.copy_(torch.tensor(host_sizes, dtype=torch.int32))
        output.fill_(-7)
        expected = torch.full_like(output, -7)
        for rank, size in enumerate(host_sizes):
            expected[rank * capacity + size : (rank + 1) * capacity].zero_()
        graph.replay()
        assert torch.equal(output.view(torch.int32), expected.view(torch.int32))


def test_mapped_combine_rejects_wrong_destination_contract():
    torch.npu.set_device(0)
    values = torch.ones(6, 5120, device="npu", dtype=torch.bfloat16)
    factors = torch.ones(6, device="npu")
    ids = torch.arange(6, device="npu", dtype=torch.int32)
    owned = torch.ones(1, 6, device="npu")
    with pytest.raises(ValueError, match="int32"):
        prefill_combine_output(
            values,
            factors,
            ids,
            owned,
            out=torch.zeros(16, 5120, device="npu"),
            output_rows=torch.ones(1, device="npu", dtype=torch.int64),
        )
