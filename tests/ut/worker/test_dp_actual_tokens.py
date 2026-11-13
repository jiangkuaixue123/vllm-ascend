# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker import model_runner_v1 as runner_module


@pytest.mark.parametrize("pad", [False, True])
def test_dp_sync_preserves_actual_counts_without_an_extra_collective(monkeypatch, pad):
    runner = SimpleNamespace(dp_size=4, dp_rank=0, vllm_config=object())
    modes = runner_module.CUDAGraphMode
    calls = []

    def all_reduce(value, group):
        calls.append(value.shape)
        assert value.device.type == "cpu"
        assert value[0, 0] == 164
        value[0] = torch.tensor([164, 4, 8160, 140])
        value[1] = 0
        if len(value) == 3:
            assert value[2, 0] == 161
            value[2] = torch.tensor([161, 0, 8160, 139])

    monkeypatch.setattr(runner_module, "should_skip_allreduce_across_dp_group", lambda *args: False)
    monkeypatch.setattr(runner_module, "get_dp_group", lambda: SimpleNamespace(cpu_group=object()))
    monkeypatch.setattr(runner_module.dist, "all_reduce", all_reduce)
    maximum, capacities, mode = runner_module.NPUModelRunner._sync_metadata_across_dp(
        runner, 164, cudagraph_mode=modes.NONE, allow_dp_padding=pad, num_actual_tokens=161
    )
    assert (maximum, mode) == (8160, modes.NONE)
    assert capacities.tolist() == ([8160] * 4 if pad else [164, 4, 8160, 140])
    assert runner._num_actual_tokens_across_dp.tolist() == [161, 0, 8160, 139]
    assert calls == [torch.Size([3, 4])]
    runner_module.NPUModelRunner._sync_metadata_across_dp(runner, 164, is_draft_model=True)
    assert runner._num_actual_tokens_across_dp is None
    assert calls == [torch.Size([3, 4]), torch.Size([2, 4])]


def test_skipped_dp_sync_cannot_reuse_target_counts(monkeypatch):
    runner = SimpleNamespace(
        dp_size=4, dp_rank=0, vllm_config=object(), _num_actual_tokens_across_dp=torch.tensor([1, 2, 3, 4])
    )
    monkeypatch.setattr(runner_module, "should_skip_allreduce_across_dp_group", lambda *args: True)
    runner_module.NPUModelRunner._sync_metadata_across_dp(runner, 24, num_actual_tokens=17)
    assert runner._num_actual_tokens_across_dp is None
