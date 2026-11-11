# SPDX-License-Identifier: Apache-2.0
"""The process-global dispatcher must not leak across nested target and draft calls."""

from types import SimpleNamespace

import pytest
from torch import nn

from vllm_ascend.models.deepseek_v41 import fused_moe


@pytest.mark.parametrize("fail", [False, True])
def test_nested_runner_comm_geometry(monkeypatch, fail):
    kind = fused_moe.MoECommType.ALLTOALL
    ctx = SimpleNamespace(moe_comm_type=kind, moe_comm_method=128)
    monkeypatch.setattr(fused_moe, "_EXTRA_CTX", ctx)

    def runner(n):
        obj = fused_moe.V41MoERunner.__new__(fused_moe.V41MoERunner)
        nn.Module.__init__(obj)
        obj.comm_methods = {kind: n}
        return obj

    target, draft = runner(384), runner(128)

    def call_stock(runner, value):
        assert ctx.moe_comm_method == value
        if runner is target:
            if fail:
                with pytest.raises(RuntimeError, match="native error"):
                    draft._forward_impl(128)
            else:
                assert draft._forward_impl(128) == 128
            assert ctx.moe_comm_method == 384
        elif fail:
            raise RuntimeError("native error")
        return value

    monkeypatch.setattr(fused_moe.AscendMoERunner, "_forward_impl", call_stock)
    assert target._forward_impl(384) == 384
    assert ctx.moe_comm_method == 128
