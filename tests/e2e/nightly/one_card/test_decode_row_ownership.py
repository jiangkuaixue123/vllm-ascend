# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The target block owns masked residuals while preserving every live-row bit."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.hc_post import hc_post_fused
from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.models.deepseek_v41.core import hc_post, make_identity_pre_mix
from vllm_ascend.models.deepseek_v41.moe import DeepseekV41MoE


def exact(actual, expected):
    assert actual is not None and actual.shape == expected.shape and actual.dtype == expected.dtype
    both_nan = actual.isnan() & expected.isnan()
    assert torch.equal(actual.view(torch.int16)[~both_nan], expected.view(torch.int16)[~both_nan])


@pytest.mark.parametrize("rows", [3, 6, 12])
@torch.inference_mode()
def test_masked_hc_post_nonfinite_and_graph_refresh(rows):
    sub = torch.randn(rows, 5120, dtype=torch.bfloat16, device="npu")
    residual = torch.randn(rows, 4, 5120, dtype=sub.dtype, device=sub.device)
    post = torch.rand(rows, 4, device=sub.device)
    comb = torch.rand(rows, 4, 4, device=sub.device)
    storage = torch.ones(rows + 3, dtype=torch.bool, device=sub.device)
    mask = storage[3:]
    hc_post_fused(sub, residual, post, comb, row_mask=mask)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = hc_post_fused(sub, residual, post, comb, row_mask=mask)
    pointer = output.data_ptr()
    for step in range(16):
        mask.copy_(torch.arange(rows, device=sub.device) % 4 != step % 4)
        if step % 7 == 0:
            mask.fill_(step % 2 == 0)
        sub.copy_(torch.randn_like(sub))
        residual.copy_(torch.randn_like(residual))
        sub[:, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0], device=sub.device)
        residual[:, :, 4:8] = torch.tensor([2**26, 1, -(2**26), 1], device=sub.device)[None, :, None]
        graph.replay()
        expected = torch.where(mask[:, None, None], hc_post(sub, residual, post, comb), 0)
        exact(output, expected)
        assert output.data_ptr() == pointer
        assert torch.all(output.view(torch.int16)[~mask] == 0)
    assert hc_post_fused(sub, residual, post, comb, row_mask=mask.float()) is None


class Block:
    pre_and_mixes = model_module.DeepseekV41Block.pre_and_mixes
    post_residual = model_module.DeepseekV41Block.post_residual
    mixes = model_module.DeepseekV41Block.mixes
    __call__ = model_module.DeepseekV41Block.forward

    def __init__(self, mask, parameters, eps=1e-6):
        self.__dict__.update(parameters)
        self.config = SimpleNamespace(rms_norm_eps=eps, hc_eps=1e-6, hc_sinkhorn_iters=20)
        self.attn_norm = model_module.DeepseekV41RMSNorm(5120, eps).to(mask.device)
        self.ffn_norm = model_module.DeepseekV41RMSNorm(5120, eps).to(mask.device)
        self.attn_norm.weight.fill_(1)
        self.ffn_norm.weight.fill_(1)
        self.attn = lambda value, runtime: torch.where(
            mask[:, None], value * 1.25, 0 if runtime.block_row_mask is None else float("nan")
        )
        self.moe = SimpleNamespace(
            hidden_size=5120,
            _routed_output=lambda value, **_: torch.where(mask[:, None], value.float() * 0.75, float("nan")),
            _shared_output=lambda value: value.float() * 0.25,
        )
        self.ffn = SimpleNamespace(
            forward_decode=lambda value, active, **kwargs: DeepseekV41MoE.forward_decode(
                self.moe, value, active, **kwargs
            )
        )


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("eps", [1e-6, 1e-20])
@torch.inference_mode()
def test_native_hc_block_chain_matches_legacy_live_rows(rows, eps):
    mask = torch.ones(rows, dtype=torch.bool, device="npu")
    states = torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device=mask.device)
    parameters = {}
    for name in ("attn", "ffn"):
        parameters[f"hc_{name}_fn"] = torch.randn(24, 20480, device=mask.device) / 256
        parameters[f"hc_{name}_scale"] = torch.rand(3, device=mask.device)
        parameters[f"hc_{name}_base"] = torch.rand(24, device=mask.device)
    legacy, owner = Block(mask, parameters, eps), Block(mask, parameters, eps)

    def run(block, owns):
        runtime = SimpleNamespace(decode=SimpleNamespace(active=mask), block_row_mask=mask if owns else None)
        value = torch.where(mask[:, None, None], states, 0) if owns else states
        pre = make_identity_pre_mix(value)
        for _ in range(3):
            value, pre = block(value, pre, runtime, mask)
        return value

    run(legacy, False)
    run(owner, True)
    old_graph, new_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(old_graph):
        expected = run(legacy, False)
    with torch.npu.graph(new_graph):
        actual = run(owner, True)
    for step in range(16):
        mask.copy_(torch.arange(rows, device=mask.device) % 3 != step % 3)
        if step % 5 == 0:
            mask.fill_(step % 2 == 0)
        states.copy_(torch.randn_like(states))
        old_graph.replay()
        new_graph.replay()
        exact(actual[mask], expected[mask])
        assert torch.all(actual.view(torch.int16)[~mask] == 0)


@pytest.mark.parametrize("rows", [3, 6, 12])
@torch.inference_mode()
def test_model_entry_and_engram_reestablish_zero_rows(rows):
    mask = torch.ones(rows, dtype=torch.bool, device="npu")
    embedded = torch.randn(rows, 5120, dtype=torch.bfloat16, device=mask.device)
    observed = []

    class Layer:
        engram = SimpleNamespace(
            forward_decode=lambda x, values, active, **_: torch.where(active[:, None, None], x + 2, float("nan"))
        )

        def __call__(self, x, pre, runtime, active):
            assert runtime.block_row_mask is active
            observed.append(x)
            return x, pre

    layer = Layer()
    model = SimpleNamespace(
        sequence_parallel=False,
        config=SimpleNamespace(hc_mult=4),
        layers=[layer],
        embed=lambda _: embedded,
        _start_hc=lambda x: (x[:, None, :].repeat(1, 4, 1), x.new_ones((rows, 4))),
        _decode_engram_values={(rows, 0): torch.empty(0, device=mask.device)},
        _capture_aux=lambda values, _, x, __: values.append(x[:, 0]),
        _finish_hc=lambda x, pre, count: x[:, 0],
        aux_hidden_state_layers=True,
    )
    metadata = SimpleNamespace(decode=SimpleNamespace(active=mask, grouped_row_start=0))
    ids = torch.arange(rows, device=mask.device)
    model_module.DeepseekV41ForCausalLM._forward_decode(model, ids, metadata)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, auxiliary = model_module.DeepseekV41ForCausalLM._forward_decode(model, ids, metadata)
    captured = observed[-1]
    for step in range(16):
        mask.copy_(torch.arange(rows, device=mask.device) % 3 == step % 3)
        embedded.copy_(torch.randn_like(embedded))
        embedded[~mask] = float("nan")
        graph.replay()
        expected = torch.where(mask[:, None], embedded + 2, 0)
        exact(output, expected)
        exact(auxiliary[0], expected)
        assert torch.all(captured.view(torch.int16)[~mask] == 0)


@pytest.mark.parametrize("rows", [3, 6, 12])
@torch.inference_mode()
def test_hc_fallback_and_default_keep_consumer_contract(monkeypatch, rows):
    args = (
        torch.randn(rows, 5120, dtype=torch.bfloat16, device="npu"),
        torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device="npu"),
        torch.rand(rows, 4, device="npu"),
        torch.rand(rows, 4, 4, device="npu"),
    )
    mask = torch.arange(rows, device="npu") % 2 == 0
    runtime = SimpleNamespace(decode=object(), block_row_mask=None)
    block = SimpleNamespace()
    exact(model_module.DeepseekV41Block.post_residual(block, *args, runtime), hc_post(*args))
    monkeypatch.setattr(model_module, "hc_post_fused", lambda *_, **__: None)
    runtime.block_row_mask = mask
    expected = torch.where(mask[:, None, None], hc_post(*args), 0)
    exact(model_module.DeepseekV41Block.post_residual(block, *args, runtime), expected)
