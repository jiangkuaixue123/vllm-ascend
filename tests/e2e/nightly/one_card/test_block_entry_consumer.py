# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialized residuals, auxiliary readers and prepared next-block HC inputs."""

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open
from torch import nn

from vllm_ascend.models.deepseek_v41.attention import DeepseekV41RMSNorm
from vllm_ascend.models.deepseek_v41.core import make_identity_pre_mix
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Block, DeepseekV41ForCausalLM
from vllm_ascend.utils import enable_custom_op


class AttentionProbe(nn.Module):
    def forward(self, x, runtime):
        runtime.attention_output_mask = runtime.decode.active.roll(1)
        return x.roll(13, dims=-1)


class FFNProbe(nn.Module):
    def forward_decode(self, x, active, *, mask_owned_by_block=False):
        return torch.where(active[:, None], x.roll(-7, dims=-1), float("nan"))


class EngramProbe(nn.Module):
    def forward_decode(self, hidden, values, active, *, sequence_parallel):
        return torch.where(active[:, None, None], (hidden + 0.125).bfloat16(), 0)


def block(layer_id):
    model = Path("/home/admin/model").resolve()
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]

    def read(name):
        key = f"layers.{layer_id}.{name}"
        with safe_open(str(model / index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key).npu().contiguous()

    layer = DeepseekV41Block.__new__(DeepseekV41Block)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(hidden_size=5120, rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20)
    layer.attn, layer.ffn, layer.engram = AttentionProbe(), FFNProbe(), None
    for kind in ("attn", "ffn"):
        for field in ("fn", "scale", "base"):
            layer.register_parameter(f"hc_{kind}_{field}", nn.Parameter(read(f"hc_{kind}_{field}")))
        norm = DeepseekV41RMSNorm(5120, 1e-20).to(device="npu", dtype=torch.bfloat16)
        norm.weight.copy_(read(f"{kind}_norm.weight"))
        setattr(layer, f"{kind}_norm", norm)
    return layer.eval()


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("engram_boundary", [False, True])
@torch.inference_mode()
def test_three_block_materialized_and_prepared_consumers(rows, engram_boundary):
    assert enable_custom_op()
    layers = [block(i) for i in (3, 4, 5)]
    if engram_boundary:
        layers[1].engram = EngramProbe()
    x = torch.randn(rows, 4, 5120, device="npu", dtype=torch.bfloat16)
    active = torch.arange(rows, device="npu") % 3 != 1

    def flow(joint):
        hidden = torch.where(active[:, None, None], x, 0)
        pre = make_identity_pre_mix(hidden)
        runtime = SimpleNamespace(
            decode=SimpleNamespace(active=active), block_row_mask=active, attention_output_mask=None
        )
        prepared, aux, prepared_edges = None, [], 0
        for index, layer in enumerate(layers):
            if layer.engram is not None:
                assert prepared is None
                hidden = torch.where(active[:, None, None], (hidden + 0.125).bfloat16(), 0)
            # The real DSpark reader observes the materialized HC residual here.
            aux.append(hidden.mean(dim=1))
            following = layers[index + 1] if joint and index + 1 < len(layers) else None
            if following is not None and following.engram is not None:
                following = None
            if following is None and prepared is None:
                hidden, pre = layer(hidden, pre, runtime, active)
            else:
                result = layer(hidden, pre, runtime, active, prepared_attn=prepared, next_block=following)
                if following is None:
                    hidden, pre = result
                    prepared = None
                else:
                    hidden, pre, prepared = result
                    assert prepared is not None
                    prepared_edges += 1
        assert prepared is None
        if joint:
            assert prepared_edges == (1 if engram_boundary else 2)
        return hidden, pre, *aux

    def check_components(actual):
        for a, b in zip(actual, flow(False)):
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)

    check_components(flow(True))
    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList(layers)
    model.sequence_parallel = False
    model.aux_hidden_state_layers = (1, 2, 3)
    model.embed, model.norm = nn.Identity(), layers[-1].ffn_norm
    model._decode_engram_values = {(rows, 1): None}
    model._start_hc = MethodType(lambda self, embedded: (x, make_identity_pre_mix(x)), model)
    input_ids = torch.arange(rows, device="npu")
    metadata = SimpleNamespace(decode=SimpleNamespace(active=active))

    def driver():
        output, aux = model._forward_decode(input_ids, metadata)
        return output, *aux

    def check(actual):
        hidden, pre, *aux = flow(False)
        expected = model._finish_hc(hidden, pre, rows), *aux
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, torch.where(active[:, None], b, 0), rtol=0, atol=0, equal_nan=True)

    check(driver())
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        driver()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = driver()
    for step in range(4):
        x.copy_(torch.randn_like(x) * (step + 1))
        layers[1].hc_attn_base.add_(step * 0.01)
        active.copy_(torch.arange(rows, device="npu") % 3 != step % 3)
        if step == 3:
            active.zero_()
            x.fill_(float("nan"))
        graph.replay()
        check(output)
