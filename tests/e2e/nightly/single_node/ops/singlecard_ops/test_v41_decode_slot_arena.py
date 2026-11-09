# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the target decode entry, true cache writes/index owners and native A3 attention on changing graphs."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v41 import attention, native_attention
from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

LAYERS = (2, 3, 20, 21, 22, 24, 25)
PHYS = 4096


class Fixture:
    def __init__(self, rows, alias, max_model_len=16384, shard=1, module=attention, tp=4):
        self.rows = rows
        self.tp = tp
        self.offset = rows * shard
        self.max_model_len = max_model_len
        self.module = module
        self.current_layer = None
        config = DeepseekV41TextConfig()
        self.layer_ids = (
            LAYERS
            if max_model_len == 16384
            else tuple(i for i in range(config.num_hidden_layers) if config.compress_ratios[i])
        )
        full = rows * tp
        self.values = torch.empty(full, 512, device="npu", dtype=torch.bfloat16)
        self.layers = []
        self.metadata = {}
        self.builders = {}
        self.common = {}
        self.tables = {}
        source20 = None
        gen = torch.Generator().manual_seed(307)
        for layer_id in self.layer_ids:
            layer = module.DeepseekV41Attention.__new__(module.DeepseekV41Attention)
            nn.Module.__init__(layer)
            layer.layer_id = layer_id
            layer.owns_kv = layer_id in config.kv_source_layer_ids
            layer.owns_index = layer_id in config.index_source_layer_ids
            layer.ratio = config.compress_ratios[layer_id]
            layer.tp, layer.dsa_cp, layer.uses_local_kv = tp, True, True
            layer.heads, layer.dim, layer.index_heads, layer.rope_dim = 64, 512, 32, 64
            layer.config = SimpleNamespace(
                sliding_window=128,
                index_n_heads=32,
                index_head_dim=128,
                index_topk=512,
                candidate_source_layer_id=20,
                candidate_block_size=8,
                candidate_topk_blocks=2048,
            )
            layer.sequence_group = object()
            layer._sequence_inputs = lambda x, phase: (self.values, self.offset)
            layer.wq_a = layer.wkv = lambda x: (x.clone(), None)
            layer.wq_b = lambda x: (x.repeat(1, 64), None)
            layer.q_norm = layer.kv_norm = nn.Identity()
            layer.frequencies = torch.linspace(0.01, 1, 32, device="npu")
            layer.indexer = SimpleNamespace(
                wq_b=lambda x: (x.repeat(1, 8), None),
                weights_proj=lambda x: (x[:, :32].contiguous(), None),
                wk=lambda x: x[:, :128].contiguous(),
                k_norm=lambda x: x.clone(),
            )
            layer.compressor = SimpleNamespace(
                wkv=lambda x: x.clone(), wgate=lambda x: x.clone(), norm=lambda x: x.clone()
            )
            layer._project_output = lambda output, dtype: output.flatten(1)
            layer.attn_sink = torch.linspace(-1, 1, 64, device="npu")
            layer.attention_kernel = module.DeepseekV41NativeAttention()
            cache = module.DeepseekV41Cache.__new__(module.DeepseekV41Cache)
            nn.Module.__init__(cache)
            cache.dim, cache.index_dim, cache.block_size, cache.source_ratio = (
                512,
                128,
                128,
                layer.ratio if layer.owns_kv else 0,
            )
            cache.token_history = False
            key = torch.randn(PHYS, 1024 if cache.source_ratio == 2 else 512, generator=gen).bfloat16().npu()
            if cache.source_ratio == 2:
                value = torch.randn(PHYS, 1024, generator=gen).npu().view(torch.bfloat16)
            else:
                value = torch.randn(PHYS, 1024 if cache.source_ratio else 2, generator=gen).bfloat16().npu()
            if layer_id == 20:
                source20 = value
            if layer_id == 21 and alias:
                key = source20  # Raw writes now overlap the existing source's compressed field.
            cache.kv_cache = (key, value)
            cache.native_layout = (layer.attention_kernel, 64, tp, 640)
            layer.cache = cache
            builder = module.DeepseekV41MetadataBuilder.__new__(module.DeepseekV41MetadataBuilder)
            builder.decode_graph_enabled = True
            builder.capture_sizes = {full}
            builder.max_model_len = max_model_len
            builder.block_size = 128
            builder.query_width = 6
            builder.native_layers = [cache]
            builder.native_decode_plans = {}
            builder.decode_buffers = {}
            count = full // 6
            table = (torch.arange(count)[:, None] * 3 + torch.arange(max_model_len // 128) * 7 + layer_id) % (
                PHYS // 128
            )
            self.tables[layer_id] = table.int()
            common = SimpleNamespace(
                num_input_tokens=full,
                num_actual_tokens=full,
                num_reqs=count,
                max_query_len=6,
                max_seq_len=max_model_len,
                seq_lens=torch.zeros(count, dtype=torch.long, device="npu"),
                slot_mapping=torch.full((full,), -1, dtype=torch.long, device="npu"),
                block_table_tensor=table.int().npu(),
            )
            md = builder.build_for_cudagraph_capture(common)
            cache.metadata = lambda md=md: md
            self.metadata[layer_id], self.builders[layer_id], self.common[layer_id] = md, builder, common
            self.layers.append(layer)

    def update(self, step):
        full = self.rows * self.tp
        gen = torch.Generator().manual_seed(901 + step)
        x = torch.randn(full, 512, generator=gen).bfloat16()
        if step == 3:
            x.view(torch.int16)[self.offset + 1, :4] = torch.tensor([-32768, 0x7FC2, 0x7F80, 1], dtype=torch.int16)
        self.values.copy_(x)
        for lid in self.layer_ids:
            common = self.common[lid]
            groups = full // 6
            endpoints = (
                (0, 3, 1029, 2050) if self.max_model_len == 16384 else (0, 3, 8192, 16384, 16385, 153599, 153600)
            )
            end = torch.full((groups,), endpoints[step], dtype=torch.long)
            if step > 1:
                end = (end + torch.arange(groups) % 3).clamp(max=self.max_model_len)
            table = self.tables[lid].roll(step, dims=1).clone()
            if step == 2:
                table[:, 0] = -1
                table[:, 2] = PHYS // 128
            positions = end[:, None] + torch.arange(6) - 6
            safe = positions.clamp(0, self.max_model_len - 1)
            block = table.gather(1, safe // 128).long()
            slots = (
                (block * 128 + safe % 128)
                .masked_fill((positions < 0) | (block < 0) | (block >= PHYS // 128), -1)
                .flatten()
            )
            if step == 2:
                slots[::6] = -1  # A dead first row must not hide its group's live raw history.
            if step == 3 and lid == 20:
                slots[1::6] = -1
            common.seq_lens.copy_(end)
            common.slot_mapping.copy_(slots)
            common.block_table_tensor.copy_(table)
            assert (
                self.builders[lid]._update_decode(common, bounds=list(range(0, full + 1, 6)))
                is self.metadata[lid].decode
            )

    def run(self):
        runtime = self.module.DeepseekV41Runtime.for_decode(self.metadata[2])
        local = self.values[self.offset : self.offset + self.rows]
        native = self.module.DeepseekV41NativeAttention
        raw_outputs = []
        self.physical_layers = []

        def observe(method, physical):
            def call(kernel, *args, **kwargs):
                output = method(kernel, *args, **kwargs)
                # The layer rotates this result in place after the native call.
                raw_outputs.append(output.clone())
                if physical:
                    self.physical_layers.append(self.current_layer)
                return output

            return call

        outputs = []
        with (
            patch.object(native, "__call__", observe(native.__call__, False)),
            patch.object(native, "decode_with_physical_slots", observe(native.decode_with_physical_slots, True)),
        ):
            for layer in self.layers:
                self.current_layer = layer.layer_id
                outputs.append(layer(local, runtime))
        assert len(raw_outputs) == len(outputs) == len(self.layer_ids)
        return tuple(raw_outputs + outputs)


def equal(left, right):
    assert len(left) == len(right)
    for index, (a, b) in enumerate(zip(left, right)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"output {index} differs"


@torch.inference_mode()
@patch.object(native_attention, "try_selected_fia", lambda *args, **kwargs: None)
def check_graph_case(rows, alias, max_model_len, shard, reference=attention, observations=None, tp=4):
    # Isolate arena addressing by using the same shared-KV operator on both sides.
    # FIA uses different arithmetic; its real TP consumers have separate coverage.
    init_device_properties_triton()
    assert attention.enable_custom_op()
    # Eager parent/child and independently captured parent/child carry independent evolving K/V states.
    engines = [Fixture(rows, alias, max_model_len, shard, module, tp) for module in (reference, attention) * 2]
    for f in engines:
        f.update(0)
    graphs = {}
    index_owners = {2, 8, 14, 20, 24, 28, 32, 36}
    expected = [
        (lid, lid in index_owners or (alias and lid == 22)) for lid in engines[1].layer_ids if not (alias and lid == 21)
    ]
    for i, f in enumerate(engines):
        events = []
        original = f.module.stage_decode_arena
        # A separately supplied parent makes its own long-context guard decision.
        observe_real = bool(i % 2) or reference is not attention

        def observe(*args, events=events, fixture=f, stage=original):
            events.append((fixture.current_layer, args[-1]))
            return stage(*args)

        with patch.object(f.module, "stage_decode_arena", observe if observe_real else None):
            f.run()
            if observations is not None:
                observations.append(
                    {
                        "engine": i,
                        "capture": False,
                        "staged": events.copy(),
                        "physical_layers": f.physical_layers.copy(),
                    }
                )
            assert events == (expected if i % 2 else [])
            assert f.physical_layers == [lid for lid, _ in events]
            if i >= 2:
                events.clear()
                g = torch.npu.NPUGraph()
                with torch.npu.graph(g):
                    output = f.run()
                if observations is not None:
                    observations.append(
                        {
                            "engine": i,
                            "capture": True,
                            "staged": events.copy(),
                            "physical_layers": f.physical_layers.copy(),
                        }
                    )
                assert events == (expected if i % 2 else [])
                assert f.physical_layers == [lid for lid, _ in events]
                graphs[i] = (g, output)
    steps = (1, 2, 3, 0, 2) if max_model_len == 16384 else (1, 2, 3, 4, 5, 6, 0) * 2
    for step in steps:
        for f in engines:
            f.update(step)
        stage = None if reference is attention else reference.stage_decode_arena
        with patch.object(reference, "stage_decode_arena", stage):
            baseline = engines[0].run()
        equal(baseline, engines[1].run())
        for g, out in graphs.values():
            g.replay()
            torch.npu.synchronize()
            equal(baseline, out)
        for f in engines[1:]:
            for ref, obs in zip(engines[0].layers, f.layers):
                for a, b in zip(ref.cache.kv_cache, obs.cache.kv_cache):
                    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
        if observations is not None:
            observations.append(
                {
                    "replay_step": step,
                    "compared_layers": len(engines[0].layer_ids),
                    "raw_native_post_and_cache_bits_equal": True,
                }
            )


@pytest.mark.parametrize(
    ("rows", "alias", "max_model_len", "shard"),
    [(6, False, 16384, 1), (12, False, 16384, 1), (6, True, 16384, 1), (6, False, 153600, 0), (12, True, 153600, 3)],
)
def test_slot_arena_real_decode_graph(rows, alias, max_model_len, shard):
    if attention.stage_decode_arena is None or attention.get_ascend_device_type() != attention.AscendDeviceType.A3:
        pytest.skip("A3 with standard Triton required")
    check_graph_case(rows, alias, max_model_len, shard)


@pytest.mark.parametrize("rows,shard", [(3, 0), (3, 1), (3, 6), (3, 7), (6, 7)])
@pytest.mark.parametrize("alias", [False, True])
def test_tp8_half_and_full_query_groups(rows, shard, alias):
    if attention.stage_decode_arena is None or attention.get_ascend_device_type() != attention.AscendDeviceType.A3:
        pytest.skip("A3 with standard Triton required")
    check_graph_case(rows, alias, 153600, shard, tp=8)
