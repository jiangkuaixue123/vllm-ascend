# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401 - register torch.npu and its stock kernels.

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

BF, FP = torch.bfloat16, torch.float32


def _words(t):
    return t.view(torch.int16 if t.dtype == BF else torch.int32)


@pytest.mark.parametrize(
    ("rows", "ratio"), [(24, 2), (24, 1), (48, 2), (48, 1)], ids=["full24-r2", "full24-r1", "full48-r2", "full48-r1"]
)
@torch.inference_mode()
def test_stock_cache_scatter_replay(rows, ratio):
    assert get_ascend_device_type() == AscendDeviceType.A3 and enable_custom_op()
    physical = 257
    generator = torch.Generator().manual_seed(509)
    widths = (1024, 2048) if ratio == 2 else (512, 1024)
    initial = [
        torch.randint(-32768, 32768, (physical + 2, width), dtype=torch.int16, generator=generator).npu()
        for width in widths
    ]
    raw = (0, 0, 512, BF, False)
    low = (0, 512, 512, FP if ratio == 2 else BF, False)
    high = (1, 0, 512, FP, False)
    latent = (0, 512, 512, BF, True)
    index = (1, 512, 128, BF, True)
    fields = (raw, low, high, latent, index) if ratio == 2 else (raw, low, latent, index)
    eligible = len(fields) if ratio == 2 else len(fields) - 1
    slots = torch.zeros(rows, dtype=torch.int64, device="npu")
    active = torch.zeros(rows, dtype=torch.bool, device="npu")
    complete = torch.zeros_like(active)
    md = attention.DeepseekV41DecodeMetadata(None, slots, None, active, 16384, 6, 0)
    inputs = [torch.zeros((rows, w), dtype=t, device="npu") for _, _, w, t, _ in fields]
    addresses = [x.data_ptr() for x in (slots, active, complete, *inputs)]
    context = SimpleNamespace(cudagraph_runtime_mode=attention.CUDAGraphMode.FULL, is_draft_model=False)
    arms = []
    for _ in range(3):
        backing = [t.clone() for t in initial]
        cache = attention.DeepseekV41Cache.__new__(attention.DeepseekV41Cache)
        torch.nn.Module.__init__(cache)
        cache.dim, cache.index_dim, cache.source_ratio = 512, 128, ratio
        cache.kv_cache = tuple(t[1:-1].view(BF) for t in backing)
        cache.prefix = f"model.layers.{2 if ratio == 2 else 20}.attn.cache"
        arms.append((cache, backing))

    def forward(cache):
        for (bank, offset, _, _, gated), values in zip(fields, inputs):
            cache.decode_write(md, values, bank, offset, complete if gated else active)

    with (
        patch.object(attention, "is_forward_context_available", return_value=True),
        patch.object(attention, "get_forward_context", return_value=context),
    ):
        graphs = []
        for i, (cache, _) in enumerate(arms[:2]):
            # A non-FULL dispatch of the real method records the unmodified parent operator.
            context.cudagraph_runtime_mode = attention.CUDAGraphMode.FULL if i else attention.CUDAGraphMode.NONE
            forward(cache)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with (
                patch.object(attention, "_scatter_decode_field", wraps=attention._scatter_decode_field) as terminal,
                patch.object(attention.logger, "info", wraps=attention.logger.info) as log,
                torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True),
            ):
                forward(cache)
            torch.npu.synchronize()
            hits = [
                c.args
                for c in log.call_args_list
                if c.args and str(c.args[0]).startswith("V4.1 stock cache scatter capture")
            ]
            assert terminal.call_count == len(hits) == (eligible if i else 0)
            assert all(
                call[1] == cache.prefix and call[2] == ratio and call[5:7] == (rows, physical) and call[-1] == 1024
                for call in hits
            )
            graphs.append(graph)
        for _, backing in arms:
            for target, original in zip(backing, initial):
                target.copy_(original)

        touched = set()
        for step in range(5):
            positions = (torch.arange(rows) * 5 + step) % physical
            positions[:6] = torch.tensor([0, 0, physical - 1, physical - 1, -1, physical])
            live = torch.arange(rows) % 4 != 3
            live[:4] = True
            if step in (0, 4):
                live.zero_()
            if step == 3:
                live[:2] = False
            enabled = (torch.arange(rows) + step) % 2 == 0
            enabled[2:4] = True
            slots.copy_(positions)
            active.copy_(live)
            complete.copy_(enabled)
            originals = []
            for values in inputs:
                words = _words(values)
                host = torch.randint(-31000, 31000, words.shape, dtype=words.dtype, generator=generator)
                if words.dtype == torch.int16:
                    special = [0, -32768, 1, -32767, 0x7F80, -128, 0x7F81, 0x7FC2, -127, -1]
                else:
                    special = [0, -(2**31), 1, 1 - 2**31, 0x7F800000, -0x800000, 0x7F800001, 0x7FC12345, -0x7FFFFF, -1]
                host[:, : len(special)] = torch.tensor(special, dtype=words.dtype)
                host[3].copy_(host[2])
                originals.append(host)
                words.copy_(host)
            for graph in graphs:
                graph.replay()
            forward(arms[2][0])
            torch.npu.synchronize()
            touched.update(int(s) + 1 for s in positions[live] if 0 <= int(s) < physical)
            touched.add(1)
            untouched = [j for j in range(physical + 2) if j not in touched]
            for bank, seed in enumerate(initial):
                baseline = arms[0][1][bank]
                assert torch.equal(baseline, arms[1][1][bank]) and torch.equal(baseline, arms[2][1][bank])
                assert torch.equal(arms[1][1][bank][untouched], seed[untouched])
                if (ratio == 2 and bank == 0) or (ratio == 1 and bank == 1):
                    assert torch.equal(arms[1][1][bank][:, 640:], seed[:, 640:])
            assert [x.data_ptr() for x in (slots, active, complete, *inputs)] == addresses
            assert (
                torch.equal(slots.cpu(), positions)
                and torch.equal(active.cpu(), live)
                and torch.equal(complete.cpu(), enabled)
            )
            assert all(torch.equal(_words(v).cpu(), ref) for v, ref in zip(inputs, originals))
