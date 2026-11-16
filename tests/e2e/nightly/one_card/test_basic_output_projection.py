# SPDX-License-Identifier: Apache-2.0
"""Exercise the real FIA storage format through the fused BF16 send producer."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.models.common.ops.selected_fia import try_selected_fia
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Attention, _quantize_projection_input
from vllm_ascend.ops.triton.deepseek_v41_fp8_qdq import pack_prefill_projection_input


def reference(x, tp):
    send = x.view(x.shape[0], tp, 64 // tp, 512).transpose(0, 1).contiguous()
    return _quantize_projection_input(send.view(-1, 8 // tp, 4096)).view(-1, 64 // tp, 512)


def fia(q, kv, sink, slots, positions, frequencies):
    out = try_selected_fia(q, kv, sink, slots, 512**-0.5, sink_source_dtype=torch.bfloat16, mask_empty_rows=False)
    out = out[:, 0]
    DeepseekV41Attention.rotate(
        SimpleNamespace(rope_dim=64, frequencies=frequencies), out.unsqueeze(0), positions, True
    )
    return out


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("rows", [3, 6, 12])
@torch.inference_mode()
def test_real_fia_output_and_graph(rows, tp):
    torch.npu.set_device(0)
    q = torch.randn(rows, 1, 64, 512, device="npu", dtype=torch.bfloat16)
    kv = torch.randn(rows, 640, 512, device=q.device, dtype=q.dtype)
    sink = torch.zeros(64, device=q.device, dtype=torch.float32)
    slots = torch.arange(640, device=q.device, dtype=torch.int32).view(1, 1, 640).repeat(rows, 1, 1)
    positions = torch.arange(rows, device=q.device)
    frequencies = 160000.0 ** (-torch.arange(32, device=q.device).float() / 32)

    def run():
        out = fia(q, kv, sink, slots, positions, frequencies)
        assert out.is_contiguous() and int(torch_npu.get_npu_format(out)) == int(torch_npu.Format.NCHW)
        packed = pack_prefill_projection_input(out, tp)
        assert packed is not None
        return packed, reference(out, tp)

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        packed, expected = run()
    for step in range(4):
        q.copy_(torch.randn_like(q) * (step + 1))
        kv.copy_(torch.randn_like(kv) / (step + 1))
        positions.add_(17103)
        graph.replay()
        assert torch.equal(packed.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("rows", [3, 6, 12])
@torch.inference_mode()
def test_basic_storage_all_words(rows, tp):
    torch.npu.set_device(0)
    n = rows * 64 * 512
    words = torch.arange(65536, dtype=torch.int32).short().repeat((n + 65535) // 65536)[:n]
    x = torch_npu.npu_format_cast(words.view(torch.bfloat16).reshape(rows, 64, 512).npu(), torch_npu.Format.NCHW)
    assert x.is_contiguous() and int(torch_npu.get_npu_format(x)) == int(torch_npu.Format.NCHW)
    before = x.clone()
    actual = pack_prefill_projection_input(x, tp)
    assert actual is not None
    assert torch.equal(actual.view(torch.int16), reference(x, tp).view(torch.int16))
    assert torch.equal(before.view(torch.int16), x.view(torch.int16))
