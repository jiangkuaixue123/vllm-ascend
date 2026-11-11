# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.models.deepseek_v41.native_ops import native_engram_statistics


@torch.inference_mode()
def test_engram_statistics_keep_identical_rows_under_batching_and_graph_replay():
    if not get_device_config().supports_fixed_engram_reductions:
        pytest.skip("Fixed Engram reductions require Ascend A5.")
    pytest.importorskip("triton")
    gen = torch.Generator().manual_seed(81)
    h = torch.randn((2, 268, 4, 5120), generator=gen).bfloat16()
    kv = torch.randn((2, 268, 5, 5120), generator=gen).bfloat16()
    h[0, 0].zero_()
    w = torch.randn((4, 5120), generator=gen)
    hidden, projected, weight = h.npu(), kv.npu(), w.npu()

    def statistics(x, key_value):
        key = key_value[..., :4, :]
        assert key.stride(-3) == 5 * 5120
        stats = native_engram_statistics(x, key, weight)
        assert stats is not None
        return torch.stack(stats, -1)

    joined = statistics(hidden, projected)
    for i in range(2):
        isolated = statistics(hidden[i : i + 1], projected[i : i + 1])
        assert torch.equal(isolated, joined[i : i + 1])
    assert torch.equal(statistics(hidden.flip(0), projected.flip(0)).flip(0), joined)
    h64, k64 = h.double(), kv[..., :4, :].double()
    dot_terms = h64 * w.double() * k64
    expected = torch.stack((h64.square().mean(-1), k64.square().mean(-1), dot_terms.sum(-1)), -1)
    magnitude = torch.stack((expected[..., 0], expected[..., 1], dot_terms.abs().sum(-1)), -1)
    error = (joined.cpu().double() - expected).abs()
    assert torch.all(error <= 32 * torch.finfo(torch.float32).eps * magnitude)

    # Decode replay uses the same input addresses for different request data.
    static_h = hidden[:1, :12].clone()
    static_kv = projected[:1, :12].clone()
    statistics(static_h, static_kv)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = statistics(static_h, static_kv)
    for i in (0, 1, 0):
        static_h.copy_(hidden[i : i + 1, :12])
        static_kv.copy_(projected[i : i + 1, :12])
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(output, joined[i : i + 1, :12])
