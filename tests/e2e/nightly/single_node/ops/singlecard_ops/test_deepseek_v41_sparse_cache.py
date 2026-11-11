# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.models.deepseek_v41 import compact_kv, graph_ops, mixed_mla
from vllm_ascend.models.deepseek_v41.native_ops import initialize_native_ops


@pytest.mark.parametrize(
    "encoding,width",
    [
        (compact_kv.CacheEncoding.WINDOW_MXFP8, 512),
        (compact_kv.CacheEncoding.COMPRESSED_E4_FP4, 512),
        (compact_kv.CacheEncoding.INDEX_MXFP4, 128),
    ],
)
@torch.inference_mode()
def test_compact_kv_multirow_device_tiles(encoding, width):
    if not get_device_config().supports_compact_kv_epilog:
        pytest.skip("Requires the A5 compact KV writer.")
    initialize_native_ops()
    random = torch.Generator().manual_seed(268)
    rows = torch.randn((268, width), generator=random).bfloat16()
    if encoding == compact_kv.CacheEncoding.WINDOW_MXFP8:
        packed = compact_kv.pack_window(rows)
    else:
        compressed = encoding == compact_kv.CacheEncoding.COMPRESSED_E4_FP4
        packed = compact_kv.pack_fp4(rows, 16 if compressed else 32, compressed)
    # Use multiple rows on each AIV, cross page boundaries and skip inactive slots.
    slots = torch.randperm(411, generator=random)[: rows.size(0)]
    slots[-2:] = torch.tensor([-1, 411])
    expected = torch.full((411, packed.size(1)), 0xA5, dtype=torch.uint8)
    cache = expected.npu()
    expected[slots[:-2]] = packed[:-2]
    compact_kv.store_compact_kv(cache, rows.npu(), slots.npu(), encoding)
    assert torch.equal(cache.cpu(), expected)


@torch.inference_mode()
def test_mixed_mla_reads_compact_rows_through_paged_target():
    if not get_device_config().supports_mixed_quant_sparse_mla:
        pytest.skip("Requires the A5 mixed sparse attention provider.")
    # Uniform logits give each visible row and the zero-valued sink the same weight.
    # The two cache banks use different encodings and noncontiguous physical rows.
    width = 512
    columns = torch.where(torch.arange(width) % 2 == 0, 1.0, -1.0)
    factors = torch.arange(384) % 4 - 2
    raw = compact_kv.pack_window((factors[:, None] * columns).bfloat16())
    compressed = compact_kv.pack_fp4(((1 - factors[:, None]) * columns).bfloat16(), 16, True)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    raw_ids = torch.full((3, 128), -1, dtype=torch.long)
    cmp_ids = torch.full((3, 512), -1, dtype=torch.long)
    raw_ids[0, :2] = torch.tensor([253, 3])
    raw_ids[1, :3] = torch.tensor([258, 5, 7])
    cmp_ids[0, :1] = 256
    cmp_ids[1, :4] = torch.tensor([0, 132, 129, 258])
    q = torch.zeros((3, 64, width), dtype=torch.bfloat16, device="npu")
    assert mixed_mla.supports_mixed_mla(q, 64)
    attn = SimpleNamespace(attn_sink=torch.zeros(64, device=q.device), softmax_scale=width**-0.5, rope_head_dim=64)
    spans = (raw_ids >= 0).sum(-1, keepdim=True)
    output = graph_ops._sparse(
        q, raw.npu(), compressed.npu(), raw_ids.npu(), cmp_ids.npu(), attn, spans.npu(), lut.npu()
    )
    for i in range(q.size(0)):
        raw_values = compact_kv.unpack_window(raw[raw_ids[i, raw_ids[i] >= 0]], width)
        cmp_values = compact_kv.unpack_fp4(compressed[cmp_ids[i, cmp_ids[i] >= 0]], width, 16, True, lut)
        keys = torch.cat((raw_values, cmp_values)).float()
        # The Mixed reader uses one E8M0 scale per 64 non-RoPE channels.
        # Reconstruct its keys independently, including the second rounding of FP4 values.
        groups = keys[:, :448].unflatten(-1, (7, 64))
        scales = (groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil().exp2()
        decoded = (groups / scales).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scales
        keys[:, :448] = decoded.flatten(-2)
        expected = (keys.sum(0) / (keys.size(0) + 1)).bfloat16().expand(q.size(1), -1)
        torch.testing.assert_close(output[i].cpu(), expected, rtol=0.008, atol=0.004)
