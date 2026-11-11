# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.compact_kv import (
    CacheEncoding,
    pack_fp4,
    pack_window,
    store_compact_kv,
    store_mixed_kv,
    unpack_fp4,
    unpack_window,
)
from vllm_ascend.models.deepseek_v41.eager_kernels import act_quant
from vllm_ascend.models.deepseek_v41.graph_ops import LayerPages, _fp4


@pytest.mark.parametrize("width,block,e4scale", [(128, 32, False), (512, 16, True)])
def test_fp4_page_roundtrip_keeps_quantized_bf16_bits(width, block, e4scale):
    lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0))
    x = torch.randn((8, width), generator=torch.Generator().manual_seed(47)).bfloat16()
    x[0].mul_(2.0**-21)
    x[1, :2] = torch.tensor([0.0, -0.0])
    expected = x.clone()
    _fp4(expected, lut, block, e4scale)
    packed = pack_fp4(x, block, e4scale)
    assert packed.dtype == torch.uint8
    assert packed.shape == (8, width // 2 + width // block)
    assert torch.equal(unpack_fp4(packed, width, block, e4scale, lut).view(torch.int16), expected.view(torch.int16))


def test_raw_page_and_planner_match_physical_storage():
    x = torch.randn((7, 512), generator=torch.Generator().manual_seed(95)).bfloat16()
    x[:2].mul_(2.0**-22)
    expected = x.clone()
    act_quant(expected, 32, inplace=True)
    raw = pack_window(x)
    assert raw.shape == (7, 528)
    assert torch.equal(unpack_window(raw, 512).view(torch.int16), expected.view(torch.int16))
    config = SimpleNamespace(
        head_dim=512, is_kv_source=True, compress_ratio=2, indexer=SimpleNamespace(index_head_dim=128)
    )
    state = SimpleNamespace(
        history=SimpleNamespace(num_slots=128, block_size=128, tail_groups=1), max_tokens=0, device="cpu"
    )
    pages = LayerPages(config, state)
    assert sum(buf.numel() * buf.element_size() for buf in vars(pages).values()) == LayerPages.bytes_per_block(
        config, 128
    )


@pytest.mark.parametrize(
    "encoding,dim",
    [(CacheEncoding.WINDOW_MXFP8, 512), (CacheEncoding.COMPRESSED_E4_FP4, 512), (CacheEncoding.INDEX_MXFP4, 128)],
)
def test_compact_store_uses_row_slots_and_preserves_guards(encoding, dim):
    x = torch.randn((6, dim), generator=torch.Generator().manual_seed(79)).bfloat16()
    if encoding == CacheEncoding.WINDOW_MXFP8:
        packed = pack_window(x)
    else:
        e4 = encoding == CacheEncoding.COMPRESSED_E4_FP4
        packed = pack_fp4(x, 16 if e4 else 32, e4)
    slots = torch.tensor([9, 3, -1, 7, 14, 1])
    cache = torch.full((12, packed.size(1)), 0xA5, dtype=torch.uint8)
    expected = cache.clone()
    expected[slots[[0, 1, 3, 5]]] = packed[[0, 1, 3, 5]]
    store_compact_kv(cache, x, slots, encoding)
    assert torch.equal(cache, expected)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_mixed_store_uses_versioned_writer_and_original_storage(monkeypatch, dtype):
    cache = torch.empty((2, 128, 1, 584), dtype=dtype)
    x = torch.arange(512 * 3).to(torch.bfloat16).view(512, 3).T
    slots = torch.tensor([0, 19, 127, 17, 128, 37])[::2]
    calls = []

    def writer(bank, values, indices, mode):
        assert bank.shape == (256, 584) and bank.dtype == torch.uint8
        assert bank.data_ptr() == cache.data_ptr() and bank.is_contiguous()
        assert values.is_contiguous() and indices.is_contiguous()
        assert torch.equal(values.view(torch.int16), x.contiguous().view(torch.int16))
        assert torch.equal(indices, slots)
        calls.append(mode)

    monkeypatch.setattr(torch.ops._C_ascend, "kv_compress_epilog_v2", writer, raising=False)
    store_mixed_kv(cache, x, slots)
    assert calls == [3]
    with pytest.raises(ValueError, match="128"):
        store_mixed_kv(cache[:, :64].contiguous(), x, slots)
