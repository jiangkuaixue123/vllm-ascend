# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.compact_kv import pack_fp4, pack_window, unpack_fp4, unpack_window
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
