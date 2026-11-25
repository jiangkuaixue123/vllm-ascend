# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.hy_v4_flash.attention import AscendHYV4FlashAttention


@pytest.mark.parametrize("tokens", [1, 6, 24, 48, 96])
def test_packed_rotary_requires_paired_decode_rows(tokens):
    device = SimpleNamespace(type="npu")
    cache = SimpleNamespace(dtype=torch.bfloat16, ndim=2, shape=(131584, 64), is_contiguous=lambda: True, device=device)
    rotary = SimpleNamespace(
        cos_sin_cache=cache,
        head_size=64,
        rotary_dim=64,
        is_neox_style=False,
        mrope_interleaved=True,
        mrope_section=(12, 10, 10),
    )
    layer = SimpleNamespace(
        rotary_emb=rotary,
        absorption_weights=SimpleNamespace(key=object()),
        latent_dim=512,
        rope_dim=64,
        _is_decode_verification=lambda: True,
    )

    def operand(rows):
        return SimpleNamespace(device=device, shape=(rows, 64), dtype=torch.bfloat16, stride=lambda axis: 1)

    positions = SimpleNamespace(shape=(3, tokens))
    can_pack = AscendHYV4FlashAttention._can_pack_query
    assert can_pack(layer, positions, operand(tokens), operand(tokens))
    assert not can_pack(layer, positions, operand(tokens), operand(tokens + 1))
    layer._is_decode_verification = lambda: False
    assert not can_pack(layer, positions, operand(tokens), operand(tokens))
