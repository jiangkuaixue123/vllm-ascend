# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Attention
from vllm_ascend.models.deepseek_v41.attention_ops import apply_rotary_emb_real


@pytest.mark.parametrize("shape,rope_dim", [((1, 7, 128), 64), ((2, 7, 3, 96), 32), ((1, 0, 512), 64)])
@pytest.mark.parametrize("inverse", [False, True])
def test_rotate_reference_fallback(shape, rope_dim, inverse):
    # A strided tail must update its backing storage and preserve adjacent elements.
    rows, width = shape[1], shape[-1]
    backing = torch.randn(*shape[:-1], 2 * width, device="cpu").bfloat16()
    expected = backing.clone()
    positions = torch.arange(rows, device="cpu") * 511
    freq = 10000.0 ** (-torch.arange(0, rope_dim, 2, device="cpu").float() / rope_dim)
    phases = positions.float()[:, None] * freq
    apply_rotary_emb_real(expected[..., ::2][..., -rope_dim:], phases.cos(), phases.sin(), inverse)
    x = backing[..., ::2]
    result = DeepseekV41Attention.rotate(SimpleNamespace(frequencies=freq, rope_dim=rope_dim), x, positions, inverse)
    assert result is x
    assert torch.equal(backing, expected)


def test_rotate_preserves_position_shape_check():
    x = torch.ones(1, 6, 128, dtype=torch.bfloat16, device="cpu")
    layer = SimpleNamespace(frequencies=torch.ones(32, device="cpu"), rope_dim=64)
    with pytest.raises(ValueError, match="cosine and sine"):
        DeepseekV41Attention.rotate(layer, x, torch.arange(5, device="cpu"))
