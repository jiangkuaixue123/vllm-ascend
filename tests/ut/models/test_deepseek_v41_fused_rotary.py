# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.models.deepseek_v41.rotary import rotate_rows


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("token_shape,heads,dim", [((6,), 1, 128), ((2, 5), 3, 512)])
def test_partial_rotary_preserves_prefix_and_uses_each_tokens_phase(token_shape, heads, dim, inverse):
    gen = torch.Generator().manual_seed(37)
    width = 64
    shape = (*token_shape, heads, dim) if heads > 1 else (*token_shape, dim)
    x = torch.randn(*shape, 2, generator=gen).bfloat16()[..., 1]
    initial = x.clone()
    angles = torch.randn(*token_shape, width // 2, generator=gen)
    phases = torch.stack((angles.cos(), angles.sin()), -1)
    z = torch.view_as_complex(initial[..., -width:].float().reshape(*shape[:-1], -1, 2))
    rotation = torch.polar(torch.ones_like(angles), -angles if inverse else angles)
    if heads > 1:
        rotation = rotation.unsqueeze(-2)
    expected = torch.view_as_real(z * rotation).flatten(-2).bfloat16()
    assert rotate_rows(x, phases, width, inverse) is x
    assert torch.equal(x[..., :-width], initial[..., :-width])
    torch.testing.assert_close(x[..., -width:], expected, rtol=0, atol=0)
