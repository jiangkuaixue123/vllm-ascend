# SPDX-License-Identifier: Apache-2.0
"""Rotate the interleaved tail of each V4.1 token, in place."""

import torch


def rotate_rows(x, phases, width, inverse=False):
    """Leading phase dimensions identify tokens; an optional head dimension shares each phase."""
    if not x.numel():
        return x
    if x.device.type == "npu":
        from .native_ops import initialize_native_ops

        initialize_native_ops()
        # The Ascend 950 kernel broadcasts [tokens, 1, 1, width] over heads.
        # Pass the full head to preserve the untouched prefix and its row stride.
        dense = x.contiguous()
        tokens = phases.numel() // width
        bsnd = dense.view(tokens, 1, -1, x.size(-1))
        cos, sin = (p.repeat_interleave(2, dim=-1).reshape(tokens, 1, 1, width) for p in phases.unbind(-1))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            bsnd, cos, -sin if inverse else sin, "interleave", [x.size(-1) - width, x.size(-1)]
        )
        if dense is not x:
            x.copy_(dense)
    else:
        tail = x[..., -width:]
        pairs = torch.view_as_complex(tail.float().contiguous().unflatten(-1, (-1, 2)))
        phase = torch.view_as_complex(phases)
        if x.ndim == phases.ndim:
            phase = phase.unsqueeze(-2)
        tail.copy_(torch.view_as_real(pairs * (phase.conj() if inverse else phase)).flatten(-2))
    return x
