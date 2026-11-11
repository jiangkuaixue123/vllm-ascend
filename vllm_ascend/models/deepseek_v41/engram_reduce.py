# SPDX-License-Identifier: Apache-2.0
"""Fixed-width FP32 reductions for the V4.1 Engram gate on Ascend A5."""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk(H, K, W, BLOCK: tl.constexpr):
    channel = tl.arange(0, BLOCK)
    h = tl.load(H + channel).to(tl.float32)
    k = tl.load(K + channel).to(tl.float32)
    w = tl.load(W + channel)
    return tl.sum(h * h, 0), tl.sum(k * k, 0), tl.sum((h * w) * k, 0)


@triton.jit
def _two_chunks(H, K, W, BLOCK: tl.constexpr):
    h0, k0, d0 = _chunk(H, K, W, BLOCK)
    h1, k1, d1 = _chunk(H + BLOCK, K + BLOCK, W + BLOCK, BLOCK)
    return h0 + h1, k0 + k1, d0 + d1


@triton.jit
def _engram_stats(
    H,
    K,
    W,
    Out,
    H_ROW: tl.constexpr,
    H_COPY: tl.constexpr,
    K_ROW: tl.constexpr,
    K_COPY: tl.constexpr,
    W_COPY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program owns all channels of one (token, HC copy). Its FP32 reduction
    # never crosses program boundaries and has no dependence on batch geometry.
    row = tl.program_id(0).to(tl.int64)
    copy = tl.program_id(1)
    h = H + row * H_ROW + copy * H_COPY
    k = K + row * K_ROW + copy * K_COPY
    w = W + copy * W_COPY
    ha, ka, da = _two_chunks(h, k, w, BLOCK)
    hb, kb, db = _two_chunks(h + 2 * BLOCK, k + 2 * BLOCK, w + 2 * BLOCK, BLOCK)
    hc, kc, dc = _chunk(h + 4 * BLOCK, k + 4 * BLOCK, w + 4 * BLOCK, BLOCK)
    out = Out + (row * 4 + copy) * 3
    tl.store(out, ((ha + hb) + hc) * (1.0 / 5120))
    tl.store(out + 1, ((ka + kb) + kc) * (1.0 / 5120))
    tl.store(out + 2, (da + db) + dc)


def gate_statistics(hidden, key, weight):
    """Return the two square means and the weighted dot sum per token and HC copy."""
    # The key shares a [tokens, 5*5120] WKV allocation with one value per token.
    # Coalescing B,T preserves its 5*5120 token stride and the residual's optional broadcast HC.
    h = hidden.reshape(-1, 4, 5120)
    k = key.reshape(-1, 4, 5120)
    out = torch.empty((*hidden.shape[:-1], 3), dtype=torch.float32, device=hidden.device)
    if h.shape[0]:
        _engram_stats[(h.shape[0], 4)](
            h, k, weight, out, h.stride(0), h.stride(1), k.stride(0), k.stride(1), weight.stride(0), 1024
        )
    return out.unbind(-1)
