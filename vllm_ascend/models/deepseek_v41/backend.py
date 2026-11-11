"""DP-local text backbone with an explicit EP device group.

Attention, indexer, Engram gate and hyper-connections adapted from DeepSeek's
frozen inference/model.py. The HF model and checkpoint use the following license.

MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from __future__ import annotations

import math
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .config import text_model_args
from .ngram import EngramLayout, NgramHashState
from .eager_kernels import act_quant, compact_linear, fp4_act_quant, hc_split_sinkhorn, sparse_attn
from .ep import ExpertPlacement

fp8_block_size = 32
fp4_block_size = 32
scale_fmt = "ue8m0"
scale_dtype = torch.float8_e8m0fnu


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch's default dtype, restoring it even if the body raises."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


class Linear(nn.Module):
    """A raw checkpoint matrix plus its scale. FP4 stays byte-packed."""
    def __init__(self, in_features, out_features, dtype=torch.float8_e4m3fn):
        super().__init__()
        self.packed_fp4 = dtype == torch.float4_e2m1fn_x2
        if self.packed_fp4:
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.uint8), requires_grad=False)
            self.scale = nn.Parameter(torch.empty(out_features, in_features // 32, dtype=torch.uint8), requires_grad=False)
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
            self.scale = nn.Parameter(torch.empty((out_features + 31) // 32, in_features // 32, dtype=torch.uint8), requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
            self.register_parameter("scale", None)

    def forward(self, x):
        if self.scale is None:
            return F.linear(x, self.weight)
        return compact_linear(x, self.weight, self.scale, packed_fp4=self.packed_fp4)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared
    value. The gate is a normalized dot product of stream against key.
    """

    def __init__(self, args: SimpleNamespace, layer_id: int, layout, make_embedding):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6

        self.embed = make_embedding(layer_id, layout.num_embeddings[self.layer_hash_index], layout.head_dim)
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = Linear(n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1))
        self.eps = args.norm_eps
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L], False shuts
        the gate so those positions pass through untouched."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h, eps = x.float(), self.eps
        # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        magnitude = dot.abs().clamp_min(self.clamp_value).sqrt()
        gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, one row per position.

    With original_seq_len > 0 this applies YaRN: dimensions whose wavelength already fits inside the
    training context keep their frequency, those far beyond it are divided by `factor`, and the
    `beta_fast`..`beta_slow` band in between is faded across with a linear ramp.
    """
    with torch.device("cpu"):
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len > 0:
            # the dim whose wavelength completes `rotations` turns over the training context
            def corrected_dim(rotations):
                return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(corrected_dim(beta_fast)), 0)
            high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
            smooth = 1 - ramp
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        freqs = torch.outer(torch.arange(seqlen), freqs)
        phases = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(phases).to(device)

def apply_rotary_emb(x, freqs_cis, inverse=False):
    pairs = x.float().unflatten(-1, (-1, 2))
    if x.ndim == 4:
        freqs_cis = freqs_cis.unsqueeze(1)
    if x.device.type == "cpu":
        phase = torch.view_as_complex(freqs_cis)
        rotated = torch.view_as_real(torch.view_as_complex(pairs) * (phase.conj() if inverse else phase))
    else:
        # Keep the table and the multiply on device, using real FP32 operations
        # for the complex rotation. Match the fused product before BF16 rounding.
        cos, sin = freqs_cis.unbind(-1)
        if inverse:
            sin = -sin
        real, imag = pairs.unbind(-1)
        rot_r = torch.addcmul(-(imag * sin), real, cos)
        rot_i = torch.addcmul(imag * cos, real, sin)
        rotated = torch.stack((rot_r, rot_i), dim=-1)
    x.copy_(rotated.flatten(-2))
    return x


@lru_cache(16)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int, device):
    """Which sliding-window cache slots each query attends to; -1 marks a slot holding nothing.

    The cache is a ring of `window_size` slots. Prefill needs one row per query, each seeing its own
    causal window. A decode step has a single query that sees the whole ring, listed oldest first.
    Order within a row does not matter to `sparse_attn`, which handles every slot independently.
    """
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
        idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size, device=device), torch.arange(oldest, device=device)])
        idxs = torch.where(idxs > start_pos, -1, idxs)  # ring still filling
    # sparse_attn needs real [b, m, topk] int32 memory, hence the materializing expand
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate.

    Returns the latent before RoPE, or None while a group is still filling up -- so during decode it
    only yields every `compress_ratio` steps, holding the partial group in `kv_state`/`score_state`.
    Pre-RoPE is deliberate: the indexer needs the unrotated form, so Attention rotates afterwards.
    """

    def __init__(self, args: SimpleNamespace, layer_id: int):
        super().__init__()
        compress_ratio = args.compress_ratios[layer_id]
        head_dim = args.head_dim
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, args.norm_eps)
        # ratio 1 is a plain projection, so it stays in the checkpoint's bf16; the softmax pooling
        # above ratio 1 runs in fp32, so those weights are promoted to fp32 to match
        self.wkv = Linear(args.dim, head_dim, dtype=torch.float32 if compress_ratio > 1 else torch.bfloat16)
        if compress_ratio > 1:
            self.wgate = Linear(args.dim, head_dim, dtype=torch.float32)
            # tail of an incomplete group, carried across decode steps
            self.kv_state: torch.Tensor
            self.score_state: torch.Tensor
            state_shape = (args.max_batch_size, compress_ratio, head_dim)
            self.register_buffer("kv_state", torch.zeros(state_shape, dtype=torch.float32), persistent=False)
            self.register_buffer(
                "score_state", torch.full(state_shape, -torch.inf, dtype=torch.float32), persistent=False
            )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                kv, self.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
                score, self.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # one token per step: fill a slot, and pool only when the group just completed
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))


class Indexer(torch.nn.Module):
    """Keeps the `index_topk` best compressed positions per query.

    A small side attention: fp4 query heads against one shared key per compressed position, scores
    rectified then combined by `weights_proj`. With a candidate source this is the second of two
    levels; `select_candidate_blocks` is the first.
    """

    def __init__(self, args: SimpleNamespace, layer_id: int, shared):
        super().__init__()
        # the index keys are derived from the compressor's latent, so only a layer that compresses
        # its own KV can produce them; every other indexer reads them from that layer's cache
        self.shared = shared
        self.owns_k = layer_id in args.kv_source_layers
        self.compress_ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.index_head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.freqs_cis: torch.Tensor | None = None
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.index_head_dim, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)
            self.k_cache: torch.Tensor
            self.register_buffer(
                "k_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len // self.compress_ratio, args.index_head_dim),
                persistent=False,
            )

    def forward(self, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor, start_pos: int, offset: int):
        """`latent` is this layer's RoPE-free compressed latent, None when this layer does not
        compress or when its current group is still incomplete. An index-key owner turns it into
        index keys here, which has to happen before Attention overwrites that same storage with
        the RoPE'd, quantized values."""
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        # Append index keys when a compression group completes.
        if self.owns_k and latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = self.k_norm(self.wk(latent))
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant(k, fp4_block_size, True)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        if self.owns_k:
            self.shared.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        fp4_act_quant(q, fp4_block_size, True)

        index_k = self.shared.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        # how many compressed positions each query can see: a block becomes visible once the query
        # has passed its last token. One query per decode step, so there it is just a number.
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            self.shared.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            # level two: score with our own weights, but only inside the source's candidate blocks
            index_score = index_score.masked_fill(~self.shared.candidates, -torch.inf)

        # top-k by score, re-sorted into position order; unreachable -> -1, rest shifted by offset
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query.

    `logits` is [..., n_positions] with positions the query cannot reach already at -inf, which is
    what makes a block score of -inf mean "not reachable yet". `compress_lens` is a plain int during
    decode, or broadcasts against logits' leading dims during prefill. Returns a bool mask shaped
    like `logits`, so the layers consuming it just mask and never think about blocks again.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so pin it in: it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Attention(nn.Module):
    """Latent attention over two KV sources at once, concatenated into one `sparse_attn` call: a
    sliding window of raw KV, plus -- when compress_ratio > 0 -- `index_topk` compressed positions
    reaching further back. Q and the output projection are both low-rank, the latter grouped.

    compress_ratio > 0 does not mean the layer compresses its own KV: only kv_source_layers do,
    the rest read that same cache.
    """

    def __init__(self, layer_id: int, args: SimpleNamespace, shared):
        super().__init__()
        self.shared = shared
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        is_backbone = layer_id < args.n_layers
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor: Compressor | None = None
        self.indexer: Indexer | None = None
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id, shared)

        self.window_kv_cache: torch.Tensor
        self.register_buffer(
            "window_kv_cache",
            torch.zeros(args.max_batch_size, args.window_size, self.head_dim),
            persistent=False,
        )
        if self.is_kv_source:
            self.compress_kv_cache: torch.Tensor
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len // self.compress_ratio,
                    self.head_dim,
                ),
                persistent=False,
            )
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
            self.window_kv_cache.device,
        )
        self.freqs_cis: torch.Tensor
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _window_kv(self, x, freqs_cis, start_pos):
        """This layer's sliding-window K and the window positions every query may attend to. The K
        stays fp8, quantized over the whole post-RoPE vector, RoPE tail included."""
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)
        if start_pos == 0:  # prefill: attend over this chunk, seeding the ring buffer for decode
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            window_kv = kv
        else:  # decode: one token into the ring buffer, attend over the whole window
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device)

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_len):
        """Which compressed positions each query attends to. Index sources run their own indexer;
        the layers in between reuse the result their source published."""
        if not self.is_index_source:
            return self.shared.topk_idxs

        bsz, seqlen, _ = x.size()
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            idxs = self.indexer(x, qr, latent, start_pos, offset)
        self.shared.topk_idxs = idxs
        return idxs

    def _compress_kv(self, x, qr, start_pos, offset):
        """The shared compressed KV and the compressed positions every query may attend to. This
        layer compresses its own KV only when it is a source; otherwise it just reads the cache."""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            self.shared.compress_kv = self.compress_kv_cache
        # the indexer needs the latent before RoPE, so it runs before the cache is written
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
        if latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs)
            # Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
            fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
        # read after the write, so this does not depend on the slice aliasing the cache
        return self.shared.compress_kv[:bsz, :compress_len], idxs

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # wo_a is block-diagonal over groups (each projects only its own heads), hence einsum not
        # Linear. convert.py dequantizes it to bf16; quantize/dequantize o before einsum.
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """MoE gating. The correction bias steers expert selection only; the routing weights come from the
    unbiased scores. Image-span tokens use a separate bias (training `noaux_tc_for_vl`)."""

    def __init__(self, layer_id: int, args: SimpleNamespace):
        super().__init__()
        n_routed_experts, n_activated_experts = args.n_routed_experts, args.n_activated_experts
        self.dim = args.dim
        self.topk = n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
        self.bias_vl = None

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [n, dim]; image_mask: [n] bool, True for tokens inside an image span."""
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # the bias picks experts but does not scale them: weights come from the raw scores
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20  # not norm_eps, matches training
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """One SwiGLU FFN. The clamps come straight from training, where they keep fp8/fp4 activations in
    range: the up branch is clamped on both sides, the gate branch only from above."""

    def __init__(self, dim: int, inter_dim: int, dtype=torch.float8_e4m3fn, swiglu_limit=0.0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    def __init__(self, layer_id, args, placement):
        super().__init__()
        if args.n_routed_experts % placement.size or args.n_shared_experts != 1:
            raise ValueError("Unsupported expert placement")
        count = args.n_routed_experts // placement.size
        self.placement = placement
        self.experts_start_idx = placement.rank * count
        self.experts_end_idx = self.experts_start_idx + count
        self.dim = args.dim
        self.gate = Gate(layer_id, args)
        self.experts = nn.ModuleList(
            Expert(args.dim, args.moe_inter_dim, dtype=torch.float4_e2m1fn_x2, swiglu_limit=args.swiglu_limit)
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(args.n_routed_experts)
        )
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

    def forward(self, x, image_mask=None):
        tokens = x.reshape(-1, self.dim)
        weights, indices = self.gate(tokens, image_mask) if tokens.size(0) else (
            tokens.new_empty((0, self.gate.topk), dtype=torch.float32),
            tokens.new_empty((0, self.gate.topk), dtype=torch.int64),
        )
        y = self.placement.routed(self.experts, self.experts_start_idx, self.experts_end_idx,
                                  tokens, indices, weights, self.placement.step_width)
        if tokens.size(0):
            y += self.shared_experts(tokens)
        return y.to(x.dtype).reshape_as(x)


class Block(nn.Module):
    """A block whose residual stream is `hc_mult` parallel copies (Hyper-Connections).

    Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer input) and
    `hc_post` (expand back out, mixing the residual in through `comb`). `hc_mixes` derives all three
    coefficient sets from the stream itself, `comb` made doubly stochastic by Sinkhorn.

    The coefficients a sublayer computes are used by the *next* one -- see `forward`.
    """

    attention_cls = Attention

    def __init__(
        self,
        layer_id: int,
        args: SimpleNamespace,
        engram_layout, make_embedding, placement, shared,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = self.attention_cls(layer_id, args, shared)
        self.ffn = MoE(layer_id, args, placement)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(args, layer_id, engram_layout, make_embedding)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc]. Returns the
        pre / post / comb coefficients, split out of one projection of the flattened stream."""
        # normalized over the whole flattened hc*d stream, one statistic per token
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
        """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
        x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]"""
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        *attn_args,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`pre_mix` collapses the hc_mult copies down to one input for this block's attention. Each
        sub-block's own `hc_mixes` produces the mix for the *next* one, so attention uses what the
        previous layer's FFN produced and the FFN uses what this attention produced.

        image_mask: [b, s] bool, True inside image spans (selects the VL routing bias)."""
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, *attn_args)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x, image_mask)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = vocab_size
        # bf16 in the checkpoint, kept as fp32 here so the logits come out in fp32 directly
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, full_logits=False):
        """x: [b, s, d]. Generation only needs the last position, so that is the default."""
        if not full_logits:
            x = x[:, -1]
        logits = F.linear(x.float(), self.weight)
        return logits


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """initial one-hot mix"""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing. Layers run in order and
    every source writes before its consumers read, so one slot each is enough and nothing needs
    resetting between forwards. Sources: compress_kv and index_k from kv_source_layers, topk_idxs
    from index_source_layers, candidates from candidate_source_layer."""

    def __init__(self):
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None


class TextBackbone(nn.Module):
    """One active sequence per DP rank. A prefill at zero starts new request state."""
    def __init__(self, text_config, tokenizer, *, device, max_seq_len, ep_group,
                 engram_embedding_factory=None, paged_cache=False):
        super().__init__()
        # Physical pages own the target KV, compressor, index and n-gram history
        # in the paged model. Only the serial eager model uses a private row.
        args = text_model_args(text_config, max_batch_size=0 if paged_cache else 1, max_seq_len=max_seq_len)
        self.device = torch.device(device)
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.position = 0
        self.placement = ExpertPlacement(ep_group)
        self.engram_hash = None
        layout = None
        if args.engram_layer_ids:
            layout = EngramLayout.from_args(args)
            with torch.device("cpu"):
                self.engram_hash = NgramHashState(args, layout, tokenizer)
            if engram_embedding_factory is None:
                from .engram import HostEngramEmbedding
                def engram_embedding_factory(layer, rows, dim):
                    return HostEngramEmbedding(rows, dim)

        self.shared = SharedAttentionRuntime()
        with torch.device(self.device), set_dtype(torch.bfloat16):
            self.embed = nn.Embedding(args.vocab_size, args.dim, _weight=torch.empty(args.vocab_size, args.dim))
            self.layers = nn.ModuleList(Block(i, args, layout, engram_embedding_factory, self.placement, self.shared)
                                        for i in range(args.n_layers))
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.head = ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        self.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, input_ids, start_pos=0):
        if input_ids.ndim != 2 or input_ids.size(0) != 1 or not input_ids.size(1):
            raise ValueError("Expected one nonempty sequence per DP rank")
        length = input_ids.size(1)
        if start_pos < 0 or start_pos + length > self.max_seq_len:
            raise ValueError("Position outside this rank's eager cache")
        if start_pos and (start_pos != self.position or length != 1):
            raise ValueError("Decode must continue this rank's single active sequence by one token")
        hashes = None
        if self.engram_hash is not None:
            with torch.device("cpu"):
                hashes = self.engram_hash(input_ids.to("cpu"), start_pos)
        self.placement.step_width = self.placement.token_width(length, self.device)
        h = self.embed(input_ids).unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        pre = make_identity_pre_mix(h, self.hc_mult)
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
            h, pre = layer(h, start_pos, pre, None)
        self.position = start_pos + length
        return self.norm(layer.hc_pre(h, pre))

    @torch.inference_mode()
    def idle_forward(self):
        """Serve peers' routed experts without touching this rank's live KV or n-gram cache."""
        self.placement.step_width = self.placement.token_width(0, self.device)
        empty = self.embed.weight.new_empty((0, self.dim))
        for layer in self.layers:
            layer.ffn(empty)
        return empty
