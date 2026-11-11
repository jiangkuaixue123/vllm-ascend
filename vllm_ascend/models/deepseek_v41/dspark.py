"""DeepSeek V4.1 DSpark from the frozen official inference/model.py (MIT, DeepSeek).

Each DP worker owns replicated attention/heads and its own 1/EP routed experts.
Only accepted target hidden states can enter the request's projected main-KV cache.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn

from vllm_ascend.models.deepseek_v41 import backend as base
from vllm_ascend.models.deepseek_v41 import weights as checkpoint
from vllm_ascend.models.deepseek_v41.config import text_model_args
from vllm_ascend.models.deepseek_v41.eager_kernels import act_quant, sparse_attn


class DraftOutput(NamedTuple):
    token_ids: torch.Tensor  # [requests, block_size], excluding the target's seed
    logits: torch.Tensor  # [requests, block_size, vocab], including sequential Markov bias
    confidence: torch.Tensor  # [requests, block_size], raw learned confidence logits


def _rotate(x, phases, inverse=False):
    """Rotate a flat token batch whose rows can have different absolute positions."""
    pairs = x.float().unflatten(-1, (-1, 2))
    # A head axis exists for Q/O, but not the shared latent KV.
    if phases.ndim == pairs.ndim - 1:
        phases = phases.unsqueeze(-3)
    if x.device.type == "cpu":
        z = torch.view_as_complex(phases)
        rotated = torch.view_as_real(torch.view_as_complex(pairs) * (z.conj() if inverse else z))
    else:
        c, s = phases.unbind(-1)
        if inverse:
            s = -s
        re, im = pairs.unbind(-1)
        rotated = torch.stack((torch.addcmul(-(im * s), re, c), torch.addcmul(im * c, re, s)), -1)
    x.copy_(rotated.flatten(-2))


class DSparkAttention(base.Attention):
    def __init__(self, layer_id, args, shared):
        super().__init__(layer_id, args, shared)
        if self.compress_ratio:
            raise ValueError("DSpark layers use uncompressed sliding-window attention")
        self.pages = getattr(shared, "draft_pages", None)
        self.page_layer = layer_id - args.n_layers
        if self.pages is None:
            self.window_kv_cache = torch.zeros(
                args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.bfloat16
            )
        else:
            del self.window_kv_cache

    def commit(self, main, slots, positions):
        kv = self.kv_norm(self.wkv(main))
        _rotate(kv[..., -self.rope_head_dim :], self.freqs_cis[positions])
        if self.pages is not None:
            self.pages.write(self.page_layer, slots, kv)
        else:
            act_quant(kv, inplace=True)
            flattened = self.window_kv_cache.view(-1, self.head_dim)
            flattened.index_copy_(0, slots * self.window_kv_cache.size(1) + positions, kv)

    def forward(self, x, positions, slots):
        b, k = x.shape[:2]
        win = self.window_size
        offsets = torch.arange(k, device=x.device)
        phases = self.freqs_cis[positions[:, None] + 1 + offsets]
        q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(-1, (self.n_local_heads, self.head_dim))
        _rotate(q[..., -self.rope_head_dim :], phases)
        kv = self.kv_norm(self.wkv(x))
        _rotate(kv[..., -self.rope_head_dim :], phases)
        act_quant(kv, inplace=True)

        # Arrange main KVs in the same physical-ring order as HF, then append draft
        # block KVs immediately after the live prefix. This preserves softmax tile order.
        if self.pages is not None:
            history = self.pages.read(self.page_layer, slots)
        else:
            ring = torch.arange(win, device=x.device).view(1, -1)
            cycle = torch.div(positions[:, None], win, rounding_mode="floor") * win
            absolute = cycle + ring
            absolute = torch.where(absolute > positions[:, None], absolute - win, absolute).clamp_min(0)
            history = self.window_kv_cache[slots[:, None], absolute]
        all_kv = torch.cat((history, kv), dim=1)
        length = (positions[:, None] + 1).clamp_max(win)
        indices = torch.arange(win + k, device=x.device).expand(b, -1)
        indices = torch.where(indices < length, indices, torch.where(indices < length + k, win + indices - length, -1))
        o = sparse_attn(q, all_kv, self.attn_sink, indices[:, None].expand(-1, k, -1), self.softmax_scale)
        _rotate(o[..., -self.rope_head_dim :], phases, inverse=True)
        o = o.reshape(b, k, self.n_local_groups, -1)
        act_quant(o, inplace=True)
        grouped = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        return self.wo_b(torch.einsum("bsgd,grd->bsgr", o, grouped).flatten(2))


class MarkovHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = nn.Embedding(
            args.vocab_size, args.dspark_markov_rank, _weight=torch.empty(args.vocab_size, args.dspark_markov_rank)
        )
        self.head = base.ParallelHead(args.vocab_size, args.dspark_markov_rank)

    def forward(self, ids):
        embedded = self.embed(ids)
        return self.head(embedded, full_logits=True), embedded


class ConfidenceHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = base.Linear(args.dim + args.dspark_markov_rank, 1, dtype=torch.float32)

    def forward(self, hidden, markov):
        return self.proj(torch.cat((hidden, markov), dim=-1).float()).squeeze(-1)


class DSparkBlock(base.Block):
    attention_cls = DSparkAttention

    def __init__(self, stage, args, placement, shared):
        super().__init__(args.n_layers + stage, args, None, None, placement, shared)
        if stage == 0:
            self.main_proj = base.Linear(args.dim * len(args.dspark_target_layer_ids), args.dim)
            self.main_norm = base.RMSNorm(args.dim, args.norm_eps)
        if stage == args.n_mtp_layers - 1:
            self.norm = base.RMSNorm(args.dim, args.norm_eps)
            self.markov_head = MarkovHead(args)
            self.confidence_head = ConfidenceHead(args)


def greedy_block(last, target_head, hidden, seed_ids):
    """Condition each position's Markov bias on its previous chosen token."""
    logits = target_head(last.norm(hidden), full_logits=True)
    previous = seed_ids
    tokens, embeddings, adjusted = [], [], []
    for offset in range(logits.size(1)):
        bias, embed = last.markov_head(previous)
        row = logits[:, offset] + bias
        previous = row.argmax(-1)
        tokens.append(previous)
        embeddings.append(embed)
        adjusted.append(row)
    return DraftOutput(
        torch.stack(tokens, 1), torch.stack(adjusted, 1), last.confidence_head(hidden, torch.stack(embeddings, 1))
    )


class DSparkDraft(nn.Module):
    def __init__(
        self, text_config, target_backbone, *, device, max_requests, max_model_len, ep_group=None, draft_pages=None
    ):
        super().__init__()
        self.draft_pages = draft_pages
        self.max_model_len = max_model_len
        self.max_requests = max_requests
        self.vocab_size = text_config.vocab_size
        self.target_layers = tuple(text_config.dspark_target_layer_ids)
        self.k = text_config.dspark_block_size
        self.noise = text_config.dspark_noise_token_id
        args = text_model_args(text_config, max_batch_size=max_requests, max_seq_len=max_model_len + self.k)
        args.n_routed_experts = args.dspark_n_routed_experts
        args.n_activated_experts = args.dspark_n_activated_experts
        self.placement = base.ExpertPlacement(ep_group)
        self.hc_mult = args.hc_mult
        self.expert_count = args.n_routed_experts
        # Shared target embedding and LM head remain registered only on the target,
        # maintaining a one-to-one mapping between our parameters and raw mtp.* keys.
        self._target = weakref.ref(target_backbone)
        with torch.device(device), base.set_dtype(torch.bfloat16):
            runtime = base.SharedAttentionRuntime()
            runtime.draft_pages = draft_pages
            self.mtp = nn.ModuleList(DSparkBlock(i, args, self.placement, runtime) for i in range(args.n_mtp_layers))
        self.requires_grad_(False)

    @torch.inference_mode()
    def commit_target(self, aux, slots, positions):
        """Commit only accepted target input rows, including all prompt tokens."""
        if isinstance(aux, (tuple, list)):
            aux = torch.cat(aux, dim=-1)
        slots, positions = slots.long(), positions.long()
        if aux.ndim != 2 or aux.shape[0] != slots.numel() or slots.shape != positions.shape or slots.ndim != 1:
            raise ValueError("expected one slot and absolute target position per auxiliary row")
        if not slots.numel():
            return
        pages = getattr(self, "draft_pages", None)
        limit = self.max_requests if pages is None else pages.num_slots
        if not ((slots >= 0) & (slots < limit) & (positions >= 0) & (positions < self.max_model_len)).all():
            raise ValueError("target commit outside the request or position range")
        x = self.mtp[0].main_norm(self.mtp[0].main_proj(aux))
        for block in self.mtp:
            block.attn.commit(x, slots, positions)

    @torch.inference_mode()
    def propose(self, seed_ids, slots, positions):
        """Greedy five-token block from the target's next seed and last committed position."""
        pages = getattr(self, "draft_pages", None)
        expected = seed_ids.shape if pages is None else (*seed_ids.shape, pages.window)
        if seed_ids.ndim != 1 or not seed_ids.numel() or slots.shape != expected or positions.shape != seed_ids.shape:
            raise ValueError("Expected one seed, position and KV address row per request")
        if pages is None:
            valid_slots = (slots >= 0) & (slots < self.max_requests)
        else:
            valid_slots = ((slots >= 0) & (slots < pages.num_slots)).all(dim=1)
        valid = (
            valid_slots
            & (positions >= 0)
            & (positions + self.k + 1 < self.max_model_len)
            & (seed_ids >= 0)
            & (seed_ids < self.vocab_size)
        )
        if not valid.all():
            raise ValueError("a seed, request slot, or complete draft block is out of range")
        return self(seed_ids, slots, positions)

    def forward(self, seed_ids, slots, positions):
        """Tensor path for fixed-width graph inputs checked by the request scheduler."""
        positions, slots = positions.long(), slots.long()
        target = self._target()
        if target is None:
            raise RuntimeError("the target model must outlive its shared DSpark heads")
        draft_input = torch.cat((seed_ids[:, None], seed_ids.new_full((len(seed_ids), self.k - 1), self.noise)), dim=1)
        h = target.embed(draft_input).unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        mix = base.make_identity_pre_mix(h, self.hc_mult)
        self.placement.step_width = len(seed_ids) * self.k
        for block in self.mtp:
            h, mix = block(h, positions, mix, None, slots)
        last = self.mtp[-1]
        h = last.hc_pre(h, mix)
        return greedy_block(last, target.head, h, seed_ids)

    def load_weights(self, directory, *, receipt=None):
        """Validate all local parameter layouts before copying any original shard rows."""
        refs, header_hashes = checkpoint.read_checkpoint(directory)
        source = {name: ref for name, ref in refs.items() if name.startswith("mtp.")}
        params = dict(self.named_parameters())
        local, conversions, actions = {}, {}, Counter()
        for name, ref in source.items():
            if name.endswith(".ffn.gate.bias_vl"):
                actions["vision_gate_only"] += 1
            elif (
                checkpoint.EXPERT.match(name)
                and checkpoint.expert_owner(name, self.expert_count, self.placement.size) != self.placement.rank
            ):
                actions["other_ep_rank"] += 1
            elif name.endswith(".attn.wo_a.scale"):
                conversions[name.removesuffix(".scale") + ".weight"] = ref
                actions["wo_a_scale"] += 1
            else:
                local[name] = ref
                actions["loaded"] += 1
        if local.keys() != params.keys():
            raise ValueError(f"DSpark checkpoint/architecture mismatch: {sorted(local.keys() ^ params.keys())[:8]}")
        for name, ref in local.items():
            p = params[name]
            if name in conversions:
                s = conversions[name]
                if (
                    ref.dtype != "F8_E4M3"
                    or s.dtype != "F8_E8M0"
                    or p.dtype != torch.bfloat16
                    or tuple(p.shape) != ref.shape
                    or ref.shape != tuple(v * 32 for v in s.shape)
                ):
                    raise ValueError(f"invalid converted projection {name}")
            elif name.endswith((".markov_head.head.weight", ".confidence_head.proj.weight")):
                if ref.dtype != "BF16" or p.dtype != torch.float32 or tuple(p.shape) != ref.shape:
                    raise ValueError(f"invalid BF16 to FP32 head {name}")
            else:
                checkpoint._parameter_contract(name, ref, p)
        with ExitStack() as stack, torch.no_grad():
            files = {
                p: stack.enter_context(p.open("rb", buffering=0))
                for p in {r.file for r in (*local.values(), *conversions.values())}
            }
            for name, ref in local.items():
                checkpoint._copy_tensor(ref, params[name], files, conversions.get(name))
        index = Path(directory) / checkpoint.INDEX_NAME
        summary = dict(
            index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None,
            mtp_shard_headers={ref.file.name: header_hashes[ref.file.name] for ref in source.values()},
            raw_mtp_keys=len(source),
            actions=dict(actions),
            ep_rank=self.placement.rank,
            ep_size=self.placement.size,
            tensors=len(local),
            parameter_bytes=sum(p.numel() * p.element_size() for p in params.values()),
            target_layers=self.target_layers,
            block_size=self.k,
            converted=sorted(conversions),
            shared_target_parameters=["embed.weight", "head.weight"],
        )
        if receipt is not None:
            Path(receipt).write_text(json.dumps(summary, indent=2) + "\n")
        return summary
