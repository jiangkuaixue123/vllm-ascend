# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny end-to-end checks of the trained V4.1 draft and its packed cache."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from vllm_ascend.models.deepseek_v41 import dspark, moe
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Metadata, RequestSpan
from vllm_ascend.models.deepseek_v41.attention_ops import fp8_quant_dequant
from vllm_ascend.models.deepseek_v41.core import hc_mixes


@pytest.mark.parametrize("length, expected", [(0, []), (3, [0, 1, 2]), (4, [0, 1, 2, 3]), (6, [4, 5, 2, 3])])
def test_official_ring_slot_order(length, expected):
    assert dspark.ring_positions(length, 4, device="cpu").tolist() == expected


def test_every_draft_query_sees_later_noise_and_accepted_context():
    q = torch.zeros(5, 1, 32, dtype=torch.bfloat16)
    context = torch.ones(3, 32, dtype=torch.bfloat16)
    query = torch.zeros(5, 32, dtype=torch.bfloat16)
    query[-1] = 6
    # All logits are zero. Three context + five query + one sink in the denominator.
    output = dspark.block_attention(q, context, query, torch.zeros(1), 1)
    torch.testing.assert_close(output, torch.ones_like(q), rtol=0, atol=0)


def test_draft_builder_uses_rejection_adjusted_lengths_instead_of_target_upper_bound():
    common = SimpleNamespace(
        num_actual_tokens=10,
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 5, 10]),
        seq_lens=torch.tensor([11, 8]),
        seq_lens_cpu_upper_bound=torch.tensor([9, 4]),
        block_table_tensor=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        slot_mapping=torch.arange(10),
    )
    builder = object.__new__(dspark.DSparkMetadataBuilder)
    metadata = builder.build_for_drafting(common)
    assert [span.start_pos for span in metadata.requests] == [6, 3]
    assert metadata.decode is None


class TinyLinear(nn.Module):
    def __init__(self, input_size, output_size, return_bias=True, params_dtype=None, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=params_dtype or torch.bfloat16))
        self.return_bias = return_bias

    def forward(self, x):
        value = F.linear(x, self.weight)
        return (value, None) if self.return_bias else value


class TinyVocab(nn.Embedding):
    def __init__(self, vocab_size, width, params_dtype=None, **kwargs):
        super().__init__(vocab_size, width, dtype=params_dtype or torch.bfloat16)


class TinyLogits(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

    def forward(self, head, states):
        return F.linear(states.float(), head.weight.float())


@pytest.fixture
def draft(monkeypatch, tmp_path):
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(moe, "tensor_model_parallel_all_reduce", lambda x: x)
    for module in (dspark, moe):
        for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
            monkeypatch.setattr(module, name, TinyLinear)
    monkeypatch.setattr(dspark, "VocabParallelEmbedding", TinyVocab)
    monkeypatch.setattr(dspark, "ParallelLMHead", TinyVocab)
    monkeypatch.setattr(dspark, "LogitsProcessor", TinyLogits)
    group = SimpleNamespace(world_size=1, rank_in_group=0, all_reduce=lambda x: x)
    comm = moe.DeepseekV41MoECommunication(group, group)
    expert_cls = dspark.DSparkMoE
    monkeypatch.setattr(dspark, "DSparkMoE", lambda c, q, p: expert_cls(c, q, p, communication=comm))
    config = SimpleNamespace(
        hidden_size=32,
        vocab_size=16,
        head_dim=32,
        num_attention_heads=2,
        o_groups=1,
        index_head_dim=32,
        q_lora_rank=32,
        o_lora_rank=32,
        qk_rope_head_dim=4,
        sliding_window=4,
        compress_rope_theta=10000,
        rope_theta=10000,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        dspark_markov_rank=4,
        dspark_block_size=5,
        dspark_noise_token_id=15,
        dspark_target_layer_ids=[0, 1, 2],
        num_hidden_layers=3,
        compress_ratios=[0] * 6,
        dspark_n_routed_experts=2,
        dspark_num_experts_per_tok=2,
        n_shared_experts=1,
        hidden_act="silu",
        moe_intermediate_size=64,
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, model=str(tmp_path)),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        quant_config=None,
        cache_config=SimpleNamespace(block_size=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = dspark.DeepseekV41DSparkForCausalLM(vllm_config=cfg)
    finally:
        torch.set_default_dtype(previous)
    generator = torch.Generator().manual_seed(678)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "norm.weight" in name:
                value.fill_(1)
            else:
                value.copy_(torch.randn(value.shape, generator=generator) * 0.15)
    for stage in model.mtp:
        stage.attn.cache.kv_cache = torch.zeros(2, 8, 4, 1, stage.attn.cache.width)
    return model


def run_block(model, ids, lengths, tables):
    spans = [RequestSpan(i * 5, i * 5 + 5, length + 5) for i, length in enumerate(lengths)]
    md = DeepseekV41Metadata(spans, torch.tensor(tables), torch.empty(0, dtype=torch.long), len(ids))
    pos = torch.cat([torch.arange(length, length + 5) for length in lengths])
    for stage in model.mtp:
        stage.attn.cache.metadata = lambda: md
    return model(torch.tensor(ids), pos)


@torch.no_grad()
def test_three_stages_paged_context_and_markov_outputs_are_request_local(draft):
    generator = torch.Generator().manual_seed(827)
    aux = torch.randn((9, 96), generator=generator).bfloat16()
    context = draft.combine_hidden_states(aux)
    pos = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2])
    # The two disjoint requests use non-contiguous physical pages.
    slots = torch.tensor([12, 13, 14, 15, 24, 25, 4, 5, 6])
    draft.precompute_and_store_context_kv(context, pos, [slots] * 3)
    for stage in draft.mtp:
        projected = stage.attn.project_kv(context, pos)
        torch.testing.assert_close(stage.attn.cache.pages()[0][slots, :32], projected.float(), rtol=0, atol=0)
    ids = [2, 15, 15, 15, 15, 7, 15, 15, 15, 15]
    snapshots = [s.attn.cache.kv_cache.clone() for s in draft.mtp]
    packed = run_block(draft, ids, [6, 3], [[3, 6, 0], [1, 5, 0]])
    independent = torch.cat((run_block(draft, ids[:5], [6], [[3, 6, 0]]), run_block(draft, ids[5:], [3], [[1, 5, 0]])))
    torch.testing.assert_close(packed, independent, rtol=0, atol=0)
    for stage, previous in zip(draft.mtp, snapshots):
        torch.testing.assert_close(stage.attn.cache.kv_cache, previous, rtol=0, atol=0)
    logits = draft.compute_logits(packed)
    expected = F.linear(draft.mtp[-1].norm(packed).float(), draft.mtp[-1].head.weight.float())
    torch.testing.assert_close(logits, expected)
    embedding = draft.markov_embed(torch.tensor(ids))
    torch.testing.assert_close(
        draft.markov_bias(embedding), F.linear(embedding.float(), draft.mtp[-1].markov_head.head.weight)
    )
    expected_conf = (
        F.linear(torch.cat((packed, embedding), -1).float(), draft.mtp[-1].confidence_head.proj.weight)
        .squeeze(-1)
        .sigmoid()
    )
    torch.testing.assert_close(draft.compute_confidence(packed, embedding), expected_conf)
    assert draft.get_draft_kv_cache_layer_names() == [f"mtp.{i}.attn.cache" for i in range(3)]
    assert draft.get_draft_attn_causal() == [False] * 3
    with pytest.raises(ValueError, match="complete anchor-first"):
        bad = DeepseekV41Metadata([RequestSpan(0, 4, 8)], torch.tensor([[1, 2]]), torch.empty(0), 4)
        for stage in draft.mtp:
            stage.attn.cache.metadata = lambda: bad
        draft(torch.tensor(ids[:4]), torch.arange(4, 8))


def dense_norm(x, norm):
    return (norm.weight * (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps))).to(x.dtype)


def dense_rotary(x, positions, attn, inverse=False):
    tail = x[..., -attn.rope_dim :].float().unflatten(-1, (-1, 2))
    phase = positions.float()[:, None] * attn.frequencies
    if x.ndim == 3:
        phase = phase[:, None, :]
    even, odd = tail.unbind(-1)
    sn = -phase.sin() if inverse else phase.sin()
    rotated = torch.stack((even * phase.cos() - odd * sn, even * sn + odd * phase.cos()), -1).flatten(-2)
    return torch.cat((x[..., : -attn.rope_dim], rotated.to(x.dtype)), -1)


def dense_kv(attn, x, positions):
    raw = dense_norm(F.linear(x, attn.wkv.weight), attn.kv_norm)
    return fp8_quant_dequant(dense_rotary(raw, positions, attn), 32)


def dense_attention(attn, x, main_x, positions):
    # Deliberately construct the official chronological ring and concatenated
    # query keys from raw context; never access the plugin's paged cache.
    qr = dense_norm(F.linear(x, attn.wq_a.weight), attn.q_norm)
    q = dense_rotary(F.linear(qr, attn.wq_b.weight).view(-1, attn.heads, attn.dim), positions, attn)
    context = dense_kv(attn, main_x, torch.arange(len(main_x)))
    ring = {i % attn.config.sliding_window: value for i, value in enumerate(context)}
    kv = torch.cat((torch.stack([ring[i] for i in sorted(ring)]), dense_kv(attn, x, positions)))
    scores = torch.einsum("shd,kd->shk", q.float(), kv.float()) * attn.dim**-0.5
    maximum = scores.amax(-1, keepdim=True)
    probs = (scores - maximum).exp()
    denom = probs.sum(-1, keepdim=True) + (attn.attn_sink.view(1, -1, 1) - maximum).exp()
    output = (torch.einsum("shk,kd->shd", probs.bfloat16().float(), kv.float()) / denom).bfloat16()
    output = dense_rotary(output, positions, attn, inverse=True).view(len(x), attn.groups, -1)
    output = fp8_quant_dequant(output, 32)
    weights = attn.wo_a.weight.view(attn.groups, attn.config.o_lora_rank, -1)
    return F.linear(torch.einsum("sgd,grd->sgr", output, weights).flatten(1), attn.wo_b.weight)


def dense_expert(expert, x, weight):
    gate = F.linear(x, expert.w1.weight).float().clamp(max=expert.swiglu_limit)
    up = F.linear(x, expert.w3.weight).float().clamp(-expert.swiglu_limit, expert.swiglu_limit)
    return F.linear((F.silu(gate) * up * weight).bfloat16(), expert.w2.weight)


def dense_moe(module, x):
    gate = module.gate
    scores = F.softplus(F.linear(x.float(), gate.weight) / gate.temperature).sqrt()
    ids = (scores + gate.bias).topk(gate.top_k).indices
    selected = scores.gather(-1, ids)
    weights = selected / selected.sum(-1, keepdim=True) * gate.scaling_factor
    output = torch.zeros_like(x, dtype=torch.float32)
    for row in range(len(x)):
        for expert_id, weight in sorted(zip(ids[row].tolist(), weights[row])):
            output[row] += dense_expert(module.experts[expert_id], x[row], weight).float()
    return (output + dense_expert(module.shared_experts, x, 1).float()).bfloat16()


def dense_three_stage_hidden(model, ids, aux):
    first = model.mtp[0]
    main_x = dense_norm(F.linear(aux, first.main_proj.weight), first.main_norm)
    h = F.embedding(ids, first.embed.weight).unsqueeze(1).repeat(1, model.config.hc_mult, 1)
    pre = torch.zeros(h.shape[:-1], dtype=torch.float32)
    pre[:, 0] = 1
    positions = torch.arange(len(aux), len(aux) + len(ids))
    for stage in model.mtp:
        for part in ("attn", "ffn"):
            next_pre, post, comb = hc_mixes(
                h,
                getattr(stage, f"hc_{part}_fn"),
                getattr(stage, f"hc_{part}_scale"),
                getattr(stage, f"hc_{part}_base"),
                model.config.rms_norm_eps,
                model.config.hc_eps,
                model.config.hc_sinkhorn_iters,
            )
            x = dense_norm((h.float() * pre[:, :, None]).sum(1).bfloat16(), getattr(stage, f"{part}_norm"))
            y = dense_attention(stage.attn, x, main_x, positions) if part == "attn" else dense_moe(stage.ffn, x)
            residual = (h.float()[:, :, None, :] * comb[:, :, :, None]).sum(1)
            h = (post[:, :, None] * y.float()[:, None, :] + residual).bfloat16()
            pre = next_pre
    return (h.float() * pre[:, :, None]).sum(1).bfloat16()


@torch.no_grad()
def test_three_stage_dense_oracle_after_rejected_tail_and_markov_chain(draft):
    generator = torch.Generator().manual_seed(174)
    aux = torch.randn(9, 96, generator=generator).bfloat16()
    context = draft.combine_hidden_states(aux)
    slots = torch.tensor([12, 13, 14, 15, 20, 21, 22, 23, 4])
    draft.precompute_and_store_context_kv(context, torch.arange(9), [slots] * 3)
    # Requests accepted only positions [0,6); the cached speculative tail
    # [6,9) is still physically present. Query 6 must ignore that tail.
    ids = torch.tensor([2, 15, 15, 15, 15])
    actual = run_block(draft, ids.tolist(), [6], [[3, 5, 1]])
    expected = dense_three_stage_hidden(draft, ids, aux[:6])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    normalized = dense_norm(expected, draft.mtp[-1].norm)
    logits = F.linear(normalized.float(), draft.mtp[-1].head.weight.float())
    actual_logits = draft.compute_logits(actual)
    token = actual_token = ids[0]
    for row in range(len(ids)):
        markov = F.embedding(token, draft.mtp[-1].markov_head.embed.weight)
        expected_logits = logits[row] + F.linear(markov.float(), draft.mtp[-1].markov_head.head.weight)
        actual_markov = draft.markov_embed(actual_token)
        adjusted = actual_logits[row] + draft.markov_bias(actual_markov)
        torch.testing.assert_close(adjusted, expected_logits, rtol=0, atol=0)
        confidence = F.linear(
            torch.cat((expected[row], markov)).float(), draft.mtp[-1].confidence_head.proj.weight
        ).sigmoid()
        torch.testing.assert_close(
            draft.compute_confidence(actual[row], actual_markov), confidence.squeeze(), rtol=0, atol=0
        )
        token, actual_token = expected_logits.argmax(), adjusted.argmax()
        assert token == actual_token


@torch.no_grad()
def test_idle_draft_warmup_preserves_cache_and_real_proposal_requires_metadata(draft, monkeypatch):
    ctx = SimpleNamespace(in_profile_run=False, is_draft_model=True, attn_metadata=None, draft_attn_metadatas=[])
    monkeypatch.setattr(dspark, "get_forward_context", lambda: ctx)
    originals = []
    for stage in draft.mtp:
        stage.attn.cache.metadata = lambda: None
        stage.attn.cache.kv_cache.fill_(7)
        originals.append(stage.attn.cache.kv_cache.clone())
    draft.precompute_and_store_context_kv(torch.zeros(5, 32, dtype=torch.bfloat16), torch.arange(5))
    draft(torch.tensor([2, 15, 15, 15, 15]), torch.arange(5))
    for stage, previous in zip(draft.mtp, originals):
        torch.testing.assert_close(stage.attn.cache.kv_cache, previous, rtol=0, atol=0)
    ctx.draft_attn_metadatas = None
    with pytest.raises(RuntimeError, match="Missing cache slots"):
        draft.precompute_and_store_context_kv(torch.zeros(5, 32, dtype=torch.bfloat16), torch.arange(5))
    with pytest.raises(RuntimeError, match="Missing cache slots"):
        draft.precompute_and_store_context_kv(
            torch.zeros(5, 32, dtype=torch.bfloat16), torch.arange(5), [torch.arange(5), None, torch.arange(5)]
        )
    with pytest.raises(RuntimeError, match="requires the draft layer's paged metadata"):
        draft(torch.tensor([2, 15, 15, 15, 15]), torch.arange(5))


@torch.no_grad()
def test_loader_reads_authoritative_draft_tensors_and_rejects_missing(draft, tmp_path):
    params = dict(draft.named_parameters())
    saved = {k: v.detach().clone() for k, v in params.items()}
    save_file(saved, str(tmp_path / "draft.safetensors"))
    index = {k: "draft.safetensors" for k in params}
    path = tmp_path / "quant_model_weights.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": index}))
    for value in params.values():
        value.zero_()

    def forbidden_iterator():
        raise AssertionError("Never load the entire target checkpoint through the generic iterator")
        yield

    assert draft.load_weights(forbidden_iterator()) == set(saved)
    for key, value in params.items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    del index["mtp.2.confidence_head.proj.weight"]
    path.write_text(json.dumps({"weight_map": index}))
    with pytest.raises(KeyError, match="confidence_head"):
        draft.load_weights(forbidden_iterator())
