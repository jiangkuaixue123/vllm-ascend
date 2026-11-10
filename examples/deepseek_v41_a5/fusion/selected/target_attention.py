# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_epilogues import compressor_pool_norm
from vllm_ascend.ops.triton.deepseek_v41_indexer import indexer_reduce
from vllm_ascend.ops.triton.deepseek_v41_kv import kv_norm_rope_qdq
from vllm_ascend.ops.triton.deepseek_v41 import fp8_quant_dequant, fp4_quant_dequant
from vllm_ascend.models.deepseek_v41.attention import _OUTPUT_QDQ_TILE_ROWS


def _quantize_projection_input(output):
    """Consume the attention output; E4M3 quantization groups only its last dimension."""
    if output.shape[0] <= _OUTPUT_QDQ_TILE_ROWS:
        return fp8_quant_dequant(output, 32)
    for tile in output.split(_OUTPUT_QDQ_TILE_ROWS):
        fp8_quant_dequant(tile, 32, inplace=True)
    return output


def forward(self, hidden_states, runtime):
    if runtime.decode is not None:
        return self._decode_forward(hidden_states, runtime)
    metadata = self.cache.metadata()
    gathered, offset = self._sequence_inputs(hidden_states, "eager")
    queries_input = hidden_states if self.dsa_cp else gathered
    start = offset if self.dsa_cp else 0
    qr = self.q_norm(self.wq_a(queries_input)[0])
    queries = self.wq_b(qr)[0].view(-1, self.heads, self.dim)
    raw_kv = self.wkv(gathered)[0]
    output = torch.zeros_like(queries)
    for request_idx, request in enumerate(runtime.requests):
        begin, end = (request.begin, request.end)
        if begin == end:
            continue
        positions = torch.arange(request.start_pos, request.seq_len, device=hidden_states.device)
        kv = raw_kv[begin:end]
        phases = positions.float().unsqueeze(-1) * self.frequencies
        kv = kv_norm_rope_qdq(kv, self.kv_norm.weight, phases.cos(), phases.sin(), self.kv_norm.eps)
        first = max(0, request.start_pos - self.config.sliding_window + 1)
        if metadata is not None:
            slots = self.cache.slots(metadata, request_idx, positions)
            self.cache.write_slots(slots, kv, 0, 0)
            history = torch.arange(first, request.seq_len, device=hidden_states.device)
            kv = self.cache.read(metadata, request_idx, history, 0, 0, self.dim).to(kv.dtype)
        if self.ratio:
            shared = runtime.shared[request_idx]
            self._compressed(gathered[begin:end], positions, metadata, request_idx, request, shared)
        lo, hi = (max(begin, start), min(end, start + queries_input.shape[0]))
        if lo >= hi:
            continue
        rows = slice(lo - start, hi - start)
        positions = positions[lo - begin : hi - begin]
        q = queries[rows]
        self.rotate(q.unsqueeze(0), positions)
        width = min(self.config.sliding_window, request.seq_len)
        window = (positions - self.config.sliding_window + 1).clamp_min(0)
        window = window.unsqueeze(-1) + torch.arange(width, device=hidden_states.device)
        indices = torch.where(window <= positions.unsqueeze(-1), window - first, -1)
        if self.ratio:
            compressed_indices = self._index(queries_input[rows], qr[rows], positions, shared)
            indices = torch.cat(
                (indices, torch.where(compressed_indices >= 0, compressed_indices + kv.shape[0], -1)), -1
            )
            kv = torch.cat((kv, shared.compressed), 0)
        else:
            kv = kv[: kv.shape[0] - (end - hi)]
        attended = self.attention_kernel(
            q.unsqueeze(0),
            kv.unsqueeze(0),
            self.attn_sink,
            indices.int().unsqueeze(0),
            self.dim ** (-0.5),
            mode="DSA target indexed prefill" if self.ratio else "SWA target causal prefill",
            swa_window=None if self.ratio else self.config.sliding_window,
        )[0]
        self.rotate(attended.unsqueeze(0), positions, inverse=True)
        output[rows] = attended
    return self._project_output(output, hidden_states.dtype)


def _decode_forward(self, local_states, runtime):
    full_md = self.cache.metadata().decode
    gathered, offset = self._sequence_inputs(local_states, "uniform_decode")
    if full_md is None or gathered.shape[0] != full_md.active.shape[0]:
        raise ValueError("V4.1 graph token rows must match the decode metadata bucket")
    local_md = full_md.shard(offset, local_states.shape[0]) if self.sequence_group is not None else full_md
    x, md = (local_states, local_md) if self.dsa_cp else (gathered, full_md)
    full_pos = (full_md.seq_lens - 1).clamp_min(0)
    pos = full_pos[offset : offset + x.shape[0]] if self.dsa_cp else full_pos
    qr = self.q_norm(self.wq_a(x)[0])
    q = self.wq_b(qr)[0].view(x.shape[0], self.heads, self.dim)
    raw = self.wkv(gathered)[0]
    self.rotate(q.unsqueeze(0), pos)
    phases = full_pos.float().unsqueeze(-1) * self.frequencies
    raw = kv_norm_rope_qdq(raw, self.kv_norm.weight, phases.cos(), phases.sin(), self.kv_norm.eps)
    self.cache.decode_write(full_md, raw, 0, 0, full_md.active)
    extra = 0
    if self.ratio:
        self._decode_compressed(gathered, full_pos, full_md, runtime)
        if self.owns_kv:
            runtime.source_decode = md
        self._decode_index(x, qr, pos, md, runtime)
        extra = runtime.decode_topk.shape[-1]
    window = self.config.sliding_window
    columns = torch.arange(min(window, md.max_model_len) + extra, device=x.device)[None, :]
    width = md.seq_lens.clamp(0, window)[:, None]
    raw_pos = (md.seq_lens - window).clamp_min(0)[:, None] + columns
    in_window = (columns < width) & md.active[:, None]
    selected = self.cache.decode_read(md, raw_pos, 0, 0, self.dim)
    valid = in_window
    if extra:
        ordinal = columns - width
        indices = runtime.decode_topk.gather(1, ordinal.clamp(0, extra - 1))
        compressed = (ordinal >= 0) & (ordinal < extra) & (indices >= 0)
        comp = runtime.source_cache.decode_read(
            runtime.source_decode, (indices + 1) * self.ratio - 1, 0, self.dim, self.dim
        )
        selected = torch.where(compressed[..., None], comp, selected)
        valid = valid | compressed & md.active[:, None]
    ids = torch.where(valid, columns, -1).to(torch.int32).unsqueeze(1)
    attended = self.attention_kernel(
        q.unsqueeze(1),
        selected.to(x.dtype),
        self.attn_sink,
        ids,
        self.dim ** (-0.5),
        mode="DSA target indexed decode" if self.ratio else "SWA target indexed decode",
        uniform_decode=True,
    )[:, 0]
    self.rotate(attended.unsqueeze(0), pos, inverse=True)
    output = self._project_output(attended, x.dtype)
    return torch.where(local_md.active[:, None], output, 0)


def _compressed(self, x, positions, metadata, request_idx, request, shared):
    if not self.owns_kv:
        if shared.compressed is None or shared.index_keys is None:
            raise RuntimeError("Compressed attention source did not execute")
        return
    ratio, dim = (self.ratio, self.dim)
    compressor = self.compressor
    raw = compressor.wkv(x.float() if ratio > 1 else x)
    scores = compressor.wgate(x.float()) if ratio > 1 else None
    completed = positions[(positions + 1) % ratio == 0]
    if metadata is None:
        group_kv = raw[: completed.numel() * ratio]
        group_scores = scores[: completed.numel() * ratio] if scores is not None else None
    else:
        slots = self.cache.slots(metadata, request_idx, positions)
        self.cache.write_slots(slots, raw, 0, dim)
        if scores is not None:
            self.cache.write_slots(slots, scores, 1, 0)
        group_positions = (completed.unsqueeze(-1) - torch.arange(ratio - 1, -1, -1, device=x.device)).flatten()
        group_kv = self.cache.read(metadata, request_idx, group_positions, 0, dim, dim)
        group_scores = (
            self.cache.read(metadata, request_idx, group_positions, 1, 0, dim) if scores is not None else None
        )
    if completed.numel():
        if ratio == 2:
            latent = compressor_pool_norm(
                group_kv.unflatten(0, (-1, 2)),
                group_scores.unflatten(0, (-1, 2)),
                compressor.norm.weight,
                compressor.norm.eps,
                x.dtype,
            )
        else:
            if ratio > 1:
                latent = (
                    (group_kv.unflatten(0, (-1, ratio)) * group_scores.unflatten(0, (-1, ratio)).softmax(1))
                    .sum(1)
                    .to(x.dtype)
                )
            else:
                latent = group_kv.to(x.dtype)
            latent = compressor.norm(latent)
        keys = self.indexer.k_norm(self.indexer.wk(latent))
        self.rotate(keys.unsqueeze(0), completed + 1 - ratio)
        keys = fp4_quant_dequant(keys, 32, scale_format="e8m0")
        self.rotate(latent.unsqueeze(0), completed + 1 - ratio)
        latent = fp4_quant_dequant(latent, 16, scale_format="e4m3")
        if metadata is not None:
            slots = self.cache.slots(metadata, request_idx, completed)
            self.cache.write_slots(slots, latent, 0, dim)
            self.cache.write_slots(slots, keys, 1, dim)
    else:
        latent = x.new_empty((0, dim))
        keys = x.new_empty((0, self.config.index_head_dim))
    if metadata is None:
        shared.compressed, shared.index_keys = (latent, keys)
    else:
        ends = torch.arange(ratio - 1, request.seq_len, ratio, device=x.device)
        shared.compressed = self.cache.read(metadata, request_idx, ends, 0, dim, dim).to(x.dtype)
        shared.index_keys = self.cache.read(metadata, request_idx, ends, 1, dim, self.config.index_head_dim).to(x.dtype)


def _decode_compressed(self, x, pos, md, runtime):
    if not self.owns_kv:
        return
    ratio, dim = (self.ratio, self.dim)
    raw = self.compressor.wkv(x.float() if ratio > 1 else x)
    complete = (pos + 1) % ratio == 0
    self.cache.decode_write(md, raw, 0, dim, md.active)
    if ratio > 1:
        score = self.compressor.wgate(x.float())
        self.cache.decode_write(md, score, 1, 0, md.active)
        relative = torch.arange(ratio - 1, -1, -1, device=x.device)
        group_pos = pos[:, None] - relative
        values = self.cache.decode_read(md, group_pos, 0, dim, dim)
        scores = self.cache.decode_read(md, group_pos, 1, 0, dim)
        if ratio == 2:
            latent = compressor_pool_norm(
                values, scores, self.compressor.norm.weight, self.compressor.norm.eps, x.dtype
            )
        else:
            raw = (values * scores.softmax(1)).sum(1).to(x.dtype)
            latent = self.compressor.norm(raw)
    else:
        latent = self.compressor.norm(raw)
    group_start = (pos + 1 - ratio).clamp_min(0)
    keys = self.indexer.k_norm(self.indexer.wk(latent))
    self.rotate(keys.unsqueeze(0), group_start)
    self.rotate(latent.unsqueeze(0), group_start)
    keys = fp4_quant_dequant(keys, 32, scale_format="e8m0")
    latent = fp4_quant_dequant(latent, 16, scale_format="e4m3")
    self.cache.decode_write(md, latent, 0, dim, complete)
    self.cache.decode_write(md, keys, 1, dim, complete)
    runtime.source_cache = self.cache
    runtime.source_decode = md


def _index(self, x, qr, positions, shared):
    if not self.owns_index:
        if shared.topk is None:
            raise RuntimeError("Attention index source did not execute")
        return shared.topk
    count = shared.compressed.shape[0]
    if count == 0:
        shared.topk = torch.empty((x.shape[0], 0), device=x.device, dtype=torch.long)
        return shared.topk
    indexer = self.indexer
    q = indexer.wq_b(qr)[0].view(-1, self.index_heads, self.config.index_head_dim)
    weights = indexer.weights_proj(x)[0] * (self.config.index_head_dim ** (-0.5) * self.config.index_n_heads ** (-0.5))
    row_bytes = self.index_heads * max(
        count * max(q.element_size(), weights.element_size()), self.config.index_head_dim * 4
    )
    tile_rows = max(1, _INDEX_SCORE_TILE_BYTES // row_bytes)
    k = min(self.config.index_topk, count)
    topk = torch.empty((x.shape[0], k), device=x.device, dtype=torch.long)
    candidate_layer = self.config.candidate_source_layer_id
    if self.layer_id == candidate_layer:
        shared.candidates = torch.empty((x.shape[0], count), device=x.device, dtype=torch.bool)
    elif 0 <= candidate_layer < self.layer_id and shared.candidates is None:
        raise RuntimeError("Candidate source did not execute")
    columns = torch.arange(count, device=x.device)
    for start in range(0, x.shape[0], tile_rows):
        rows = slice(start, start + tile_rows)
        query = q[rows]
        self.rotate(query.unsqueeze(0), positions[rows])
        query = fp4_quant_dequant(query, 32, scale_format="e8m0")
        per_head = torch.einsum("shd,td->sht", query, shared.index_keys)
        per_head.relu_()
        per_head = per_head.to(torch.promote_types(per_head.dtype, weights.dtype))
        scores = per_head.mul_(weights[rows].unsqueeze(-1)).sum(1)
        del per_head
        if self.tp > 1 and (not self.dsa_cp):
            scores = tensor_model_parallel_all_reduce(scores)
        lengths = ((positions[rows] + 1) // self.ratio).unsqueeze(-1)
        scores.masked_fill_(columns >= lengths, -torch.inf)
        if self.layer_id == candidate_layer:
            shared.candidates[rows] = select_candidate_blocks(
                scores, lengths, self.config.candidate_topk_blocks, self.config.candidate_block_size
            )
        elif 0 <= candidate_layer < self.layer_id:
            scores.masked_fill_(~shared.candidates[rows], -torch.inf)
        indices = topk_columns(scores, k).sort(-1).values
        topk[rows] = torch.where(indices < lengths, indices, -1)
    shared.topk = topk
    return shared.topk


def _decode_index(self, x, qr, pos, md, runtime):
    if not self.owns_index:
        if runtime.decode_topk is None:
            raise RuntimeError("Missing V4.1 decode index source")
        return
    capacity = md.max_model_len // self.ratio
    if not capacity:
        runtime.decode_topk = torch.empty((x.shape[0], 0), dtype=torch.int64, device=x.device)
        return
    groups = torch.arange(capacity, device=x.device).unsqueeze(0)
    ends = ((groups + 1) * self.ratio - 1).expand(x.shape[0], -1)
    indexer = self.indexer
    q = indexer.wq_b(qr)[0].view(-1, self.index_heads, self.config.index_head_dim)
    self.rotate(q.unsqueeze(0), pos)
    q = fp4_quant_dequant(q, 32, scale_format="e8m0")
    weights = indexer.weights_proj(x)[0] * (self.config.index_head_dim ** (-0.5) * self.config.index_n_heads ** (-0.5))
    parts = []
    for tile in ends.split(_DECODE_INDEX_TILE_KEYS, dim=-1):
        keys = runtime.source_cache.decode_read(runtime.source_decode, tile, 1, self.dim, self.config.index_head_dim)
        per_head = torch.einsum("bhd,btd->bht", q, keys.to(x.dtype))
        parts.append(indexer_reduce(per_head, weights))
    scores = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    if self.tp > 1 and (not self.dsa_cp):
        scores = tensor_model_parallel_all_reduce(scores)
    lengths = torch.where(md.active, md.seq_lens // self.ratio, 0)[:, None]
    scores.masked_fill_(groups >= lengths, -torch.inf)
    source = self.config.candidate_source_layer_id
    if self.layer_id == source:
        runtime.decode_candidates = select_candidate_blocks(
            scores, lengths, self.config.candidate_topk_blocks, self.config.candidate_block_size
        )
    elif 0 <= source < self.layer_id:
        scores.masked_fill_(~runtime.decode_candidates, -torch.inf)
    indices = topk_columns(scores, min(self.config.index_topk, capacity)).sort(-1).values
    runtime.decode_topk = torch.where(indices < lengths, indices, -1)


def _project_output(self, output, dtype):
    if self.dsa_cp:
        send = output.reshape(-1, self.tp, self.local_heads, self.dim).transpose(0, 1).contiguous()
        send = send.reshape(-1, self.local_heads, self.dim)
        output = torch.empty_like(send)
        dist.all_to_all_single(output, send, group=self.sequence_group.device_group)
    output = output.view(-1, self.groups, self.local_heads * self.dim // self.groups)
    output = _quantize_projection_input(output)
    projection = self.wo_a.weight.view(self.groups, self.config.o_lora_rank, -1)
    output = torch.einsum("sgd,grd->sgr", output, projection).flatten(1)
    if self.sequence_group is not None:
        output = self.wo_b.quant_method.apply(self.wo_b, output, bias=None)
        output = self.sequence_group.reduce_scatter(output.float(), dim=0)
    else:
        output = self.wo_b(output)[0].float()
        if self.tp > 1:
            output = tensor_model_parallel_all_reduce(output)
    return output.to(dtype)
