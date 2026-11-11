"""Tensor-only V4.1 attention over vLLM physical pages. Based on the frozen MIT HF inference model."""

from functools import cache
from importlib.util import find_spec

import torch

from vllm_ascend.models.deepseek_v41.compact_kv import cache_row_bytes, pack_fp4, pack_window, unpack_fp4, unpack_window
from vllm_ascend.models.deepseek_v41.decode_state import PagedHistory
from vllm_ascend.models.deepseek_v41.eager_kernels import _fp4_level, act_quant, power_of_two_scale
from vllm_ascend.models.deepseek_v41.rotary import rotate_rows as _rotate


def _fp4(x, lut, width=32, e4scale=False):
    groups = x.float().unflatten(-1, (-1, width))
    maximum = groups.abs().amax(-1, keepdim=True)
    if e4scale:
        scales = (maximum.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        scales = power_of_two_scale(maximum.clamp_min(6 * 2.0**-126), 6.0)
    codes = _fp4_level((groups / scales).clamp(-6.0, 6.0))
    x.copy_((lut[codes.long()] * scales).flatten(-2))


class LayerPages:
    @staticmethod
    def layout(attn):
        """Yield table name, tokens per stored row, row width and element type."""
        yield "window", 1, cache_row_bytes(attn.head_dim, 8, 32), torch.uint8
        if attn.is_kv_source:
            yield "compressed", attn.compress_ratio, cache_row_bytes(attn.head_dim, 4, 16), torch.uint8
            yield "index", attn.compress_ratio, cache_row_bytes(attn.indexer.index_head_dim, 4, 32), torch.uint8
            if attn.compress_ratio == 2:
                yield "first_kv", 2, attn.head_dim, torch.float32
                yield "first_score", 2, attn.head_dim, torch.float32

    @staticmethod
    def rows_per_block(name, ratio, block_size, tail_groups):
        return tail_groups if name in ("first_kv", "first_score") else block_size // ratio

    def __init__(self, attn, inputs, *, physical_slots=None):
        count = inputs.history.num_slots if physical_slots is None else physical_slots
        block_size = inputs.history.block_size
        for name, ratio, width, dtype in self.layout(attn):
            per_block = self.rows_per_block(name, ratio, block_size, inputs.history.tail_groups)
            rows = count // block_size * per_block + inputs.max_tokens
            setattr(self, name, torch.zeros((rows, width), device=inputs.device, dtype=dtype))

    @classmethod
    def bytes_per_block(cls, attn, block_size, max_draft_tokens=0):
        tail = PagedHistory.groups_per_block(block_size, max_draft_tokens)
        return sum(
            cls.rows_per_block(name, ratio, block_size, tail) * width * dtype.itemsize
            for name, ratio, width, dtype in cls.layout(attn)
        )


def _store_source(attn, x, state, pages):
    count = x.shape[0]
    ratio = attn.compress_ratio
    compressor = attn.compressor
    if ratio == 1:
        latent = compressor.norm(compressor.wkv(x))
    elif ratio == 2:
        value = compressor.wkv(x.float())
        score = compressor.wgate(x.float())
        previous, dest = state.tail_plan[:count].unbind(1)
        dest = dest.contiguous()
        local = previous >= 0
        batch = previous.clamp_min(0)
        cached = (-1 - previous).clamp_min(0)
        # A long query beginning on an odd position can wrap and overwrite
        # the needed old tail slot. Gather before issuing either in-place store.
        prev_value = torch.where(local[:, None], value[batch], pages.first_kv.index_select(0, cached))
        prev_score = torch.where(local[:, None], score[batch], pages.first_score.index_select(0, cached))
        pages.first_kv.index_copy_(0, dest, value)
        pages.first_score.index_copy_(0, dest, score)
        pair = torch.stack((prev_value, value), dim=1)
        scores = torch.stack((prev_score, score), dim=1)
        latent = compressor.norm((pair * scores.softmax(1)).sum(1).to(x.dtype))
    else:
        raise ValueError("Expected checkpoint compression ratio one or two")
    dest = state.write_addresses(count, ratio)
    pos = (state.pos[:count] // ratio) * ratio
    freq = attn.freqs_cis.index_select(0, pos)
    idx = attn.indexer
    key = _rotate(idx.k_norm(idx.wk(latent)), freq, idx.rope_head_dim)
    observer = getattr(attn, "_compressed_observer", None)
    trace_write = observer is not None and observer.captures(count)
    if trace_write:
        observer.tap("index_write_input", key)
    index_data = pack_fp4(key, 32, False)
    pages.index.index_copy_(0, dest, index_data)
    if trace_write:
        observer.tap("index_write_data", index_data)
        observer.tap("compressed_write_slots", dest)
    del index_data
    _rotate(latent, freq, attn.rope_head_dim)
    if trace_write:
        observer.tap("compressed_write_input", latent)
    compressed_data = pack_fp4(latent, 16, True)
    pages.compressed.index_copy_(0, dest, compressed_data)
    if trace_write:
        observer.tap("compressed_write_data", compressed_data)


def _candidate_blocks(score, lengths, limit, block_size):
    # Keep the same full-width topk operand and force the last visible block,
    # as in the HF selector. Store IDs instead of a dense mask per query.
    width = score.size(-1)
    blocks = torch.nn.functional.pad(score, (0, -width % block_size), value=-torch.inf)
    blocks = blocks.unflatten(-1, (-1, block_size)).amax(-1)
    last = (lengths - 1) // block_size
    blocks.masked_fill_(torch.arange(blocks.size(1), device=score.device) == last, torch.inf)
    values, ids = blocks.topk(min(limit, blocks.size(1)), dim=-1)
    return torch.where(values > -torch.inf, ids, -1).int()


@cache
def _has_native_paged_index(device):
    return torch.npu.get_device_name(device) == "Ascend950DT_9574" and find_spec("triton") is not None


def _index_scores(query, weight, state, source, rows, lengths, width, ratio, block_size, blocks):
    if (
        width > 4096
        and query.device.type == "npu"
        and query.shape[1:] == (32, 128)
        and block_size == 8
        and hasattr(state, "pages_for_rows")
        and _has_native_paged_index(query.device)
    ):
        from vllm_ascend.models.deepseek_v41.index_score_kernel import paged_index_scores

        return paged_index_scores(
            query,
            weight,
            source.index,
            state.pages_for_rows(rows),
            lengths,
            width,
            ratio,
            state.history.block_size,
            blocks,
            block_size,
        )
    count = query.size(0)
    # Keep the original logical order for torch.topk, including equal scores.
    # The extra column absorbs masked lanes in the final partial block.
    out = query.new_full((count, width + 1), -torch.inf)
    scan = width if blocks is None else blocks.size(1) * block_size
    for start in range(0, scan, 1024):
        seq = torch.arange(start, min(start + 1024, scan), device=query.device)
        if blocks is None:
            logical = seq.expand(count, -1)
        else:
            block = blocks[:, seq // block_size].long()
            logical = block * block_size + seq % block_size
        visible = (logical >= 0) & (logical < lengths) & (logical < width)
        last_token = ((logical.clamp(0, width - 1) + 1) * ratio - 1).clamp_max(state.history.max_len - 1)
        coords = state.physical(last_token, rows, ratio)
        keys = unpack_fp4(source.index[coords], query.size(-1), 32, False, state.fp4_lut)
        score = torch.bmm(query, keys.transpose(1, 2))
        score = (score.relu_() * weight.unsqueeze(-1)).sum(1).masked_fill_(~visible, -torch.inf)
        if blocks is None:
            out[:, start : start + seq.size(0)].copy_(score)
        else:
            out.scatter_(1, torch.where(visible, logical, width), score)
    return out[:, :width].contiguous()


def _select(attn, x, qr, freq, state, source, context, candidate):
    count = x.shape[0]
    idx = attn.indexer
    ratio = attn.compress_ratio
    width = max(1, context // ratio)
    lengths = (state.pos[:count, None] + 1) // ratio
    limit = idx.candidate_topk_blocks * idx.candidate_block_size
    if width <= idx.index_topk and width <= limit:
        logical = torch.arange(width, device=x.device).expand(count, -1)
        return torch.where(logical < lengths, logical, -1), None
    if idx.uses_candidates and width > limit and candidate is None:
        raise RuntimeError("Missing V4.1 source candidate blocks")
    q = idx.wq_b(qr).unflatten(-1, (idx.n_local_heads, idx.index_head_dim))
    _rotate(q, freq, idx.rope_head_dim)
    _fp4(q, state.fp4_lut)
    weight = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads**-0.5)
    selected, block_ids = [], []
    for start in range(0, count, 8):
        rows = slice(start, min(start + 8, count))
        blocks = candidate[rows] if idx.uses_candidates and candidate is not None else None
        score = _index_scores(
            q[rows], weight[rows], state, source, rows, lengths[rows], width, ratio, idx.candidate_block_size, blocks
        )
        if idx.is_candidate_source and width > limit:
            block_ids.append(
                _candidate_blocks(score, lengths[rows], idx.candidate_topk_blocks, idx.candidate_block_size)
            )
        indices = score.topk(min(width, idx.index_topk), dim=-1, sorted=False).indices
        indices = indices.sort(dim=-1).values
        selected.append(torch.where(indices < lengths[rows], indices, -1))
    if idx.is_candidate_source:
        candidate = torch.cat(block_ids) if block_ids else None
    return torch.cat(selected), candidate


def _raw_window(state, count, context, window_size):
    ids, spans = state.raw_windows[window_size]
    width = window_size
    if state.raw_span_hint is not None and context != state.history.max_len:
        width = min(max(context, state.raw_span_hint), window_size)
    if width < window_size:
        # An inactive lane follows the shorter, eager prefill shape.
        return ids[:count, :width], spans[:count].clamp_max(width)
    return ids[:count], spans[:count]


def _sparse(q, window, compressed, win_ids, comp_ids, attn, raw_spans, cache_lut=None):
    """HF 64-key online softmax, including the BF16 exponent rounding and sink logit."""
    count, heads, dim = q.shape
    result = torch.empty_like(q)
    observer = getattr(attn, "_window_observer", None)
    compressed_trace = getattr(attn, "_compressed_observer", None)
    for start in range(0, count, 32):
        rows = slice(start, min(start + 32, count))
        w = win_ids[rows]
        valid = w >= 0
        values = window[w.clamp_min(0)]
        if compressed_trace is not None and compressed_trace.captures(count):
            compressed_trace.tap("raw_ids", w)
            compressed_trace.tap("raw_data", values)
        if observer is not None and count == observer.width:
            observer.gathered(attn.layer_id, w, values)
        if cache_lut is not None:
            values = unpack_window(values, dim)
        if comp_ids is not None:
            c = comp_ids[rows]
            # HF appends compressed entries directly after each query's raw
            # window. A batch can contain different prompt lengths and decodes.
            # A different 64-key tile boundary changes BF16 probability rounding.
            # Read the compacted order. This also handles padded graph lanes
            # without scattering into a flattened [queries, keys] buffer.
            raw_count = w.size(1)
            stride = raw_count + c.size(1)
            column = torch.arange(stride, device=w.device).unsqueeze(0)
            span = raw_spans[rows]
            source = torch.where(column < span, column, column + raw_count - span).clamp(0, stride - 1)
            in_bounds = column < span + c.size(1)
            valid = torch.cat((valid, c >= 0), dim=1).gather(1, source) & in_bounds
            previous = compressed[c.clamp_min(0)]
            if compressed_trace is not None and compressed_trace.captures(count):
                compressed_trace.tap("comp_ids", c)
                compressed_trace.tap("comp_data", previous)
            if cache_lut is not None:
                previous = unpack_fp4(previous, dim, 16, True, cache_lut)
            values = torch.cat((values, previous), dim=1)
            values = values.gather(1, source.unsqueeze(-1).expand(-1, -1, dim))
        v = values.float().masked_fill(~valid.unsqueeze(-1), 0.0)
        qq = q[rows].float()
        scores = torch.bmm(qq, v.transpose(1, 2)) * attn.softmax_scale
        scores.masked_fill_(~valid.unsqueeze(1), -torch.inf)
        maximum = qq.new_full((qq.size(0), heads, 1), -1e30)
        denom = torch.zeros_like(maximum)
        numer = torch.zeros_like(qq)
        for begin in range(0, scores.size(-1), 64):
            tile = scores[:, :, begin : begin + 64]
            new_max = torch.maximum(maximum, tile.amax(-1, keepdim=True))
            prev = (maximum - new_max).exp()
            prob = (tile - new_max).exp()
            numer = numer * prev + torch.bmm(prob.bfloat16().float(), v[:, begin : begin + 64])
            denom = denom * prev + prob.sum(-1, keepdim=True)
            maximum = new_max
        denom += (attn.attn_sink.view(1, heads, 1) - maximum).exp()
        result[rows] = numer / denom
    return result


def paged_attention(attn, x, state, pages, shared, context):
    """A single token-major pass, including mixed prefill, greedy decode and verified draft tokens."""
    count = x.size(1)
    x = x[0]
    pos = state.pos[:count]
    freq = attn.freqs_cis.index_select(0, pos)
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    _rotate(q, freq, attn.rope_head_dim)
    kv = _rotate(attn.kv_norm(attn.wkv(x)), freq, attn.rope_head_dim)
    packed = pack_window(kv)
    pages.window.index_copy_(0, state.slots[:count], packed)
    observer = getattr(attn, "_window_observer", None)
    if observer is not None:
        observer.projections(attn.layer_id, q, kv, packed, state.slots[:count], pages.window)
    del packed
    win_ids, raw_spans = _raw_window(state, count, context, attn.window_size)
    compressed_trace = getattr(attn, "_compressed_observer", None)
    if compressed_trace is not None and compressed_trace.captures(count):
        compressed_trace.tap("query", q)
        compressed_trace.tap("kv", kv)
        compressed_trace.tap("raw_spans", raw_spans)
    active = state.active[:count, None]
    comp_ids = compressed = None
    if attn.compress_ratio:
        ratio = attn.compress_ratio
        if attn.is_kv_source:
            _store_source(attn, x, state, pages)
            shared["source"] = pages
        source = shared["source"]
        if attn.is_index_source:
            shared["selected"], shared["candidate"] = _select(
                attn, x, qr, freq, state, source, context, shared.get("candidate")
            )
        indices = shared["selected"]
        if compressed_trace is not None and compressed_trace.captures(count):
            compressed_trace.tap("selected_logical", indices)
        last = ((indices.clamp_min(0) + 1) * ratio - 1).clamp_max(state.history.max_len - 1)
        comp_ids = torch.where((indices >= 0) & active, state.physical(last, slice(0, count), ratio), -1)
        compressed = source.compressed
    out = _sparse(q, pages.window, compressed, win_ids, comp_ids, attn, raw_spans, state.fp4_lut)
    if compressed_trace is not None and compressed_trace.captures(count):
        compressed_trace.tap("sparse_output", out)
    if observer is not None:
        observer.tap(attn.layer_id, "sparse_output", out)
    _rotate(out, freq, attn.rope_head_dim, True)
    out = out.reshape(count, attn.n_local_groups, -1)
    act_quant(out, 32, inplace=True)
    if compressed_trace is not None and compressed_trace.captures(count):
        compressed_trace.tap("wo_a_input", out)
    if observer is not None:
        observer.tap(attn.layer_id, "wo_a_input", out)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    out = torch.einsum("tgd,grd->tgr", out, wo_a)
    if compressed_trace is not None and compressed_trace.captures(count):
        compressed_trace.tap("wo_a_output", out)
    if observer is not None:
        observer.tap(attn.layer_id, "wo_a_output", out)
    return attn.wo_b(out.flatten(1)).unsqueeze(0)
