# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1's three trained DSpark stages, stored under checkpoint mtp.*."""

import json
import os
import time
from collections.abc import Sequence
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder

from vllm_ascend.models.common.ops.draft_attention_graph import DraftAttentionGraph

from .attention import (
    DeepseekV41Attention,
    DeepseekV41Backend,
    DeepseekV41Cache,
    DeepseekV41Metadata,
    DeepseekV41RMSNorm,
    RequestSpan,
    rotary_frequencies,
)
from .attention_ops import fp8_quant_dequant
from .core import hc_pre, make_identity_pre_mix
from .engram import IndexedMmapStore
from .fused_weights import load_indexed_v41_weights
from .model import DeepseekV41Block, read_parameter_slice
from .moe import DeepseekV41MoE
from .native_attention import DeepseekV41NativeAttention

_STAGING_QUERY_ROWS = 5
_STAGING_WINDOW = 128
_STAGING_HEAD_DIM = 512
_STAGING_MAX_REQUESTS = 8
_WITNESS_BATCH_SIZE = 4


class DSparkRequestSpans(Sequence):
    """Keep exact device lengths until an eager consumer needs Python spans."""

    def __init__(self, bounds, seq_lens):
        self.query_bounds = tuple(bounds)
        self.device_seq_lens = seq_lens
        self._spans = None

    def __len__(self):
        return len(self.query_bounds) - 1

    @property
    def materialized(self):
        return self._spans is not None

    def __getitem__(self, index):
        if self._spans is None:
            lengths = self.device_seq_lens.cpu().tolist()
            self._spans = [
                RequestSpan(lo, hi, length) for lo, hi, length in zip(self.query_bounds, self.query_bounds[1:], lengths)
            ]
        return self._spans[index]


class DSparkMetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    # The initial proposal implementation uses packed, variable-length EP MoE.
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        common = common_attn_metadata
        bounds = common.query_start_loc_cpu[: common.num_reqs + 1].tolist()
        # The device lengths include rejected-token corrections. The attention
        # graph consumes them directly; eager fallback resolves the exact spans
        # once, never substituting the target's optimistic host upper bounds.
        requests = DSparkRequestSpans(bounds, common.seq_lens[: common.num_reqs])
        return DeepseekV41Metadata(requests, common.block_table_tensor, common.slot_mapping, common.num_actual_tokens)

    def build_for_drafting(self, common_attn_metadata, draft_index=1, **kwargs):
        return self.build(0, common_attn_metadata)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        # Target capture can build metadata for a cache group containing a draft
        # layer. The independent draft proposer never records that layer.
        return self.build(0, common_attn_metadata)


class DSparkBackend(DeepseekV41Backend):
    @staticmethod
    def get_name():
        return "DEEPSEEK_V41_DSPARK"

    @staticmethod
    def get_builder_cls():
        return DSparkMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return DSparkAttention


class DSparkCache(DeepseekV41Cache):
    def get_attn_backend(self):
        return DSparkBackend


def ring_positions(context_length, window_size, *, device):
    """Logical positions in official KV ring slot order, including after wrap."""
    slots = torch.arange(min(context_length, window_size), device=device)
    return context_length - 1 - (context_length - 1 - slots).remainder(window_size)


def block_attention(q, context_kv, query_kv, sink, scale, kernel=None):
    """Every query sees the accepted context and every KV of its *own* block."""
    kv = torch.cat((context_kv, query_kv), dim=0).unsqueeze(0)
    indices = torch.arange(kv.shape[1], dtype=torch.int32, device=q.device)
    indices = indices.view(1, 1, -1).expand(1, q.shape[0], -1)
    if kernel is None:
        kernel = DeepseekV41NativeAttention()
    return kernel(q.unsqueeze(0), kv, sink, indices, scale, mode="SWA draft bidirectional")[0]


def try_stage_block_inputs(q, query_kv, metadata, cache, *, window_size, block_size, dtype):
    """Stage full rings together with request-owned KV rows and logical indices.

    Eligibility uses existing host metadata and tensor geometry only. In particular,
    no device scalar is read to decide whether to take this eager-only path. Short,
    padded or partial proposals keep the original per-request implementation.
    """
    batch = len(metadata.requests)
    if not (
        1 < batch <= _STAGING_MAX_REQUESTS
        and block_size == _STAGING_QUERY_ROWS
        and window_size == cache.block_size == _STAGING_WINDOW
        and q.ndim == 3
        and query_kv.ndim == 2
        and q.shape[0] == query_kv.shape[0] == metadata.num_actual_tokens == batch * block_size
        and q.shape[-1] == query_kv.shape[-1] == _STAGING_HEAD_DIM
        and q.dtype == query_kv.dtype == dtype == torch.bfloat16
        and q.device == query_kv.device
        and q.is_contiguous()
        and query_kv.is_contiguous()
        and all(
            span.begin == row * block_size and span.end == (row + 1) * block_size and span.start_pos >= window_size
            for row, span in enumerate(metadata.requests)
        )
    ):
        return None
    table = metadata.block_table_tensor
    if not (
        table.ndim == 2
        and table.shape[0] >= batch
        and table.dtype in (torch.int32, torch.int64)
        and table.device == q.device
        and all((span.start_pos - 1) // cache.block_size < table.shape[1] for span in metadata.requests)
    ):
        return None
    field = cache.field(0, 0, q.shape[-1])
    if not (
        field.ndim == 2
        and field.shape[1] == q.shape[-1]
        and field.device == q.device
        and field.dtype == dtype
        and field.stride(1) == 1
        and field.stride(0) >= field.shape[1]
    ):
        return None

    # The accepted history was already copied by DSparkMetadataBuilder. This one
    # small H2D vector replaces per-request ring construction; include it in timing.
    starts = torch.tensor([span.start_pos for span in metadata.requests], dtype=torch.long, device=q.device)
    ring_slots = torch.arange(window_size, device=q.device)
    history = starts[:, None] - 1 - (starts[:, None] - 1 - ring_slots).remainder(window_size)
    pages = table[:batch].gather(1, history // cache.block_size).long()
    slots = pages * cache.block_size + history % cache.block_size
    # Match cache.read's advanced indexing, including its signed-index semantics.
    # Do not use index_select: a narrowed cache field can materialize the full pool.
    context = field[slots]
    kv = torch.cat((context, query_kv.view(batch, block_size, q.shape[-1])), dim=1)
    indices = torch.arange(window_size + block_size, dtype=torch.int32, device=q.device)
    indices = indices.view(1, 1, -1).expand(1, block_size, -1)
    return kv, indices


def is_draft_dummy():
    ctx = get_forward_context()
    return getattr(ctx, "in_profile_run", False) or (
        getattr(ctx, "is_draft_model", False)
        and ctx.attn_metadata is None
        and getattr(ctx, "draft_attn_metadatas", None) == []
    )


class DSparkAttention(DeepseekV41Attention):
    """TP-head attention over paged target context and a bidirectional draft block."""

    def __init__(self, config, vllm_config, prefix):
        nn.Module.__init__(self)
        self.tp = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % self.tp or config.o_groups % self.tp:
            raise ValueError("DSpark heads and output groups must divide TP")
        self.heads = self.local_heads = config.num_attention_heads // self.tp
        self.groups = config.o_groups // self.tp
        self.dsa_cp = False
        self.attention_kernel = DeepseekV41NativeAttention()
        self._staging_b4_witness_pending = True
        self._attention_graph = DraftAttentionGraph() if getattr(vllm_config, "lora_config", None) is None else None
        self.sequence_group = None
        self.dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.config = config
        quant = vllm_config.quant_config
        args = {"bias": False, "quant_config": quant}
        self.attn_sink = nn.Parameter(torch.empty(self.heads, dtype=torch.float32))
        self.wq_a = ReplicatedLinear(config.hidden_size, config.q_lora_rank, prefix=prefix + ".wq_a", **args)
        self.q_norm = DeepseekV41RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank, config.num_attention_heads * self.dim, prefix=prefix + ".wq_b", **args
        )
        self.wkv = ReplicatedLinear(config.hidden_size, self.dim, prefix=prefix + ".wkv", **args)
        self.kv_norm = DeepseekV41RMSNorm(self.dim, config.rms_norm_eps)
        self.wo_a = ColumnParallelLinear(
            config.num_attention_heads * self.dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=prefix + ".wo_a",
        )
        self.wo_b = RowParallelLinear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            reduce_results=False,
            prefix=prefix + ".wo_b",
            **args,
        )
        self.cache = DSparkCache(config, vllm_config, prefix + ".cache")
        self.register_buffer(
            "frequencies", rotary_frequencies(config, False).to(self.attn_sink.device), persistent=False
        )

    def project_kv(self, states, positions):
        kv = self.kv_norm(self.wkv(states)[0])
        self.rotate(kv.unsqueeze(0), positions.reshape(-1))
        return fp8_quant_dequant(kv, 32)

    def store_context(self, states, positions, slots):
        kv = self.project_kv(states, positions)
        if slots is None:
            return
        if slots.shape != positions.shape or states.shape[0] != slots.numel():
            raise ValueError("DSpark context positions, cache slots and hidden rows must align")
        # Slots for discarded / padded context tokens must never change slot 0.
        active = (slots >= 0).nonzero().flatten()
        self.cache.write_slots(slots.index_select(0, active), kv.index_select(0, active), 0, 0)

    def _record_staging_b4_witness(self, q, query_kv, metadata, staged):
        """Log one completed B4 invocation using host metadata only, then close."""
        self._staging_b4_witness_pending = False

        def geometry(tensor):
            return {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }

        source = Path(__file__).resolve()
        record = {
            "schema": "dspark-staging-b4-witness-v2",
            "pid": os.getpid(),
            "host_monotonic_ns": time.monotonic_ns(),
            "cache_prefix": self.cache.prefix,
            "source_path": str(source),
            "batch": len(metadata.requests),
            "num_actual_tokens": metadata.num_actual_tokens,
            "window_size": self.config.sliding_window,
            "block_size": self.config.dspark_block_size,
            "spans": [[span.begin, span.end, span.start_pos] for span in metadata.requests],
            "q": geometry(q),
            "query_kv": geometry(query_kv),
            "block_table": geometry(metadata.block_table_tensor),
            "staging_used": staged is not None,
            # Reaching this point means the batched call or every fallback call
            # returned. A call error exits forward before this bounded record.
            "completed_native_calls": (
                1 if staged is not None else sum(span.begin < metadata.num_actual_tokens for span in metadata.requests)
            ),
            "staged_kv": None if staged is None else geometry(staged[0]),
            "staged_indices": None if staged is None else geometry(staged[1]),
            "witness_closed": not self._staging_b4_witness_pending,
        }
        marker = "DSPARK_BATCH_ATTENTION_RUNTIME %s" if staged is not None else "DSPARK_STAGING_B4_WITNESS %s"
        logger.info(marker, json.dumps(record, sort_keys=True, separators=(",", ":")))

    def forward(self, hidden, positions):
        positions = positions.reshape(-1)
        graph = getattr(self, "_attention_graph", None)
        if (
            graph is not None
            and hidden.device.type == "npu"
            and torch.is_inference_mode_enabled()
            and not is_draft_dummy()
        ):
            metadata = self.cache.metadata()
            attended = graph(self, hidden, positions, metadata)
            if attended is not None:
                if isinstance(metadata.requests, DSparkRequestSpans) and not metadata.requests.materialized:
                    logger.info_once(
                        "DSPARK_DEVICE_HISTORY_RUNTIME pid=%d stage=%s rows=%d",
                        os.getpid(),
                        self.cache.prefix,
                        hidden.shape[0],
                        scope="process",
                    )
                logger.info_once(
                    "DSPARK_ATTN_GRAPH_RUNTIME pid=%d stage=%s rows=%d",
                    os.getpid(),
                    self.cache.prefix,
                    hidden.shape[0],
                    scope="process",
                )
                return self._project_output(attended, hidden.dtype)
        qr = self.q_norm(self.wq_a(hidden)[0])
        q = self.wq_b(qr)[0].view(-1, self.heads, self.dim)
        self.rotate(q.unsqueeze(0), positions)
        query_kv = self.project_kv(hidden, positions)
        metadata = self.cache.metadata()
        if metadata is None:
            if not is_draft_dummy():
                raise RuntimeError("DSpark proposal requires the draft layer's paged metadata")
            attended = torch.zeros_like(q)
        else:
            staged = try_stage_block_inputs(
                q,
                query_kv,
                metadata,
                self.cache,
                window_size=self.config.sliding_window,
                block_size=self.config.dspark_block_size,
                dtype=hidden.dtype,
            )
            if staged is not None:
                kv, indices = staged
                batch = len(metadata.requests)
                # Staging proves complete contiguous blocks and request-owned KV.
                # Native physical_indices adds each row's padded page offset.
                attended = self.attention_kernel(
                    q.view(batch, self.config.dspark_block_size, self.heads, self.dim),
                    kv,
                    self.attn_sink,
                    indices.expand(batch, -1, -1),
                    self.dim**-0.5,
                    mode="SWA draft bidirectional",
                ).reshape_as(q)
            else:
                attended = torch.zeros_like(q)
                for row, span in enumerate(metadata.requests):
                    if span.begin >= metadata.num_actual_tokens:
                        continue
                    if span.end - span.begin != self.config.dspark_block_size or span.start_pos < 0:
                        raise ValueError("DSpark requires a complete anchor-first query block per request")
                    rows = slice(span.begin, span.end)
                    history = ring_positions(span.start_pos, self.config.sliding_window, device=q.device)
                    context = self.cache.read(metadata, row, history, 0, 0, self.dim).to(hidden.dtype)
                    attended[rows] = block_attention(
                        q[rows],
                        context,
                        query_kv[rows],
                        self.attn_sink,
                        self.dim**-0.5,
                        self.attention_kernel,
                    )
            if self._staging_b4_witness_pending and len(metadata.requests) == _WITNESS_BATCH_SIZE:
                self._record_staging_b4_witness(q, query_kv, metadata, staged)
        self.rotate(attended.unsqueeze(0), positions, inverse=True)
        return self._project_output(attended, hidden.dtype)


class DSparkMoE(DeepseekV41MoE):
    """The draft block has all DP rows; SP experts own contiguous local rows."""

    def forward(self, states):
        comm = self.communication
        if not comm.sequence_parallel:
            return super().forward(states)
        tp = comm.tp_group
        padded = F.pad(states, (0, 0, 0, -states.shape[0] % tp.world_size))
        width = padded.shape[0] // tp.world_size
        local = padded.narrow(0, tp.rank_in_group * width, width).contiguous()
        output = super().forward(local)
        if not width:
            return output
        return tp.all_gather(output, dim=0)[: states.shape[0]]


class DSparkStage(DeepseekV41Block):
    # The HC timing is exactly the target block's. Only its two sublayers,
    # checkpoint paths, and the number of routed experts differ.
    def __init__(self, config, vllm_config, stage):
        nn.Module.__init__(self)
        self.config = config
        prefix = f"mtp.{stage}"
        self.attn = DSparkAttention(config, vllm_config, prefix + ".attn")
        expert_config = copy(config)
        expert_config.n_routed_experts = config.dspark_n_routed_experts
        expert_config.num_experts_per_tok = config.dspark_num_experts_per_tok
        self.ffn = DSparkMoE(expert_config, vllm_config.quant_config, prefix + ".ffn")
        self.attn_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        width = (config.hc_mult + 2) * config.hc_mult
        for part in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{part}_fn",
                nn.Parameter(torch.empty(width, config.hc_mult * config.hidden_size, dtype=torch.float32)),
            )
            self.register_parameter(f"hc_{part}_scale", nn.Parameter(torch.empty(3, dtype=torch.float32)))
            self.register_parameter(f"hc_{part}_base", nn.Parameter(torch.empty(width, dtype=torch.float32)))


class DSparkMarkov(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dspark_markov_rank, dtype=torch.bfloat16)
        self.head = nn.Linear(config.dspark_markov_rank, config.vocab_size, bias=False, dtype=torch.float32)


class DSparkConfidence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size + config.dspark_markov_rank, 1, bias=False, dtype=torch.float32)


class DeepseekV41DSparkForCausalLM(nn.Module):
    """Full-vocabulary draft consumed by the Ascend DSpark block proposer."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        spec = vllm_config.speculative_config
        self.config = config = vllm_config.model_config.hf_text_config
        if (
            spec is None
            or spec.num_speculative_tokens != config.dspark_block_size
            or not config.dspark_target_layer_ids
        ):
            raise ValueError("V4.1 DSpark needs its trained query block and target auxiliary layers")
        count = len(config.compress_ratios) - config.num_hidden_layers
        if count <= 0 or any(config.compress_ratios[config.num_hidden_layers :]):
            raise ValueError("DSpark checkpoint must have uncompressed attention in every draft stage")
        self.mtp = nn.ModuleList([DSparkStage(config, vllm_config, stage) for stage in range(count)])
        self.mtp[0].main_proj = ReplicatedLinear(
            config.hidden_size * len(config.dspark_target_layer_ids),
            config.hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix="mtp.0.main_proj",
            return_bias=False,
        )
        self.mtp[0].main_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mtp[0].embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix="mtp.0.embed")
        self.mtp[-1].norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mtp[-1].head = ParallelLMHead(
            config.vocab_size, config.hidden_size, params_dtype=torch.float32, prefix=f"mtp.{count - 1}.head"
        )
        self.mtp[-1].markov_head = DSparkMarkov(config)
        self.mtp[-1].confidence_head = DSparkConfidence(config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.model_path = Path(vllm_config.model_config.model)

    def embed_input_ids(self, input_ids):
        return self.mtp[0].embed(input_ids)

    def combine_hidden_states(self, aux_hidden_states):
        return self.mtp[0].main_norm(self.mtp[0].main_proj(aux_hidden_states))

    def get_draft_kv_cache_layer_names(self):
        return [stage.attn.cache.prefix for stage in self.mtp]

    def get_draft_attn_causal(self):
        return [False] * len(self.mtp)

    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None):
        if not context_states.numel():
            return
        if context_slot_mapping is None:
            if not is_draft_dummy():
                raise RuntimeError("Missing cache slots for real DSpark context")
            context_slot_mapping = [None] * len(self.mtp)
        if len(context_slot_mapping) != len(self.mtp):
            raise ValueError("DSpark needs one context slot mapping per stage")
        if any(slots is None for slots in context_slot_mapping) and not is_draft_dummy():
            raise RuntimeError("Missing cache slots for a real DSpark stage")
        for stage, slots in zip(self.mtp, context_slot_mapping):
            stage.attn.store_context(context_states, context_positions, slots)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        hidden = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = hidden.unsqueeze(-2).repeat(1, self.config.hc_mult, 1)
        pre = make_identity_pre_mix(hidden)
        for stage in self.mtp:
            hidden, pre = stage(hidden, pre, positions)
        return hc_pre(hidden, pre)

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.logits_processor(self.mtp[-1].head, self.mtp[-1].norm(hidden_states).float())

    def compute_draft_logits(self, hidden_states):
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, token_ids):
        return token_ids

    def markov_embed(self, token_ids):
        return self.mtp[-1].markov_head.embed(token_ids.long())

    def markov_bias(self, embedding):
        return self.mtp[-1].markov_head.head(embedding.float())

    def compute_confidence(self, hidden_states, markov_embed):
        features = torch.cat((hidden_states, markov_embed), dim=-1)
        return self.mtp[-1].confidence_head.proj(features.float()).squeeze(-1).sigmoid()

    @torch.no_grad()
    def load_weights(self, weights):
        # The generic iterator includes backbone and very large Engram tensors.
        # Only open the shards that own this EP/TP rank's trained draft weights.
        with IndexedMmapStore(self.model_path) as store:
            loaded = load_indexed_v41_weights(
                self,
                store.weight_map,
                get_tensor_model_parallel_rank(),
                get_tensor_model_parallel_world_size(),
                read_parameter_slice,
            )
        logger.info("Loaded %d indexed V4.1 DSpark parameters on this TP/EP rank", len(loaded))
        return loaded
