"""Ascend V4.1 text generation with eager DP8/EP8 execution."""
import os

import torch
from torch import nn
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.tokenizers import cached_tokenizer_from_config

from .backend import TextBackbone


class AscendDeepseekV41ForCausalLM(nn.Module):
    def __new__(cls, *, vllm_config=None, prefix=""):
        if cls is AscendDeepseekV41ForCausalLM and vllm_config is not None:
            if (
                vllm_config.scheduler_config.max_num_seqs > 1
                or vllm_config.speculative_config is not None
                or not vllm_config.model_config.enforce_eager
                or getattr(getattr(vllm_config, "cache_config", None), "enable_prefix_caching", False)
            ):
                from .graph_model import DeepseekV41Stage2ForCausalLM

                cls = DeepseekV41Stage2ForCausalLM
        return super().__new__(cls)

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        p = vllm_config.parallel_config
        s = vllm_config.scheduler_config
        m = vllm_config.model_config
        if not (p.data_parallel_size == 8 and p.enable_expert_parallel
                and p.tensor_parallel_size == p.pipeline_parallel_size == 1
                and getattr(p, "prefill_context_parallel_size", 1) == 1):
            raise ValueError("Stage 1 requires DP8/EP8, TP1/PP1/PCP1")
        if (not m.enforce_eager or vllm_config.cache_config.enable_prefix_caching
                or s.max_num_seqs != 1 or vllm_config.speculative_config is not None):
            raise ValueError("Stage 1 requires eager, one sequence/rank, prefix cache and speculation off")
        group = get_ep_group()
        self.dp_rank = get_dp_group().rank_in_group
        if group.world_size != 8:
            raise RuntimeError(f"Expected eight EP members, got {group.world_size}")
        local = m.hf_text_config.n_routed_experts // group.world_size
        self.expert_range = (group.rank_in_group * local, (group.rank_in_group + 1) * local)
        tokenizer = cached_tokenizer_from_config(model_config=m)
        self.backbone = TextBackbone(
            m.hf_text_config, tokenizer, device=torch.device("npu", torch.npu.current_device()),
            max_seq_len=m.max_model_len, ep_group=group.device_group,
        )
        # Register one small scheduler-visible cache group. Its metadata gives
        # the real query length apart from DP padding. Compressed KV, indexes
        # and n-gram history live in the DP rank's bounded TextBackbone buffers.
        self.metadata_key = f"{prefix + '.' if prefix else ''}model.layers.0.self_attn.attn"
        self.metadata_layer = Attention(1, 128, 128 ** -0.5, num_kv_heads=1,
                                        cache_config=vllm_config.cache_config, prefix=self.metadata_key)
        self.dim = self.backbone.dim
        self.trace = None
        trace_dir = os.getenv("VLLM_ASCEND_DSV41_TRACE_DIR")
        if trace_dir:
            from .diagnostics import EagerTrace
            inputs = os.environ.get("VLLM_ASCEND_DSV41_TRACE_INPUTS")
            if not inputs:
                raise ValueError("VLLM_ASCEND_DSV41_TRACE_INPUTS must name the trace's input directory")
            self.trace = EagerTrace(trace_dir, inputs, self.dp_rank, group.rank_in_group)

    def embed_input_ids(self, input_ids):
        return self.backbone.embed(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if input_ids is None or inputs_embeds is not None or intermediate_tensors is not None:
            raise ValueError("Stage 1 accepts text token IDs on its single pipeline stage")
        context = get_forward_context()
        md = context.attn_metadata
        mappings = getattr(context, "slot_mapping", None)
        slots = mappings.get(self.metadata_key) if isinstance(mappings, dict) else None
        out = self.backbone.embed.weight.new_zeros((input_ids.numel(), self.dim))
        if md is None or (slots is not None and not (slots >= 0).any().item()):
            self.backbone.idle_forward()
            return out
        if not isinstance(md, dict) or self.metadata_key not in md:
            raise RuntimeError("Missing per-request scheduler metadata for the eager cache group")
        data = md[self.metadata_key]
        # Only one request is admitted on each DP rank. Use the end of query
        # zero, since num_input_tokens can include other ranks' alignment pads.
        offsets = getattr(data, "query_start_loc", None)
        if offsets is not None:
            n = int(offsets[1]) if len(offsets) >= 2 else 0
        elif slots is not None:
            # Some backends expose only the cache slot mapping. With the
            # full-attention scheduling group all real tokens form a prefix.
            active = (slots >= 0).tolist()
            n = sum(active)
            if not all(active[:n]) or any(active[n:]):
                raise RuntimeError("Cannot determine real query length from scheduler slots")
        elif getattr(data, "actual_seq_lengths_q", None) is not None:
            n = int(data.actual_seq_lengths_q[0])
        else:
            raise RuntimeError(f"No query lengths or slots on metadata {type(data).__name__}")
        if n == 0:
            self.backbone.idle_forward()
            return out
        if n < 0 or n > input_ids.numel() or n > positions.numel():
            raise RuntimeError("Invalid scheduler query length")
        pos = positions[:n].tolist()
        if pos != list(range(pos[0], pos[0] + n)):
            raise RuntimeError("Expected one contiguous prompt or one continuing decode token")
        if self.trace is not None:
            self.trace.begin(input_ids[:n], pos[0], self.backbone.layers)
        hidden = self.backbone(input_ids[:n].reshape(1, n), start_pos=pos[0])
        out[:n] = hidden[0]
        return out

    def compute_logits(self, hidden_states):
        logits = self.backbone.head(hidden_states, full_logits=True)
        if self.trace is not None:
            self.trace.end(logits)
        return logits
