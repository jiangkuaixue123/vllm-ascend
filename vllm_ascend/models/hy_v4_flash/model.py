# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_ep_group, get_tp_group
from vllm.logger import logger
from vllm.models.hy_v4_flash.model import HYV4FlashDecoderLayer, HYV4FlashForCausalLM, HYV4FlashModel

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_shard

from .attention import AscendHYV4FlashAttention, FlashAttentionBackend
from .dfly_config import METHOD
from .engram import FlashEngram, FlashNgramHashState
from .ihc import FlashHCHeadLayer, FlashHCLayer
from .local_vocabulary import LocalVocabulary
from .moe import AscendFlashMoE
from .ordered_verification_scatter import matches_native_scatter

FLASHCOMM_950_PEER_BYTES = 768 * 1024
# Largest per-rank token payload measured on the dedicated A5 copy communicator.
_MAX_SMALL_CCU_GATHER_BYTES = 288 * 1024


def validate_runtime(vllm_config: VllmConfig) -> None:
    model = vllm_config.model_config
    config = model.hf_text_config
    config.validate_flash_config()
    parallel = vllm_config.parallel_config
    for field in (
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, field) != 1:
            raise ValueError(f"HYV4 Flash Ascend requires {field}=1")
    if parallel.enable_eplb or parallel.enable_dbo:
        raise ValueError("HYV4 Flash Ascend does not support EPLB or dual batch overlap")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("HYV4 Flash Ascend requires MRV2")
    if vllm_config.compilation_config.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        raise ValueError("HYV4 Flash Ascend supports eager or FULL_DECODE_ONLY ACL graphs")
    if model.dtype != torch.bfloat16:
        raise ValueError("HYV4 Flash Ascend requires BF16 compute")
    if not FlashAttentionBackend.supports_kv_cache_dtype(vllm_config.cache_config.cache_dtype):
        raise ValueError("HYV4 Flash Ascend requires BF16 or FP8 KV cache")
    if model.quantization == "fp8":
        block = getattr(vllm_config.quant_config, "weight_block_size", None)
        if tuple(block or ()) != (128, 128):
            raise ValueError("HYV4 Flash Ascend requires FP8 weights with 128x128 block scales")
    elif model.quantization is not None:
        raise ValueError("HYV4 Flash Ascend accepts BF16 or native block-wise FP8 checkpoints")
    if get_ascend_config().enable_fused_mc2:
        raise ValueError("HYV4 Flash Ascend requires fused MC2 disabled")
    spec = vllm_config.speculative_config
    if spec is not None and getattr(spec, "custom_method", None) != METHOD:
        raise ValueError("HYV4 Flash Ascend requires DFly for speculative decoding")
    # Packed cache pages include IndexPool and position state. Mooncake V2 pull
    # copies these replicated MLA pages whole, including across unequal TP sizes.
    transfer = vllm_config.kv_transfer_config
    if transfer is not None and (
        transfer.kv_connector != "MooncakeConnectorV2" or transfer.get_from_extra_config("use_layerwise", False)
    ):
        raise ValueError("HYV4 Flash Ascend PD requires MooncakeConnectorV2 in pull mode")
    if vllm_config.lora_config is not None:
        raise ValueError("HYV4 Flash Ascend does not support LoRA")
    if config.kv_lora_rank != 512 or config.qk_rope_head_dim != 64:
        raise ValueError("HYV4 Flash requires a 512-wide MLA latent and 64 rotary dimensions")
    if config.engram_layer_ids and not config.enable_ihc:
        raise ValueError("HYV4 Flash Engram requires iHC")
    if not (config.learnable_sink and config.gated_mla and config.gating_type == "elementwise"):
        raise ValueError("HYV4 Flash requires learned sinks and elementwise MLA gates")
    if (
        getattr(config, "gate_lora_rank", None) is not None
        or getattr(config, "o_lora_rank", None) is not None
        or getattr(config, "gate_groups", 1) != 1
        or getattr(config, "o_groups", 1) != 1
        or getattr(config, "full_nope", False)
        or not getattr(config, "qk_norm", True)
        or config.qk_rope_head_dim != config.swa_qk_rope_head_dim
    ):
        raise ValueError("HYV4 Flash Ascend requires ungrouped MLA projections and equal rotary widths")
    for field in ("num_attention_heads", "swa_num_attention_heads"):
        if getattr(config, field) % parallel.tensor_parallel_size:
            raise ValueError(f"{field} must be divisible by tensor_parallel_size")


def gather_verify_rows(hidden: torch.Tensor, copy_group: dist.ProcessGroup | None) -> torch.Tensor:
    if (
        copy_group is None
        or hidden.dtype != torch.bfloat16
        or not 0 < hidden.numel() * hidden.element_size() <= _MAX_SMALL_CCU_GATHER_BYTES
    ):
        return sp_all_gather(hidden)
    result = hidden.new_empty((copy_group.size() * hidden.shape[0], *hidden.shape[1:]))
    dist.all_gather_into_tensor(result, hidden.contiguous(), group=copy_group)
    logger.info_once("HYV4 Flash gathers verification token shards with CCU.")
    return result


class AscendHYV4FlashDecoderLayer(HYV4FlashDecoderLayer):
    attention_cls = AscendHYV4FlashAttention
    hc_layer_cls = FlashHCLayer
    moe_cls = AscendFlashMoE

    def __init__(self, config, vllm_config, cache_config=None, quant_config=None, prefix="", topk_indices_buffer=None):
        super().__init__(config, vllm_config, cache_config, quant_config, prefix, topk_indices_buffer)
        self.copy_group = None
        self.sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe and self.block_type == "moe"
        if self.sequence_parallel:
            layers = config.mlp_layer_types
            engram_layers = config.engram_layer_ids or ()
            index = self.layer_idx
            # Engram consumes full token rows before the decoder layer.
            self.input_is_sharded = index > 0 and layers[index - 1] != "dense" and index + 1 not in engram_layers
            self.keep_output_sharded = (
                index + 1 < len(layers) and layers[index + 1] != "dense" and index + 2 not in engram_layers
            )
            self.self_attn.o_proj.reduce_results = False
            self.self_attn.sequence_parallel_output = True
            if self.enable_ihc:
                self.hc_attn_layer.hc_pre.sequence_parallel_input = self.input_is_sharded
                self.hc_mlp_layer.hc_pre.sequence_parallel_input = True

    def _forward_normal(self, positions, hidden_states, residual):
        if not self.sequence_parallel:
            return super()._forward_normal(positions, hidden_states, residual)

        num_tokens = positions.shape[-1]
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.input_is_sharded:
            hidden_states = gather_verify_rows(hidden_states, self.copy_group)[:num_tokens]
        else:
            residual = sp_shard(residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        residual = hidden_states + residual
        hidden_states = self.mlp(self.post_attention_layernorm(residual))
        if not self.keep_output_sharded:
            hidden_states = gather_verify_rows(hidden_states, self.copy_group)[:num_tokens]
            residual = gather_verify_rows(residual, self.copy_group)[:num_tokens]
        return hidden_states, residual

    def _forward_ihc(self, positions, hidden_states):
        if not self.sequence_parallel:
            return super()._forward_ihc(positions, hidden_states)

        num_tokens = positions.shape[-1]
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        if self.input_is_sharded:
            hidden_states = gather_verify_rows(hidden_states, self.copy_group)[:num_tokens]
        else:
            residual = sp_shard(residual)
            post_gates = sp_shard(post_gates)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)

        hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_mlp_layer.pre(hidden_states)
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)
        if not self.keep_output_sharded:
            hidden_states = gather_verify_rows(hidden_states, self.copy_group)[:num_tokens]
        return hidden_states, None


class AscendHYV4FlashModel(HYV4FlashModel):
    decoder_layer_cls = AscendHYV4FlashDecoderLayer
    hc_head_cls = FlashHCHeadLayer
    engram_cls = FlashEngram
    engram_hash_cls = FlashNgramHashState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.parallel_config.use_sequence_parallel_moe and get_ascend_device_type() == AscendDeviceType.A5:
            # Large prefill transfers can select the multi-channel HCCL path.
            # Cap each peer's payload on the shared TP and routed EP groups.
            for group in (get_tp_group(), get_ep_group()):
                group.device_communicator.max_sequence_collective_peer_bytes = FLASHCOMM_950_PEER_BYTES
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.copy_group = None
        if (
            vllm_config.additional_config.get("hy4flash_ccu_copy_gathers", False)
            and get_ascend_device_type() == AscendDeviceType.A5
            and vllm_config.parallel_config.tensor_parallel_size == 8
            and vllm_config.parallel_config.data_parallel_size == 1
            and vllm_config.parallel_config.pipeline_parallel_size == 1
            and dist.get_world_size() == 8
        ):
            tp = get_tp_group()
            self.copy_group = tp.device_communicator.enable_ccu_copy_group(tp.ranks)
            for layer in self.layers:
                layer.copy_group = self.copy_group
            if vllm_config.additional_config.get("hy4flash_ordered_verify_scatter", False) and matches_native_scatter(
                tp, self.copy_group
            ):
                for layer in self.layers:
                    if layer.sequence_parallel:
                        layer.self_attn.ordered_scatter_group = self.copy_group
                        layer.self_attn.ordered_scatter_rank = tp.rank_in_group
        self.uses_dfly = vllm_config.speculative_config is not None
        self.local_vocabulary = None

    def embed_input_ids(self, input_ids):
        if self.local_vocabulary is not None and self.local_vocabulary.can_embed(input_ids):
            return self.local_vocabulary.embed(input_ids)
        # Async speculative batches can contain -1 at discarded query slots.
        return super().embed_input_ids(input_ids.clamp_min(0) if self.uses_dfly else input_ids)


class AscendHYV4FlashForCausalLM(HYV4FlashForCausalLM):
    model_cls = AscendHYV4FlashModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        parallel = vllm_config.parallel_config
        if (
            vllm_config.additional_config.get("hy4flash_local_vocabulary", False)
            and get_ascend_device_type() is AscendDeviceType.A5
            and parallel.tensor_parallel_size == 8
            and parallel.data_parallel_size == parallel.pipeline_parallel_size == 1
            and vllm_config.speculative_config is not None
            and LocalVocabulary.supports(self.model.embed_tokens, self.lm_head)
        ):
            self.model.local_vocabulary = LocalVocabulary(
                self.model.embed_tokens,
                self.lm_head,
                max_decode_tokens=min(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    vllm_config.scheduler_config.max_num_seqs * vllm_config.uniform_decode_query_len,
                ),
                copy_group=self.model.copy_group,
                sharded_verification=vllm_config.additional_config.get("hy4flash_sharded_greedy_verify", False)
                and vllm_config.lora_config is None,
            )

    @staticmethod
    def get_model_state_cls():
        from .model_state import FlashModelState

        return FlashModelState
