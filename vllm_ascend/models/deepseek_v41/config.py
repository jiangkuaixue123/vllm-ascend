from transformers import PretrainedConfig


class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v41_text"

    def __init__(self, max_position_embeddings=1048576, rope_scaling=None, rope_parameters=None,
                 rope_theta=10000, **kwargs):
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = dict(rope_parameters or rope_scaling or {"rope_type": "default"})
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        super().__init__(**kwargs)


class DeepseekV41Config(PretrainedConfig):
    """Expose the frozen text topology to vLLM; the custom loader owns quantization."""

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}

    def __init__(self, text_config=None, vision_config=None, quantization_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        self.text_config = text_config if text_config is not None else DeepseekV41TextConfig()
        # Preserve the disk contract while keeping the generic FP8 re-packer out
        # of this model. The loader retains packed FP4 and the shared host tables.
        self.checkpoint_quantization = quantization_config or kwargs.pop("checkpoint_quantization", None)
        self.checkpoint_vision_config = vision_config or kwargs.pop("checkpoint_vision_config", None)
        super().__init__(**kwargs)


def text_model_args(text_config, *, max_batch_size, max_seq_len):
    """Translate checkpoint text_config names to the text and MTP block constructors."""
    from types import SimpleNamespace

    # The original checkpoint declares the matrix storage in its outer
    # quantization_config. The checkpoint loader validates this FP8/MXFP4 schema.
    values = dict(max_batch_size=max_batch_size, max_seq_len=max_seq_len,
                  dtype="fp8", expert_dtype="fp4", gate_temp=getattr(text_config, "gate_temp", 1.0))
    direct = (
        "vocab_size", "n_routed_experts", "n_shared_experts", "norm_topk_prob", "swiglu_limit",
        "q_lora_rank", "head_dim", "o_groups", "o_lora_rank", "compress_ratios", "compress_rope_theta",
        "index_n_heads", "index_head_dim", "index_topk", "candidate_topk_blocks", "candidate_block_size",
        "hc_mult", "hc_sinkhorn_iters", "hc_eps", "engram_layer_ids", "engram_num_embeddings",
        "engram_max_ngram_size", "engram_vocab_size", "engram_n_heads", "engram_head_dim",
        "engram_compressed_vocab_size", "dspark_block_size", "dspark_noise_token_id", "dspark_target_layer_ids",
        "dspark_markov_rank", "dspark_n_routed_experts",
    )
    renamed = {
        "dim": "hidden_size", "moe_inter_dim": "moe_intermediate_size", "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads", "rope_head_dim": "qk_rope_head_dim", "norm_eps": "rms_norm_eps",
        "n_activated_experts": "num_experts_per_tok", "score_func": "scoring_func",
        "route_scale": "routed_scaling_factor", "window_size": "sliding_window",
        "kv_source_layers": "kv_source_layer_ids", "index_source_layers": "index_source_layer_ids",
        "candidate_source_layer": "candidate_source_layer_id", "engram_pad_id": "engram_pad_token_id",
        "n_mtp_layers": "num_nextn_predict_layers", "dspark_n_activated_experts": "dspark_num_experts_per_tok",
    }
    values.update({field: getattr(text_config, field) for field in direct})
    values.update({field: getattr(text_config, source) for field, source in renamed.items()})
    rope = text_config.rope_parameters
    values.update(rope_theta=rope["rope_theta"], original_seq_len=rope["original_max_position_embeddings"],
                  rope_factor=rope["factor"], beta_fast=rope["beta_fast"], beta_slow=rope["beta_slow"])
    return SimpleNamespace(**values)
