# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

from transformers import AutoConfig, PretrainedConfig


class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v41_text"

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=5120,
        moe_intermediate_size=2304,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        hidden_act="silu",
        swiglu_limit=10.0,
        rms_norm_eps=1e-20,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=False,
        max_position_embeddings=1048576,
        rope_theta=10000,
        rope_scaling=None,
        n_routed_experts=384,
        n_shared_experts=1,
        num_experts_per_tok=6,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        sliding_window=128,
        compress_ratios=None,
        compress_rope_theta=160000,
        kv_source_layer_ids=None,
        index_source_layer_ids=None,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram_layer_ids=None,
        engram_num_embeddings=None,
        engram_max_ngram_size=4,
        engram_vocab_size=16000000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=99092,
        num_nextn_predict_layers=3,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=None,
        dspark_markov_rank=256,
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
        **kwargs,
    ):
        """Restore text, per-layer compression, expert routing and Engram settings from the checkpoint."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = (
            deepcopy(rope_scaling)
            if rope_scaling is not None
            else {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            }
        )
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        self.compress_ratios = (
            list(compress_ratios) if compress_ratios is not None else [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        )
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = list(kv_source_layer_ids) if kv_source_layer_ids is not None else [2, 8, 14, 20]
        self.index_source_layer_ids = (
            list(index_source_layer_ids) if index_source_layer_ids is not None else [2, 8, 14, 20, 24, 28, 32, 36]
        )
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.engram_layer_ids = list(engram_layer_ids) if engram_layer_ids is not None else [1, 14]
        self.engram_num_embeddings = (
            list(engram_num_embeddings) if engram_num_embeddings is not None else [384006168, 384016682]
        )
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = (
            list(dspark_target_layer_ids) if dspark_target_layer_ids is not None else [37, 38, 39]
        )
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"

    def __init__(
        self,
        num_hidden_layers=32,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=2816,
        patch_size=14,
        rope_theta=10000,
        downsample_ratio=3,
        max_image_tokens=1024,
        min_pixels=295936,
        max_wh_ratio=None,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio
        super().__init__(**kwargs)


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"
    sub_configs = {
        "text_config": DeepseekV41TextConfig,
        "vision_config": DeepseekV41VisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=129264,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        for name, value in (
            ("text_config", text_config),
            ("vision_config", vision_config),
        ):
            config_class = self.sub_configs[name]
            if value is None:
                value = config_class()
            elif isinstance(value, dict):
                value = config_class(**deepcopy(value))
            elif not isinstance(value, config_class):
                raise TypeError(f"{name} must be a dict or {config_class.__name__}")
            setattr(self, name, value)
        self.image_token_id = image_token_id
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def register_config():
    for cls in (DeepseekV41Config, DeepseekV41TextConfig, DeepseekV41VisionConfig):
        AutoConfig.register(cls.model_type, cls, exist_ok=True)
