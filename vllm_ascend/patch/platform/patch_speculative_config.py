from contextlib import contextmanager
from copy import copy

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

from vllm_ascend.models.deepseek_v41.config import DSPARK_ARCHITECTURE, DeepseekV41Config, DeepseekV41TextConfig

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override
_orig_update_arch = SpeculativeConfig.update_arch_

# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _orig_hf_config_override(hf_config)
    architectures = hf_config.architectures or ()
    if hf_config.model_type == "qwen3" and "DSparkDraftModel" in architectures:
        dflash_config = hf_config.dflash_config
        hf_config.update(
            {
                "architectures": ["Qwen3DSparkModel"],
                "mask_token_id": dflash_config["mask_token_id"],
                "target_layer_ids": dflash_config["target_layer_ids"],
            }
        )
    return hf_config


@contextmanager
def _temporarily_disable_dspark_dcp(self: SpeculativeConfig):
    target_parallel_config = self.target_parallel_config
    if getattr(self, "method", None) != "dspark" or target_parallel_config.decode_context_parallel_size <= 1:
        yield
        return

    guard_parallel_config = copy(target_parallel_config)
    guard_parallel_config.decode_context_parallel_size = 1
    self.target_parallel_config = guard_parallel_config
    try:
        yield
    finally:
        self.target_parallel_config = target_parallel_config


def _dspark_post_init(self):
    # TODO: This block can be deleted after the upstream supports the overlay of mla dcp and dspark
    with _temporarily_disable_dspark_dcp(self):
        _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


def _update_draft_arch(self):
    # Upstream dispatches embedded DSpark checkpoints to the V4 draft class.
    # V4.1 stores its text/draft configuration inside the multimodal wrapper.
    # Select the trained V4.1 architecture before vLLM inspects the class.
    if self.method == "dspark" and isinstance(self.target_model_config.hf_text_config, DeepseekV41TextConfig):
        if self.model != self.target_model_config.model:
            raise ValueError("V4.1 DSpark uses the draft stages embedded in the target checkpoint")
        if self.draft_tensor_parallel_size not in (None, self.target_parallel_config.tensor_parallel_size):
            raise ValueError("V4.1 DSpark requires the target tensor-parallel group size")
        draft = self.draft_model_config.hf_config
        text = draft.text_config
        if self.num_speculative_tokens not in (None, text.dspark_block_size):
            raise ValueError(f"V4.1 DSpark requires its trained block of {text.dspark_block_size} tokens")
        self.num_speculative_tokens = text.dspark_block_size
        draft.model_type = DeepseekV41Config.model_type
        draft.architectures = [DSPARK_ARCHITECTURE]
        draft.dspark_block_size = text.dspark_block_size
        draft.dspark_target_layer_ids = text.dspark_target_layer_ids
        draft.ptd_token_id = text.dspark_noise_token_id
        draft.sample_from_anchor = True
    return _orig_update_arch(self)


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.update_arch_ = _update_draft_arch
SpeculativeConfig.__post_init__ = _dspark_post_init
