# SPDX-License-Identifier: Apache-2.0
"""Select the V4.1 draft architecture for the installed core's DSpark method."""

from functools import wraps

ARCHITECTURE = "DeepseekV41DSparkDraftModel"


def is_v41_spec(spec):
    if spec is None or spec.method != "dspark":
        return False
    config = getattr(getattr(spec, "target_model_config", None), "hf_config", None)
    return getattr(config, "model_type", None) == "deepseek_v41"


def register_spec_config():
    from vllm import ModelRegistry
    from vllm.config import SpeculativeConfig

    previous = SpeculativeConfig.update_arch_
    if getattr(previous, "_handles_v41_dspark", False):
        return
    ModelRegistry.register_model(ARCHITECTURE, f"{__package__}.dspark:DSparkDraft")

    @wraps(previous)
    def set_architecture(self):
        if is_v41_spec(self):
            hf = self.draft_model_config.hf_config
            text = self.target_model_config.hf_text_config
            if self.num_speculative_tokens not in (None, text.dspark_block_size):
                raise ValueError("V4.1 DSpark requires one complete checkpoint block")
            hf.model_type = "deepseek_v41"
            hf.architectures = [ARCHITECTURE]
            hf.n_predict = hf.dspark_block_size = text.dspark_block_size
            hf.pard_token = hf.dspark_noise_token_id = text.dspark_noise_token_id
            hf.dspark_target_layer_ids = list(text.dspark_target_layer_ids)
            hf.eagle_aux_hidden_state_layer_ids = list(text.dspark_target_layer_ids)
        return previous(self)

    set_architecture._handles_v41_dspark = True
    SpeculativeConfig.update_arch_ = set_architecture
