from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.distributed.afd_transfer.afd_connector.metadata import \
    AFDConnectorMetadata
from vllm.model_executor.models.deepseek_v2 import (DeepseekV2Model,
                                                    _get_llama_4_scaling,
                                                    apply_dbo_yield)
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context

from vllm_ascend.ops.weight_prefetch import \
    maybe_prefetch_mla_preprocess_weights


def forward_m2n(self, hidden_states, residual, positions, afd_metadata,
                llama_4_scaling):
    recv_handle = None
    forward_ctx = get_forward_context()
    afd_connector = afd_metadata.afd_connector

    for layer in islice(self.layers, self.start_layer, self.end_layer):
        afd_metadata.afd_stage_idx = forward_ctx.ubatch_idx

        if recv_handle is not None:
            for work in recv_handle:
                work.wait()

        if layer.layer_idx > 0:
            maybe_prefetch_mla_preprocess_weights(layer.self_attn,
                                                  hidden_states)
            hidden_states = afd_connector.recv_ffn_output(
                hidden_states=hidden_states,
                metadata=None,
            )

        current_hidden, residual, topk_weights, topk_ids, router_logits = \
            layer.compute_attn_output(positions, hidden_states, residual,
                                      llama_4_scaling)

        metadata = AFDConnectorMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=afd_metadata.afd_stage_idx,
            seq_len=hidden_states.shape[0],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            num_ubatches=forward_ctx.num_ubatches,
            connector_data=None,
        )

        afd_connector.configure_metadata(metadata,
                                         config=self.config,
                                         batch_size=self.max_num_reqs)

        hidden_states, send_attn_handle = afd_connector.send_attn_output(
            hidden_states=current_hidden,
            metadata=metadata,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            dynamic_scales=None,
        )

        if send_attn_handle is not None:
            metadata.connector_data.handle = send_attn_handle

        hidden_states = apply_dbo_yield(hidden_states)

    hidden_states = afd_connector.recv_ffn_output(hidden_states=hidden_states,
                                                  metadata=afd_metadata)
    return hidden_states, residual


def forward(
    self,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
):
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Compute llama 4 scaling once per forward pass if enabled
    # Note(wxy): This is a hack fix to avoid graph mode error for torch 2.8
    # We'll find a better way to remove this patch.
    try:
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling")
    except AttributeError:
        llama_4_scaling_config = None
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    # support afd
    forward_ctx = get_forward_context()
    afd_metadata = forward_ctx.afd_metadata if forward_ctx is not None else None

    if afd_metadata is not None:
        hidden_states, residual = self.forward_m2n(hidden_states, residual, positions, afd_metadata,
                                                   llama_4_scaling)
    else:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual,
                                            llama_4_scaling)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual
        })

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


DeepseekV2Model.forward_m2n = forward_m2n
DeepseekV2Model.forward = forward
