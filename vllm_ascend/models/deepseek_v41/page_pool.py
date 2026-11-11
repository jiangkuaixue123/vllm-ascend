# SPDX-License-Identifier: Apache-2.0
"""Account for the paged target's physical tensors in the standard vLLM KV planner."""

from dataclasses import replace

from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import FullAttentionSpec


class AutoPagedMetadata(Attention):
    def __init__(self, *args, private_page_bytes, **kwargs):
        super().__init__(*args, **kwargs)
        self.private_page_bytes = private_page_bytes

    def get_kv_cache_spec(self, vllm_config):
        marker = super().get_kv_cache_spec(vllm_config)
        if not isinstance(marker, FullAttentionSpec) or marker.page_size_padded is not None:
            raise ValueError("V4.1 expects a single full-history attention group")
        return replace(marker, page_size_padded=marker.page_size_bytes + self.private_page_bytes)

    def workload_budget(self, budget, vllm_config):
        scheduler = vllm_config.scheduler_config
        if (
            vllm_config.cache_config.enable_prefix_caching
            or scheduler.watermark
            or scheduler.scheduler_cls is not None
            or vllm_config.max_concurrent_batches != 1
            or vllm_config.kv_transfer_config is not None
            or vllm_config.ec_transfer_config is not None
        ):
            # These modes can retain or reserve blocks beyond the active rows.
            return budget
        spec = self.get_kv_cache_spec(vllm_config)
        # FullAttention's request table covers min(tokens + speculative lookahead,
        # max_model_len). The V1 block pool also reserves one null block.
        per_request = spec.max_num_blocks_per_req(vllm_config, vllm_config.model_config.max_model_len)
        blocks = scheduler.max_num_seqs * per_request + 1
        return min(budget, blocks * spec.page_size_bytes)

    def device_marker_config(self, config):
        if config.num_blocks < 1 or len(config.kv_cache_groups) != 1 or len(config.kv_cache_tensors) != 1:
            raise ValueError("V4.1 expects one populated scheduler KV cache group")
        group = config.kv_cache_groups[0]
        tensor = config.kv_cache_tensors[0]
        spec = group.kv_cache_spec
        if group.layer_names != [self.layer_name] or tensor.shared_by != group.layer_names:
            raise ValueError("V4.1 scheduler did not bind the target's full-history group")
        if not isinstance(spec, FullAttentionSpec) or tensor.size != config.num_blocks * spec.page_size_bytes:
            raise ValueError("V4.1 scheduler returned an inconsistent KV cache allocation")
        marker = replace(spec, page_size_padded=None)
        if spec.page_size_bytes != marker.page_size_bytes + self.private_page_bytes:
            raise ValueError("V4.1 KV page size differs from the model's physical layout")
        # The other bytes of this very same plan are allocated as model-owned
        # window, compressor and indexer tensors. Keep the scheduler's plan intact.
        return replace(
            config,
            kv_cache_groups=[replace(group, kv_cache_spec=marker)],
            kv_cache_tensors=[replace(tensor, size=config.num_blocks * marker.page_size_bytes)],
        )
