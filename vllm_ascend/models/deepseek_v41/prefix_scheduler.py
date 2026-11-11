# SPDX-License-Identifier: Apache-2.0
"""Use the stock V1 scheduler with the committed DSpark projection on each full page."""

import json
from functools import cache

from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.kv_cache_interface import FullAttentionSpec


def reuse_projected_blocks(manager):
    coordinator = manager.coordinator
    if not isinstance(coordinator, UnitaryKVCacheCoordinator) or not isinstance(
        coordinator.kv_cache_spec, FullAttentionSpec
    ):
        raise ValueError("V4.1 DSpark requires one full-history prefix cache group")
    # Unlike an EAGLE cache with missing draft inputs at its end, this physical
    # group stores all three accepted DSpark target projections. The ordinary
    # num_tokens - 1 cap still leaves a real query to produce target logits.
    coordinator.eagle_group_ids.discard(0)
    coordinator.single_type_managers[0].use_eagle = False


def scope_prompt_hashes(request, window):
    # The initial HF prompt appends sparse keys after min(prompt_len, window)
    # raw columns; later decode uses the whole window. An old short prompt that
    # generated a full block must not seed a longer prompt with the same IDs.
    span = min(request.num_prompt_tokens, window)
    if span < window or request.cache_salt is not None:
        request.cache_salt = json.dumps([span, request.cache_salt], separators=(",", ":"))
        if request.block_hashes:
            request.block_hashes.clear()
            request.update_block_hashes()


class _PrefixScheduler:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.vllm_config.speculative_config is not None:
            reuse_projected_blocks(self.kv_cache_manager)

    def add_request(self, request):
        scope_prompt_hashes(request, self.vllm_config.model_config.hf_text_config.sliding_window)
        return super().add_request(request)


@cache
def _prefix_scheduler_type(name, base):
    return type(name, (_PrefixScheduler, base), {"__module__": __name__})


def __getattr__(name):
    # Resolve the core alias when vLLM resolves the configured class. Platform
    # plugins can replace the default scheduler after this module was imported.
    if name == "V41PrefixScheduler":
        from vllm.v1.core.sched.scheduler import Scheduler

        return _prefix_scheduler_type(name, Scheduler)
    if name == "V41AsyncPrefixScheduler":
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        return _prefix_scheduler_type(name, AsyncScheduler)
    raise AttributeError(name)
