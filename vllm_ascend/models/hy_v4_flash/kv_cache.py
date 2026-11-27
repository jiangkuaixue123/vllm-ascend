# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec, SlidingWindowMLASpec

from vllm_ascend.utils import is_hidden_state_cache_spec


@dataclass(frozen=True, kw_only=True)
class Hy4SlidingWindowCacheSpec(SlidingWindowMLASpec):
    """Bounded MLA with the same single-tensor C8 pages as Hy4 sparse layers."""


def group_hy4_cache_specs(
    specs: dict[str, KVCacheSpec],
    *,
    disable_hybrid: bool | None,
) -> list[KVCacheGroupSpec] | None:
    window_layers = {name: spec for name, spec in specs.items() if isinstance(spec, Hy4SlidingWindowCacheSpec)}
    if not window_layers:
        return None
    if disable_hybrid:
        raise ValueError("HYV4 Flash bounded window KV requires the hybrid KV cache manager.")
    sparse_layers = {name: spec for name, spec in specs.items() if name not in window_layers}
    if (
        not sparse_layers
        or not all(is_hidden_state_cache_spec(spec) for spec in sparse_layers.values())
        or len(set(sparse_layers.values())) != 1
        or len(set(window_layers.values())) != 1
        or len({(spec.block_size, spec.page_size_bytes) for spec in specs.values()}) != 1
    ):
        raise ValueError("HYV4 Flash bounded window KV requires uniform sparse and sliding-window pages.")
    sparse = next(iter(sparse_layers.values()))
    window = next(iter(window_layers.values()))
    window_names = list(window_layers)
    # The groups share the scheduler's block pool. Keep each sliding group
    # within the width of the sparse group so it cannot enlarge a physical block.
    window_group_count = cdiv(len(window_names), len(sparse_layers))
    return [
        KVCacheGroupSpec(layer_names=list(sparse_layers), kv_cache_spec=sparse),
        *(
            KVCacheGroupSpec(layer_names=window_names[index::window_group_count], kv_cache_spec=window)
            for index in range(window_group_count)
        ),
    ]
