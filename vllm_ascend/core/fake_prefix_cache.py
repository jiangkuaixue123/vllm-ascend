#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any

from vllm.logger import logger

from vllm_ascend import envs


def maybe_apply_fake_prefix_cache(
    request: Any,
    block_size: int,
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
    load_kv_async: bool,
) -> int:
    """Raise external prefix-cache hits to a configured debug target.

    The fake tokens are represented as externally-computed tokens so the
    scheduler asks the KV cache manager to allocate readable blocks for the
    skipped prefix. Their contents are not valid; this is for perf debug only.
    """
    ratio = envs.VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO
    if ratio is None:
        return num_external_computed_tokens
    if not 0 <= ratio <= 1:
        raise ValueError("VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO must be in [0, 1]")

    external_before = num_external_computed_tokens
    if ratio == 0 or load_kv_async:
        logger.info(
            "Fake prefix cache debug: request_id=%s ratio=%s "
            "load_kv_async=%s block_size=%s prompt_tokens=%s total_tokens=%s "
            "local_computed=%s external_before=%s external_after=%s "
            "applied=False",
            getattr(request, "request_id", None),
            ratio,
            load_kv_async,
            block_size,
            request.num_prompt_tokens,
            request.num_tokens,
            num_local_computed_tokens,
            external_before,
            num_external_computed_tokens,
        )
        return num_external_computed_tokens

    max_hit_tokens = max(request.num_tokens - 1, 0)
    target_hit_tokens = min(int(request.num_prompt_tokens * ratio), max_hit_tokens)
    target_hit_tokens = target_hit_tokens // block_size * block_size
    current_hit_tokens = num_local_computed_tokens + num_external_computed_tokens
    if target_hit_tokens <= current_hit_tokens:
        logger.info(
            "Fake prefix cache debug: request_id=%s ratio=%s "
            "load_kv_async=%s block_size=%s prompt_tokens=%s total_tokens=%s "
            "target_hit_tokens=%s current_hit_tokens=%s local_computed=%s "
            "external_before=%s external_after=%s applied=False",
            getattr(request, "request_id", None),
            ratio,
            load_kv_async,
            block_size,
            request.num_prompt_tokens,
            request.num_tokens,
            target_hit_tokens,
            current_hit_tokens,
            num_local_computed_tokens,
            external_before,
            num_external_computed_tokens,
        )
        return num_external_computed_tokens
    num_external_computed_tokens += target_hit_tokens - current_hit_tokens
    logger.info(
        "Fake prefix cache debug: request_id=%s ratio=%s "
        "load_kv_async=%s block_size=%s prompt_tokens=%s total_tokens=%s "
        "target_hit_tokens=%s current_hit_tokens=%s local_computed=%s "
        "external_before=%s external_after=%s applied=True",
        getattr(request, "request_id", None),
        ratio,
        load_kv_async,
        block_size,
        request.num_prompt_tokens,
        request.num_tokens,
        target_hit_tokens,
        current_hit_tokens,
        num_local_computed_tokens,
        external_before,
        num_external_computed_tokens,
    )
    return num_external_computed_tokens
