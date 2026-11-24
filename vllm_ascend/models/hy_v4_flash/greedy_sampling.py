# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify plain greedy DFly requests from per-rank maximum scores and token IDs."""

from vllm.distributed import get_tp_group
from vllm.logger import logger
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled

from .greedy_verification import choose


def plain_greedy_batch(runner, batch, grammar):
    sampler = runner.sampler
    rejection = runner.rejection_sampler
    if (
        grammar is not None
        or runner.batch_sharder is not None
        or batch.num_draft_tokens == 0
        or batch.num_reqs == 0
        or sampler is None
        or rejection is None
        or runner.speculator is None
        or runner.speculator.draft_logits is not None
        or rejection.sampler is not sampler
        or rejection.watermark_key is not None
        or rejection.synthetic_conditional_rates is not None
        or rejection.enable_adaptive_verification
        or rejection.use_block_verification
        or rejection.fly_window_size
        or sampler.return_sampling_mask
        or sampler.compute_nans
        or sampler.trace_replay_state is not None
    ):
        return False
    requests = batch.idx_mapping_np
    return (
        (sampler.sampling_states.temperature.np[requests] == 0).all()
        and not sampler.needs_logits_processing[requests].any()
        and sampler.get_logprobs_dims(requests) is None
    )


def sample(runner, hidden, batch, vocabulary):
    prefill_lengths = runner.sampler.req_states.prefill_len.gpu
    drafts, _ = gather_draft_sampled(
        batch.input_ids,
        batch.positions,
        batch.logits_indices,
        batch.expanded_idx_mapping,
        batch.expanded_local_pos,
        prefill_lengths,
    )
    local_scores = vocabulary.project(hidden[batch.logits_indices])
    group = get_tp_group()
    sampled, counts = choose(
        local_scores,
        group.device_group,
        vocabulary.start,
        group.world_size,
        batch.cu_num_logits,
        drafts,
        runner.rejection_sampler.num_speculative_steps + 1,
    )
    counts, rejected = get_num_sampled_and_rejected(
        counts,
        batch.seq_lens,
        batch.cu_num_logits,
        batch.idx_mapping,
        prefill_lengths,
    )
    logger.info_once("HYV4 Flash verifies greedy drafts from TP-local target vocabulary winners.")
    output = SamplerOutput(
        sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None, num_sampled=counts, num_rejected=rejected
    )
    return output, counts, rejected
