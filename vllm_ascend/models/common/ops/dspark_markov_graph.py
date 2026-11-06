# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the unchanged greedy Markov chain without capturing any collectives."""

from dataclasses import dataclass

import torch

_BLOCK_SIZE = 5
_GRAPH_BATCHES = (1, 2, 4, 8)
_ND_FORMAT = 2


def greedy_markov_chain(logits, draft_ids, embedding, projection):
    """Preserve the original sequential embedding, FP32 GEMM, add and argmax."""
    for index in range(_BLOCK_SIZE):
        markov = embedding(draft_ids[:, index].long())
        bias = projection(markov.float())
        logits[:, index].add_(bias)
        draft_ids[:, index + 1].copy_(logits[:, index].argmax(dim=-1))


def supports_markov_graph(logits, draft_ids, embedding, projection):
    return (
        logits.device.type == "npu"
        and torch.is_inference_mode_enabled()
        and logits.ndim == 3
        and logits.shape[0] in _GRAPH_BATCHES
        and logits.shape[1] == _BLOCK_SIZE
        and logits.shape[2] > 0
        and logits.dtype == torch.float32
        and draft_ids.shape == (logits.shape[0], _BLOCK_SIZE + 1)
        and draft_ids.dtype == torch.int64
        and embedding.weight.ndim == projection.weight.ndim == 2
        and embedding.weight.shape == projection.weight.shape
        and embedding.weight.shape[0] == logits.shape[2]
        and embedding.weight.dtype == torch.bfloat16
        and projection.weight.dtype == torch.float32
        and projection.bias is None
        and embedding.padding_idx is None
        and embedding.max_norm is None
        and all(
            tensor.device == logits.device and tensor.is_contiguous() and not tensor.is_neg() and not tensor.is_conj()
            for tensor in (logits, draft_ids, embedding.weight, projection.weight)
        )
    )


@dataclass
class _MarkovGraphEntry:
    logits: torch.Tensor
    draft_ids: torch.Tensor
    graph: object
    signature: tuple


class DSparkMarkovGraph:
    """Own four batch-size entries, with private input/output and graph buffers."""

    def __init__(self):
        self.entries = {}

    def __call__(self, logits, draft_ids, embedding, projection):
        if not supports_markov_graph(logits, draft_ids, embedding, projection):
            return False
        if torch.npu.is_current_stream_capturing():
            return False
        from torch_npu import get_npu_format

        if any(get_npu_format(t) != _ND_FORMAT for t in (logits, draft_ids, embedding.weight, projection.weight)):
            return False
        signature = (
            logits.device,
            torch.npu.current_stream(logits.device).npu_stream,
            tuple(logits.shape),
            tuple(embedding.weight.shape),
            embedding.weight.data_ptr(),
            projection.weight.data_ptr(),
            bool(torch.npu.matmul.allow_hf32),
        )
        batch = logits.shape[0]
        entry = self.entries.get(batch)
        if entry is not None and entry.signature != signature:
            return False
        if entry is None:
            static_logits, static_ids = logits.clone(), draft_ids.clone()
            # Resolve native kernels once, then restore values before recording.
            greedy_markov_chain(static_logits, static_ids, embedding, projection)
            static_logits.copy_(logits)
            static_ids.copy_(draft_ids)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                greedy_markov_chain(static_logits, static_ids, embedding, projection)
            entry = _MarkovGraphEntry(static_logits, static_ids, graph, signature)
            self.entries[batch] = entry
        # Recording also mutates the private buffers. Refresh before every replay,
        # including the first, and preserve both original output side effects.
        entry.logits.copy_(logits)
        entry.draft_ids.copy_(draft_ids)
        entry.graph.replay()
        logits.copy_(entry.logits)
        draft_ids.copy_(entry.draft_ids)
        return True
