# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.deepseek_v41.core import hc_split_sinkhorn
from vllm_ascend.ops.triton import hc_sinkhorn


def inputs(rows, width=4):
    generator = torch.Generator().manual_seed(42)
    channels = (width + 2) * width
    mixes = torch.randn(rows, channels, generator=generator)
    # Imbalanced, asymmetric residual logits distinguish row and column normalization.
    if width == 4:
        mixes[:, 8:] += torch.tensor([8, 5, -9, 0, 9, -8, 2, 0, -6, 7, 3, 1, -2, 0, 9, 5])
    return mixes.npu(), torch.tensor([0.5, -1.5, 1.0]).npu(), torch.randn(channels, generator=generator).npu()


def assert_outputs(actual, reference):
    pre, post, comb = actual
    torch.testing.assert_close(pre, reference[0], rtol=0, atol=0)
    torch.testing.assert_close(post, reference[1], rtol=0, atol=0)
    torch.testing.assert_close(comb, reference[2], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(hc_sinkhorn.hc_sinkhorn4 is None, reason="Ascend Triton HC unavailable or disabled")
@pytest.mark.parametrize("rows", [1, 5, 6, 12, 40])
@torch.inference_mode()
def test_hc_sinkhorn_eager_and_full_graph_replay(rows):
    args = inputs(rows)
    with patch.object(hc_sinkhorn, "hc_sinkhorn4", wraps=hc_sinkhorn.hc_sinkhorn4) as fused:
        # Warm up the real Triton kernel before capture and verify its use in the normal model path.
        eager = hc_split_sinkhorn(*args, 4)
        fused.assert_called_once()
        with torch.inference_mode(False), torch.no_grad():
            reference = hc_split_sinkhorn(*args, 4)
        assert_outputs(eager, reference)

        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            captured = hc_split_sinkhorn(*args, 4)
        assert fused.call_count == 2
        previous = reference[2]
        for delta in (0.25, -2.0):
            # Change individual columns in place so the second replay must consume fresh logits.
            args[0][:, 8::3].add_(delta)
            graph.replay()
            with torch.inference_mode(False), torch.no_grad():
                reference = hc_split_sinkhorn(*args, 4)
            assert not torch.equal(previous, reference[2])
            assert_outputs(captured, reference)
            previous = reference[2]
        assert fused.call_count == 2


@pytest.mark.parametrize(("width", "rounds", "eps"), [(2, 20, 1e-6), (4, 1, 1e-6), (4, 20, 0.0)])
@torch.inference_mode()
def test_other_hc_configurations_use_reference(width, rounds, eps):
    args = inputs(3, width)
    with patch.object(hc_sinkhorn, "hc_sinkhorn4", side_effect=AssertionError("Unsupported fused configuration")):
        actual = hc_split_sinkhorn(*args, width, rounds, eps)
    with torch.inference_mode(False), torch.no_grad():
        reference = hc_split_sinkhorn(*args, width, rounds, eps)
    assert_outputs(actual, reference)


@torch.inference_mode()
def test_hc_sinkhorn_empty_or_disabled():
    empty_args = inputs(0)
    assert hc_split_sinkhorn(*empty_args, 4)[2].shape == (0, 4, 4)
    args = inputs(3)
    with patch.object(hc_sinkhorn, "hc_sinkhorn4", None):
        actual = hc_split_sinkhorn(*args, 4)
    with torch.inference_mode(False), torch.no_grad():
        reference = hc_split_sinkhorn(*args, 4)
    assert_outputs(actual, reference)
