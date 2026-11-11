# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from vllm_ascend.model_loader.deepseek_v41 import DeepseekV41ModelLoader
from vllm_ascend.models.deepseek_v41.graph_model import DeepseekV41Stage2ForCausalLM, PagedACLGraphWrapper
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41ForCausalLM


@pytest.mark.parametrize(
    ("sequences", "spec", "eager", "expected"),
    [
        (1, None, True, AscendDeepseekV41ForCausalLM),
        (4, None, True, DeepseekV41Stage2ForCausalLM),
        (1, object(), True, DeepseekV41Stage2ForCausalLM),
        (1, None, False, DeepseekV41Stage2ForCausalLM),
    ],
)
def test_registered_class_initializes_the_selected_type_once(sequences, spec, eager, expected):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=sequences),
        speculative_config=spec,
        model_config=SimpleNamespace(enforce_eager=eager),
    )
    with (
        patch.object(AscendDeepseekV41ForCausalLM, "__init__", autospec=True, return_value=None) as eager_init,
        patch.object(DeepseekV41Stage2ForCausalLM, "__init__", autospec=True, return_value=None) as paged_init,
    ):
        model = AscendDeepseekV41ForCausalLM(vllm_config=config, prefix="target")
    assert type(model) is expected
    assert isinstance(model, AscendDeepseekV41ForCausalLM)
    if expected is DeepseekV41Stage2ForCausalLM:
        paged_init.assert_called_once_with(model, vllm_config=config, prefix="target")
        eager_init.assert_not_called()
        assert expected.ascend_full_graph_wrapper_cls is PagedACLGraphWrapper
    else:
        eager_init.assert_called_once_with(model, vllm_config=config, prefix="target")
        paged_init.assert_not_called()
        assert not hasattr(expected, "ascend_full_graph_wrapper_cls")


def test_finalization_follows_generic_weight_loading(monkeypatch, tmp_path):
    config, model_config = object(), object()
    monkeypatch.setenv("VLLM_ASCEND_DSV41_RESULTS_DIR", str(tmp_path))
    calls = []

    class Paged(nn.Module):
        def finalize_after_load(self, *, receipt_dir):
            calls.append((get_current_vllm_config(), receipt_dir))
            self.new_child = nn.Identity()

    loader = DeepseekV41ModelLoader(load_config=None)
    for model in (nn.Identity(), Paged()):
        with patch.object(BaseModelLoader, "load_model", return_value=model) as generic:
            assert loader.load_model(config, model_config, "target") is model
            generic.assert_called_once_with(config, model_config, "target")
    assert calls == [(config, str(tmp_path))]
    assert not model.new_child.training


def test_uniform_full_decode_warmup_uses_capture_context(monkeypatch):
    import vllm_ascend.ops  # noqa: F401  (the worker initializes custom ops before the runner)
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    target = object.__new__(DeepseekV41Stage2ForCausalLM)
    nn.Module.__init__(target)
    target.history = SimpleNamespace(max_len=4096)
    target._warming_full_decode_graph = False
    target._inside_full_wrapper = True
    staged_length = 1
    target._prepare = lambda *_: (staged_length, 0, -1)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model = target
    widths = []

    def observe_dummy(num_tokens, *, cudagraph_runtime_mode, is_graph_capturing=False, **kwargs):
        if is_graph_capturing and cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return
        with patch.object(torch.npu, "is_current_stream_capturing", return_value=is_graph_capturing):
            widths.append(target._stage_attention_context(None, None))

    runner._dummy_run = observe_dummy
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        for mode, uniform, expected in (
            (CUDAGraphMode.FULL, True, [4096, 4096]),
            (CUDAGraphMode.FULL, False, [1, 4096]),
            (CUDAGraphMode.PIECEWISE, True, [1]),
        ):
            desc = BatchDescriptor(num_tokens=6, uniform=uniform)
            runner._warmup_and_capture(desc, mode, num_warmups=1)
            assert widths == expected
            widths.clear()
            # An ordinary memory profile still uses only its inactive scratch row.
            assert target._stage_attention_context(None, None) == 1

        def fail_dummy(*args, **kwargs):
            raise RuntimeError("injected warmup failure")

        runner._dummy_run = fail_dummy
        with pytest.raises(RuntimeError, match="injected warmup failure"):
            runner._warmup_and_capture(BatchDescriptor(num_tokens=6, uniform=True), CUDAGraphMode.FULL, num_warmups=1)
        staged_length = 11
        assert target._stage_attention_context(None, None) == 11
