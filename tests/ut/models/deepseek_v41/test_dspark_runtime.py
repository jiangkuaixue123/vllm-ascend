# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport

from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Cache
from vllm_ascend.models.deepseek_v41.config import DSPARK_ARCHITECTURE, DeepseekV41Config
from vllm_ascend.models.deepseek_v41.dspark import DSparkBackend, DSparkCache
from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.spec_decode import get_spec_decode_method, llm_base_proposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDeepseekV41DSparkProposer, AscendDSparkProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_config.json"


def make_spec_config():
    target = DeepseekV41Config.from_json_file(FIXTURE)
    draft = deepcopy(target)
    # Upstream selects the generic V4 DSpark architecture before update_arch_.
    draft.model_type = "deepseek_v4"
    draft.architectures = ["DSparkDraftModel"]
    return SimpleNamespace(
        method="dspark",
        model="embedded-checkpoint",
        target_model_config=SimpleNamespace(model="embedded-checkpoint", hf_text_config=target.text_config),
        target_parallel_config=SimpleNamespace(tensor_parallel_size=8),
        draft_model_config=SimpleNamespace(hf_config=draft),
        num_speculative_tokens=5,
        draft_tensor_parallel_size=None,
    )


def test_embedded_draft_uses_v41_registry_and_preserves_quantized_checkpoint_metadata(monkeypatch):
    spec = make_spec_config()
    spec.num_speculative_tokens = None
    draft = spec.draft_model_config.hf_config
    quant = deepcopy(draft.quantization_config)
    inspected = []
    monkeypatch.setattr(
        patch_speculative_config,
        "_orig_update_arch",
        lambda config: inspected.extend(config.draft_model_config.hf_config.architectures),
    )
    patch_speculative_config._update_draft_arch(spec)
    assert inspected == [DSPARK_ARCHITECTURE]
    assert draft.model_type == "deepseek_v41" and draft.quantization_config == quant
    assert draft.sample_from_anchor and spec.num_speculative_tokens == draft.dspark_block_size == 5
    assert draft.ptd_token_id == 128799 and draft.dspark_target_layer_ids == [37, 38, 39]
    assert spec.target_model_config.hf_text_config is not draft.text_config


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("num_speculative_tokens", 4),
        ("num_speculative_tokens", 6),
        ("model", "unrelated-checkpoint"),
        ("draft_tensor_parallel_size", 1),
    ],
)
def test_embedded_draft_rejects_incompatible_proposals(field, invalid):
    spec = make_spec_config()
    setattr(spec, field, invalid)
    with pytest.raises(ValueError, match="V4.1 DSpark"):
        patch_speculative_config._update_draft_arch(spec)


@pytest.mark.parametrize(
    ("arch", "expected"),
    [(DSPARK_ARCHITECTURE, AscendDeepseekV41DSparkProposer), ("DSparkDraftModel", AscendDSparkProposer)],
)
def test_proposer_dispatch_preserves_other_dspark_models(monkeypatch, arch, expected):
    monkeypatch.setattr(AscendDeepseekV41DSparkProposer, "__init__", lambda *args: None)
    monkeypatch.setattr(AscendDSparkProposer, "__init__", lambda *args: None)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(architectures=[arch]))
    )
    assert type(get_spec_decode_method("dspark", config, "cpu", None)) is expected


def test_full_verification_graph_excludes_only_the_independent_draft_layers():
    proposer = object.__new__(AscendDeepseekV41DSparkProposer)
    assert not proposer.share_target_weights and AscendDSparkProposer.share_target_weights
    proposer._draft_attn_layer_names = {"mtp.0.attn.cache", "mtp.1.attn.cache"}
    assert proposer.owns_draft_group(["mtp.0.attn.cache", "mtp.1.attn.cache"])
    assert not proposer.owns_draft_group(["layers.0.attn.cache", "mtp.1.attn.cache"])
    assert not proposer.owns_draft_group([])


def test_proposer_discovers_real_draft_caches_before_target_graph_negotiation(monkeypatch):
    spec = make_spec_config()
    config = spec.target_model_config.hf_text_config
    context = {}
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context, cudagraph_capture_sizes=[24, 48]),
        cache_config=SimpleNamespace(block_size=64),
        model_config=SimpleNamespace(enforce_eager=False, max_model_len=1024),
        num_speculative_tokens=5,
    )
    target_name = "layers.0.attn.cache"
    target = DeepseekV41Cache(config, vllm_config, target_name)
    names = [f"mtp.{stage}.attn.cache" for stage in range(3)]

    class Draft(nn.Module):
        def get_draft_kv_cache_layer_names(self):
            return names

    def load_draft():
        for name in names:
            DSparkCache(config, vllm_config, name)
        return Draft()

    proposer = object.__new__(AscendDeepseekV41DSparkProposer)
    proposer.vllm_config = vllm_config
    proposer.maybe_eager_context = nullcontext()
    proposer.supports_mm_inputs = False
    proposer.parallel_drafting = True
    proposer.pass_hidden_states_to_model = True
    proposer.parallel_drafting_hidden_state_tensor = None
    proposer.runner, proposer.device = None, "cpu"
    proposer.max_query_tokens, proposer.max_num_tokens = 10, 16
    proposer._get_model = load_draft
    monkeypatch.setattr(llm_base_proposer, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr("vllm_ascend.ascend_config.get_ascend_config", lambda: SimpleNamespace(draft_window_size=None))
    proposer.load_model(target)
    assert proposer._draft_attn_layer_names == set(names)
    assert proposer.owns_draft_group(names) and not proposer.owns_draft_group([target_name])
    group = SimpleNamespace(layer_names=[target_name, *names], kv_cache_spec=target.get_kv_cache_spec(vllm_config))
    proposer.initialize_attn_backend(SimpleNamespace(kv_cache_groups=[group]), [64])
    assert proposer.attn_layer_names == names and proposer._layer_group_idx == [0, 0, 0]
    assert len(proposer.draft_attn_groups) == 1 and proposer.draft_attn_groups[0].backend is DSparkBackend
    assert set(proposer.draft_attn_groups[0].layer_names) == set(names)
    assert (
        proposer.draft_attn_groups[0].get_metadata_builder().get_cudagraph_support(vllm_config, group.kv_cache_spec)
        == AttentionCGSupport.NEVER
    )


def test_unpadded_rejections_compact_each_physical_context_slot_mapping_once():
    proposer = object.__new__(AscendDeepseekV41DSparkProposer)
    proposer.token_arange_np = np.arange(12)
    proposer.runner = SimpleNamespace(actual_seq_lengths_q=[6, 12], attn_state=None, decode_token_per_req=6)
    proposer.draft_attn_groups = [SimpleNamespace(kv_cache_group_id=gid) for gid in range(4)]
    primary = torch.tensor([*range(80, 86), *range(160, 166)])
    other = torch.tensor([*range(180, 186), *range(360, 366)])
    proposer._per_group_slot_mappings = {0: primary, 1: primary.view(-1), 2: other, 3: other.view(-1)}
    bounds = torch.tensor([0, 6, 12], dtype=torch.int32)
    lengths = torch.tensor([26, 76], dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc=bounds,
        query_start_loc_cpu=bounds,
        _seq_lens_cpu=lengths,
        slot_mapping=primary,
        num_reqs=2,
        num_input_tokens=12,
        block_table_tensor=torch.tensor([[5, 6], [12, 13]]),
        positions=torch.arange(12),
        positions_cpu=None,
        attn_state=None,
        decode_token_per_req=6,
        num_computed_tokens_cpu=None,
        _num_computed_tokens_cpu=None,
        token_to_req=None,
        is_prefilling=None,
        group_len=None,
        group_key_idx=None,
        group_key_cache_idx=None,
        req_ids_tensor=None,
    )
    # The first request rejects all five drafts; the second accepts all five.
    compact, indices = proposer.prepare_inputs(common, [[2], [4] * 6], [5, 5])
    assert indices.tolist() == [0, 6, 7, 8, 9, 10, 11]
    assert compact.query_start_loc_cpu.tolist() == [0, 1, 7]
    assert compact.seq_lens.tolist() == [21, 76]
    assert primary.tolist() == [80, 160, 161, 162, 163, 164, 165] + [-1] * 5
    assert other.tolist() == [180, 360, 361, 362, 363, 364, 365] + [-1] * 5


def test_v1_graph_resolver_preserves_q6_tp8_buckets_with_eager_draft():
    compilation = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[24, 48],
        max_cudagraph_capture_size=48,
    )
    parallel = SimpleNamespace(tensor_parallel_size=8, use_sequence_parallel_moe=True)
    runner = SimpleNamespace(
        parallel_config=parallel,
        vllm_config=SimpleNamespace(parallel_config=parallel),
        compilation_config=compilation,
        uniform_decode_query_len=6,
        max_num_reqs=8,
        kv_cache_config=None,
        cudagraph_dispatcher=Mock(),
        use_aclgraph=False,
        speculative_config=None,
    )
    runner.cudagraph_dispatcher.get_capture_descs.return_value = []
    target_backend = Mock()
    target_backend.__name__ = "target"
    target_backend.get_builder_cls.return_value.get_cudagraph_support.return_value = AttentionCGSupport.UNIFORM_BATCH
    NPUModelRunner._check_and_update_cudagraph_mode(
        runner, [{target_backend}, set()], [SimpleNamespace(kv_cache_spec=None)] * 2
    )
    assert compilation.cudagraph_capture_sizes == [24, 48]
    runner.cudagraph_dispatcher.initialize_cudagraph_keys.assert_called_once_with(CUDAGraphMode.FULL_DECODE_ONLY, 6)


@pytest.mark.parametrize("proposer_type", [AscendDSparkProposer, AscendDeepseekV41DSparkProposer])
@pytest.mark.parametrize("is_profile", [False, True])
def test_idle_dp_draft_still_runs_its_ep_forward_without_reusing_live_cache_slots(
    monkeypatch, proposer_type, is_profile
):
    proposer = object.__new__(proposer_type)
    proposer.runner = Mock()
    proposer.runner._sync_metadata_across_dp.return_value = (10, torch.tensor([10, 10]), None)
    proposer.num_query_per_req = proposer.num_speculative_tokens = 5
    proposer.max_query_tokens = 40
    proposer.use_cuda_graph = False
    proposer.vllm_config = object()
    proposer._context_positions_buffer = torch.arange(10)
    proposer._dflash_hidden_states = proposer.hidden_states = torch.zeros(10, 2)
    proposer.token_indices_to_sample = torch.zeros(5, dtype=torch.int32)
    proposer._get_positions = lambda count: torch.arange(count)
    proposer._pad_draft_buffers = Mock()
    proposer.model = Mock()
    proposer.input_ids = torch.zeros(10, dtype=torch.int64)
    proposer._context_slot_mapping_buffers = [torch.tensor([17, 23])] * 3
    proposer._runnable = Mock(
        side_effect=lambda **args: proposer.build_model_inputs_first_pass(
            args["num_input_tokens"], proposer._context_slot_mapping_buffers
        )
    )
    context = Mock(side_effect=lambda *a, **k: nullcontext())
    monkeypatch.setattr("vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context", context)
    proposer.dummy_run(num_tokens=5, num_reqs=1, is_profile=is_profile)
    proposer.runner._sync_metadata_across_dp.assert_called_once_with(5, is_draft_model=True)
    actual = 0 if proposer_type is AscendDeepseekV41DSparkProposer and not is_profile else 10
    assert context.call_args.kwargs["num_actual_tokens"] == actual
    assert context.call_args.kwargs["num_tokens"] == 10
    torch.testing.assert_close(context.call_args.kwargs["num_tokens_across_dp"], torch.tensor([10, 10]))
    if is_profile:
        proposer._runnable.assert_not_called()
        proposer.model.assert_called_once()
        states, positions = proposer.model.precompute_and_store_context_kv.call_args.args
        assert states.shape[0] == positions.numel() == 10
    else:
        proposer._runnable.assert_called_once()
        states, positions, slots = proposer.model.precompute_and_store_context_kv.call_args.args
        assert states.shape[0] == positions.numel() == 10
        if proposer_type is AscendDeepseekV41DSparkProposer:
            assert slots is None
        else:
            assert len(slots) == len(proposer._context_slot_mapping_buffers)
            for actual_slots, expected_slots in zip(slots, proposer._context_slot_mapping_buffers):
                torch.testing.assert_close(actual_slots, expected_slots)


def test_target_auxiliary_projection_and_sequential_markov_block(monkeypatch):
    class ToyDraft(nn.Module):
        def __init__(self):
            super().__init__()
            self.combined = []
            self.inputs = []

        def combine_hidden_states(self, states):
            self.combined.append(states)
            return states.reshape(-1, 3, 2).sum(1)

        def forward(self, input_ids, positions, inputs_embeds):
            self.inputs.append(input_ids)
            return input_ids.new_zeros(input_ids.numel(), 2, dtype=torch.float32)

        def compute_logits(self, hidden):
            return hidden.new_zeros(hidden.shape[0], 7)

        def markov_embed(self, ids):
            return ids

        def markov_bias(self, ids):
            return nn.functional.one_hot((ids + 1) % 7, num_classes=7).float()

    proposer = object.__new__(AscendDeepseekV41DSparkProposer)
    proposer.model = ToyDraft()
    proposer.method, proposer.runner = "dspark", None
    proposer.num_speculative_tokens, proposer.parallel_drafting = 5, True
    proposer._share_mtp_indices = proposer._enable_probabilistic_draft_probs = False
    proposer.dynamic_spec = None
    proposer.input_ids = torch.tensor([1, 6, 6, 6, 6, 3, 6, 6, 6, 6])
    proposer._get_positions = lambda count: torch.arange(count)
    proposer._dspark_draft_buffer = torch.zeros(2, 6, dtype=torch.long)
    proposer._dspark_seed_buffer = torch.tensor([1, 3])
    proposer._context_slot_mapping_buffers = None
    proposer.build_model_inputs_first_pass = Mock()
    monkeypatch.setattr(llm_base_proposer, "get_ascend_config", lambda: SimpleNamespace(enable_reduce_sample=False))
    monkeypatch.setattr(llm_base_proposer, "lmhead_tp_enable", lambda: False)
    aux = torch.arange(60).reshape(10, 6).float()
    result = proposer._combine_target_hidden_states(proposer.model, aux)
    torch.testing.assert_close(result, aux.reshape(10, 3, 2).sum(1))
    assert len(proposer.model.combined) == 1
    draft_ids = proposer._run_merged_draft(10, 2, torch.arange(10), torch.arange(10), None, [], 10)
    proposer.build_model_inputs_first_pass.assert_called_once_with(10, None)
    assert draft_ids.tolist() == [[2, 3, 4, 5, 6], [4, 5, 6, 0, 1]]
    assert len(proposer.model.inputs) == 1
