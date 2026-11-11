# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm import SamplingParams
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.request import Request

from vllm_ascend.models.deepseek_v41.decode_state import DecodeInputs, PagedHistory
from vllm_ascend.models.deepseek_v41.graph_model import DeepseekV41Stage2ForCausalLM
from vllm_ascend.models.deepseek_v41.graph_ops import LayerPages
from vllm_ascend.models.deepseek_v41.page_pool import AutoPagedMetadata


def make_target(override=None):
    cfg = SimpleNamespace(
        quant_config=None,
        cache_config=SimpleNamespace(block_size=4, num_gpu_blocks_override=override, enable_prefix_caching=False),
        model_config=SimpleNamespace(
            max_model_len=8, original_max_model_len=8, hf_text_config=SimpleNamespace(model_type="deepseek_v41_text")
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4, watermark=0, scheduler_cls=None, disable_hybrid_kv_cache_manager=False
        ),
        max_concurrent_batches=1,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
    )
    attentions = [
        SimpleNamespace(head_dim=32, compress_ratio=r, is_kv_source=r != 0, indexer=SimpleNamespace(index_head_dim=16))
        for r in (0, 2, 1)
    ]
    marker = object.__new__(AutoPagedMetadata if override is None else Attention)
    nn.Module.__init__(marker)
    marker.layer_name = "model.layers.0.self_attn.attn"
    marker.kv_sharing_target_layer_name = None
    marker.attn_type = AttentionType.DECODER
    marker.sliding_window = None
    marker.kv_cache_dtype = "auto"
    marker.kv_cache_torch_dtype = torch.bfloat16
    marker.head_size = marker.head_size_v = 128
    marker.num_kv_heads = 1
    marker.private_page_bytes = sum(LayerPages.bytes_per_block(a, 4) for a in attentions)
    target = object.__new__(DeepseekV41Stage2ForCausalLM)
    nn.Module.__init__(target)
    target.metadata_layer = marker
    target.history = PagedHistory(8, 4, override)
    target.stage = DecodeInputs(target.history, 6, "cpu", [])
    target.backbone = SimpleNamespace(layers=[SimpleNamespace(attn=a) for a in attentions])
    target.layer_pages = [LayerPages(a, target.stage) for a in attentions]
    return target, cfg


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def private_bytes(target):
    return sum(tensor_bytes(vars(tables).values()) for tables in target.layer_pages)


def plan(target, config, budgets):
    marker = target.metadata_layer
    specs = [{marker.layer_name: marker.get_kv_cache_spec(config)} for _ in budgets]
    return get_kv_cache_configs(config, specs, budgets)


def test_full_page_charge_and_stock_rank_minimum():
    target, cfg = make_target()
    marker = target.metadata_layer
    spec = marker.get_kv_cache_spec(cfg)
    cost = spec.page_size_bytes
    assert cost == spec.unpadded_page_size_bytes + marker.private_page_bytes
    assert marker.workload_budget(99 * cost, cfg) == 9 * cost
    assert marker.workload_budget(7 * cost + 19, cfg) == 7 * cost + 19
    budgets = [marker.workload_budget(n * cost, cfg) for n in (11, 8, 10, 20, 18, 7, 9, 16)]
    configs = plan(target, cfg, budgets)
    assert [c.num_blocks for c in configs] == [7] * 8
    config = configs[0]
    scratch = private_bytes(target)
    device = target.prepare_device_kv_cache(config, max_model_len=8)
    assert config.kv_cache_tensors[0].size == 7 * cost
    assert private_bytes(target) - scratch + device.kv_cache_tensors[0].size == 7 * cost
    assert target.stage.history is target.history and target.history.num_blocks == 7
    assert list(target.history.plan([3], [27], [[6]], [1]).slots) == [27]
    with pytest.raises(ValueError, match="outside"):
        target.history.plan([0], [28], [[7]], [1])
    active_storage = target.layer_pages[0].window.data_ptr()
    target.prepare_device_kv_cache(config, max_model_len=8)
    assert target.layer_pages[0].window.data_ptr() == active_storage
    with pytest.raises(ValueError, match="max seq len"):
        plan(target, cfg, [cost])


def test_all_request_rows_fit_with_stock_speculative_lookahead():
    target, cfg = make_target()
    marker = target.metadata_layer
    cost = marker.get_kv_cache_spec(cfg).page_size_bytes
    config = plan(target, cfg, [marker.workload_budget(100 * cost, cfg)])[0]
    manager = KVCacheManager(config, 8, 4, 4, enable_caching=False, use_eagle=True)
    requests = []
    for row in range(4):
        req = Request(str(row), [1] * 7, SamplingParams(max_tokens=1), pooling_params=None)
        assert manager.allocate_slots(req, 7, num_lookahead_tokens=5, has_scheduled_reqs=row > 0) is not None
        requests.append(req)
    assert manager.block_pool.get_num_free_blocks() == 0
    manager.free(requests[1])
    assert manager.block_pool.get_num_free_blocks() == 2
    req = Request("replacement", [2] * 8, SamplingParams(max_tokens=1), pooling_params=None)
    assert manager.allocate_slots(req, 8, num_lookahead_tokens=5) is not None


def test_preserve_nonstandard_admission_and_explicit_override():
    target, cfg = make_target()
    budget = 123456789
    for obj, attr, value in (
        (cfg.cache_config, "enable_prefix_caching", True),
        (cfg.scheduler_config, "watermark", 0.1),
        (cfg.scheduler_config, "scheduler_cls", object()),
        (cfg, "max_concurrent_batches", 2),
        (cfg, "kv_transfer_config", object()),
        (cfg, "ec_transfer_config", object()),
    ):
        prior = getattr(obj, attr)
        setattr(obj, attr, value)
        assert target.metadata_layer.workload_budget(budget, cfg) == budget
        setattr(obj, attr, prior)

    target, cfg = make_target(145)
    marker = target.metadata_layer
    spec = marker.get_kv_cache_spec(cfg)
    assert type(marker) is Attention and spec.page_size_padded is None
    config = plan(target, cfg, [spec.page_size_bytes])[0]
    assert config.num_blocks == 145
    original = target.layer_pages[0].window.data_ptr()
    assert target.prepare_device_kv_cache(config, max_model_len=8) is config
    assert target.layer_pages[0].window.data_ptr() == original


def test_failed_allocation_restores_scratch_and_can_retry(monkeypatch):
    from vllm_ascend.models.deepseek_v41 import graph_model

    target, cfg = make_target()
    cost = target.metadata_layer.get_kv_cache_spec(cfg).page_size_bytes
    config = plan(target, cfg, [3 * cost])[0]
    scratch = private_bytes(target)
    second = target.backbone.layers[1].attn

    def allocate(attn, inputs, *, physical_slots=None):
        if attn is second and physical_slots is not None:
            raise RuntimeError("injected allocation failure")
        return LayerPages(attn, inputs, physical_slots=physical_slots)

    with monkeypatch.context() as m:
        m.setattr(graph_model, "LayerPages", allocate)
        with pytest.raises(RuntimeError, match="injected allocation failure"):
            target.prepare_device_kv_cache(config, max_model_len=8)
    assert target.history.num_blocks == 0 and target.history is target.stage.history
    assert private_bytes(target) == scratch
    target.prepare_device_kv_cache(config, max_model_len=8)
    assert target.history.num_blocks == 3


def test_registered_worker_binds_only_the_remaining_marker_bytes(monkeypatch):
    import vllm_ascend.ops  # noqa: F401  (initialize the platform's custom ops)
    # isort: split
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker

    target, cfg = make_target()
    marker = target.metadata_layer
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model, runner.vllm_config, runner.model_config, runner.cache_config = (
        target,
        cfg,
        cfg.model_config,
        cfg.cache_config,
    )
    runner.max_model_len = 6
    runner.device = torch.device("cpu")
    runner.use_compress = runner.use_sparse = runner.use_hybrid_blocks = runner.sparse_kv_offload_enabled = False
    runner.runner_only_attn_layers = set()
    runner.shared_kv_cache_layers = {}
    runner.kv_caches = []
    runner.compilation_config = cfg.compilation_config = SimpleNamespace(
        static_forward_context={marker.layer_name: marker}
    )
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner, worker.vllm_config, worker.model_config, worker.cache_config = (
        runner,
        cfg,
        cfg.model_config,
        cfg.cache_config,
    )
    worker._apply_kv_offload_decode_memory_constraints = lambda budget: budget
    specs = runner.get_kv_cache_spec()
    cost = specs[marker.layer_name].page_size_bytes
    assert worker._apply_kv_cache_memory_constraints(cost * 50) == 9 * cost
    monkeypatch.setattr("vllm_ascend.worker.worker.maybe_apply_startup_plan", lambda _: None)
    runner.profile_run = lambda: None
    worker.init_snapshot = SimpleNamespace(free_memory=cost * 50)
    cfg.cache_config.kv_cache_memory_bytes = cost * 5 + 1
    assert worker.determine_available_memory() == cfg.cache_config.kv_cache_memory_bytes
    cfg.cache_config.kv_cache_memory_bytes = None
    worker.device = runner.device
    worker.requested_memory, runner.model_memory_usage = 20 * cost, cost
    profile = SimpleNamespace(
        weights_memory=cost,
        non_torch_increase=cost,
        before_profile=SimpleNamespace(torch_peak=0),
        after_profile=SimpleNamespace(free_memory=40 * cost),
    )
    monkeypatch.setattr("vllm_ascend.worker.worker.memory_profiling", lambda *a, **kw: nullcontext(profile))
    monkeypatch.setattr(torch.npu, "memory_stats", lambda _: {"allocated_bytes.all.peak": cost})
    assert worker.determine_available_memory() == 9 * cost
    config = get_kv_cache_configs(cfg, [specs], [3 * cost])[0]
    group = config.kv_cache_groups[0]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            SimpleNamespace(
                backend=AscendAttentionBackend, kv_cache_spec=group.kv_cache_spec, layer_names=group.layer_names
            )
        ]
    )
    scratch = private_bytes(target)
    kv_caches = runner.initialize_kv_cache_tensors(config)
    bound = kv_caches[marker.layer_name]
    assert marker.kv_cache is bound and runner.kv_caches[0] is bound
    assert target.history.max_len == 6
    assert private_bytes(target) - scratch + tensor_bytes(bound) == config.kv_cache_tensors[0].size
