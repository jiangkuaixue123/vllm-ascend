# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full target EP16 consumers of exact BF16 row masking with native W8A8."""

import json
import os
import statistics
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.distributed.distributed_c10d import _get_default_group

from tests.e2e.nightly.single_node.ops.multicard_ops_a3.test_dispatch_ffn_route_before_down import (
    NUMERICAL_ATOL,
    NUMERICAL_RTOL,
)
from vllm_ascend.models.deepseek_v41 import moe as moe_module
from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig
from vllm_ascend.models.deepseek_v41.moe import (
    DeepseekV41Gate,
    DeepseekV41MoE,
    DeepseekV41MoECommunication,
    DeepseekV41MoEExpert,
)
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    enable_custom_op,
)

TP = 4
EP = 16
EXPERTS = 384
LOCAL_EXPERTS = EXPERTS // EP
TOP_K = 6
HIDDEN = 5120
INTERMEDIATE = 2304
TIMED_CALLS = 40
TIMING_SAMPLES = 5
WEIGHT_NZ_MODE = 1


class NativeQuantConfig:
    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(AscendW8A8DynamicLinearMethod())


class NativeExperts(nn.Module):
    def __init__(self, gate, rows, active):
        super().__init__()
        rank = dist.get_rank()
        generator = torch.Generator(device="cpu").manual_seed(513 + rank)
        self.gate, self.active = gate, active
        self.use_fixed_routing = True
        self.fixed_indices = (
            torch.arange(rows * TOP_K, device="npu", dtype=torch.int32).reshape(rows, TOP_K) + rank * TOP_K
        ) % EXPERTS
        self.fixed_weights = torch.full((rows, TOP_K), 1.0 / TOP_K, device="npu", dtype=torch.float32)
        w1 = torch.randint(
            -16, 17, (LOCAL_EXPERTS, HIDDEN, 2 * INTERMEDIATE), generator=generator, dtype=torch.int8, device="cpu"
        ).npu()
        w2 = torch.randint(
            -16, 17, (LOCAL_EXPERTS, INTERMEDIATE, HIDDEN), generator=generator, dtype=torch.int8, device="cpu"
        ).npu()
        self.w1, self.w2 = (torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ) for w in (w1, w2))
        self.scale1 = (
            torch.rand(LOCAL_EXPERTS, 2 * INTERMEDIATE, generator=generator, device="cpu") * 0.01 + 0.002
        ).npu()
        self.scale2 = (torch.rand(LOCAL_EXPERTS, HIDDEN, generator=generator, device="cpu") * 0.01 + 0.002).npu()
        self.output = torch.empty(rows, HIDDEN, dtype=torch.float32, device="npu")
        self.counts = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device="npu")
        self.group = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    def forward(self, hidden_states, router_logits):
        if self.use_fixed_routing:
            weights, indices = self.fixed_weights, self.fixed_indices
        else:
            weights, indices = self.gate.select_experts(hidden_states, router_logits, TOP_K, True)
        torch.ops._C_ascend.dispatch_ffn_combine(
            hidden_states,
            [self.w1],
            [self.w2],
            indices.to(torch.int32),
            [self.scale1],
            [self.scale2],
            [],
            [],
            weights,
            self.group,
            hidden_states.shape[0] * EP * TOP_K,
            self.output,
            self.counts,
            x_active_mask=self.active,
            swiglu_limit=10.0,
            route_before_down=True,
        )
        return self.output


def make_moe(rows, active):
    rank = dist.get_rank()
    config = DeepseekV41TextConfig(
        hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE, n_routed_experts=EXPERTS, num_experts_per_tok=TOP_K
    )
    model = DeepseekV41MoE.__new__(DeepseekV41MoE)
    nn.Module.__init__(model)
    model.use_fused_runner = True
    model.hidden_size, model.n_routed_experts = HIDDEN, EXPERTS
    model.communication = DeepseekV41MoECommunication(
        SimpleNamespace(rank_in_group=rank // TP, world_size=EP // TP),
        SimpleNamespace(rank_in_group=rank, world_size=EP),
        SimpleNamespace(rank_in_group=rank % TP, world_size=TP),
        sequence_parallel=True,
        replicated_shared=True,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("npu"):
            model.gate = DeepseekV41Gate(config)
            model.shared_experts = DeepseekV41MoEExpert(
                config,
                NativeQuantConfig(),
                "test.target.shared",
                shared=True,
                replicated_shared=True,
                sequence_parallel=True,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    generator = torch.Generator(device="cpu").manual_seed(419)
    model.gate.weight.copy_(torch.randn(EXPERTS, HIDDEN, generator=generator, device="cpu") * 0.01)
    model.gate.bias.copy_(torch.linspace(-0.01, 0.01, EXPERTS, device="cpu"))
    for linear in (model.shared_experts.w1, model.shared_experts.w2, model.shared_experts.w3):
        linear.weight.copy_(
            torch.randint(-8, 9, linear.weight.shape, generator=generator, dtype=torch.int8, device="cpu")
        )
        linear.weight_scale.copy_(
            torch.rand(linear.weight_scale.shape, generator=generator, device="cpu") * 0.005 + 0.002
        )
        linear.weight_offset.zero_()
        linear.quant_method.process_weights_after_loading(linear)
    model.experts = NativeExperts(model.gate, rows, active)
    return model.eval()


def assert_shared_exact(actual, expected):
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


def capture(call):
    call()
    torch.npu.synchronize()
    dist.barrier()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = call()
    return graph, output


def time_complete(call):
    dist.barrier()
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    call()
    submitted = time.perf_counter_ns()
    torch.npu.synchronize()
    completed = time.perf_counter_ns()
    values = torch.tensor([(submitted - started) / 1e6, (completed - started) / 1e6], device="npu")
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return values.cpu().tolist()


@torch.inference_mode()
def run_case(rows):
    """Preserve routing inputs for correctness, then include real routing in complete-stage timing."""
    rank = dist.get_rank()
    torch.manual_seed(611 + rank)
    states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16)
    active = torch.ones(rows, dtype=torch.bool, device="npu")
    model = make_moe(rows, active)
    original_shared = model._shared_output
    parts = {}

    def shared(value):
        result = original_shared(value)
        parts["shared"] = result
        return result

    model._shared_output = shared

    # Compare eager and graph outputs with the same preselected route tensors.
    eager = []
    graphs, outputs, shared_outputs = [], [], []
    for optimized in (False, True):
        with pytest.MonkeyPatch.context() as patch:
            if not optimized:
                patch.setattr(
                    moe_module, "mask_decode_rows", lambda value, mask: torch.where(mask[:, None], value, 0.0)
                )
            eager.append(model.forward_decode(states, active).cpu())
            graph, output = capture(lambda: model.forward_decode(states, active))
            graphs.append(graph)
            outputs.append(output)
            shared_outputs.append(parts["shared"])
    torch.testing.assert_close(eager[1], eager[0], rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
    for case, active_dp in enumerate((4, 1, 0, 4)):
        active.copy_((torch.arange(rows, device="npu") % 5 != 1) & (rank // TP < active_dp))
        states.mul_(0.9).add_(0.01 * (case + 1))
        model.experts.fixed_indices.add_(13).remainder_(EXPERTS)
        model.experts.fixed_weights.mul_(0.95)
        model.shared_experts.w1.weight_scale.mul_(1.01)
        model.experts.scale2.mul_(1.005)
        graphs[0].replay()
        torch.npu.synchronize()
        expected = outputs[0].cpu()
        expected_shared = shared_outputs[0].cpu()
        expected_counts = model.experts.counts.cpu()
        graphs[1].replay()
        torch.npu.synchronize()
        torch.testing.assert_close(outputs[1].cpu(), expected, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
        assert_shared_exact(shared_outputs[1].cpu(), expected_shared)
        torch.testing.assert_close(model.experts.counts.cpu(), expected_counts, rtol=0, atol=0)
        assert torch.count_nonzero(outputs[1][~active].cpu()) == 0
        assert model.experts.fixed_indices.dtype == torch.int32

    # Both timed graphs include the unchanged FP32 gate and real top-k selection.
    model.experts.use_fixed_routing = False
    active.fill_(True)
    functions = []
    for optimized in (False, True):
        with pytest.MonkeyPatch.context() as patch:
            if not optimized:
                patch.setattr(
                    moe_module, "mask_decode_rows", lambda value, mask: torch.where(mask[:, None], value, 0.0)
                )

            def complete(model=model, states=states, active=active):
                result = None
                for _ in range(TIMED_CALLS):
                    result = model.forward_decode(states, active)
                return result

            graph, _ = capture(complete)
            functions.append(graph.replay)
    samples = [[], []]
    for repeat in range(TIMING_SAMPLES):
        for index in (repeat % 2, 1 - repeat % 2):
            samples[index].append(time_complete(functions[index]))
    medians = [list(map(statistics.median, zip(*sample))) for sample in samples]
    dist.barrier()
    if rank == 0:
        print(
            "TARGET_ROW_MASK_MOE_COMPONENT "
            + json.dumps(
                {
                    "status": "PASS",
                    "world_size": EP,
                    "local_experts": LOCAL_EXPERTS,
                    "top_k": TOP_K,
                    "rows": rows,
                    "native_graph_cases": 4,
                    "calls": TIMED_CALLS,
                    "metrics": ["host_ms", "ready_ms"],
                    "parent": medians[0],
                    "candidate": medians[1],
                    "scope": "gate, selection, target EP16 MC2, native shared expert, FP32 merge and BF16 mask",
                    "weights": "synthetic native W8A8 weights; fixed routing for paired numerical comparisons",
                    "rtol": NUMERICAL_RTOL,
                    "atol": NUMERICAL_ATOL,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == EP
        enable_custom_op()
        torch.npu.config.allow_internal_format = True
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("vllm_ascend.utils.get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=WEIGHT_NZ_MODE))
            for rows in (6, 12):
                run_case(rows)
    finally:
        dist.destroy_process_group()
