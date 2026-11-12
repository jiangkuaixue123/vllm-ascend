# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real EP16 draft experts, mixed DP widths, mask refresh and graph replay."""

import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open
from torch import nn
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.models.common.ops import draft_ffn_graph as graph_module
from vllm_ascend.models.deepseek_v41 import fused_experts, moe
from vllm_ascend.models.deepseek_v41.dspark import DSparkMoE
from vllm_ascend.ops.fused_moe import prepare_finalize
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.linear import AscendReplicatedLinear
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op

TP_SIZE = 4
EP_SIZE = 16
LOCAL_EXPERTS = 8
HIDDEN = 5120
INTERMEDIATE = 2304
CALLS = 6
SAMPLES = 5


class NativeQuantConfig:
    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(w8a8_dynamic.AscendW8A8DynamicLinearMethod())


class TPGroup:
    world_size = TP_SIZE

    def __init__(self, group):
        self.group = group
        self.rank_in_group = dist.get_rank() % TP_SIZE

    def all_gather(self, value, dim):
        assert dim == 0
        output = value.new_empty((value.shape[0] * TP_SIZE, value.shape[1]))
        dist.all_gather_into_tensor(output, value.contiguous(), group=self.group)
        return output


class NativeRouted(nn.Module):
    """Use production prepare, routing, fused EP operation and finalization."""

    def __init__(self, gate, read):
        super().__init__()
        self.gate = gate
        weights, scales = [[], []], [[], []]
        for expert in range(dist.get_rank() * LOCAL_EXPERTS, (dist.get_rank() + 1) * LOCAL_EXPERTS):
            prefix = f"mtp.0.ffn.experts.{expert}."
            weights[0].append(torch.cat([read(prefix + name + ".weight") for name in ("w1", "w3")]).t())
            weights[1].append(read(prefix + "w2.weight").t())
            scales[0].append(torch.cat([read(prefix + name + ".weight_scale") for name in ("w1", "w3")]).flatten())
            scales[1].append(read(prefix + "w2.weight_scale").flatten())
        for index in range(2):
            self.register_buffer(
                f"weight{index + 1}",
                torch_npu.npu_format_cast(torch.stack(weights[index]).npu(), ACL_FORMAT_FRACTAL_NZ),
            )
            self.register_buffer(f"scale{index + 1}", torch.stack(scales[index]).bfloat16().float().npu())
        self.prepare = object.__new__(prepare_finalize.PrepareAndFinalizeWithMC2)
        self.prepare.tp_size, self.prepare.tp_rank = TP_SIZE, dist.get_rank() % TP_SIZE
        self.comm = object.__new__(fused_experts.DeepseekV41FusedMC2Comm)
        self.comm.moe_config = SimpleNamespace(
            ep_size=EP_SIZE,
            experts_per_token=3,
            num_experts=128,
            ep_group=SimpleNamespace(rank_in_group=dist.get_rank()),
        )
        group = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(dist.get_rank())
        self.comm.token_dispatcher = SimpleNamespace(moe_all_to_all_group_name=group)
        self.comm.swiglu_limit = 10.0
        self.comm.expert_token_nums = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device="npu")

    def forward(self, hidden_states, router_logits):
        prepared = self.prepare.prepare(hidden_states, router_logits, replace_allreduce=True)
        routes, indices = self.gate.select_experts(prepared.hidden_states, prepared.router_logits, 3, True)
        inputs = build_fused_experts_input(
            hidden_states=prepared.hidden_states,
            topk_weights=routes,
            topk_ids=indices,
            w1=self.weight1,
            w2=self.weight2,
            w1_scale=self.scale1,
            w2_scale=self.scale2,
            quant_type=fused_experts.QuantType.W8A8,
            dynamic_eplb=False,
            mc2_mask=prepared.mc2_mask,
        )
        output = self.comm.fused_experts(inputs).routed_out
        return self.prepare.finalize(output, False)


def make_layer(patch, group):
    path = Path("/home/admin/model")
    index = json.loads((path / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]

    def read(name):
        with safe_open(path / index[name], framework="pt", device="cpu") as reader:
            return reader.get_tensor(name)

    configuration = json.loads((path / "config.json").read_text())
    config = SimpleNamespace(**configuration.get("text_config", configuration))
    config.n_routed_experts, config.num_experts_per_tok = 128, 3
    patch.setattr(moe, "ReplicatedLinear", AscendReplicatedLinear)
    patch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: dist.get_rank() % TP_SIZE)
    patch.setattr(w8a8_dynamic, "maybe_trans_nz", lambda value: torch_npu.npu_format_cast(value, ACL_FORMAT_FRACTAL_NZ))
    patch.setattr(fused_experts, "get_ascend_config", lambda: SimpleNamespace(mega_moe_max_tokens=65536))
    layer = DSparkMoE.__new__(DSparkMoE)
    nn.Module.__init__(layer)
    layer.hidden_size, layer.n_routed_experts, layer.use_fused_runner = HIDDEN, 128, True
    layer.communication = moe.DeepseekV41MoECommunication(
        SimpleNamespace(rank_in_group=dist.get_rank() // TP_SIZE, world_size=4),
        SimpleNamespace(rank_in_group=dist.get_rank(), world_size=EP_SIZE),
        TPGroup(group),
        sequence_parallel=True,
        replicated_shared=True,
    )
    with torch.device("npu"):
        layer.gate = moe.DeepseekV41Gate(config)
        layer.shared_experts = moe.DeepseekV41MoEExpert(
            config,
            NativeQuantConfig(),
            "mtp.0.ffn.shared_experts",
            shared=True,
            replicated_shared=True,
            sequence_parallel=True,
        )
    for name, parameter in layer.gate.named_parameters():
        parameter.copy_(read("mtp.0.ffn.gate." + name))
    for name, parameter in layer.shared_experts.named_parameters():
        parameter.copy_(read("mtp.0.ffn.shared_experts." + name))
    for projection in (layer.shared_experts.w1, layer.shared_experts.w3, layer.shared_experts.w2):
        projection.quant_method.process_weights_after_loading(projection)
    layer.experts = NativeRouted(layer.gate, read)
    layer._ffn_graph = graph_module.DraftFFNGraph()
    return layer.eval()


def exact(actual, expected):
    torch.testing.assert_close(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16), rtol=0, atol=0)


def time_calls(layer, states, graph):
    layer._ffn_graph = graph
    layer(states)
    samples = []
    for _ in range(SAMPLES):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        for _ in range(CALLS):
            layer(states)
        submitted = time.perf_counter_ns()
        torch.npu.synchronize()
        ready = time.perf_counter_ns()
        result = torch.tensor([(submitted - start) / 1e6, (ready - start) / 1e6], device="npu", dtype=torch.float32)
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
        samples.append(result.cpu().tolist())
    return [statistics.median(sample[index] for sample in samples) for index in range(2)]


@torch.inference_mode()
def run():
    """Exercise differing DP widths without changing the original eager gate."""
    patch = pytest.MonkeyPatch()
    groups = [dist.new_group(list(range(start, start + TP_SIZE))) for start in range(0, EP_SIZE, TP_SIZE)]
    context = SimpleNamespace(
        is_draft_model=True, in_profile_run=False, moe_comm_type=graph_module.MoECommType.FUSED_MC2
    )
    patch.setattr(graph_module, "is_forward_context_available", lambda: True)
    patch.setattr(graph_module, "get_forward_context", lambda: context)
    patch.setattr(prepare_finalize, "_EXTRA_CTX", context)
    try:
        layer = make_layer(patch, groups[dist.get_rank() // TP_SIZE])
        graph = layer._ffn_graph
        for counts in ((5, 0, 5, 0), (20, 10, 5, 0), (40, 35, 20, 0)):
            padded = (max(counts) + TP_SIZE - 1) // TP_SIZE * TP_SIZE
            context.padded_num_tokens = padded
            for step in range(4):
                tokens = counts[(dist.get_rank() // TP_SIZE + step) % 4]
                context.mc2_mask = torch.arange(padded, device="npu") < tokens
                original_mask = context.mc2_mask
                generator = torch.Generator().manual_seed(91 + dist.get_rank() // TP_SIZE + step)
                states = torch.randn(tokens, HIDDEN, generator=generator).bfloat16().npu() * (0.0 if step == 2 else 0.1)
                full = F.pad(states, (0, 0, 0, -tokens % TP_SIZE))
                width = full.shape[0] // TP_SIZE
                local = full.narrow(0, dist.get_rank() % TP_SIZE * width, width).contiguous()
                # Gate GEMM is outside the changed region in both production paths.
                logits = layer.gate.routing_logits(local, decode=False, image_mask=None)
                expected = layer._forward_experts(local, logits).clone()
                actual = graph(layer, local, logits)
                exact(actual, expected)
                assert context.mc2_mask is original_mask
                assert list(graph.groups[padded].entries) == list(range(padded // TP_SIZE + 1))
                actual.fill_(7)
                layer.shared_experts.w1.weight_scale.mul_(1.001)
                layer.experts.scale2.mul_(1.001)
                layer.gate.bias.add_(0.0001)
            # Timings include unchanged eager gate and original TP all-gather.
            tokens = max(counts)
            context.mc2_mask = torch.arange(padded, device="npu") < tokens
            states = torch.randn(tokens, HIDDEN, device="npu", dtype=torch.bfloat16) * 0.1
            parent = time_calls(layer, states, None)
            candidate = time_calls(layer, states, graph)
            if dist.get_rank() == 0:
                print(
                    "DSPARK_FFN_GRAPH_COMPONENT "
                    + json.dumps(
                        dict(
                            status="PASS",
                            tokens=tokens,
                            padded=padded,
                            mixed_dp_cases=4,
                            calls=CALLS,
                            metrics=["host_ms", "ready_ms"],
                            parent=parent,
                            candidate=candidate,
                        )
                    ),
                    flush=True,
                )
    finally:
        patch.undo()


if __name__ == "__main__":
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == EP_SIZE
        torch.npu.config.allow_internal_format = True
        torch.set_default_dtype(torch.bfloat16)
        enable_custom_op()
        run()
    finally:
        dist.destroy_process_group()
