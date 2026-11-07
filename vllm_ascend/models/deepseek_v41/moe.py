# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness-first packed EP execution for the DeepSeek V4.1 text MoE.

The equations follow the official inference/model.py at revision
314113f8a7dadb388215396ba5fbee2c610a04d6. Routing weights are applied before the
down projection, including its activation quantization. This implementation
supports replicated or sequence-sharded input rows with static expert placement.
"""

from collections.abc import Sequence
from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.deepseek_v41.config import DeepseekV41TextConfig
from vllm_ascend.models.deepseek_v41.core import (
    clipped_swiglu,
    route_sqrtsoftplus,
)


def local_expert_ids(num_experts: int, ep_size: int, ep_rank: int) -> range:
    if num_experts < 1 or ep_size < 1 or num_experts % ep_size:
        raise ValueError("The number of experts must be positive and divisible by EP")
    if not 0 <= ep_rank < ep_size:
        raise ValueError("EP rank is outside the expert parallel group")
    count = num_experts // ep_size
    return range(ep_rank * count, (ep_rank + 1) * count)


def unpack_dp_tokens(gathered: torch.Tensor, token_counts: Sequence[int]) -> torch.Tensor:
    """Remove equal-size gather padding, preserving DP rank and token order."""
    if not token_counts or min(token_counts) < 0:
        raise ValueError("DP token counts must be a nonempty nonnegative sequence")
    padded_count = max(token_counts)
    if gathered.shape[0] != len(token_counts) * padded_count:
        raise ValueError("Gathered rows do not match the padded DP token counts")
    return torch.cat(
        [gathered.narrow(0, rank * padded_count, count) for rank, count in enumerate(token_counts)],
        dim=0,
    )


def select_dp_tokens(gathered: torch.Tensor, token_counts: Sequence[int], dp_rank: int) -> torch.Tensor:
    if not 0 <= dp_rank < len(token_counts) or min(token_counts) < 0:
        raise ValueError("Invalid DP rank or token counts")
    if gathered.shape[0] != sum(token_counts):
        raise ValueError("Reduced rows do not match the unpadded DP token counts")
    return gathered.narrow(0, sum(token_counts[:dp_rank]), token_counts[dp_rank])


def local_expert_contributions(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_indices: torch.Tensor,
    experts: Sequence[nn.Module | None],
    expert_ids: range,
) -> torch.Tensor:
    """Accumulate each owned expert once per packed batch in FP32."""
    if (
        hidden_states.ndim != 2
        or routing_indices.ndim != 2
        or routing_weights.shape != routing_indices.shape
        or routing_indices.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError("Expected packed states [tokens, hidden] and routes [tokens, k]")
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_id in expert_ids:
        expert = experts[expert_id]
        if expert is None:
            raise ValueError(f"Owned expert {expert_id} has no module")
        token, slot = torch.where(routing_indices == expert_id)
        if token.numel() == 0:
            continue
        contribution = expert(hidden_states[token], routing_weights[token, slot].unsqueeze(-1))
        output.index_add_(0, token, contribution.float())
    return output


def local_decode_contributions(
    states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: Sequence[nn.Module | None],
    expert_ids: range,
) -> torch.Tensor:
    """Evaluate the original top-k routes with fixed shapes for graph replay."""
    output = torch.zeros_like(states, dtype=torch.float32)
    for expert_id in expert_ids:
        expert = experts[expert_id]
        if expert is None:
            raise ValueError(f"Owned expert {expert_id} has no module")
        matches = indices == expert_id
        selected = matches.any(dim=-1, keepdim=True)
        # Top-k contains distinct expert IDs, so this selects at most one weight.
        # Leave all rows in the GEMM and mask nonroutes after W2. The routed
        # weight still enters the expert before W2 activation quantization.
        factor = torch.where(matches, weights, 0.0).sum(dim=-1, keepdim=True)
        contribution = expert(states, torch.where(selected, factor, 1.0))
        output.add_(torch.where(selected, contribution.float(), 0.0))
    return output


class PackedMoECommunication(Protocol):
    dp_rank: int
    ep_rank: int
    ep_size: int
    sequence_parallel: bool
    replicated_shared: bool

    def gather(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]: ...

    def reduce(self, contributions: torch.Tensor) -> torch.Tensor: ...

    def gather_decode(self, states: torch.Tensor) -> torch.Tensor: ...

    def gather_sequence(self, states: torch.Tensor, *, fixed: bool) -> torch.Tensor: ...

    def reduce_sequence(self, contributions: torch.Tensor, local_rows: int) -> torch.Tensor: ...

    def shared_input(self, states: torch.Tensor) -> torch.Tensor: ...

    def shared_output(self, output: torch.Tensor, local_rows: int) -> torch.Tensor: ...


class DeepseekV41MoECommunication:
    """Preserve token ownership across the physical TP, DP and EP groups."""

    def __init__(self, dp_group, ep_group, tp_group=None, *, sequence_parallel=False, replicated_shared=False):
        self.dp_group = dp_group
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.dp_rank = dp_group.rank_in_group
        self.ep_rank = ep_group.rank_in_group
        self.ep_size = ep_group.world_size
        self.sequence_parallel = sequence_parallel
        self.replicated_shared = replicated_shared
        if (sequence_parallel or replicated_shared) and (tp_group is None or tp_group.world_size < 2):
            raise ValueError("Shared-expert DP and sequence parallelism require a physical TP group")

    def _token_counts(self, local_count: int) -> tuple[int, ...]:
        if self.dp_group.world_size == 1:
            return (local_count,)
        # DPMetadata describes runner padding, whereas the model may remove
        # padding before MoE. Every rank must take this same size collective.
        local_size = torch.tensor([local_count], dtype=torch.int64, device="cpu")
        sizes = [torch.empty_like(local_size) for _ in range(self.dp_group.world_size)]
        dist.all_gather(sizes, local_size, group=self.dp_group.cpu_group)
        return tuple(size.tolist()[0] for size in sizes)

    def gather(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        counts = self._token_counts(hidden_states.shape[0])
        if len(counts) != self.dp_group.world_size or min(counts) < 0:
            raise ValueError("DP token metadata does not match the DP group")
        if hidden_states.shape[0] != counts[self.dp_rank]:
            raise ValueError("MoE expects full local-DP tokens replicated across TP")
        if self.dp_group.world_size == 1 or max(counts) == 0:
            return hidden_states, counts
        padded = F.pad(hidden_states, (0, 0, 0, max(counts) - hidden_states.shape[0]))
        gathered = self.dp_group.all_gather(padded.contiguous(), dim=0)
        return unpack_dp_tokens(gathered, counts), counts

    def reduce(self, contributions: torch.Tensor) -> torch.Tensor:
        if contributions.shape[0] == 0:
            return contributions
        return self.ep_group.all_reduce(contributions)

    def gather_decode(self, states: torch.Tensor) -> torch.Tensor:
        """All DP ranks have the same positive bucket size, including idle ranks."""
        if self.dp_group.world_size == 1:
            return states
        return self.dp_group.all_gather(states.contiguous(), dim=0)

    def gather_sequence(self, states: torch.Tensor, *, fixed: bool) -> torch.Tensor:
        """Gather unique TP token shards in EP-rank order, padding unequal DP batches."""
        local_rows = states.shape[0]
        rows = local_rows if fixed else max(self._token_counts(local_rows))
        if not rows:
            return states
        padded = F.pad(states, (0, 0, 0, rows - local_rows)) if rows != local_rows else states
        return self.ep_group.all_gather(padded.contiguous(), dim=0)

    def reduce_sequence(self, contributions: torch.Tensor, local_rows: int) -> torch.Tensor:
        # The gather and scatter use the same EP group/order. Each token owner
        # receives the sum of its experts exactly once, then drops only EP padding.
        return self.ep_group.reduce_scatter(contributions.contiguous(), dim=0)[:local_rows]

    def shared_input(self, states: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel and not self.replicated_shared:
            return self.tp_group.all_gather(states.contiguous(), dim=0)
        if self.replicated_shared and not self.sequence_parallel:
            tp = self.tp_group.world_size
            pad = (-states.shape[0]) % tp
            padded = F.pad(states, (0, 0, 0, pad)) if pad else states
            rows = padded.shape[0] // tp
            return padded.narrow(0, self.tp_group.rank_in_group * rows, rows).contiguous()
        return states

    def shared_output(self, output: torch.Tensor, local_rows: int) -> torch.Tensor:
        if self.sequence_parallel and not self.replicated_shared:
            return self.tp_group.reduce_scatter(output.float().contiguous(), dim=0).to(output.dtype)
        if self.replicated_shared and not self.sequence_parallel:
            return self.tp_group.all_gather(output.contiguous(), dim=0)[:local_rows]
        return output


class DeepseekV41Gate(nn.Module):
    def __init__(self, config: DeepseekV41TextConfig, *, vision_routing: bool = False):
        super().__init__()
        if config.scoring_func != "sqrtsoftplus" or config.topk_method != "noaux_tc":
            raise ValueError("V4.1 text routing requires sqrtsoftplus/noaux_tc")
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.scaling_factor = config.routed_scaling_factor
        self.temperature = getattr(config, "gate_temp", 1.0)
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )

        self.bias_vl = nn.Parameter(torch.empty_like(self.bias), requires_grad=False) if vision_routing else None
        self._draft_router_graph = None

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.shape[0]:
            return self.weight.new_empty((0, self.weight.shape[0]))
        return F.linear(hidden_states.float(), self.weight)

    def routing_logits(self, states, *, decode, image_mask):
        logits = self.logits(states)
        if self.bias_vl is None or decode:
            return logits
        # Gather the modality alongside the token's router logits so it follows
        # exactly the same EP gather, including padding on an idle DP replica.
        if image_mask is None:
            image_mask = logits.new_zeros(logits.shape[0])
        return torch.cat((logits, image_mask.reshape(-1, 1).to(logits.dtype)), dim=-1)

    def select_experts(self, hidden_states, gating_output, topk, renormalize):
        graph = getattr(self, "_draft_router_graph", None)
        if graph is not None:
            selected = graph(self, gating_output, topk, renormalize)
            if selected is not None:
                return selected
        bias = self.bias
        if self.bias_vl is not None and gating_output.shape[-1] == bias.numel() + 1:
            logits, modality = gating_output.split((bias.numel(), 1), dim=-1)
            gating_output = logits
            bias = torch.where(modality.bool(), self.bias_vl, bias)
        return route_sqrtsoftplus(
            gating_output,
            bias,
            topk,
            renormalize,
            self.scaling_factor,
            self.temperature,
        )

    def forward(self, hidden_states: torch.Tensor):
        return self.select_experts(hidden_states, self.logits(hidden_states), self.top_k, self.renormalize)


class DeepseekV41MoEExpert(nn.Module):
    """Checkpoint-named projections with routing before down quantization."""

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        *,
        shared: bool = False,
        replicated_shared: bool = False,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        hidden, intermediate = config.hidden_size, config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        sharded = shared and not replicated_shared
        self.reduce_shared = sharded and not sequence_parallel
        up_class = ColumnParallelLinear if sharded else ReplicatedLinear
        down_class = RowParallelLinear if sharded else ReplicatedLinear
        common = dict(bias=False, quant_config=quant_config, return_bias=False)
        up_options = {} if sharded else {"disable_tp": True}
        down_options = {"input_is_parallel": True, "reduce_results": False} if sharded else {"disable_tp": True}
        self.w1 = up_class(hidden, intermediate, prefix=f"{prefix}.w1", **common, **up_options)
        self.w2 = down_class(intermediate, hidden, prefix=f"{prefix}.w2", **common, **down_options)
        self.w3 = up_class(hidden, intermediate, prefix=f"{prefix}.w3", **common, **up_options)

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor | None = None) -> torch.Tensor:
        activation = clipped_swiglu(self.w1(hidden_states), self.w3(hidden_states), self.swiglu_limit)
        if routing_weights is not None:
            activation = activation * routing_weights
        output = self.w2(activation.to(hidden_states.dtype))
        if self.reduce_shared:
            output = tensor_model_parallel_all_reduce(output.float())
        return output.to(hidden_states.dtype)


class DeepseekV41MoE(nn.Module):
    def __init__(
        self,
        config: DeepseekV41TextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        *,
        communication: PackedMoECommunication | None = None,
        vision_routing: bool = False,
    ):
        super().__init__()
        if config.n_shared_experts != 1 or config.hidden_act != "silu":
            raise ValueError("V4.1 requires one shared SwiGLU expert")
        self.use_fused_runner = communication is None
        if communication is None:
            parallel = get_current_vllm_config().parallel_config
            if (
                parallel.enable_eplb
                or parallel.enable_elastic_ep
                or parallel.expert_placement_strategy != "linear"
                or parallel.enable_dbo
                or parallel.prefill_context_parallel_size != 1
            ):
                raise ValueError("Reference V4.1 MoE requires static linear EP without PCP/DBO")
            communication = DeepseekV41MoECommunication(
                get_dp_group(),
                get_ep_group(),
                get_tp_group(),
                sequence_parallel=parallel.use_sequence_parallel_moe,
                replicated_shared=get_ascend_config().enable_shared_expert_dp,
            )
            expected_ep_size = parallel.tensor_parallel_size * parallel.data_parallel_size
            if communication.ep_size != expected_ep_size or (
                expected_ep_size > 1 and not parallel.enable_expert_parallel
            ):
                raise ValueError("V4.1 expert parallel group must span TP x DP")
        self.communication = communication
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.expert_ids = local_expert_ids(config.n_routed_experts, communication.ep_size, communication.ep_rank)
        self.n_local_experts = len(self.expert_ids)
        self.experts_start_idx, self.experts_end_idx = (
            self.expert_ids.start,
            self.expert_ids.stop,
        )
        self.gate = DeepseekV41Gate(config, vision_routing=vision_routing)
        if self.use_fused_runner:
            # Import the registered Ascend ops in the worker process only.
            from .fused_moe import build_fused_experts

            self.experts = build_fused_experts(
                config, quant_config, f"{prefix}.experts", self.gate, communication.sequence_parallel
            )
        else:
            # The injected communication path is the CPU numerical reference.
            self.experts = nn.ModuleList(
                [
                    DeepseekV41MoEExpert(config, quant_config, f"{prefix}.experts.{expert_id}")
                    if expert_id in self.expert_ids
                    else None
                    for expert_id in range(config.n_routed_experts)
                ]
            )
        self.shared_experts = DeepseekV41MoEExpert(
            config,
            quant_config,
            f"{prefix}.shared_experts",
            shared=True,
            replicated_shared=communication.replicated_shared,
            sequence_parallel=communication.sequence_parallel,
        )
        logger.info_once(
            "V4.1 MoE layout: sequence_parallel=%s replicated_shared=%s shared_w2_shape=%s",
            communication.sequence_parallel,
            communication.replicated_shared,
            tuple(self.shared_experts.w2.weight.shape),
        )

    def _routed_output(self, states: torch.Tensor, *, decode: bool, image_mask=None) -> torch.Tensor:
        comm = self.communication
        if self.use_fused_runner:
            logger.info_once(
                "V4.1 executing Ascend fused MoE runner: experts=%d top_k=%d sequence_parallel=%s decode=%s",
                self.n_routed_experts,
                self.gate.top_k,
                comm.sequence_parallel,
                decode,
                scope="process",
            )
            logits = self.gate.routing_logits(states, decode=decode, image_mask=image_mask)
            output = self.experts(hidden_states=states, router_logits=logits)
            if output.shape != states.shape:
                raise RuntimeError(f"V4.1 fused MoE returned {output.shape} for local token shard {states.shape}")
            return output.float()
        reference_image_routing = self.gate.bias_vl is not None and not decode
        if reference_image_routing:
            row_types = states.new_zeros(states.shape[0]) if image_mask is None else image_mask
            states_to_gather = torch.cat((states, row_types.reshape(-1, 1).to(states.dtype)), dim=1)
        else:
            states_to_gather = states
        if comm.sequence_parallel:
            logger.info_once("V4.1 MoE executing sequence-parallel EP gather/reduce-scatter")
            gathered = comm.gather_sequence(states_to_gather, fixed=decode)
        elif decode:
            gathered = comm.gather_decode(states_to_gather)
        else:
            gathered, counts = comm.gather(states_to_gather)
        if gathered.shape[0] == 0:
            return states.float()
        if reference_image_routing:
            gathered, mask = gathered.split((self.hidden_size, 1), dim=1)
            logits = self.gate.routing_logits(gathered, decode=False, image_mask=mask)
            weights, indices = self.gate.select_experts(gathered, logits, self.gate.top_k, self.gate.renormalize)
        else:
            weights, indices = self.gate(gathered)
        evaluate = local_decode_contributions if decode else local_expert_contributions
        partial = evaluate(gathered, weights, indices, self.experts, self.expert_ids)
        if comm.sequence_parallel:
            return comm.reduce_sequence(partial, states.shape[0])
        reduced = comm.reduce(partial)
        if decode:
            slots = states.shape[0]
            return reduced.narrow(0, comm.dp_rank * slots, slots)
        return select_dp_tokens(reduced, counts, comm.dp_rank)

    def _shared_output(self, states: torch.Tensor) -> torch.Tensor:
        comm = self.communication
        if comm.replicated_shared:
            logger.info_once("V4.1 MoE executing replicated shared expert on TP token shards")
        return comm.shared_output(self.shared_experts(comm.shared_input(states)), states.shape[0])

    def forward(self, hidden_states: torch.Tensor, image_mask=None) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("V4.1 MoE requires packed [tokens, hidden_size] inputs")
        output = self._routed_output(hidden_states, decode=False, image_mask=image_mask)
        if hidden_states.shape[0] > 0:
            output = output + self._shared_output(hidden_states).float()
        return output.to(hidden_states.dtype)

    def forward_decode(self, hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Run the same collectives on every rank using fixed local decode slots."""
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != self.hidden_size
            or not hidden_states.shape[0]
            or token_mask.shape != hidden_states.shape[:1]
            or token_mask.dtype != torch.bool
            or token_mask.device != hidden_states.device
        ):
            raise ValueError("Expected fixed decode states [slots, hidden] and bool slot mask")
        valid = token_mask.unsqueeze(-1)
        states = torch.where(valid, hidden_states, 0.0)
        output = self._routed_output(states, decode=True) + self._shared_output(states).float()
        return torch.where(valid, output.to(states.dtype), 0.0)
