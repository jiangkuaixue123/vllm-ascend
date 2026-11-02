# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text inference for indexed Ascend W8A8 checkpoints.

The model equations derive only from the official HF reference revision
314113f8a7dadb388215396ba5fbee2c610a04d6. Text inference keeps Engram
tables on the host and uses separate TP attention and DP/EP expert domains.
"""

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_ascend.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41RMSNorm,
    DeepseekV41Runtime,
    RequestSpan,
)
from vllm_ascend.models.deepseek_v41.core import (
    hc_mixes,
    hc_post,
    hc_pre,
    make_identity_pre_mix,
)
from vllm_ascend.models.deepseek_v41.engram import (
    DeepseekV41Engram,
    EngramHasher,
    EngramLayout,
    IndexedMmapStore,
    MmapEngramEmbedding,
)
from vllm_ascend.models.deepseek_v41.moe import DeepseekV41MoE

MAX_DECODE_GRAPH_ROWS = 8


def _decode_capture_sizes(compilation):
    sizes = tuple(compilation.cudagraph_capture_sizes or ())
    if (
        compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
        or not sizes
        or any(type(size) is not int or not 1 <= size <= MAX_DECODE_GRAPH_ROWS for size in sizes)
        or len(set(sizes)) != len(sizes)
    ):
        raise ValueError("V4.1 requires eager or FULL_DECODE_ONLY with unique capture sizes in [1, 8]")
    return frozenset(sizes)


def reduce_engram(values, group):
    return group.all_reduce(values)


class EngramQueryUnrotation(nn.Module):
    """Recover the original gate basis from row-vector residuals x_rot=x@Q."""

    def __init__(self, matrix):
        super().__init__()
        if matrix.dtype != torch.float32 or matrix.ndim != 2:
            raise ValueError("Engram rotation must be an FP32 matrix")
        self.register_buffer("matrix", matrix, persistent=False)

    def forward(self, hidden_states):
        if self.matrix.dtype != torch.float32:
            raise RuntimeError("Engram query rotation must retain FP32 precision")
        return hidden_states.float() @ self.matrix.T


class DeepseekV41Block(nn.Module):
    def __init__(self, config, vllm_config, layer_id):
        super().__init__()
        self.config = config
        prefix = f"layers.{layer_id}"
        self.attn = DeepseekV41Attention(config, vllm_config, layer_id, f"{prefix}.attn")
        self.ffn = DeepseekV41MoE(config, vllm_config.quant_config, f"{prefix}.ffn")
        self.attn_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.engram = None
        width = (config.hc_mult + 2) * config.hc_mult
        for name in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{name}_fn",
                nn.Parameter(torch.empty(width, config.hc_mult * config.hidden_size, dtype=torch.float32)),
            )
            self.register_parameter(f"hc_{name}_base", nn.Parameter(torch.empty(width, dtype=torch.float32)))
            self.register_parameter(f"hc_{name}_scale", nn.Parameter(torch.empty(3, dtype=torch.float32)))

    def mixes(self, x, name):
        return hc_mixes(
            x,
            getattr(self, f"hc_{name}_fn"),
            getattr(self, f"hc_{name}_scale"),
            getattr(self, f"hc_{name}_base"),
            self.config.rms_norm_eps,
            self.config.hc_eps,
            self.config.hc_sinkhorn_iters,
        )

    def forward(self, hidden_states, pre_mix, runtime):
        if hidden_states.shape[0] == 0:
            # Other DP replicas can be active. Participate in every expert
            # collective while avoiding empty NPU GEMMs and TP reductions.
            self.ffn(hidden_states.new_empty((0, self.config.hidden_size)))
            return hidden_states, pre_mix
        attn_pre, post, comb = self.mixes(hidden_states, "attn")
        attended = self.attn(self.attn_norm(hc_pre(hidden_states, pre_mix)), runtime)
        hidden_states = hc_post(attended, hidden_states, post, comb)
        ffn_pre, post, comb = self.mixes(hidden_states, "ffn")
        ffn_input = self.ffn_norm(hc_pre(hidden_states, attn_pre))
        if getattr(runtime, "decode", None) is not None:
            output = self.ffn.forward_decode(ffn_input, runtime.decode.active)
        else:
            output = self.ffn(ffn_input)
        return hc_post(output, hidden_states, post, comb), ffn_pre


def read_parameter_slice(reader, key, parameter, module, tp_rank, tp_size):
    """Read only the local TP slice; no arbitrary shard iteration or duplicates."""
    source = reader.get_slice(key)
    source_shape = tuple(source.get_shape())
    target_shape = tuple(parameter.shape)
    slices = [slice(None)] * len(source_shape)
    if isinstance(module, VocabParallelEmbedding):
        indices = module.shard_indices
        slices[0] = slice(indices.org_vocab_start_index, indices.org_vocab_end_index)
        value = source[tuple(slices)]
        parameter.zero_()
        parameter[: value.shape[0]].copy_(value)
        return
    if source_shape == target_shape:
        value = source[tuple(slices)]
    else:
        if isinstance(module, ColumnParallelLinear):
            axis = 0
        elif isinstance(module, RowParallelLinear) and key.endswith(".weight"):
            axis = 1
        elif key.endswith(".attn_sink"):
            axis = 0
        else:
            raise ValueError(f"No TP mapping for {key}: {source_shape} -> {target_shape}")
        if source_shape[axis] != target_shape[axis] * tp_size:
            raise ValueError(f"Invalid TP partition for {key}")
        slices[axis] = slice(tp_rank * target_shape[axis], (tp_rank + 1) * target_shape[axis])
        value = source[tuple(slices)]
    if value.shape != parameter.shape:
        raise ValueError(f"Checkpoint shape mismatch for {key}: {value.shape} != {parameter.shape}")
    parameter.copy_(value)


class DeepseekV41ForCausalLM(nn.Module):
    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        parallel = vllm_config.parallel_config
        if parallel.pipeline_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
            raise ValueError("V4.1 reference path currently requires PP1 and DCP1")
        if parallel.enable_eplb:
            raise ValueError("V4.1 reference experts do not implement EPLB migration")
        if vllm_config.speculative_config is not None:
            raise ValueError("V4.1 DSpark integration has not been implemented")
        self.decode_enabled = not vllm_config.model_config.enforce_eager
        self._decode_capture_sizes = (
            _decode_capture_sizes(vllm_config.compilation_config) if self.decode_enabled else frozenset()
        )
        self._decode_engram_values: dict[tuple[int, int], torch.Tensor] = {}
        config = self.config
        self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix="embed")
        self.layers = nn.ModuleList(
            [DeepseekV41Block(config, vllm_config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            params_dtype=torch.float32,
            prefix="head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.model_path = Path(vllm_config.model_config.model)
        self.engram_store = None
        self.engram_hasher = None
        layout = EngramLayout.from_config(config)
        if layout is not None:
            self.engram_store = IndexedMmapStore(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                vllm_config.model_config.tokenizer,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.engram_hasher = EngramHasher.from_tokenizer(
                layout,
                tokenizer,
                compressed_vocab_size=config.engram_compressed_vocab_size,
                pad_token_id=config.engram_pad_token_id,
            )
            for layer_id in layout.layer_ids:
                prefix = f"layers.{layer_id}.engram"
                embedding = MmapEngramEmbedding(
                    self.engram_store,
                    prefix + ".embed",
                    tp_rank=get_tensor_model_parallel_rank(),
                    tp_size=get_tensor_model_parallel_world_size(),
                    tp_group=get_tp_group(),
                )
                projection = nn.Linear(
                    layout.hash_columns * layout.head_dim,
                    (config.hc_mult + 1) * config.hidden_size,
                    bias=False,
                )
                self.layers[layer_id].engram = DeepseekV41Engram(
                    embedding,
                    projection,
                    torch.empty(config.hc_mult, config.hidden_size),
                    torch.empty(config.hc_mult, config.hidden_size),
                    eps=config.rms_norm_eps,
                    tp_reduce=reduce_engram,
                )
        self.engram_layer_ids = tuple(config.engram_layer_ids)
        self._weights_loaded = False

    def _configure_engram_rotation(self):
        if not self.engram_layer_ids:
            return
        contract = getattr(self.vllm_config.model_config.hf_config, "engram_rotation_config", None)
        if contract is None:
            raise ValueError("Indexed INT8 Engram requires an explicit rotation contract")
        if contract == {
            "value_projection_rotated": False,
            "value_basis": "original",
            "key_and_gate_basis": "original",
            "runtime_delta_rotation": False,
        }:
            for layer_id in self.engram_layer_ids:
                self.layers[layer_id].engram.set_rotation_contract("unrotated")
            return
        if contract != {
            "value_projection_rotated": True,
            "value_basis": "quarot_global",
            "key_and_gate_basis": "original",
            "runtime_delta_rotation": False,
        }:
            raise ValueError(f"Unsupported Engram rotation contract: {contract}")
        description = json.loads((self.model_path / "quant_model_description.json").read_text())
        relative = description["optional"]["quarot"]["rotation_map"]["global_rotation"]
        path = (self.model_path / relative).resolve()
        if not path.is_relative_to(self.model_path.resolve()):
            raise ValueError("Engram rotation file escapes checkpoint directory")
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            matrix = reader.get_tensor("global_rotation")
        if (
            matrix.dtype != torch.float32
            or matrix.shape != (self.config.hidden_size, self.config.hidden_size)
            or not torch.isfinite(matrix).all()
        ):
            raise ValueError("Invalid checkpoint global rotation matrix")
        transform = EngramQueryUnrotation(matrix.to(self.embed.weight.device))
        for layer_id in self.engram_layer_ids:
            engram = self.layers[layer_id].engram
            engram.query_transform = transform
            engram.set_rotation_contract("external")
        logger.info("V4.1 Engram uses shared FP32 query inverse rotation; value is pre-rotated")

    def embed_input_ids(self, input_ids):
        return self.embed(input_ids)

    def _hash_tokens(self, input_ids, runtime, metadata):
        if self.engram_hasher is None:
            return None
        if not runtime.requests:
            layout = self.engram_hasher.layout
            return torch.empty(
                (0, len(layout.layer_ids), layout.hash_columns),
                dtype=torch.int64,
                device="cpu",
            )
        cache = self.layers[0].attn.cache
        history_column = cache.dim + cache.index_dim
        hashes = []
        cpu_ids = input_ids.cpu()
        for request_idx, request in enumerate(runtime.requests):
            history = []
            if request.start_pos:
                positions = torch.arange(
                    max(0, request.start_pos - self.config.engram_max_ngram_size + 1),
                    request.start_pos,
                    device=input_ids.device,
                )
                history = cache.read(metadata, request_idx, positions, 1, history_column, 1)
                history = history.flatten().long().cpu()
            state = self.engram_hasher.new_state(history, position=request.start_pos)
            hashes.append(
                self.engram_hasher.hash_tokens(
                    cpu_ids[request.begin : request.end],
                    state,
                    start_position=request.start_pos,
                )
            )
        if metadata is not None:
            cache.write(metadata, input_ids.unsqueeze(-1), 1, history_column)
        return torch.cat(hashes, dim=0)

    def prepare_decode_graph_inputs(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None) -> int:
        """Stage each live request's Engram rows before replay and return its batch's live count."""
        if input_ids is None or input_ids.ndim != 1 or positions.shape != input_ids.shape:
            raise ValueError("V4.1 decode graph expects one position per padded token")
        bucket = input_ids.shape[0]
        if bucket not in self._decode_capture_sizes:
            raise ValueError(f"V4.1 decode graph has no configured {bucket}-row bucket")
        cache = self.layers[0].attn.cache
        decode = cache.metadata().decode
        if decode is None or decode.active.shape != input_ids.shape or decode.slot_mapping.shape != input_ids.shape:
            raise RuntimeError("Missing V4.1 full decode metadata")
        valid = decode.active & (decode.slot_mapping >= 0)
        if self.engram_hasher is None:
            return int(valid.sum().cpu())
        # Copy the small header once. Logical request rows may have holes and
        # change order; only the current block table identifies each history.
        header = (
            torch.stack((valid.long(), positions.long(), input_ids.long(), decode.slot_mapping.long()), dim=1)
            .cpu()
            .tolist()
        )
        live = [row for row, (active, _, _, _) in enumerate(header) if active]
        layout = self.engram_hasher.layout
        history_column = cache.dim + cache.index_dim
        if live:
            logger.info_once("V4.1 staging an active token for full decode graph replay")
            physical = [header[row][3] for row in live]
            if len(set(physical)) != len(live) or max(physical) >= cache.pages()[1].shape[0]:
                raise ValueError("Invalid or duplicate live Engram cache slot")
            if any(not 0 <= header[row][1] < decode.max_model_len for row in live):
                raise ValueError("Live Engram position outside the decode context")
            lookback = min(layout.max_ngram_size - 1, max(header[row][1] for row in live))
            histories = [[] for _ in range(bucket)]
            if lookback:
                if bucket == 1:
                    # Keep the single-row path's exact, bounded read.
                    pos = header[0][1]
                    past = torch.arange(pos - lookback, pos, device=input_ids.device)
                    histories[0] = cache.read(decode, 0, past, 1, history_column, 1).flatten().long().cpu().tolist()
                else:
                    shifts = torch.arange(lookback, 0, -1, device=input_ids.device)
                    past = positions.long()[:, None] - shifts
                    histories = cache.decode_read(decode, past, 1, history_column, 1).squeeze(-1).long().cpu().tolist()
            rows = []
            for row in live:
                _, pos, token, _ = header[row]
                count = min(pos, lookback)
                state = self.engram_hasher.new_state(histories[row][-count:] if count else [], position=pos)
                rows.append(self.engram_hasher.hash_tokens([token], state, start_position=pos))
            hashes = torch.cat(rows, dim=0)
        else:
            hashes = torch.empty(
                (0, len(layout.layer_ids), layout.hash_columns),
                dtype=torch.int64,
                device="cpu",
            )
        for index, layer_id in enumerate(self.engram_layer_ids):
            engram = self.layers[layer_id].engram
            key = (bucket, layer_id)
            self._decode_engram_values[key] = engram.prepare_decode(
                hashes[:, index],
                bucket_size=bucket,
                device=input_ids.device,
                slots=live,
                out=self._decode_engram_values.get(key),
            )
        if live:
            # A padded row may contain -1; it must never index a real cache page.
            if len(live) == bucket:
                cache.write(decode, input_ids.unsqueeze(-1), 1, history_column)
            else:
                rows = torch.tensor(live, dtype=torch.long, device=input_ids.device)
                slots = decode.slot_mapping.index_select(0, rows).long()
                cache.pages()[1][slots, history_column] = input_ids.index_select(0, rows).float()
        return len(live)

    def _forward_decode(self, input_ids, metadata):
        runtime = DeepseekV41Runtime.for_decode(metadata)
        hidden_states = self.embed(input_ids).unsqueeze(-2).repeat(1, self.config.hc_mult, 1)
        pre_mix = make_identity_pre_mix(hidden_states)
        for layer_id, layer in enumerate(self.layers):
            if layer.engram is not None:
                hidden_states = layer.engram.forward_decode(
                    hidden_states,
                    self._decode_engram_values[(input_ids.shape[0], layer_id)],
                    runtime.decode.active,
                )
            hidden_states, pre_mix = layer(hidden_states, pre_mix, runtime)
        output = self.norm(hc_pre(hidden_states, pre_mix))
        return torch.where(runtime.decode.active[:, None], output, 0)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or input_ids is None:
            raise ValueError("V4.1 text path requires token IDs and no pipeline tensors")
        metadata = self.layers[0].attn.cache.metadata()
        if (
            getattr(self, "decode_enabled", False)
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            return self._forward_decode(input_ids, metadata)
        actual_tokens = metadata.num_actual_tokens if metadata else input_ids.numel()
        runtime = DeepseekV41Runtime(metadata.requests if metadata else [RequestSpan(0, actual_tokens, actual_tokens)])
        ids = input_ids[:actual_tokens]
        hashes = self._hash_tokens(ids, runtime, metadata)
        if actual_tokens == 0:
            hidden_states = self.embed.weight.new_empty((0, self.config.hidden_size))
        else:
            hidden_states = self.embed(ids) if inputs_embeds is None else inputs_embeds[:actual_tokens]
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.config.hc_mult, 1)
        pre_mix = make_identity_pre_mix(hidden_states)
        for layer_id, layer in enumerate(self.layers):
            if layer.engram is not None and actual_tokens:
                index = self.engram_layer_ids.index(layer_id)
                hidden_states = layer.engram(hidden_states, hashes[:, index])
            hidden_states, pre_mix = layer(hidden_states, pre_mix, runtime)
        output = (
            self.norm(hc_pre(hidden_states, pre_mix))
            if actual_tokens
            else hidden_states.new_empty((0, self.config.hidden_size))
        )
        if actual_tokens < input_ids.numel():
            output = torch.cat(
                (
                    output,
                    output.new_zeros(input_ids.numel() - actual_tokens, self.config.hidden_size),
                ),
                0,
            )
        return output

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.head, hidden_states.float())

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The generic iterator materializes gigantic host tables and includes
        # stale duplicate tensors. Use the checkpoint's authoritative index.
        store = self.engram_store or IndexedMmapStore(self.model_path)
        parameters = dict(self.named_parameters())
        modules = dict(self.named_modules())
        by_shard = defaultdict(list)
        for name, parameter in parameters.items():
            if name not in store.weight_map:
                raise KeyError(f"Required V4.1 parameter absent from index: {name}")
            by_shard[store.weight_map[name]].append((name, parameter))
        loaded = set()
        for shard, items in sorted(by_shard.items()):
            path = (self.model_path / shard).resolve()
            if not path.is_relative_to(self.model_path.resolve()):
                raise ValueError("Checkpoint shard escapes model directory")
            with safe_open(str(path), framework="pt", device="cpu") as reader:
                for name, parameter in items:
                    module = modules[name.rsplit(".", 1)[0]]
                    read_parameter_slice(
                        reader,
                        name,
                        parameter,
                        module,
                        get_tensor_model_parallel_rank(),
                        get_tensor_model_parallel_world_size(),
                    )
                    loaded.add(name)
            logger.info("V4.1 loaded indexed shard %s (%d local tensors)", shard, len(items))
        if loaded != set(parameters):
            raise RuntimeError(f"Unloaded V4.1 parameters: {set(parameters) - loaded}")
        self._configure_engram_rotation()
        self._weights_loaded = True
        logger.info(
            "V4.1 loaded all %d local parameters; Engram tables use host mmap",
            len(loaded),
        )
        return loaded
