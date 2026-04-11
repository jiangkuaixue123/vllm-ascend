# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Dict, Tuple
from unittest.mock import patch
from collections import defaultdict

import numpy as np
import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.attention.utils import using_paged_attention

from ..utils import weak_ref_tensors

ACLGRAPH_CAPTURE_DEBUG_ENV = "VLLM_ASCEND_DEBUG_SFA_UBATCH_GRAPH"


def _aclgraph_capture_debug_enabled() -> bool:
    return bool(int(os.getenv(ACLGRAPH_CAPTURE_DEBUG_ENV, "0")))


def _tensor_capture_str(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}=None"
    return (f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} ptr={tensor.data_ptr()}")


def _attn_capture_str(attn_metadata: Any) -> str:
    if attn_metadata is None:
        return "attn=None"

    parts = [
        f"id={id(attn_metadata)}",
        f"num_input_tokens={getattr(attn_metadata, 'num_input_tokens', None)}",
        f"num_actual_tokens={getattr(attn_metadata, 'num_actual_tokens', None)}",
    ]
    for name in ("slot_mapping", "block_tables", "cum_query_lens", "seq_lens",
                 "cos", "sin"):
        parts.append(
            _tensor_capture_str(name, getattr(attn_metadata, name, None)))
    return "attn={" + ", ".join(parts) + "}"


def _forward_context_capture_str(forward_context: Any) -> str:
    if forward_context is None:
        return "forward_context=None"
    return (
        "forward_context={"
        f"id={id(forward_context)}, "
        f"runtime={getattr(forward_context, 'cudagraph_runtime_mode', None)}, "
        f"capturing={getattr(forward_context, 'capturing', None)}, "
        f"ubatch_idx={getattr(forward_context, 'ubatch_idx', None)}, "
        f"num_ubatches={getattr(forward_context, 'num_ubatches', None)}, "
        f"num_tokens={getattr(forward_context, 'num_tokens', None)}, "
        f"batch_descriptor={getattr(forward_context, 'batch_descriptor', None)}"
        "}"
    )


def _capture_debug_log(message: str, *args) -> None:
    if _aclgraph_capture_debug_enabled():
        logger.info("[ACLGRAPH-CAPTURE-DEBUG] " + message, *args)


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Optional[Any] = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 cudagraph_options: Optional[CUDAGraphOptions] = None):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry]\
                                                                        = {}

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"aclgraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        _capture_debug_log(
            "wrapper entry mode=%s runtime=%s key=%s %s %s %s %s %s %s",
            self.runtime_mode,
            aclgraph_runtime_mode,
            batch_descriptor,
            _forward_context_capture_str(forward_context),
            _tensor_capture_str("input_ids", kwargs.get("input_ids")),
            _tensor_capture_str("positions", kwargs.get("positions")),
            _tensor_capture_str("inputs_embeds", kwargs.get("inputs_embeds")),
            _tensor_capture_str("intermediate_tensors",
                                kwargs.get("intermediate_tensors")),
            _attn_capture_str(getattr(forward_context, "attn_metadata", None)),
        )

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or \
                            aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = \
                ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            _capture_debug_log(
                "capture path mode=%s key=%s existing_entries=%s",
                self.runtime_mode,
                batch_descriptor,
                len(self.concrete_aclgraph_entries),
            )
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)",
                             self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            _capture_debug_log(
                "capture start mode=%s key=%s %s %s %s %s %s %s",
                self.runtime_mode,
                entry.batch_descriptor,
                _forward_context_capture_str(forward_context),
                _tensor_capture_str("input_ids", kwargs.get("input_ids")),
                _tensor_capture_str("positions", kwargs.get("positions")),
                _tensor_capture_str("inputs_embeds", kwargs.get("inputs_embeds")),
                _tensor_capture_str("intermediate_tensors",
                                    kwargs.get("intermediate_tensors")),
                _attn_capture_str(getattr(forward_context, "attn_metadata",
                                          None)),
            )

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            aclgraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                forward_context.capturing = True
                with torch.npu.graph(aclgraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's aclgraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.aclgraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise aclgraph mode, because
                        # the output of the last graph will not be used by
                        # any other acl graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the workspaces
            # to save memory
            global _graph_params
            global _draft_graph_params
            weak_ref_workspaces(_graph_params)
            weak_ref_workspaces(_draft_graph_params)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph

            _capture_debug_log(
                "capture end mode=%s key=%s output_type=%s input_addresses=%s",
                self.runtime_mode,
                entry.batch_descriptor,
                type(output).__name__,
                input_addresses,
            )

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}")

        logger.info_once("Replaying aclgraph")
        _capture_debug_log(
            "replay path mode=%s key=%s output_type=%s input_addresses=%s",
            self.runtime_mode,
            batch_descriptor,
            type(entry.output).__name__ if entry.output is not None else None,
            entry.input_addresses,
        )
        # In async scheduling or multi-threaded (MT) scenarios, it is possible that
        # the CPU's record event (from update_attn_params) for the iteration i completes
        # before the grph replay of iteration i-1.
        # To ensure proper ordering, we must call synchronize here before replaying,
        # so that update_attn_params only executes after the previous graph replay has fully completed.
        torch.npu.synchronize()
        entry.aclgraph.replay()
        return entry.output


def weak_ref_workspaces(params):
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(
            params.workspaces[num_tokens])


def _update_attn_pa_params(update_stream, forward_context, runtime_shape):
    graph_params = get_graph_params()
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_heads,
                scale,
                block_table,
                seq_lens,
                output,
            ) = param
            seq_lens = forward_context.attn_metadata[key].seq_lens

            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                block_table=block_table,
                context_lens=seq_lens,
                out=output,
                workspace=graph_params.workspaces.get(runtime_shape),
            )
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def _update_attn_fia_params(update_stream, forward_context, runtime_shape):
    if forward_context.is_draft_model:
        graph_params = get_draft_graph_params()
    else:
        graph_params = get_graph_params()
    # For Qwen3-next, since the kv_cache_config has already categorized
    # linear_attn and self_attn, the attn_metadata is first arranged with
    # self_attn followed by linear_attn. Therefore, using zip directly
    # filters out the update operations for linear_attn.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (query, key_cache, value, block_tables, attn_mask, block_size,
             seq_lens, query_start_loc, num_kv_heads, num_heads, scale,
             attn_output, softmax_lse) = param

            seq_lens = forward_context.attn_metadata[key].seq_lens_list
            actual_seq_lengths_q = forward_context.attn_metadata[
                key].actual_seq_lengths_q
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key_cache,
                value=value,
                block_table=block_tables,
                atten_mask=attn_mask,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=seq_lens,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale,
                sparse_mode=3,
                workspace=graph_params.workspaces.get(runtime_shape),
                out=[attn_output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def update_attn_params(update_stream, forward_context, runtime_shape,
                       vllm_config):
    if using_paged_attention(runtime_shape, vllm_config):
        _update_attn_pa_params(update_stream, forward_context, runtime_shape)
    else:
        _update_attn_fia_params(update_stream, forward_context, runtime_shape)


def update_mla_attn_params(update_stream, forward_context, runtime_shape,
                           speculative_config, is_ubatch, num_ubatches):
    if forward_context.is_draft_model:
        graph_params = get_draft_graph_params()
    elif forward_context.afd_metadata and is_ubatch:
        graph_params_dict = get_graph_params_dict()
        merged_graph_params = interleave_multiple_graph_params(*list(graph_params_dict.values()))
        graph_params = merged_graph_params
    else:
        graph_params = get_graph_params()
        logger.debug("AFD metadata is not present ,AFD metadata is present and DBO is unenabled")
        
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                    list_of_dicts_to_dict_of_lists(forward_context.attn_metadata) if is_ubatch else forward_context.attn_metadata,
                    graph_params.attn_params[runtime_shape],
                    graph_params.handles[runtime_shape],
                    graph_params.events[runtime_shape],
            ):
            (q_nope, k_nope, q_pe, k_pe, num_heads, num_kv_heads, input_layout,
            attn_mask, sparse_mode, scale, block_table, block_size,
            seq_lens_list, actual_seq_lengths, attn_output,
            softmax_lse) = param
                
            seq_lens_list = key.decode.seq_lens_list if is_ubatch else forward_context.attn_metadata[
                key].decode.seq_lens_list
            if speculative_config and speculative_config.method == "mtp" \
                    and not forward_context.is_draft_model:
                actual_seq_lengths = forward_context.attn_metadata[
                    key].decode.actual_seq_lengths_q
                spec_multiple = speculative_config.num_speculative_tokens + 1
                seq_lens_list = seq_lens_list + [0] * (
                    runtime_shape // spec_multiple - len(seq_lens_list))
                actual_seq_lengths = [
                    spec_multiple * (i + 1)
                    for i in range(runtime_shape // spec_multiple)
                ]
            elif forward_context.is_draft_model:
                actual_seq_lengths = forward_context.attn_metadata[
                    key].decode.actual_seq_lengths_q
                block_table = forward_context.attn_metadata[
                    key].decode.block_table
                # TODO: This is a hack and should be fixed in the future.
                if speculative_config.disable_padded_drafter_batch:
                    block_table = block_table[:len(actual_seq_lengths)]
                seq_lens_list = seq_lens_list + [0] * (
                    len(actual_seq_lengths) - len(seq_lens_list))
            else:
                    seq_lens_list = seq_lens_list + [0] * (runtime_shape -
                                                        len(seq_lens_list))
            torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout=input_layout,
                atten_mask=attn_mask,
                sparse_mode=sparse_mode,
                scale=scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=seq_lens_list,
                actual_seq_lengths=actual_seq_lengths,
                workspace=graph_params.workspaces.get(runtime_shape),
                out=[attn_output, softmax_lse])
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)
    logger.debug(f"update_mla_attn_params successfully executed") 


def update_attn_dcp_pcp_params(update_stream, forward_context, runtime_shape):
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    graph_params = get_graph_params()
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (q_nope, k_nope, value, num_heads, num_kv_heads, scale,
             block_table, block_size, actual_seq_lengths_kv,
             actual_seq_lengths_q, attn_output, softmax_lse, dcp_size,
             pcp_rank, dcp_rank) = param
            attn_metadata = forward_context.attn_metadata[key]
            actual_seq_lengths_kv = attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp[:,
                                                                                             pcp_rank,
                                                                                             dcp_rank]
            pad_length = runtime_shape - len(actual_seq_lengths_kv)
            if pad_length > 0:
                pad_tensor = np.zeros(pad_length,
                                      dtype=actual_seq_lengths_kv.dtype)
                actual_seq_lengths_kv = np.concatenate(
                    [actual_seq_lengths_kv, pad_tensor])

            actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
            if dcp_size > 1:
                num_heads = num_heads * dcp_size

            torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                value,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout="TND",
                atten_mask=None,
                scale=scale,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths=actual_seq_lengths_q,
                workspace=graph_params.workspaces.get(runtime_shape),
                out=[attn_output, softmax_lse])
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def update_mla_attn_dcp_pcp_params(update_stream, forward_context,
                                   runtime_shape):
    if forward_context.is_draft_model:
        graph_params = get_draft_graph_params()
    else:
        graph_params = get_graph_params()
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.        
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (
                q_nope,
                k_nope,
                q_pe,
                k_pe,
                num_heads,
                num_kv_heads,
                input_layout,
                spec_attn_mask,
                sparse_mode,
                scale,
                block_table,
                block_size,
                actual_seq_lengths,
                actual_seq_lengths_kv,
                attn_output,
                softmax_lse,
            ) = param

            decode_meta = forward_context.attn_metadata[key].decode
            seq_len = decode_meta.cp_seq_len
            if isinstance(seq_len, torch.Tensor):
                seq_len = seq_len.tolist()
            actual_seq_lengths_kv = seq_len

            pad_length = runtime_shape - len(actual_seq_lengths_kv)
            if pad_length > 0:
                actual_seq_lengths_kv = actual_seq_lengths_kv + [0] * (
                    runtime_shape - len(actual_seq_lengths_kv))

                torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout=input_layout,
                atten_mask=spec_attn_mask,
                sparse_mode=sparse_mode,
                scale=scale,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths=actual_seq_lengths,
                workspace=graph_params.workspaces.get(runtime_shape),
                out=[attn_output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]


_graph_params: Optional[GraphParams] = None
_graph_params_dict: defaultdict[int, Optional[GraphParams]] = None


def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: []
         for size in aclgraph_capture_sizes},
        {size: None
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
    )

def set_graph_params_dict(aclgraph_capture_sizes: set[int],nums_ubatch):
    global _graph_params_dict
    if _graph_params_dict is not None:
        raise ValueError("Graph dict parameters have already been set!")
    _graph_params_dict = defaultdict(lambda: None)
    for nb in range(nums_ubatch):
        _graph_params_dict[nb] = GraphParams(
            {size: []
            for size in aclgraph_capture_sizes},
            {size: None
            for size in aclgraph_capture_sizes},
            {size: []
            for size in aclgraph_capture_sizes},
            {size: []
            for size in aclgraph_capture_sizes},
        )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = workspace


def get_graph_params():
    return _graph_params


_draft_graph_params: Optional[GraphParams] = None


def set_draft_graph_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_params
    if _draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    _draft_graph_params = GraphParams(
        {size: []
         for size in aclgraph_capture_sizes},
        {size: None
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
    )


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_params
    if _draft_graph_params is not None:
        _draft_graph_params.workspaces[num_tokens] = workspace


def get_draft_graph_params():
    return _draft_graph_params

def get_graph_params_dict():
    return _graph_params_dict




def list_of_dicts_to_dict_of_lists(list_of_dicts: List[Dict]):
    """Convert [{"k1": a, "k2": b}, {"k1": c, "k2": d}] -> {"k1": [a, c], "k2": [b, d]}"""
    if not list_of_dicts:
        return {}
    keys = list_of_dicts[0].keys()
    
    result = {k: [] for k in keys}
    for d in list_of_dicts:
        for k in keys:
            result[k].append(d[k])
    ans = []        
    for key in keys:
        for v in result[key]:
            ans.append(v)  
    return ans


from typing import Dict, List, TypeVar, Generic, Any

T = TypeVar('T')

def interleave_multiple_lists(*lists: List[T]) -> List[T]:
    """
    Interleave multiple lists element by element.
    Example: [a, b], [x, y], [1, 2] -> [a, x, 1, b, y, 2]
    """
    if not lists:
        return []
    
    max_len = max(len(lst) for lst in lists) if lists else 0
    result = []
    
    for i in range(max_len):
        for lst in lists:
            if i < len(lst):
                result.append(lst[i])
    
    return result


def interleave_multiple_dict_of_lists(*dicts: Dict[int, List]) -> Dict[int, List]:
    """
    For each key present in any dictionary, interleave the corresponding lists.
    If a key exists only in some dicts, those lists are interleaved while missing ones are treated as empty.
    """
    if not dicts:
        return {}
    
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    
    new_dict = {}
    for key in all_keys:
        # Get the corresponding list from each dict, defaulting to empty list if key not found
        lists_to_interleave = [d.get(key, []) for d in dicts]
        new_dict[key] = interleave_multiple_lists(*lists_to_interleave)
    
    return new_dict

def interleave_multiple_graph_params(*params_list) -> GraphParams:
    """
    Interleave multiple GraphParams objects.
    For Dict[int, List] fields (events, handles, attn_params): interleave corresponding lists by key
    For Dict[int, Tensor] field (workspaces): keep values from the first param, add missing entries from others
    """
    if not params_list:
        return GraphParams()
    
    # Interleave events
    new_events = interleave_multiple_dict_of_lists(*(p.events for p in params_list))
    
    # Interleave handles
    new_handles = interleave_multiple_dict_of_lists(*(p.handles for p in params_list))
    
    # Interleave attn_params
    new_attn_params = interleave_multiple_dict_of_lists(*(p.attn_params for p in params_list))
    
    # Merge workspaces: start with first param's workspaces, then add missing entries from others
    new_workspaces = params_list[0].workspaces.copy()
    # for param in params_list[1:]:
    #     for key, value in param.workspaces.items():
    #         if key not in new_workspaces:
    #             new_workspaces[key] = value
    
    return GraphParams(
        events=new_events,
        workspaces=new_workspaces,
        handles=new_handles,
        attn_params=new_attn_params
    )
