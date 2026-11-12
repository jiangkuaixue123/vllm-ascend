# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch fallbacks and cross-stream lifetime/order without device imports."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def setup():
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            calls.append((self.name, "record"))
            return self.name

        def wait_event(self, event):
            calls.append((self.name, "wait", event))

    main, aux = Stream("main"), Stream("aux")

    def tensor(shape, dtype="fp32", requires_grad=False):
        value = SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=SimpleNamespace(type="npu"),
            requires_grad=requires_grad,
            layout="strided",
            contiguous=True,
            negative=False,
            conjugated=False,
            format=2,
        )
        value.is_contiguous = lambda: value.contiguous
        value.is_neg = lambda: value.negative
        value.is_conj = lambda: value.conjugated
        value.record_stream = lambda stream: calls.append((id(value), "lifetime", stream.name))
        return value

    x, previous = tensor((6, 4, 5120), "bf16"), tensor((6, 4))
    params = [
        tensor((24, 20480), requires_grad=True),
        tensor((3,), requires_grad=True),
        tensor((24,), requires_grad=True),
    ]
    layer = SimpleNamespace(
        attn=SimpleNamespace(tp=4, _query_overlap_stream=aux),
        config=SimpleNamespace(rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20),
        hc_attn_fn=params[0],
        hc_attn_scale=params[1],
        hc_attn_base=params[2],
    )
    outputs = tuple(tensor((6, 4)) for _ in range(4))
    mixed = object()

    def native(*args):
        calls.append("native")
        return outputs

    def input_only(*args):
        calls.append("input")
        return mixed

    env = {
        "torch": SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            strided="strided",
            is_inference_mode_enabled=lambda: True,
            compiler=SimpleNamespace(is_compiling=lambda: False),
            npu=SimpleNamespace(current_stream=lambda *args: main),
        ),
        "os": SimpleNamespace(getpid=lambda: 1),
        "HAS_TRITON": True,
        "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: SimpleNamespace(is_draft_model=False),
        "get_npu_format": lambda value: value.format,
        "npu_stream_switch": lambda stream: nullcontext(),
        "hc_pre_delayed": native,
        "hc_input4": input_only,
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
    }

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    path = ROOT / "vllm_ascend/models/common/ops/hc_attention_overlap.py"
    tree = RemoveImports().visit(ast.parse(path.read_text()))
    exec(compile(tree, str(path), "exec"), env)
    return env, layer, x, previous, SimpleNamespace(decode=object()), calls, outputs, mixed


def test_weights_can_require_grad_in_inference_and_lifetimes_cover_both_streams():
    env, layer, x, previous, runtime, calls, outputs, mixed = setup()
    result = env["try_start_hc_attention"](layer, x, previous, runtime)
    assert result == (mixed, outputs, "aux")
    assert calls[0] == ("main", "record")
    assert calls[6:9] == [("aux", "wait", "main"), "native", ("aux", "record")]
    assert all(
        (id(value), "lifetime", "aux") in calls
        for value in (x, previous, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
    )
    assert all((id(value), "lifetime", "main") in calls for value in outputs)
    assert calls[-1] == "input"


@pytest.mark.parametrize(
    "case",
    [
        "tp8",
        "prefill",
        "no_stream",
        "same_stream",
        "draft",
        "no_context",
        "training",
        "compiling",
        "no_triton",
        "batch_invariant",
        "format",
        "strided",
        "autograd",
    ],
)
def test_unsupported_contract_returns_before_any_device_work(case):
    env, layer, x, previous, runtime, calls, _, _ = setup()
    if case == "tp8":
        layer.attn.tp = 8
    elif case == "prefill":
        runtime.decode = None
    elif case == "no_stream":
        layer.attn._query_overlap_stream = None
    elif case == "same_stream":
        layer.attn._query_overlap_stream = env["torch"].npu.current_stream()
    elif case == "draft":
        env["get_forward_context"] = lambda: SimpleNamespace(is_draft_model=True)
    elif case == "no_context":
        env["is_forward_context_available"] = lambda: False
    elif case == "training":
        env["torch"].is_inference_mode_enabled = lambda: False
    elif case == "compiling":
        env["torch"].compiler.is_compiling = lambda: True
    elif case == "no_triton":
        env["HAS_TRITON"] = False
    elif case == "batch_invariant":
        env["envs"].VLLM_BATCH_INVARIANT = True
    elif case == "format":
        layer.hc_attn_fn.format = 29
    elif case == "strided":
        previous.contiguous = False
    elif case == "autograd":
        x.requires_grad = True
    assert env["try_start_hc_attention"](layer, x, previous, runtime) is None
    assert calls == []


def test_native_unavailable_does_not_launch_input_kernel():
    env, layer, x, previous, runtime, calls, _, _ = setup()
    env["hc_pre_delayed"] = lambda *args: None
    assert env["try_start_hc_attention"](layer, x, previous, runtime) is None
    assert "input" not in calls
