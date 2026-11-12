# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata fallbacks and the actual model dispatch, without requiring NPU imports."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def tensor(shape, dtype, **changes):
    metadata = dict(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=SimpleNamespace(type="npu"),
        contiguous=True,
        negative=False,
        conjugated=False,
        npu_format=2,
    )
    metadata.update(changes)
    value = SimpleNamespace(**metadata)
    value.is_contiguous = lambda: value.contiguous
    value.is_neg = lambda: value.negative
    value.is_conj = lambda: value.conjugated
    return value


def load_dispatcher():
    path = ROOT / "vllm_ascend/models/common/ops/hc_post.py"

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    tree = RemoveImports().visit(ast.parse(path.read_text()))
    calls = []
    output = object()
    env = {
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32", bool="bool", is_inference_mode_enabled=lambda: True),
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": SimpleNamespace(A3="A3"),
        "get_npu_format": lambda value: value.npu_format,
        "hc_post4": lambda *args: calls.append(args) or output,
    }
    exec(compile(tree, str(path), "exec"), env)
    return env, calls, output


def inputs(rows=6):
    return [
        tensor((rows, 5120), "bf16"),
        tensor((rows, 4, 5120), "bf16"),
        tensor((rows, 4), "fp32"),
        tensor((rows, 4, 4), "fp32"),
    ]


@pytest.mark.parametrize("rows", [3, 6, 12])
def test_supported_inputs_preserve_operand_order(rows):
    env, calls, output = load_dispatcher()
    args = inputs(rows)
    assert env["hc_post_fused"](*args) is output
    assert len(calls) == 1 and all(actual is expected for actual, expected in zip(calls[0], args))


@pytest.mark.parametrize(
    "index,changes",
    [
        (0, {"device": SimpleNamespace(type="cpu")}),
        (0, {"dtype": "fp32"}),
        (0, {"shape": (6, 4096)}),
        (0, {"ndim": 3}),
        (1, {"shape": (6, 3, 5120)}),
        (1, {"dtype": "fp32"}),
        (2, {"shape": (6, 3)}),
        (2, {"dtype": "bf16"}),
        (3, {"shape": (6, 16)}),
        (3, {"dtype": "bf16"}),
        (3, {"contiguous": False}),
        (1, {"npu_format": 29}),
        (2, {"negative": True}),
        (3, {"conjugated": True}),
    ],
)
def test_unsupported_metadata_falls_back(index, changes):
    env, calls, _ = load_dispatcher()
    args = inputs()
    for name, value in changes.items():
        setattr(args[index], name, value)
    assert env["hc_post_fused"](*args) is None
    assert not calls


@pytest.mark.parametrize("rows", [0, 1, 5, 24])
def test_other_shapes_fall_back(rows):
    env, calls, _ = load_dispatcher()
    assert env["hc_post_fused"](*inputs(rows)) is None
    assert not calls


def test_no_triton_and_autograd_fall_back():
    env, calls, _ = load_dispatcher()
    env["hc_post4"] = None
    assert env["hc_post_fused"](*inputs()) is None
    env["torch"].is_inference_mode_enabled = lambda: False
    assert env["hc_post_fused"](*inputs()) is None
    assert not calls


def test_kernel_failures_are_not_hidden():
    env, _, _ = load_dispatcher()

    def fail(*args):
        raise RuntimeError("kernel failure")

    env["hc_post4"] = fail
    with pytest.raises(RuntimeError, match="kernel failure"):
        env["hc_post_fused"](*inputs())


@pytest.mark.parametrize("rows", [3, 6, 12])
def test_active_mask_reaches_kernel_with_original_operands(rows):
    env, calls, output = load_dispatcher()
    args = inputs(rows)
    mask = tensor((rows,), "bool")
    assert env["hc_post_fused"](*args, mask) is output
    assert calls == [tuple(args) + (mask,)]


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": (6, 1)},
        {"dtype": "bf16"},
        {"device": SimpleNamespace(type="cpu")},
        {"contiguous": False},
        {"npu_format": 29},
        {"negative": True},
        {"conjugated": True},
    ],
)
def test_unsupported_active_mask_falls_back(changes):
    env, calls, _ = load_dispatcher()
    mask = tensor((6,), "bool")
    for name, value in changes.items():
        setattr(mask, name, value)
    assert env["hc_post_fused"](*inputs(), mask) is None
    assert not calls


@pytest.mark.parametrize("supported", [False, True])
def test_model_restores_original_mask_before_fallback_mixing(supported):
    path = ROOT / "vllm_ascend/models/deepseek_v41/model.py"
    cls = next(node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "DeepseekV41Block")
    method = next(node for node in cls.body if getattr(node, "name", None) == "post_residual")
    fused, reference, masked = object(), object(), object()
    calls = []
    args = inputs()

    class Mask:
        def __getitem__(self, key):
            return self

    mask = Mask()

    def fused_call(*values):
        assert values == tuple(args) + (mask,)
        calls.append("fused")
        return fused if supported else None

    def where(condition, value, zero):
        assert condition is mask and value is args[0] and zero == 0
        calls.append("mask")
        return masked

    def reference_call(*values):
        assert values == (masked, *args[1:])
        calls.append("reference")
        return reference

    env = {
        "hc_post_fused": fused_call,
        "hc_post": reference_call,
        "torch": SimpleNamespace(where=where),
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": SimpleNamespace(getpid=lambda: 1),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    actual = env["post_residual"](object(), *args, SimpleNamespace(decode=object()), mask)
    assert actual is (fused if supported else reference)
    assert calls == (["fused"] if supported else ["fused", "mask", "reference"])


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("local_mask", [False, True])
def test_block_only_defers_tp4_decode_masks_to_both_post_consumers(tp, decode, local_mask):
    path = ROOT / "vllm_ascend/models/deepseek_v41/model.py"
    cls = next(node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "DeepseekV41Block")
    method = next(node for node in cls.body if getattr(node, "name", None) == "forward")
    hidden, pre, mask = inputs()[1], object(), object()
    calls, post_masks = [], []

    class Attention:
        def __call__(self, states, runtime, *, defer_output_mask=False):
            calls.append(("attention", defer_output_mask))
            return states

    class FFN:
        def __call__(self, states):
            calls.append(("moe", False))
            return states

        def forward_decode(self, states, token_mask, *, defer_output_mask=False):
            assert token_mask is mask
            calls.append(("moe", defer_output_mask))
            return states

    def post_residual(sub, residual, post, comb, runtime, token_mask=None):
        post_masks.append(token_mask)
        return residual

    attention = Attention()
    attention.tp = tp
    block = SimpleNamespace(
        attn=attention,
        ffn=FFN(),
        pre_and_mixes=lambda *args: (hidden, pre, object(), object()),
        attn_norm=lambda value: value,
        ffn_norm=lambda value: value,
        post_residual=post_residual,
    )
    env = {"_HC_DECODE_MASK_TP": 4}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    runtime = SimpleNamespace(decode=SimpleNamespace(active=mask) if decode else None)
    actual, next_pre = env["forward"](block, hidden, pre, runtime, mask if local_mask else None)
    deferred = tp == 4 and decode and local_mask
    assert calls == [("attention", deferred), ("moe", deferred)]
    assert post_masks == [mask if deferred else None] * 2
    assert actual is hidden and next_pre is pre


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_actual_model_method_only_fuses_decode(decode, supported):
    path = ROOT / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Block")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "post_residual")
    fused, reference = object(), object()
    calls = []
    env = {
        "hc_post_fused": lambda *args: calls.append("fused") or (fused if supported else None),
        "hc_post": lambda *args: calls.append("reference") or reference,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    runtime = SimpleNamespace(decode=object() if decode else None)
    result = env["post_residual"](object(), *inputs(), runtime)
    assert result is (fused if decode and supported else reference)
    expected = (["fused"] if supported else ["fused", "reference"]) if decode else ["reference"]
    assert calls == expected
