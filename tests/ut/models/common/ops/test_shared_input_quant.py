# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise native-call reuse and fallback without importing an NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def setup_dispatch():
    class Linear:
        pass

    class AscendLinear(Linear):
        custom_op = None

    class Adapter:
        pass

    class Scheme:
        act_quant_type = "int8"

    layers = []
    for _ in range(2):
        layer = Linear()
        layer.quant_method = Adapter()
        layer.quant_method.quant_method = Scheme()
        layer.bias = None
        layer.return_bias = False
        layer._forward_hooks = {}
        layer._forward_pre_hooks = {}
        layer.weight, layer.weight_scale = object(), object()
        layers.append(layer)
    states = SimpleNamespace(
        device=SimpleNamespace(type="npu"), dtype="bf16", ndim=2, shape=(6, 5120), is_contiguous=lambda: True
    )
    quantized, scales = object(), object()
    calls = []

    def quantize(value, **kwargs):
        calls.append(("quantize", value, kwargs))
        return quantized, scales

    def matmul(*args, **kwargs):
        calls.append(("matmul", args, kwargs))
        return object()

    env = {
        "torch": SimpleNamespace(bfloat16="bf16", int8="int8", is_inference_mode_enabled=lambda: True),
        "torch_npu": SimpleNamespace(npu_dynamic_quant=quantize, npu_quant_matmul=matmul),
        "ReplicatedLinear": Linear,
        "AscendReplicatedLinear": AscendLinear,
        "AscendLinearMethod": Adapter,
        "AscendW8A8DynamicLinearMethod": Scheme,
        "os": SimpleNamespace(getpid=lambda: 1),
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
    }
    path = ROOT / "vllm_ascend/models/common/ops/shared_input_quant.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), "exec"), env)
    return env, layers, states, calls, quantized, scales


@pytest.mark.parametrize("rows", [6, 12])
def test_one_quantization_feeds_original_two_projection_parameters(rows):
    env, layers, states, calls, quantized, scales = setup_dispatch()
    states.shape = (rows, 5120)
    outputs = env["shared_input_projections"](*layers, states)
    assert len(outputs) == 2
    assert [call[0] for call in calls] == ["quantize", "matmul", "matmul"]
    assert calls[0][1] is states and calls[0][2] == {"dst_type": "int8"}
    for layer, call in zip(layers, calls[1:]):
        assert call[1] == (quantized, layer.weight, layer.weight_scale)
        assert call[2] == {"pertoken_scale": scales, "bias": None, "output_dtype": "bf16"}


def test_serving_ascend_class_and_custom_dispatch_fallback():
    env, layers, states, calls, _, _ = setup_dispatch()
    for layer in layers:
        layer.__class__ = env["AscendReplicatedLinear"]
    assert env["shared_input_projections"](*layers, states) is not None
    assert [call[0] for call in calls] == ["quantize", "matmul", "matmul"]
    calls.clear()
    layers[0].custom_op = object()
    assert env["shared_input_projections"](*layers, states) is None
    assert not calls


@pytest.mark.parametrize(
    "changes",
    [
        {"device": SimpleNamespace(type="cpu")},
        {"dtype": "fp32"},
        {"shape": (3, 5120)},
        {"shape": (20, 5120)},
        {"shape": (6, 1024)},
        {"ndim": 3},
        {"is_contiguous": lambda: False},
    ],
)
def test_unsupported_input_does_not_call_native_ops(changes):
    env, layers, states, calls, _, _ = setup_dispatch()
    for name, value in changes.items():
        setattr(states, name, value)
    assert env["shared_input_projections"](*layers, states) is None
    assert not calls


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "change", ["bias", "return_bias", "hooks", "pre_hooks", "scheme", "act_type", "adapter", "subclass"]
)
def test_layer_contract_mismatch_uses_normal_forward(index, change):
    env, layers, states, calls, _, _ = setup_dispatch()
    layer = layers[index]
    if change == "bias":
        layer.bias = object()
    elif change == "return_bias":
        layer.return_bias = True
    elif change == "hooks":
        layer._forward_hooks[0] = object()
    elif change == "pre_hooks":
        layer._forward_pre_hooks[0] = object()
    elif change == "scheme":
        layer.quant_method.quant_method = object()
    elif change == "act_type":
        layer.quant_method.quant_method.act_quant_type = "fp8"
    elif change == "adapter":
        layer.quant_method = object()
    else:
        layer.__class__ = type("WrappedLinear", (type(layer),), {})
    assert env["shared_input_projections"](*layers, states) is None
    assert not calls


def test_training_falls_back_and_native_errors_propagate():
    env, layers, states, calls, _, _ = setup_dispatch()
    env["torch"].is_inference_mode_enabled = lambda: False
    assert env["shared_input_projections"](*layers, states) is None
    assert not calls
    env["torch"].is_inference_mode_enabled = lambda: True

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    env["torch_npu"].npu_dynamic_quant = fail
    with pytest.raises(RuntimeError, match="native failure"):
        env["shared_input_projections"](*layers, states)
