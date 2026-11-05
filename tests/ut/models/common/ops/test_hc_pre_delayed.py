# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata dispatch and real block-method wiring, runnable without NPU/torch."""

import ast
import math
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize("restore_rvalue_bug", [False, True])
def test_binding_macro_scalar_arguments_are_abi_typed_lvalues(tmp_path, restore_rvalue_bug):
    """Compile the actual call arguments against the adapter's Ts& contract."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "csrc/torch_binding.cpp").read_text()
    function = source.split("HcPreDelayedOutputs npu_hc_pre_delayed_npu(", 1)[1].split("\n}", 1)[0]
    declarations = "\n".join(re.findall(r"constexpr (?:int64_t|double) HC_DELAYED_\w+ = [^;]+;", function))
    arguments = re.search(r"EXEC_NPU_CMD\(aclnnHcPreDelayed, (.*?)\);", function, re.S).group(1)
    if restore_rvalue_bug:
        arguments = arguments.replace("HC_DELAYED_SINKHORN_ITERS", "20").replace("HC_DELAYED_EPS", "1e-6")
    translation_unit = tmp_path / "hc_binding_arguments.cpp"
    translation_unit.write_text(
        "#include <cstdint>\n#include <tuple>\n#include <type_traits>\n"
        "template<class... Ts> auto ConvertTypes(Ts&... args) { return std::make_tuple(args...); }\n"
        "int main() {\n"
        "constexpr int64_t HC_DELAYED_WIDTH = 4;\n"
        "int64_t x=0, fn=0, scale=0, base=0, previous_pre=0; double norm_eps=1e-20;\n"
        "std::tuple<int64_t,int64_t,int64_t,int64_t,int64_t,int64_t> outputs{};\n"
        + declarations
        + "\nauto converted = ConvertTypes("
        + arguments
        + ");\n"
        "static_assert(std::is_same_v<std::tuple_element_t<5, decltype(converted)>, int64_t>);\n"
        "static_assert(std::is_same_v<std::tuple_element_t<6, decltype(converted)>, int64_t>);\n"
        "static_assert(std::is_same_v<std::tuple_element_t<7, decltype(converted)>, double>);\n"
        "static_assert(std::is_same_v<std::tuple_element_t<8, decltype(converted)>, double>);\n"
        "}\n"
    )
    result = subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", str(translation_unit)], capture_output=True, text=True, check=False
    )
    if restore_rvalue_bug:
        assert result.returncode != 0, "literal-rvalue regression must fail Ts& deduction"
    else:
        assert result.returncode == 0, result.stderr


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple
    dtype: str = "fp32"
    device: object = "npu"
    contiguous: bool = True
    label: str = ""
    offset: int = 0
    npu_format: int = 2
    negative: bool = False
    conjugated: bool = False

    def __post_init__(self):
        if isinstance(self.device, str):
            object.__setattr__(self, "device", SimpleNamespace(type=self.device))

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugated


def load_dispatcher():
    path = ROOT / "vllm_ascend/models/common/ops/hc_pre.py"
    tree = ast.parse(path.read_text())

    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    tree = RemoveImports().visit(tree)
    calls = []
    outputs = tuple(object() for _ in range(6))

    def native(*args):
        calls.append(args)
        return outputs

    torch = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        is_inference_mode_enabled=lambda: True,
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_hc_pre_delayed=native)),
    )
    env = {
        "torch": torch,
        "math": math,
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": SimpleNamespace(A3="A3"),
        "get_npu_format": lambda t: t.npu_format,
    }
    exec(compile(tree, str(path), "exec"), env)
    return env, calls, outputs


def inputs(rows=6):
    return [
        TensorMetadata((rows, 4, 5120), "bf16"),
        TensorMetadata((24, 20480)),
        TensorMetadata((3,)),
        TensorMetadata((24,)),
        TensorMetadata((rows, 4)),
        1e-20,
        1e-6,
        20,
    ]


@pytest.mark.parametrize("rows", [6, 12])
def test_native_uses_previous_pre_and_exposes_only_model_outputs(rows):
    env, calls, outputs = load_dispatcher()
    args = inputs(rows)
    assert env["hc_pre_delayed"](*args) == outputs[:4]
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0][:5], args[:5]))
    assert calls[0][5] == args[5]


@pytest.mark.parametrize("rows", [5, 10, 15, 20, 25, 30, 35, 40])
def test_draft_contract_accepts_complete_query_blocks(rows):
    env, calls, _ = load_dispatcher()
    assert env["_supports_hc_pre"](*inputs(rows), env["_DRAFT_ROWS"])
    assert not calls


@pytest.mark.parametrize("rows", [0, 1, 6, 12, 19, 41, 45])
def test_draft_incomplete_or_oversized_blocks_use_reference(rows):
    env, calls, _ = load_dispatcher()
    assert env["hc_pre_draft"](*inputs(rows)) is None
    assert not calls


@pytest.mark.parametrize("change", [{"device": "cpu"}, {"dtype": "fp32"}, {"contiguous": False}])
def test_draft_unsupported_tensor_does_not_pad_or_dispatch(change):
    env, calls, _ = load_dispatcher()
    args = inputs(5)
    args[0] = replace(args[0], **change)
    assert env["hc_pre_draft"](*args) is None
    assert not calls


@pytest.mark.parametrize(
    "index,change",
    [
        (0, {"shape": (0, 4, 5120)}),
        (0, {"shape": (1, 4, 5120)}),
        (0, {"shape": (24, 4, 5120)}),
        (0, {"shape": (6, 4, 4096)}),
        (0, {"shape": (2, 3, 4, 5120)}),
        (0, {"device": "cpu"}),
        (0, {"dtype": "fp32"}),
        (0, {"contiguous": False}),
        (1, {"dtype": "bf16"}),
        (1, {"shape": (24, 20479)}),
        (1, {"contiguous": False}),
        (1, {"npu_format": 29}),
        (2, {"shape": (4,)}),
        (2, {"dtype": "bf16"}),
        (3, {"device": "cpu"}),
        (3, {"shape": (25,)}),
        (4, {"shape": (12, 4)}),
        (4, {"dtype": "bf16"}),
        (4, {"contiguous": False}),
    ],
)
def test_unqualified_tensor_contract_never_dispatches(index, change):
    env, calls, _ = load_dispatcher()
    args = inputs()
    args[index] = replace(args[index], **change)
    assert env["hc_pre_delayed"](*args) is None
    assert not calls


@pytest.mark.parametrize(
    "index,value", [(5, 0), (5, -1), (5, 1e-100), (5, 1e100), (5, math.nan), (5, math.inf), (6, 0), (7, 19)]
)
def test_unqualified_scalar_contract_never_dispatches(index, value):
    env, calls, _ = load_dispatcher()
    args = inputs()
    args[index] = value
    assert env["hc_pre_delayed"](*args) is None
    assert not calls


def test_training_other_soc_and_uninstalled_op_use_reference():
    for change in ("training", "soc", "missing"):
        env, calls, _ = load_dispatcher()
        if change == "training":
            env["torch"].is_inference_mode_enabled = lambda: False
        elif change == "soc":
            env["get_ascend_device_type"] = lambda: "A2"
        else:
            del env["torch"].ops._C_ascend.npu_hc_pre_delayed
        assert env["hc_pre_delayed"](*inputs()) is None
        assert not calls


def test_contiguous_offset_inputs_are_not_copied_or_rejected():
    env, calls, outputs = load_dispatcher()
    args = inputs()
    args[:5] = [replace(t, offset=32) for t in args[:5]]
    assert env["hc_pre_delayed"](*args) == outputs[:4]
    assert all(a is b for a, b in zip(args[:5], calls[0][:5]))


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("flag", ["negative", "conjugated"])
def test_lazy_storage_flags_fall_back_for_every_input(index, flag):
    env, calls, _ = load_dispatcher()
    args = inputs()
    args[index] = replace(args[index], **{flag: True})
    assert env["hc_pre_delayed"](*args) is None
    assert not calls


def test_native_failure_is_not_hidden_by_an_unmeasured_fallback():
    env, _, _ = load_dispatcher()

    def fail(*args):
        raise RuntimeError("native failure")

    env["torch"].ops._C_ascend.npu_hc_pre_delayed = fail
    with pytest.raises(RuntimeError, match="native failure"):
        env["hc_pre_delayed"](*inputs())


def block_methods(env):
    env.setdefault("hc_post_fused", lambda *args: None)
    path = ROOT / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text())
    block = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Block")
    methods = [
        node
        for node in block.body
        if isinstance(node, ast.FunctionDef) and node.name in ("pre_and_mixes", "post_residual", "forward")
    ]
    code = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), env)
    return type("BlockMethods", (), {name: env[name] for name in ("pre_and_mixes", "post_residual", "forward")})


@pytest.mark.parametrize("native_available", [False, True])
def test_two_blocks_preserve_previous_pre_timing_and_norm_module_calls(native_available):
    events = []
    pre_inputs = []
    next_gates = []
    outputs = []

    def make_next(x, fn):
        gate = object()
        next_gates.append(gate)
        return TensorMetadata((6, 5120), "bf16", label=fn), gate, object(), object()

    def native(x, fn, scale, base, previous, *args):
        if not native_available:
            return None
        pre_inputs.append(previous)
        return make_next(x, fn)

    def reference_pre(x, previous):
        pre_inputs.append(previous)
        return TensorMetadata((6, 5120), "bf16", label="reference")

    def post(x, residual, gate, comb):
        out = replace(residual, label=f"post-{len(outputs)}")
        outputs.append(out)
        return out

    cls = block_methods({"hc_pre_delayed": native, "hc_pre": reference_pre, "hc_post": post})
    blocks = [cls(), cls()]
    for i, block in enumerate(blocks):
        block.config = SimpleNamespace(rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20)
        for name in ("attn", "ffn"):
            setattr(block, f"hc_{name}_fn", f"{i}.{name}")
            setattr(block, f"hc_{name}_scale", object())
            setattr(block, f"hc_{name}_base", object())

        def norm(x, *, label):
            assert x.dtype == "bf16"
            events.append(label)
            return x

        block.attn_norm = lambda x, label=f"{i}.attn_norm": norm(x, label=label)
        block.ffn_norm = lambda x, label=f"{i}.ffn_norm": norm(x, label=label)
        block.attn = lambda x, runtime: x
        block.ffn = SimpleNamespace(forward_decode=lambda x, active: x)
        block.mixes = lambda x, name, b=block: make_next(x, getattr(b, f"hc_{name}_fn"))[1:]

    identity_pre = object()
    hidden = TensorMetadata((6, 4, 5120), "bf16")
    runtime = SimpleNamespace(decode=SimpleNamespace(active=object()))
    pre = identity_pre
    for block in blocks:
        hidden, pre = block.forward(hidden, pre, runtime)
    assert pre_inputs == [identity_pre, next_gates[0], next_gates[1], next_gates[2]]
    assert pre is next_gates[3]
    # Execute the real final HC method, not a restatement of its equation.
    path = ROOT / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text())
    model_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41ForCausalLM"
    )
    finish = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "_finish_hc")
    finish_env = {"hc_pre": reference_pre}
    exec(compile(ast.Module(body=[finish], type_ignores=[]), str(path), "exec"), finish_env)
    final_norm_calls = []
    model = SimpleNamespace(sequence_parallel=False, norm=lambda x: final_norm_calls.append(x) or x)
    final = finish_env["_finish_hc"](model, hidden, pre, 6)
    assert pre_inputs[-1] is next_gates[3]
    assert final_norm_calls == [final]
    assert events == ["0.attn_norm", "0.ffn_norm", "1.attn_norm", "1.ffn_norm"]
    assert hidden is outputs[-1]


@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("runtime", [SimpleNamespace(decode=None), SimpleNamespace(), object()])
def test_prefill_and_draft_never_call_native_even_at_decode_shapes(rows, runtime):
    def unexpected(*args):
        raise AssertionError("prefill/draft entered native HC")

    marker = object()
    cls = block_methods({"hc_pre_delayed": unexpected, "hc_pre": lambda x, pre: marker})
    block = cls()
    gates = (object(), object(), object())
    block.mixes = lambda x, name: gates
    result = block.pre_and_mixes(TensorMetadata((rows, 4, 5120), "bf16"), object(), "attn", runtime)
    assert result == (marker, *gates)
