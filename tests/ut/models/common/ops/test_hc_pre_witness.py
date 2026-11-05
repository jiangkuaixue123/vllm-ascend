# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual bounded startup witness methods without NPU imports."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]
MODEL = ROOT / "vllm_ascend/models/deepseek_v41/model.py"


def make_block(*, pending=True, capturing=True, result=True, fail=False, change_hf32=False):
    tree = ast.parse(MODEL.read_text())
    block = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Block")
    names = ("pre_and_mixes", "_record_hc_native_witness")
    methods = [node for node in block.body if isinstance(node, ast.FunctionDef) and node.name in names]
    events, records = [], []
    outputs = tuple(object() for _ in range(4))

    class MatmulOptions:
        value = False

        @property
        def allow_hf32(self):
            events.append("get_hf32")
            return self.value

    options = MatmulOptions()

    def native(*args):
        events.append("native")
        if fail:
            raise RuntimeError("native failed")
        if change_hf32:
            options.value = True
        return outputs if result else None

    def log(prefix, record):
        assert prefix == "HC_NATIVE_RUNTIME_WITNESS %s"
        events.append("log")
        records.append(json.loads(record))

    context = SimpleNamespace(capturing=capturing)
    env = {
        "hc_pre_delayed": native,
        "hc_pre": lambda x, previous: "fallback_y",
        "get_forward_context": lambda: context,
        "torch": SimpleNamespace(npu=SimpleNamespace(matmul=options)),
        "sys": SimpleNamespace(modules={"vllm_ascend.vllm_ascend_C": SimpleNamespace(__file__="/isolated/binding.so")}),
        "os": SimpleNamespace(getpid=lambda: 1234, environ={"ASCEND_CUSTOM_OPP_PATH": "/parent:/new"}),
        "Path": Path,
        "json": json,
        "get_dp_group": lambda: SimpleNamespace(rank_in_group=2),
        "get_tp_group": lambda: SimpleNamespace(rank_in_group=1),
        "get_ep_group": lambda: SimpleNamespace(rank_in_group=9),
        "logger": SimpleNamespace(info=log),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(MODEL), "exec"), env)
    cls = type("ActualWitnessMethods", (), {name: env[name] for name in names})
    instance = cls()
    instance._hc_native_witness_pending = pending
    instance.config = SimpleNamespace(rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20)
    instance.hc_attn_fn = instance.hc_attn_scale = instance.hc_attn_base = object()
    instance.mixes = lambda x, name: ("fallback_pre", "fallback_post", "fallback_comb")
    return instance, events, records, outputs, options, context


def call(block, *, decode=True):
    return block.pre_and_mixes(object(), object(), "attn", SimpleNamespace(decode=object() if decode else None))


def test_only_first_successful_capture_call_records_metadata_and_then_closes():
    block, events, records, outputs, _, _ = make_block()
    assert call(block) is outputs
    assert events == ["get_hf32", "native", "get_hf32", "log"]
    assert records == [
        {
            "schema": "hc-native-startup-metadata-v1",
            "phase": "existing_startup_warmup",
            "context_capturing": True,
            "pid": 1234,
            "dp": 2,
            "tp": 1,
            "ep": 9,
            "effective_custom_opp_path": "/parent:/new",
            "hf32_before": False,
            "hf32_after": False,
            "native_call_completed": True,
            "binding_loaded": "/isolated/binding.so",
        }
    ]
    assert block._hc_native_witness_pending is False
    assert call(block) is outputs
    assert events[-1] == "native" and len(records) == 1 and events.count("get_hf32") == 2


@pytest.mark.parametrize("pending,capturing,decode", [(False, True, True), (True, False, True), (True, True, False)])
def test_other_layers_ordinary_decode_and_prefill_do_not_read_or_log(pending, capturing, decode):
    block, events, records, _, _, _ = make_block(pending=pending, capturing=capturing)
    call(block, decode=decode)
    assert not records and "get_hf32" not in events
    assert block._hc_native_witness_pending is pending


def test_fallback_keeps_pending_and_does_not_claim_native_completion():
    block, events, records, _, _, _ = make_block(result=False)
    assert call(block) == ("fallback_y", "fallback_pre", "fallback_post", "fallback_comb")
    assert not records and block._hc_native_witness_pending is True
    assert events == ["get_hf32", "native"]


def test_failed_native_does_not_emit_a_completion_record():
    block, events, records, _, _, _ = make_block(fail=True)
    with pytest.raises(RuntimeError, match="native failed"):
        call(block)
    assert not records and block._hc_native_witness_pending is True
    assert events == ["get_hf32", "native"]


def test_witness_observes_changed_flag_without_resetting_it():
    block, _, records, outputs, options, _ = make_block(change_hf32=True)
    assert call(block) is outputs
    assert records[0]["hf32_before"] is False and records[0]["hf32_after"] is True
    assert options.value is True


def test_only_first_block_constructor_enables_witness():
    tree = ast.parse(MODEL.read_text())
    block = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Block")
    init = next(node for node in block.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignment = next(
        node
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_hc_native_witness_pending" for target in node.targets
        )
    )
    code = compile(ast.Expression(assignment.value), str(MODEL), "eval")
    assert [eval(code, {"layer_id": index}) for index in range(40)] == [True] + [False] * 39
