# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU input-contract tests; native A3 output and timing require device evidence."""

import ast
import json
import os
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


def source_functions():
    # These pure tensor helpers do not need vLLM or torch_npu initialization.
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(path.read_text())
    names = {"ring_positions", "block_attention", "try_stage_block_inputs"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and all(isinstance(target, ast.Name) and target.id.startswith("_STAGING_") for target in node.targets)
        )
    ]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


@pytest.fixture
def helpers():
    return source_functions()


def make_case(histories, *, negative=False):
    batch, window, dim = len(histories), 128, 512
    generator = torch.Generator().manual_seed(185 + batch)
    backing = torch.randn(12 * window, dim * 2, generator=generator).bfloat16()
    field = backing[:, :dim]
    table = torch.randint(12, (batch, 1200), dtype=torch.int32, generator=generator)
    table[:, 0] = 0
    if negative:
        table[0, (histories[0] - 1) // window] = -1
    q = torch.randn(batch * 5, 16, dim, generator=generator).bfloat16()
    query = torch.randn(batch * 5, dim, generator=generator).bfloat16()
    spans = [SimpleNamespace(begin=i * 5, end=(i + 1) * 5, start_pos=h) for i, h in enumerate(histories)]
    metadata = SimpleNamespace(requests=spans, num_actual_tokens=batch * 5, block_table_tensor=table)
    cache = SimpleNamespace(block_size=window, field=lambda *args: field)
    return q, query, metadata, cache, field, backing


def stage(helpers, case, **kwargs):
    return helpers["try_stage_block_inputs"](
        *case[:4],
        window_size=kwargs.get("window_size", 128),
        block_size=kwargs.get("block_size", 5),
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("batch", [2, 4, 8])
@pytest.mark.parametrize("negative", [False, True])
def test_ring_query_and_native_input_words_match_serial(helpers, batch, negative):
    histories = [128, 129, 255, 256, 8192, 8193, 153593, 153594][:batch]
    case = make_case(histories, negative=negative)
    q, query, metadata, _, field, backing = case
    field.view(torch.int16)[:, :5] = torch.tensor([0, -32768, 32640, -128, 32727], dtype=torch.int16)
    query.view(torch.int16)[:, :5] = torch.tensor([-32768, 0, 32735, 32640, -128], dtype=torch.int16)
    before = backing.view(torch.int16).clone()
    staged_kv, staged_ids = stage(helpers, case)
    for row, history in enumerate(histories):
        # Scalar ring/slot oracle is independent of the vector implementation.
        ring = torch.tensor([history - 1 - (history - 1 - slot) % 128 for slot in range(128)])
        page = metadata.block_table_tensor[row][ring // 128].long()
        context = field[page * 128 + ring % 128]
        rows = slice(row * 5, (row + 1) * 5)

        def inspect(q_single, kv_single, sink, ids_single, scale, *, mode, rows=rows, row=row):
            assert q_single.shape == (1, 5, 16, 512)
            assert mode == "SWA draft bidirectional"
            assert torch.equal(q_single.view(torch.int16), q[rows].unsqueeze(0).view(torch.int16))
            actual = staged_kv[row : row + 1]
            assert actual.is_contiguous() and actual.stride() == kv_single.stride()
            assert torch.equal(actual.view(torch.int16), kv_single.view(torch.int16))
            assert staged_ids.stride() == ids_single.stride() and torch.equal(staged_ids, ids_single)
            return torch.zeros_like(q_single)

        helpers["block_attention"](q[rows], context, query[rows], torch.zeros(16), 512**-0.5, inspect)
    assert torch.equal(backing.view(torch.int16), before)


@pytest.mark.parametrize("history", [0, 1, 127])
def test_short_history_keeps_parent_path(helpers, history):
    assert stage(helpers, make_case([history, 8192])) is None


@pytest.mark.parametrize(
    "kind",
    [
        "single",
        "empty",
        "gap",
        "partial",
        "negative_start",
        "dummy",
        "padding",
        "noncontiguous_q",
        "noncontiguous_query",
        "fp32",
        "table_rows",
        "table_columns",
        "table_dtype",
        "field_stride",
        "page_size",
    ],
)
def test_unsupported_metadata_and_layout_leave_inputs_untouched(helpers, kind):
    case = list(make_case([8192, 8193]))
    q, query, md, cache, field, backing = case
    if kind == "single":
        case = list(make_case([8192]))
    elif kind == "empty":
        md.requests = []
    elif kind == "gap":
        md.requests[1].begin += 1
    elif kind == "partial":
        md.requests[1].end -= 1
    elif kind == "negative_start":
        md.requests[1].start_pos = -1
    elif kind == "dummy":
        md.num_actual_tokens = 5
    elif kind == "padding":
        case[0] = torch.cat((q, q[:5]))
    elif kind == "noncontiguous_q":
        case[0] = q.transpose(1, 2).contiguous().transpose(1, 2)
    elif kind == "noncontiguous_query":
        case[1] = query.t().contiguous().t()
    elif kind == "fp32":
        case[0] = q.float()
    elif kind == "table_rows":
        md.block_table_tensor = md.block_table_tensor[:1]
    elif kind == "table_columns":
        md.block_table_tensor = md.block_table_tensor[:, :1]
    elif kind == "table_dtype":
        md.block_table_tensor = md.block_table_tensor.float()
    elif kind == "field_stride":
        cache.field = lambda *args: field.t().contiguous().t()
    elif kind == "page_size":
        cache.block_size = 64
    before = case[-1].view(torch.int16).clone()
    assert stage(helpers, case) is None
    assert torch.equal(case[-1].view(torch.int16), before)


@pytest.mark.parametrize("rejected", range(6))
def test_rejection_adjusted_histories_and_request_isolation(helpers, rejected):
    histories = [8193 - rejected, 153594 - rejected]
    case = list(make_case(histories))
    kv, ids = stage(helpers, case)
    case[1] = case[1].clone()
    case[1].view(torch.int16)[:5].bitwise_xor_(1)
    changed, changed_ids = stage(helpers, case)
    assert torch.equal(kv[1:].view(torch.int16), changed[1:].view(torch.int16))
    assert not torch.equal(kv[0, 128:].view(torch.int16), changed[0, 128:].view(torch.int16))
    assert torch.equal(ids, changed_ids)


def test_no_device_scalar_or_collective_in_staging():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(source.read_text())
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "try_stage_block_inputs"
    )
    attributes = {node.attr for node in ast.walk(helper) if isinstance(node, ast.Attribute)}
    assert not attributes.intersection(
        {"item", "tolist", "cpu", "synchronize", "all_gather", "all_reduce", "write_slots"}
    )


def witness_fixture(histories, *, dummy=False):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSparkAttention")
    nodes = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in {"forward", "_record_staging_b4_witness", "store_context"}
    ]
    records, calls, writes = [], [], []
    scope = source_functions()
    scope.update(
        json=json,
        os=os,
        time=time,
        Path=Path,
        __file__=str(source),
        _WITNESS_BATCH_SIZE=4,
        is_draft_dummy=lambda: dummy,
        try_write_draft_context=None,
        logger=SimpleNamespace(info=lambda fmt, value: records.append(json.loads(value))),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    case = make_case(histories)
    q, query, md, cache, field, _ = case
    cache.prefix = "mtp.1.attn.cache"
    cache.metadata = lambda: None if dummy else md
    cache.read = lambda metadata, row, positions, *args: field[
        metadata.block_table_tensor[row][positions // 128].long() * 128 + positions % 128
    ]
    cache.write_slots = lambda *args: writes.append(args)

    def kernel(q_single, kv, sink, ids, scale, *, mode):
        calls.append((q_single.shape, kv.shape, ids.shape, mode))
        return torch.zeros_like(q_single)

    obj = SimpleNamespace(
        _staging_b4_witness_pending=True,
        cache=cache,
        config=SimpleNamespace(sliding_window=128, dspark_block_size=5),
        heads=16,
        dim=512,
        wq_a=lambda hidden: (hidden, None),
        q_norm=lambda hidden: hidden,
        wq_b=lambda qr: (q, None),
        project_kv=lambda hidden, positions: query,
        rotate=lambda *args, **kwargs: None,
        _project_output=lambda hidden, dtype: hidden,
        attn_sink=torch.zeros(16),
        attention_kernel=kernel,
    )
    for name in ("forward", "_record_staging_b4_witness", "store_context"):
        setattr(obj, name, MethodType(scope[name], obj))
    return obj, case, records, calls, writes, scope


def test_witness_first_b4_records_actual_dispatch_once_and_closes():
    obj, case, records, calls, _, _scope = witness_fixture([8192, 8193, 8194, 8195])
    positions = torch.arange(20)
    obj.forward(case[1], positions)
    assert len(records) == 1 and len(calls) == 1
    record = records[0]
    assert record["staging_used"] and record["completed_native_calls"] == 1 and record["witness_closed"]
    assert record["cache_prefix"] == "mtp.1.attn.cache" and record["pid"] == os.getpid()
    assert record["q"]["shape"] == [20, 16, 512]
    assert record["staged_kv"]["shape"] == [4, 133, 512]
    assert record["staged_indices"]["shape"] == [1, 5, 133]
    assert record["staged_indices"]["stride"] == [133, 0, 1]
    assert calls == [((4, 5, 16, 512), (4, 133, 512), (4, 5, 133), "SWA draft bidirectional")]

    def unexpected_record(*args):
        raise AssertionError("Closed witness must not build geometry or log")

    obj._record_staging_b4_witness = unexpected_record
    obj.forward(case[1], positions)
    assert len(records) == 1 and len(calls) == 2 and not obj._staging_b4_witness_pending


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_witness_other_batches_do_not_record(batch):
    obj, case, records, calls, _, _ = witness_fixture([8192 + n for n in range(batch)])
    obj.forward(case[1], torch.arange(batch * 5))
    assert not records and len(calls) == 1 and obj._staging_b4_witness_pending


def test_witness_b4_fallback_is_recorded_once_without_cherry_picking():
    obj, case, records, calls, _, _ = witness_fixture([0, 8192, 8193, 8194])
    obj.forward(case[1], torch.arange(20))
    assert len(records) == 1 and not records[0]["staging_used"] and len(calls) == 4
    assert records[0]["staged_kv"] is None and records[0]["staged_indices"] is None
    case[2].requests[0].start_pos = 8195
    obj.forward(case[1], torch.arange(20))
    assert len(records) == 1 and len(calls) == 5


def test_witness_dummy_and_prefill_store_context_do_not_record():
    obj, case, records, calls, writes, _ = witness_fixture([8192, 8193, 8194, 8195], dummy=True)
    obj.forward(case[1], torch.arange(20))
    assert not records and not calls and obj._staging_b4_witness_pending
    obj.store_context(case[1], torch.arange(20), torch.arange(20))
    assert len(writes) == 1 and not records and not calls and obj._staging_b4_witness_pending


def test_witness_never_certifies_a_partial_native_loop():
    obj, case, records, calls, _, _ = witness_fixture([0, 8193, 8194, 8195])
    real_kernel = obj.attention_kernel

    def fail_second(*args, **kwargs):
        if len(calls) == 1:
            raise RuntimeError("synthetic native failure")
        return real_kernel(*args, **kwargs)

    obj.attention_kernel = fail_second
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        obj.forward(case[1], torch.arange(20))
    assert len(calls) == 1 and not records and obj._staging_b4_witness_pending


def test_witness_is_host_only_and_state_is_per_instance():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSparkAttention")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_record_staging_b4_witness")
    attrs = {n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)}
    assert not attrs.intersection({"item", "tolist", "cpu", "synchronize", "all_gather", "all_reduce", "getenv"})
    assert not any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(method))
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert any(
        isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Constant)
        and n.value.value is True
        and any(isinstance(t, ast.Attribute) and t.attr == "_staging_b4_witness_pending" for t in n.targets)
        for n in init.body
    )


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_batched_forward_preserves_request_rows(batch):
    obj, case, _, calls, _, _ = witness_fixture([8192 + n for n in range(batch)])
    q = case[0]

    def echo_queries(query, kv, sink, ids, scale, *, mode):
        calls.append(query.shape)
        assert query.shape == (batch, 5, 16, 512)
        assert ids.shape == (batch, 5, 133)
        # Distinct request contents expose accidental broadcasting or reordering.
        assert torch.equal(query.reshape_as(q), q)
        return query.clone()

    obj.attention_kernel = echo_queries
    output = obj.forward(case[1], torch.arange(batch * 5))
    assert torch.equal(output, q)
    assert len(calls) == 1


def test_batched_failure_does_not_record_success():
    obj, case, records, calls, _, _ = witness_fixture([8192, 8193, 8194, 8195])

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic batched native failure")

    obj.attention_kernel = fail
    with pytest.raises(RuntimeError, match="synthetic batched native failure"):
        obj.forward(case[1], torch.arange(20))
    assert not records and not calls and obj._staging_b4_witness_pending
