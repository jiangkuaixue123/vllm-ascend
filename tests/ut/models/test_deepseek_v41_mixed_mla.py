# SPDX-License-Identifier: Apache-2.0
"""CPU address/selection oracle for the CANN V4.1 adapter; native numerics use the A5 validator."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import mixed_mla as native
from vllm_ascend.models.deepseek_v41.compact_kv import pack_fp4, pack_window, unpack_fp4, unpack_window
from vllm_ascend.models.deepseek_v41.decode_state import DecodeInputs, PagedHistory


def test_a5_selects_registered_mixed_ops_without_generic_custom_op_capability(monkeypatch):
    from vllm_ascend.device import device_config
    from vllm_ascend.device.hardware import AscendDeviceType, device_type_from_runtime_soc, device_type_from_soc_version
    from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
    from vllm_ascend.models.deepseek_v41 import compact_kv, native_ops

    a5 = AscendDeviceType.A5
    assert device_type_from_runtime_soc(260) == a5
    for name in ("Ascend950DT_9574", "ascend950dt_9582", "Ascend950"):
        assert device_type_from_soc_version(name) == a5
    assert not get_hardware_profile(a5).supports(HardwareCapability.RUNTIME_CUSTOM_OPS)
    monkeypatch.setattr(device_config, "get_device_config", lambda: device_config.DeviceConfig(a5))
    calls = []
    monkeypatch.setattr(native_ops, "initialize_native_ops", lambda: calls.append("extension"))
    monkeypatch.setattr(native, "import_module", calls.append)

    def packet(name):
        return SimpleNamespace(default=SimpleNamespace(_schema=torch._C.parse_schema(f"{name}(Tensor x) -> Tensor")))

    meta = packet("cann_ops_transformer::sparse_flash_mla_metadata")
    main = packet("cann_ops_transformer::mixed_quant_sparse_flash_mla")
    writer = SimpleNamespace(kv_compress_epilog_v2=packet("_C_ascend::kv_compress_epilog_v2"))
    cann = SimpleNamespace(sparse_flash_mla_metadata=meta, mixed_quant_sparse_flash_mla=main)
    monkeypatch.setattr(torch.ops, "cann_ops_transformer", cann)
    monkeypatch.setattr(torch.ops, "_C_ascend", writer)

    class Device:
        type = "npu"

    q = SimpleNamespace(device=Device(), dtype=torch.bfloat16, shape=(6, 64, 512))
    native._operators.cache_clear()
    try:
        assert native.supports_mixed_mla(q, 64)
        assert native._operators(q.device) == (meta.default, main.default, compact_kv.store_mixed_kv)
        assert calls == ["extension", "cann_ops_transformer"]
        del writer.kv_compress_epilog_v2
        native._operators.cache_clear()
        assert not native.supports_mixed_mla(q, 64)
        monkeypatch.setattr(device_config, "get_device_config", lambda: device_config.DeviceConfig(AscendDeviceType.A3))
        native._operators.cache_clear()
        calls.clear()
        assert not native.supports_mixed_mla(q, 64)
        assert not calls
    finally:
        native._operators.cache_clear()


class ScratchReference:
    def __init__(self):
        self.pages = {}
        self.calls = []

    def store(self, cache, values, slots):
        # Keep decoded values in a separate table so we can validate native addressing without an NPU.
        rows = cache.size(0) * cache.size(1)
        actual = torch.full((rows, native.HEAD_DIM), float("nan"), dtype=torch.bfloat16)
        live = slots >= 0
        assert slots[live].unique().numel() == int(live.sum())
        actual.index_copy_(0, slots[live].long(), values[live])
        self.pages[id(cache)] = actual

    def metadata(self, **kw):
        self.calls.append(kw)
        assert kw["layout_q"] == "TND" and kw["layout_kv"] == "PA_BBND"
        assert "quant_mode" not in kw and "rope_head_dim" not in kw
        return torch.tensor([kw["batch_size"]], dtype=torch.int32)

    def attention(self, q, **kw):
        assert kw["quant_mode"] == 2 and kw["rope_head_dim"] == 64
        assert len(q) == kw["metadata"].item()
        raw = self.pages.pop(id(kw["ori_kv"]))
        cmp = self.pages.pop(id(kw["cmp_kv"])) if "cmp_kv" in kw else None
        out = []
        for i in range(len(q)):
            n = kw["seqused_ori_kv"][i].item()
            logical = torch.arange(n)
            raw_ids = (
                kw["ori_block_table"][i, logical // native.NATIVE_PAGE_SIZE] * native.NATIVE_PAGE_SIZE
                + logical % native.NATIVE_PAGE_SIZE
            )
            vals = raw[raw_ids]
            if cmp is not None:
                selected = kw["cmp_sparse_indices"][i, 0].long()
                selected = selected[selected >= 0]
                assert (selected < kw["seqused_cmp_kv"][i]).all()
                addresses = (
                    kw["cmp_block_table"][i, selected // native.NATIVE_PAGE_SIZE] * native.NATIVE_PAGE_SIZE
                    + selected % native.NATIVE_PAGE_SIZE
                )
                vals = torch.cat((vals, cmp[addresses]))
            out.append(_read(q[i], vals, kw["sinks"], kw["softmax_scale"]))
        return torch.stack(out), torch.empty(0)


def _read(q, k, sink, scale):
    if not len(k):
        return torch.zeros_like(q)
    logits = q.float() @ k.float().T * scale
    probs = torch.cat((logits, sink[:, None]), dim=1).softmax(1)[:, : len(k)]
    return (probs @ k.float()).bfloat16()


@pytest.mark.parametrize("compressed", [False, True])
def test_compact_physical_rows_and_padding(monkeypatch, compressed):
    g = torch.Generator().manual_seed(359)
    q = torch.randn((19, 4, 512), generator=g).bfloat16()
    raw = pack_window(torch.randn((141, 512), generator=g).bfloat16())
    cmp = pack_fp4(torch.randn((340, 512), generator=g).bfloat16(), 16, True) if compressed else None
    raw_ids = torch.randint(0, len(raw), (19, 15), generator=g)
    raw_ids[::3, :4] = -1
    raw_ids[2, 2::3] = -1
    raw_ids[-1].fill_(-1)
    cmp_ids = torch.randint(0, 340, (19, 134), generator=g) if compressed else None
    if compressed:
        cmp_ids[1, 1::3] = -1
        cmp_ids[2].fill_(-1)
        cmp_ids[-1].fill_(-1)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    sink = torch.linspace(-1, 1, q.size(1))
    oracle = ScratchReference()
    monkeypatch.setattr(native, "_operators", lambda device: (oracle.metadata, oracle.attention, oracle.store))
    result = native.mixed_mla(q, raw, raw_ids, cmp, cmp_ids, sink, 512**-0.5, lut)
    assert len(oracle.calls) == 2 and all(e["has_cmp_kv"] == compressed for e in oracle.calls)
    for row in range(len(q)):
        ids = raw_ids[row]
        vals = unpack_window(raw[ids[ids >= 0]], 512)
        if compressed:
            ids = cmp_ids[row]
            vals = torch.cat((vals, unpack_fp4(cmp[ids[ids >= 0]], 512, 16, True, lut)))
        assert torch.equal(result[row], _read(q[row], vals, sink, 512**-0.5))


def test_serial_target_preserves_batch_identity(monkeypatch):
    g = torch.Generator().manual_seed(98)
    q = torch.randn((2, 3, 3, 512), generator=g).bfloat16()
    kv = torch.randn((2, 8, 512), generator=g).bfloat16()
    indices = torch.tensor([[[0, -1, 4], [2, 1, 4], [3, 1, 5]], [[3, 1, 6], [-1, 2, 4], [2, 1, -1]]])
    oracle = ScratchReference()
    sink = torch.zeros(3)
    monkeypatch.setattr(native, "_operators", lambda device: (oracle.metadata, oracle.attention, oracle.store))
    actual = native.mixed_mla_from_bf16(q, kv, indices, 2, sink, 512**-0.5)
    for b in range(2):
        for i in range(3):
            ids = indices[b, i]
            expected = _read(q[b, i], kv[b, ids[ids >= 0]], sink, 512**-0.5)
            assert torch.equal(actual[b, i], expected)
    assert not native.supports_mixed_mla(q, 64)


@pytest.mark.parametrize("ratio", [1, 2])
def test_compression_causality_across_real_pages(monkeypatch, ratio):
    history = PagedHistory(384, 128, 8, max_draft_tokens=5)
    state = DecodeInputs(history, 6, "cpu", [], max_requests=1)
    table = [5, 3, 1]
    positions = list(range(254, 260))

    def physical(pos):
        return table[pos // 128] * 128 + pos % 128

    plan = history.plan(positions, [physical(p) for p in positions], [table], [6], [254])
    state.stage_layout(plan)
    raw_ids, _ = state.raw_windows[128]
    lens = (state.pos[:, None] + 1) // ratio
    choices = torch.cat((torch.zeros_like(lens), lens // 2, lens - 1, lens), dim=1)
    latest = (choices + 1) * ratio - 1
    cmp_ids = torch.where(choices < lens, state.physical(latest, slice(0, 6), ratio), -1)
    g = torch.Generator().manual_seed(733)
    raw = torch.randn(8 * 128, 512, generator=g).bfloat16()
    cmp = torch.randn(8 * 128 // ratio, 512, generator=g).bfloat16()
    q = torch.randn(6, 2, 512, generator=g).bfloat16()
    sink = torch.tensor([-0.5, 1.0])
    ops = ScratchReference()
    monkeypatch.setattr(native, "_operators", lambda device: (ops.metadata, ops.attention, ops.store))
    got = native.mixed_mla(q, raw, raw_ids, cmp, cmp_ids, sink, 512**-0.5)
    for row, pos in enumerate(positions):
        end = (pos + 1) // ratio
        keys = [physical((k + 1) * ratio - 1) // ratio for k in (0, end // 2, end - 1)]
        vals = torch.cat((raw[[physical(i) for i in range(pos - 127, pos + 1)]], cmp[keys]))
        assert torch.equal(got[row], _read(q[row], vals, sink, 512**-0.5))
