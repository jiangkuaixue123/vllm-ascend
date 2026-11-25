# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch


def test_window_batch_respects_table_stride_query_mask_and_workspace_limit(monkeypatch):
    from vllm.models.hy_v4_flash.reference import window_candidates

    from vllm_ascend.models.hy_v4_flash import attention as module
    from vllm_ascend.models.hy_v4_flash.decode_layout import decode_positions, window_history

    requests = module.WINDOW_REQUEST_CHUNK + 1
    width, heads, window = 6, 8, 512
    layer = module.AscendHYV4FlashAttention.__new__(module.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(layer)
    layer.indexer = layer.topk_snapshot = layer.resident_pool = None
    layer.share_verify_layout = False
    layer.share_swa_c8_reads = layer.pack_window_operands = True
    layer.local_heads, layer.window, layer.max_model_len = heads, window, 131584
    lengths = torch.arange(requests) * 105 + width
    slots = torch.zeros(requests, width, dtype=torch.int32)
    slots[::2, -2:] = -1
    slots[1] = -1
    tables = torch.arange(requests * 5).view(requests, 5)[:, :3]
    meta = SimpleNamespace(decode_seq_lens=lengths, slot_mapping=slots.flatten(), block_table=tables)
    lengths, live, logical = decode_positions(lengths, slots.flatten(), width)
    history = window_history(lengths, logical, window)
    selected = window_candidates(logical.flatten(), window)
    query = torch.ones(requests * width, heads, 2, dtype=torch.bfloat16)
    out = query.new_empty((requests * width, heads))
    cache = SimpleNamespace(shape=(8, 65600), device=SimpleNamespace(type="npu"))
    consumed = []

    def gather(storage, own_table, block_size, own_history, queries, actual_window, **options):
        first = sum(consumed)
        end = first + own_table.shape[0]
        assert storage is cache and (block_size, queries, actual_window) == (64, width, window)
        torch.testing.assert_close(own_table, tables[first:end])
        torch.testing.assert_close(own_history, history[first:end])
        assert own_table.stride(0) == tables.stride(0) and own_table.shape[0] <= module.WINDOW_REQUEST_CHUNK
        consumed.append(own_table.shape[0])
        return torch.ones((queries * own_table.shape[0], 2, 2)), object()

    def attend(q, positions, values, weights, dest, *, scores):
        end = sum(consumed) * width
        start = end - q.shape[0]
        torch.testing.assert_close(positions, selected[start:end])
        assert scores.shape == (*q.shape[:-1], 2)
        dest.fill_(7)

    monkeypatch.setattr(module, "gather_c8_window_operands", gather)
    monkeypatch.setattr(module.torch_npu, "get_npu_format", lambda tensor: module.ACL_FORMAT_ND)
    layer._attend_decoded = attend
    layer._forward_decode_graph(meta, cache, 64, None, None, None, query, None, out)
    assert consumed == [module.WINDOW_REQUEST_CHUNK, 1]
    torch.testing.assert_close(out, (live.flatten() * 7).to(out.dtype)[:, None].expand_as(out))
