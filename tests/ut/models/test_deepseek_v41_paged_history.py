# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import torch

from vllm_ascend.models.deepseek_v41.decode_state import DecodeInputs, PagedHistory


class TestPagedHistory(TestCase):
    def test_interleaved_chunks_rejection_and_page_reuse(self):
        state = PagedHistory(max_len=8, block_size=2, num_blocks=10)
        reference = SimpleNamespace(
            layout=SimpleNamespace(max_ngram_size=3),
            token_map=torch.arange(32),
            multipliers=torch.tensor([[3, 5, 7]]),
            primes=torch.tensor([[[11], [13]]]),
            offsets=torch.tensor([[0, 11]]),
            pad_id=0,
        )
        tables = {"a": [4, 7, 1, 9], "b": [2, 6, 3, 8], "c": [4, 7, 1, 9]}
        sequences = {key: [] for key in tables}

        def compare(chunks):
            ids, starts, slots, ends, pages, expected = [], [], [], [], [], []
            for req, start, new in chunks:
                prefix = sequences[req][:start]
                self.assertEqual(len(prefix), start)
                for token in new:
                    pos = len(prefix)
                    two = 3 * token ^ 5 * (prefix[-1] if prefix else 0)
                    three = two ^ 7 * (prefix[-2] if len(prefix) >= 2 else 0)
                    expected.append([two % 11, 11 + three % 13])
                    ids.append(token)
                    starts.append(pos)
                    slots.append(tables[req][pos // 2] * 2 + pos % 2)
                    prefix.append(token)
                pages.append(tables[req])
                ends.append(len(ids))
                sequences[req] = prefix
            plan = state.plan(starts + [0], slots + [-1], pages, ends)
            hashes = state.hash_ids(reference, plan, ids + [0])[:, 0]
            np.testing.assert_array_equal(hashes, expected)

        compare([("a", 0, [1, 2]), ("b", 0, [3])])
        compare([("b", 1, [4, 5]), ("a", 2, [6, 7])])
        compare([("a", 4, [8, 9, 10]), ("b", 3, [11])])
        compare([("b", 4, [13]), ("a", 5, [14])])
        compare([("c", 0, [15, 16, 17]), ("b", 5, [18])])
        before = state.tokens.copy()
        idle = state.plan([0], [-1], [], [])
        self.assertFalse(idle.active.any())
        np.testing.assert_array_equal(before, state.tokens)

    def test_inconsistent_physical_slot_rejected(self):
        state = PagedHistory(max_len=8, block_size=2, num_blocks=10)
        with self.assertRaisesRegex(ValueError, "disagree"):
            state.plan([0], [2], [[4, 7]], [1])
        with self.assertRaisesRegex(ValueError, "outside"):
            state.plan([0], [20], [[10]], [1])

    def test_one_million_context_keeps_one_table_per_request(self):
        length = 1 << 20
        history = PagedHistory(length, 128, 8200, max_draft_tokens=5)
        stage = DecodeInputs(history, 4096, "cpu", [], max_requests=4)
        self.assertEqual(stage.pages.shape, (5, 8192))
        self.assertEqual(stage.pages.nbytes, 5 * 8192 * 8)
        ptr = stage.pages.data_ptr()
        # The first request spans two pages. The second is at the final
        # logical block of 1M, with its prefix retained after a cache hit.
        tables = np.tile(np.arange(1, 8193), (2, 1))
        tables[0, 2:] = -1
        positions = [127, 128, 129, length - 2, length - 1, 0]
        slots = [255, 256, 257, length + 126, length + 127, -1]
        stage.stage_layout(history.plan([126], [254], [tables[0]], [1]))
        plan = history.plan(positions, slots, tables, [3, 5])
        stage.stage_layout(plan)
        self.assertEqual(plan.block_tables.shape, (2, 8192))
        self.assertEqual(stage.page_rows[:6].tolist(), [0, 0, 0, 1, 1, 4])
        self.assertEqual(stage.pages_for_rows(slice(0, 6))[:, 1].tolist(), [2, 2, 2, 2, 2, 0])
        lookup = torch.tensor([[126, 128], [127, 128], [128, 129], [0, length - 2], [128, length - 1], [0, 1]])
        self.assertEqual(
            stage.physical(lookup, slice(0, 6)).tolist(),
            [[254, 0], [255, 256], [256, 257], [128, length + 126], [256, length + 127], [0, 1]],
        )
        idle = history.plan([0, 0], [-1, -1], [], [])
        stage.stage_layout(idle)
        self.assertEqual(idle.block_tables.shape, (0, 8192))
        self.assertFalse(stage.pages_for_rows(slice(0, 2)).any())
        self.assertEqual(stage.pages.data_ptr(), ptr)
