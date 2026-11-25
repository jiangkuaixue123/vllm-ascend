# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select and share logical retrieval positions within one target decode forward."""


class SharedRecallRows(list):
    """Logical recall and its optional rank-local physical mapping for this forward."""

    def __init__(self):
        super().__init__()
        self._table = None
        self._physical = None

    def physical_for(self, table, map_rows):
        if table is not self._table:
            self._physical = map_rows(table, self)
            self._table = table
        return self._physical


def paired_recall_layer(layer_types, index, sparse_type):
    sparse = [i for i, kind in enumerate(layer_types) if kind == sparse_type]
    # Keep the first two sparse layers and the final layer on their own trained index.
    consumers = sparse[2:-1:2]
    for source, consumer in zip(sparse[1:-1:2], consumers):
        if index in (source, consumer):
            return source, index == source
    return None


def recall_rows_for_forward(context, metadata, policy, requests, width, ratio, budget):
    """Return separate read and write handles to prevent accidental mutation by another layer."""
    source, publish = policy
    cache = context.additional_kwargs.setdefault("hy4flash_target_recall_rows", {})
    key = (id(metadata), source, requests, width, ratio, budget)
    if publish:
        rows = SharedRecallRows()
        cache[key] = rows
        return None, rows
    rows = cache.get(key)
    return (rows if rows is not None and len(rows) == requests else None), None
