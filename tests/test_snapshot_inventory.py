"""Inventory validation stays bounded by a page and checks boundaries between pages."""

import tracemalloc
from pathlib import Path

import pytest

from premixdb.engine.contracts import DocumentRecord, Frame
from premixdb.engine.identity import CodeVersion
from premixdb.engine.snapshots import Store, encode

CODE = CodeVersion("local://inventory", "a" * 40, "09" * 32)


def page(store: Store, rows: list[DocumentRecord]) -> Frame:
    data = encode(rows)
    return Frame(digest=store._put(data), bytes=len(data))


def test_inventory_iteration_keeps_one_page_in_memory(tmp_path: Path) -> None:
    store = Store(tmp_path)
    snapshot = store.capture_inputs("01" * 16, [], [], CODE)
    manifest = store._manifest(snapshot.id)
    count = 256
    manifest["summary"]["documents"] = count
    manifest["pages"] = [
        page(
            store,
            [
                DocumentRecord(key=f"{i:04}" + "x" * 40_000, content=[0] * 32, bytes=0, frames=[])
                for i in range(first, first + 8)
            ],
        )
        for first in range(0, count, 8)
    ]
    tracemalloc.start()
    try:
        assert sum(1 for _ in store._records(manifest)) == count
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000


@pytest.mark.parametrize("damage", ["duplicate", "reversed", "empty", "missing"])
def test_inventory_validation_spans_page_boundaries(tmp_path: Path, damage: str) -> None:
    store = Store(tmp_path)
    snapshot = store.capture_inputs("01" * 16, [("a", "first"), ("b", "second")], [], CODE)
    manifest = store._manifest(snapshot.id)
    records = list(store._records(manifest))
    pages = [[records[0]], [records[1]]]
    if damage == "duplicate":
        pages[1][0]["key"] = pages[0][0]["key"]
    elif damage == "reversed":
        pages.reverse()
    elif damage == "empty":
        pages.insert(1, [])
    else:
        pages.pop()
    manifest["pages"] = [page(store, rows) for rows in pages]
    with pytest.raises(RuntimeError, match="inventory"):
        list(store._records(manifest))
