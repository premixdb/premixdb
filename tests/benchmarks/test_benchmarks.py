"""CPU-only hot paths with fixed inputs and checked outputs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import (
    Benchmark,
    coordinator,
)

import premixdb as p
from premixdb._reader import permutation
from premixdb.engine.identity import CodeVersion
from premixdb.engine.plans import filter, query_identity

pytestmark = pytest.mark.performance
CODE = CodeVersion("local://benchmarks", "a" * 40, "09" * 32)


@pytest.mark.parametrize("count", [10, 1000])
def test_query_identity(benchmark: Benchmark, count: int) -> None:
    snapshots = [f"{index:064x}" for index in range(count)]
    steps = [filter("bytes", "ge", 100), filter("characters", "lt", 10_000)]
    expected = query_identity(snapshots, steps, CODE)
    result = benchmark(query_identity, snapshots, steps, CODE)
    assert result == expected
    assert len(result) == 64


def shuffled_ordinals(count: int, seed: int) -> list[int]:
    return [permutation(index, count, seed) for index in range(count)]


@pytest.mark.parametrize("count", [128, 1000])
def test_reader_shuffle(benchmark: Benchmark, count: int) -> None:
    result = benchmark(shuffled_ordinals, count, 42)
    assert sorted(result) == list(range(count))


def test_completed_selection_read_after_eviction(benchmark: Benchmark, tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, cache_bytes=0) as db:
        query = (
            db.Corpus("bench", [p.Source(str(i), "a useful document") for i in range(1000)])
            .query()
            .wait()
        )
        with patch.object(
            coordinator(db), "_execute_query", side_effect=AssertionError("reexecuted")
        ):
            result = benchmark(
                lambda: db._query(query.id).preview(limit=10, offset=990, max_characters=128)
            )
        assert len(result) == 10
        assert all(row["text"] == "a useful document" for row in result)
