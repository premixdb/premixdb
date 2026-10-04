"""Local and partitioned evidence preserve exact UTF-8 groups and stream ownership."""

import tracemalloc
from contextlib import closing
from pathlib import Path
from sqlite3 import ProgrammingError

import pytest

from premixdb.engine.identity import CodeVersion
from premixdb.engine.queries import CorpusIndex
from premixdb.engine.snapshots import Snapshot
from premixdb.engine.spill import classes
from premixdb.runtime.partitions import PartitionStore
from premixdb.runtime.pipeline import PartitionPipeline


def test_large_document_grouping_retains_one_encoded_copy() -> None:
    size = 4 * 1024 * 1024
    snapshot = Snapshot(
        "01" * 16,
        [("large", "x" * size)],
        CodeVersion("local://test", "a" * 40, "09" * 32),
    )
    index = CorpusIndex([snapshot])
    identity = next(iter(index.documents))
    tracemalloc.start()
    try:
        with closing(classes(index, "Document")) as groups:
            assert list(next(groups)) == [(identity, 0, size)]
            assert next(groups, None) is None
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 5 * 1024 * 1024


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("unit", ["Document", "Line"])
def test_exact_groups_preserve_ranges_repetition_and_database_lifetime(
    tmp_path: Path, partitioned: bool, unit: str
) -> None:
    snapshot = Snapshot(
        "01" * 16,
        [("a", "\né\né\n"), ("b", "é"), ("c", "é\r"), ("d", ""), ("e", "é")],
        CodeVersion("local://test", "a" * 40, "09" * 32),
    )
    index = CorpusIndex([snapshot])
    ids = {doc.source_key: doc.id for doc in index.documents.values()}
    expected = (
        [
            [(ids["a"], 0, 7)],
            [(ids["b"], 0, 2), (ids["e"], 0, 2)],
            [(ids["c"], 0, 3)],
            [(ids["d"], 0, 0)],
        ]
        if unit == "Document"
        else [
            [(ids["a"], 1, 3), (ids["a"], 4, 6), (ids["b"], 0, 2), (ids["e"], 0, 2)],
            [(ids["c"], 0, 3)],
        ]
    )
    expected = sorted(sorted(group) for group in expected)
    assert sorted(sorted(group) for group in index.classes(unit)) == expected
    pipeline = PartitionPipeline(PartitionStore(tmp_path)) if partitioned else None
    try:
        provider = classes if pipeline is None else pipeline.exact_classes
        with closing(provider(index, unit)) as groups:
            actual = []
            for group in groups:
                values = list(group)
                assert list(group) == values
                assert values == sorted(values)
                actual.append(values)
            assert sorted(actual) == expected
        with pytest.raises(ProgrammingError, match="closed"):
            list(group)

        # Closing an incomplete stream releases its database while a group is retained.
        with closing(provider(index, unit)) as groups:
            retained = next(groups)
            assert list(retained)
        with pytest.raises(ProgrammingError, match="closed"):
            list(retained)
    finally:
        if pipeline is not None:
            pipeline.close()
