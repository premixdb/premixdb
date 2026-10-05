"""Indexed execution must preserve predicate, population and durable-result semantics."""

from __future__ import annotations

import operator
import statistics
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from _type_support import coordinator
from pyroaring import BitMap
from test_enrichment_service import ControlledFields

import premixdb as db
from premixdb.engine.analytics import Column, Scalar
from premixdb.internal import analytics_pb2 as a
from premixdb.runtime import Coordinator, compile_query
from premixdb.runtime.profiles import output_profiles
from premixdb.schemas.ids import _decode_id
from premixdb.storage.analytics import (
    SELECTION_SUFFIX,
    load_column,
    profile_index,
    save_column,
    validate_index,
)
from premixdb.storage.profiles import ProfileKind, endpoint, value
from premixdb.v1 import profile_pb2 as p
from premixdb.v1 import query_pb2 as q

OPS = {
    q.Comparison.OPERATOR_EQ: operator.eq,
    q.Comparison.OPERATOR_NE: operator.ne,
    q.Comparison.OPERATOR_LT: operator.lt,
    q.Comparison.OPERATOR_LE: operator.le,
    q.Comparison.OPERATOR_GT: operator.gt,
    q.Comparison.OPERATOR_GE: operator.ge,
}


@pytest.mark.parametrize(
    "kind,values,thresholds",
    [
        (
            "integer",
            [None, -(2**63), -1, 0, 2**53, 2**53 + 1, 2**60, 2**60 + 1, 2**63 - 1],
            [-(2**63), -0.5, 0, 2**53, float(2**53), 2**53 + 1, 2**60 + 1, float(2**63)],
        ),
        (
            "count",
            [None, 0, 2**53, 2**53 + 1, 2**63, 2**64 - 1],
            [-1, 0, float(2**53), 2**53 + 1, 2**64 - 1, 2**64],
        ),
        (
            "number",
            [None, -1.0, -0.0, 0.1, 0.5, 1.0, float(2**53)],
            [-1, 0, 0.1, np.nextafter(0.1, 1.0), 2**53 + 1],
        ),
        ("boolean", [None, False, True, False], [False, True]),
        ("text", [None, "", "a", "é", "🌍", "a"], ["", "a", "b", "é", "🌍"]),
        ("number", [None, None], [-1.0, 0.0, 1.0]),
        ("integer", [], [-1, 0, 1]),
    ],
)
def test_lossless_comparisons(
    kind: ProfileKind, values: list[Scalar | None], thresholds: list[Scalar]
) -> None:
    column = Column.build(values, kind)
    selections = [BitMap(range(len(values))), BitMap(range(0, len(values), 2)), BitMap()]
    for selected in selections:
        for op, compare in OPS.items():
            for threshold in thresholds:
                expected = BitMap(
                    i
                    for i in selected
                    if (item := values[i]) is not None and compare(item, threshold)
                )
                assert column.select(selected, op, threshold) == expected


@pytest.mark.parametrize("fraction", [0.01, 0.5, 0.97, 1.0])
def test_adaptive_profiles_count_actual_membership(fraction: float) -> None:
    rng = np.random.default_rng(127)
    values = rng.normal(size=16_000)
    nullable: list[Scalar | None] = [
        None if i % 97 == 0 else float(x) for i, x in enumerate(values)
    ]
    column = Column.build(nullable, "number")
    selected = BitMap(i for i in range(len(values)) if rng.random() < fraction)
    actual = [float(x) for i in selected if (x := nullable[i]) is not None]
    profile = column.distribution(selected)
    assert sum(b.documents for b in profile.buckets) == len(actual)
    assert profile.numeric.documents == len(actual)
    assert profile.numeric.total == pytest.approx(sum(actual), abs=1e-10)
    assert profile.numeric.mean == pytest.approx(statistics.mean(actual), abs=1e-12)
    assert profile.numeric.standard_deviation == pytest.approx(statistics.pstdev(actual), abs=1e-12)
    assert value(profile.numeric.minimum) == min(actual)
    assert value(profile.numeric.maximum) == max(actual)
    for bucket in profile.buckets:
        low, high = value(bucket.lower), value(bucket.upper)
        assert isinstance(low, (int, float)) and isinstance(high, (int, float))
        assert bucket.documents == sum(low <= x <= high for x in actual)


def test_blocks_keep_histogram_detail_and_large_integer_variance(tmp_path: Path) -> None:
    origin = 2**60
    layout = p.FieldDistribution(
        buckets=[
            p.ProfileBucket(
                lower=endpoint("integer", origin + i * 200),
                upper=endpoint("integer", origin + (i + 1) * 200 - 1),
            )
            for i in range(60)
        ]
    )
    values = [origin + i for i in range(12_000)]
    with Coordinator(tmp_path) as service:
        indexed = a.FieldIndex(
            full=p.FieldProfile(field=q.FIELD_DATATROVE_LENGTH, documents=len(values))
        )
        projection = indexed.projections.add(projection=p.FieldDistribution.SCALAR)
        for first in (0, 6_000):
            projection.blocks.append(
                save_column(
                    service._storage,
                    Column.build(values[first : first + 6_000], "integer", layout),
                    first,
                )
            )
        for selected in (BitMap(range(12_000)), BitMap(range(0, 12_000, 2))):
            distribution = profile_index(service, indexed, selected, use_full=False).distributions[
                0
            ]
            actual = [values[i] for i in selected]
            assert 60 <= len(distribution.buckets) <= 64
            assert distribution.numeric.standard_deviation == pytest.approx(
                statistics.pstdev(actual)
            )
            assert value(distribution.numeric.minimum) == min(actual)
            assert value(distribution.numeric.maximum) == max(actual)
        broken = a.FieldIndex()
        broken.CopyFrom(indexed)
        broken.projections[0].blocks[1].first += 1
        with pytest.raises(ValueError, match="coverage"):
            validate_index(broken, 12_000)


def test_indexed_queries_match_existing_executor_and_need_no_value_scans(tmp_path: Path) -> None:
    with db.PremixDB(storage=tmp_path) as client:
        snapshot = client.Corpus(
            "indexed",
            [
                db.Source(f"https://example.org/{i}", "" if i % 7 == 0 else "é" * (i % 11))
                for i in range(91)
            ],
        )
        with patch("premixdb.runtime.enrichment.producer", ControlledFields):
            snapshot.query()._with_fields(
                [db.language.en, db.quality.educational_value, db.topic.label]
            ).wait()
        service = coordinator(client)
        for steps in (
            [db.where(db.language.en != 0.95)],
            [db.where(db.language.en.is_null())],
            [db.where(db.text.bytes > 8), db.where(db.quality.educational_value < 0.5)],
            [db.where(db.quality.educational_value < 0.5), db.where(db.text.bytes > 8)],
            [db.where(db.topic[db.Topic.SCIENCE_AND_TECH] > 0.5)],
            [db.where(db.language.en > 0.99)],
        ):
            with patch(
                "premixdb.runtime.enrichment.read_rows", side_effect=AssertionError("value scan")
            ):
                query = snapshot.query(steps=steps).wait()
            handle = service._query(_decode_id(query.id))
            with patch("premixdb.runtime.analytics.execute", return_value=None):
                _, reference = service._execute_query(compile_query(query._proto), publish=False)
            assert handle.summary() == reference.summary()
            assert [row.id for row in handle] == [row.id for row in reference]
            assert handle.provenance() == reference.provenance()
            assert list(query.profile().fields) == output_profiles(service, query._proto, reference)
        query_id, profile = query.id, query.profile()
    with db.PremixDB(storage=tmp_path, read_only=True) as client:
        assert client._query(query_id).profile() == profile


def test_eviction_restart_late_preview_and_new_queries_use_prepared_population(
    tmp_path: Path,
) -> None:
    with db.PremixDB(storage=tmp_path) as client:
        snapshot = client.Corpus("lazy", [db.Source(str(i), "é" * (i + 1)) for i in range(37)])
        query = snapshot.query(steps=[db.where(db.text.bytes > 8)]).wait()
        preview = query.preview(offset=11, limit=4, max_characters=2)
        lineage = query._provenance()
        query_id, snapshot_id = query.id, snapshot.id
        selection = coordinator(client)._storage.load(
            "query", _decode_id(query_id), a.IndexedSelection, suffix=SELECTION_SUFFIX
        )
        assert selection.ByteSize() < 1024
    with db.PremixDB(storage=tmp_path, cache_bytes=0) as client:
        with patch.object(
            coordinator(client), "_snapshot", side_effect=AssertionError("population scan")
        ):
            assert client._query(query_id).preview(offset=11, limit=4, max_characters=2) == preview
            assert client._query(query_id)._provenance() == lineage
            assert (
                client._snapshot(snapshot_id)
                .query(steps=[db.where(db.text.bytes > 10)])
                .profile()
                .output_documents
                == 32
            )
    with db.PremixDB(storage=tmp_path, read_only=True) as client:
        with patch("premixdb.runtime.analytics.execute", side_effect=AssertionError("kernel")):
            assert client._query(query_id).preview(offset=11, limit=4, max_characters=2) == preview


def test_corrupted_prepared_column_fails_instead_of_rebuilding(tmp_path: Path) -> None:
    with Coordinator(tmp_path) as service:
        column = Column.build([0.1, None, 0.5], "number")
        block = save_column(service._storage, column, 0)
        path = tmp_path / "index/objects" / block.values.blake3_digest.hex()
        payload = bytearray(path.read_bytes())
        payload[len(payload) // 2] ^= 1
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="integrity"):
            load_column(service, block)
