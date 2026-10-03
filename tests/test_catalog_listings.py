"""Browse full corpus history without running pending recipes or reading data."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import coordinator, invalid_call

import premixdb as p
import premixdb as sdk
from premixdb._ids import _decode_id, _encode_id
from premixdb._resources import Dataset, Query
from premixdb._types import CorpusListing, SnapshotListing
from premixdb.execution.catalog_reader import Catalog
from premixdb.execution.storage import ObjectStore
from premixdb.v1 import corpus_pb2 as c
from premixdb.v1 import dataset_pb2 as d
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import snapshot_pb2 as s
from premixdb.v1 import status_pb2 as status


def corpus_handle(snapshot: sdk.Snapshot | sdk.Corpus) -> p.Corpus:
    if isinstance(snapshot, p.Corpus):
        return snapshot
    return p.Corpus(snapshot._db, c.Corpus(id=_decode_id(snapshot.corpus_id)))


def test_lists_cover_old_snapshots_and_unions_and_survive_reopening(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        before = datetime.now(timezone.utc)
        first = db.corpus("c4", [p.Source("a", "first text")])
        old_query = first.query().wait()
        old_dataset = old_query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=4).wait()
        second = db.corpus("c4", [p.Source("a", "new text")], base=first)
        other = db.corpus("other", [p.Source("b", "other text")])
        union_query = first.union(second, other).query().wait()
        other_query = other.query().wait()
        mixture = union_query.mix(
            tokenizer=p.ByteTokenizer(), tokens=4, sequence_length=4, n_candidates=2
        )
        other_mixture = other_query.mix(tokenizer=p.ByteTokenizer(), tokens=4, sequence_length=4)
        after = datetime.now(timezone.utc)

        corpora = {row["name"]: row["id"] for row in db.corpus.list(limit=1000)}
        assert corpora == {"c4": first.corpus_id, "other": other.corpus_id}
        assert db.corpus("c4").id == second.id
        snapshots = corpus_handle(db.corpus("c4")).list_snapshot(limit=1000)
        assert [row["id"] for row in snapshots] == [first.id, second.id]
        for row in snapshots:
            assert str(row["timestamp"]).endswith("Z")
            assert (
                before.replace(microsecond=0)
                <= datetime.fromisoformat(str(row["timestamp"]))
                <= after
            )
        assert corpus_handle(first).list_snapshot(limit=1000) == snapshots
        assert db._create_corpus("c4").list_snapshot(limit=1000) == snapshots
        assert corpus_handle(db.corpus("c4")).list_query(limit=1000) == sorted(
            [old_query.id, union_query.id]
        )
        assert corpus_handle(other).list_query(limit=1000) == sorted(
            [other_query.id, union_query.id]
        )
        assert [value.id for value in corpus_handle(second).list_mixture(limit=1000)] == [
            mixture.id
        ]
        assert {value.id for value in corpus_handle(other).list_mixture(limit=1000)} == {
            mixture.id,
            other_mixture.id,
        }
        datasets = corpus_handle(second).list_dataset(limit=1000)
        assert {value.id for value in datasets} == {
            old_dataset.id,
            *[_encode_id(id) for id in mixture._proto.dataset_ids],
        }
        assert all(
            value.status is p.ExecutionStatus.PENDING
            for value in datasets
            if value.id != old_dataset.id
        )

        # An unchanged recapture is the same snapshot with the same first timestamp.
        again = db.corpus("c4", [p.Source("a", "new text")], base=second)
        assert again.id == second.id
        assert corpus_handle(again).list_snapshot(limit=1000) == snapshots
        expected_dataset_ids = {value.id for value in datasets}

    with p.PremixDB(storage=tmp_path, read_only=True) as db:
        with (
            patch.object(db, "_submit", side_effect=AssertionError("listing submitted work")),
            patch.object(
                db._object_reader, "read", side_effect=AssertionError("listing read data")
            ),
            patch.object(Query, "wait", side_effect=AssertionError("listing waited for a query")),
            patch.object(Dataset, "wait", side_effect=AssertionError("listing packed a dataset")),
            patch.object(db, "_dataset", side_effect=AssertionError("listing fetched a candidate")),
        ):
            source = db.corpus("c4")
            assert corpus_handle(source).list_snapshot(limit=1000) == snapshots
            assert corpus_handle(source).list_query(limit=1000) == sorted(
                [old_query.id, union_query.id]
            )
            assert [value.id for value in corpus_handle(source).list_mixture(limit=1000)] == [
                mixture.id
            ]
            assert {
                value.id for value in corpus_handle(source).list_dataset(limit=1000)
            } == expected_dataset_ids


def test_empty_corpus_and_closed_session(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        assert db.corpus.list(limit=1000) == []
        empty = db._create_corpus("empty")
        assert db.corpus.list(limit=1000) == [{"id": empty.id, "name": "empty"}]
        assert empty.list_snapshot(limit=1000) == []
        assert empty.list_query(limit=1000) == []
        assert empty.list_mixture(limit=1000) == []
        assert empty.list_dataset(limit=1000) == []
    with pytest.raises(ValueError, match="closed"):
        db.corpus.list(limit=1000)
    with pytest.raises(ValueError, match="closed"):
        empty.list_snapshot(limit=1000)


def test_paginated_lists_include_pending_and_failed_results(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("many", [p.Source("a", "text")])
        store = coordinator(db)._storage
        for i in range(130):
            corpus = c.Corpus(id=(i + 1).to_bytes(16), name=f"empty-{i}")
            store.save("corpus", corpus.id, corpus)
            snap = s.Snapshot(
                id=(i + 1).to_bytes(32),
                corpus_id=_decode_id(snapshot.corpus_id),
                status=status.STATUS_COMPLETED,
            )
            store.save("snapshot", snap.id, snap)
            query = q.Query(
                id=(i + 1).to_bytes(32),
                snapshot_ids=[snap.id, _decode_id(snapshot.id)],
                status=status.STATUS_PENDING,
            )
            store.save("query", query.id, query, suffix=".pending")
            mix = d.Mix(id=(i + 1).to_bytes(32), query_id=query.id)
            store.save("mixture", mix.id, mix)
            dataset = d.Dataset(
                id=(i + 1).to_bytes(32), query_id=query.id, status=status.STATUS_PENDING
            )
            store.save("dataset", dataset.id, dataset, suffix=".recipe")
        failed = q.Query(
            id=b"f" * 32, snapshot_ids=[_decode_id(snapshot.id)], status=status.STATUS_ERROR
        )
        store.save("query", failed.id, failed, suffix=".failed")
        assert len(db.corpus.list(limit=1000)) == 131
        assert len(corpus_handle(snapshot).list_snapshot(limit=1000)) == 131
        assert len(corpus_handle(snapshot).list_query(limit=1000)) == 131
        assert len(corpus_handle(snapshot).list_mixture(limit=1000)) == 130
        assert len(corpus_handle(snapshot).list_dataset(limit=1000)) == 130
        # No execution history is invented for old imported snapshots.
        assert (
            sum(
                row["timestamp"] is None
                for row in corpus_handle(snapshot).list_snapshot(limit=1000)
            )
            == 130
        )
        assert len(set(corpus_handle(snapshot).list_query(limit=1000))) == 131
        for listing in (
            db.corpus.list,
            corpus_handle(snapshot).list_snapshot,
            corpus_handle(snapshot).list_query,
            corpus_handle(snapshot).list_mixture,
            corpus_handle(snapshot).list_dataset,
        ):
            complete = listing(limit=1000)

            def identities(
                values: Iterable[sdk.Mix | sdk.Dataset | str | CorpusListing | SnapshotListing],
            ) -> list[str | CorpusListing | SnapshotListing]:
                return [
                    value.id if isinstance(value, (p.Mix, p.Dataset)) else value for value in values
                ]

            assert len(listing()) == 5
            assert identities(listing(limit=3, offset=4)) == identities(complete[4:7])
            assert identities(listing(limit=10, offset=125)) == identities(complete[125:135])
            assert listing(limit=0) == []
            assert listing(offset=1000) == []


def test_dataset_listing_follows_all_pages_for_one_query(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("c4", [p.Source("a", "text")])
        query = snapshot.query().wait()
        for i in range(130):
            dataset = d.Dataset(
                id=(i + 1).to_bytes(32),
                query_id=_decode_id(query.id),
                status=status.STATUS_PENDING,
            )
            coordinator(db)._storage.save("dataset", dataset.id, dataset, suffix=".recipe")
        assert len(corpus_handle(snapshot).list_dataset(limit=1000)) == 130


def test_execution_history_rejects_repeated_continuation_tokens(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        with patch.object(
            db._executor,
            "ListExecutions",
            return_value=status.ListExecutionResponse(next_page_token=b"repeat"),
        ):
            with pytest.raises(ValueError, match="repeated a continuation token"):
                db._execution_events()


def test_read_only_catalog_listings_use_metadata_only(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("c4", [p.Source("a", "abcd")])
        query = snapshot.query().wait()
        mixture = query.mix(tokenizer=p.ByteTokenizer(), tokens=4, sequence_length=4)
        snapshots = corpus_handle(snapshot).list_snapshot(limit=1000)
    with p.PremixDB(storage=tmp_path, read_only=True) as db:
        assert isinstance(db._executor, Catalog)
        with patch.object(
            db._executor._storage, "_get", side_effect=AssertionError("read bulk data")
        ):
            assert db.corpus.list(limit=1000) == [{"id": snapshot.corpus_id, "name": "c4"}]
            source = db.corpus("c4")
            assert corpus_handle(source).list_snapshot(limit=1000) == snapshots
            assert corpus_handle(source).list_query(limit=1000) == [query.id]
            assert [value.id for value in corpus_handle(source).list_mixture(limit=1000)] == [
                mixture.id
            ]
            assert len(corpus_handle(source).list_dataset(limit=1000)) == 3


def test_mixture_query_filter_and_page_tokens_are_scoped(tmp_path: Path) -> None:
    catalog = Catalog(ObjectStore(tmp_path))
    try:
        for i in range(130):
            mix = d.Mix(id=(i + 1).to_bytes(32), query_id=b"q" * 32)
            catalog._storage.metadata.save("mixture", mix.id, mix)
        first = catalog.ListMix(d.ListMixRequest(query_id=b"q" * 32))
        assert len(first.mixtures) == 128
        second = catalog.ListMix(
            d.ListMixRequest(query_id=b"q" * 32, page_token=first.next_page_token)
        )
        assert len(second.mixtures) == 2
        with pytest.raises(ValueError, match="page token"):
            catalog.ListMix(d.ListMixRequest(query_id=b"x" * 32, page_token=first.next_page_token))
        with pytest.raises(ValueError, match="32 bytes"):
            catalog.ListMix(d.ListMixRequest(query_id=b"short"))
    finally:
        catalog.close()


def test_listing_arguments_and_document_defaults(tmp_path: Path) -> None:
    db = p.PremixDB(storage=tmp_path)
    try:
        snapshot = db.corpus("documents", [p.Source(str(i), "text") for i in range(8)])
        corpus = db._create_corpus("documents")
        query = snapshot.query().wait()
        complete = corpus.list_document(limit=100)
        for listing in (corpus.list_document, query._list_document):
            assert listing() == complete[:5]
            assert listing(limit=2, offset=5) == complete[5:7]
        for listing in (
            db.corpus.list,
            corpus_handle(corpus).list_snapshot,
            corpus_handle(corpus).list_query,
            corpus_handle(corpus).list_mixture,
            corpus_handle(corpus).list_dataset,
            corpus.list_document,
        ):
            for limit, offset in ((-1, 0), (True, 0), (1.5, 0), (1, -1), (1, True)):
                with pytest.raises(ValueError):
                    invalid_call(listing, limit=limit, offset=offset)
    finally:
        db.close()
