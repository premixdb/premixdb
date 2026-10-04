"""Browse full corpus history without running pending recipes or reading data."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Never
from unittest.mock import patch

import pytest
from _type_support import coordinator, invalid_call
from google.protobuf.message import Message

import premixdb as p
from premixdb._ids import _decode_id, _encode_id
from premixdb._resources import Dataset, Query
from premixdb._types import CorpusListing, SnapshotListing
from premixdb.execution.catalog_reader import Catalog
from premixdb.execution.storage import ObjectStore
from premixdb.v1 import corpus_pb2 as c
from premixdb.v1 import data_mixture_pb2 as d
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import snapshot_pb2 as s
from premixdb.v1 import status_pb2 as status


def corpus_handle(snapshot: p.Snapshot | p.Corpus) -> p.Corpus:
    if isinstance(snapshot, p.Corpus):
        return snapshot
    return p.Corpus(snapshot._db, c.Corpus(id=_decode_id(snapshot.corpus_id)))


def test_lists_cover_old_snapshots_and_unions_and_survive_reopening(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        before = datetime.now(timezone.utc)
        first = db.Corpus("c4", [p.Source("a", "first text")])
        old_query = first.query().wait()
        old_mixture = old_query.mix(tokenizer=p.ByteTokenizer(), sequence_length=4)
        old_dataset = old_mixture[0].wait()
        second = db.Corpus("c4", [p.Source("a", "new text")], base=first)
        other = db.Corpus("other", [p.Source("b", "other text")])
        union_query = first.union(second, other).query().wait()
        other_query = other.query().wait()
        mixture = union_query.mix(
            tokenizer=p.ByteTokenizer(),
            weights=p.RegMix(),
            tokens=4,
            sequence_length=4,
            n_candidates=2,
        )
        other_mixture = other_query.mix(tokenizer=p.ByteTokenizer(), tokens=4, sequence_length=4)
        assert mixture.datasets
        assert other_mixture.datasets
        after = datetime.now(timezone.utc)

        corpora = {row["name"]: row["id"] for row in db.Corpus.list(limit=1000)}
        assert corpora == {"c4": first.corpus_id, "other": other.corpus_id}
        assert db.Corpus("c4").id == second.id
        snapshots = corpus_handle(db.Corpus("c4")).list_snapshot(limit=1000)
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
        assert corpus_handle(db.Corpus("c4")).list_query(limit=1000) == sorted(
            [old_query.id, union_query.id]
        )
        assert corpus_handle(other).list_query(limit=1000) == sorted(
            [other_query.id, union_query.id]
        )
        assert {value.id for value in corpus_handle(second).list_mixture(limit=1000)} == {
            mixture.id,
            old_mixture.id,
        }
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
        again = db.Corpus("c4", [p.Source("a", "new text")], base=second)
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
            source = db.Corpus("c4")
            assert corpus_handle(source).list_snapshot(limit=1000) == snapshots
            assert corpus_handle(source).list_query(limit=1000) == sorted(
                [old_query.id, union_query.id]
            )
            assert {value.id for value in corpus_handle(source).list_mixture(limit=1000)} == {
                mixture.id,
                old_mixture.id,
            }
            assert {
                value.id for value in corpus_handle(source).list_dataset(limit=1000)
            } == expected_dataset_ids


def test_empty_corpus_and_closed_session(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        assert db.Corpus.list(limit=1000) == []
        empty = db._create_corpus("empty")
        assert db.Corpus.list(limit=1000) == [{"id": empty.id, "name": "empty"}]
        assert empty.list_snapshot(limit=1000) == []
        assert empty.list_query(limit=1000) == []
        assert empty.list_mixture(limit=1000) == []
        assert empty.list_dataset(limit=1000) == []
    with pytest.raises(ValueError, match="closed"):
        db.Corpus.list(limit=1000)
    with pytest.raises(ValueError, match="closed"):
        empty.list_snapshot(limit=1000)


@pytest.mark.parametrize("read_only", [False, True])
def test_named_corpus_reads_one_head_and_latest_refreshes(tmp_path: Path, read_only: bool) -> None:
    with p.PremixDB(storage=tmp_path) as writer:
        first = writer.Corpus("head", [p.Source("a", "first")])
        with p.PremixDB(storage=tmp_path, read_only=read_only) as reader:
            handle = p.Corpus(
                reader,
                c.Corpus(
                    id=_decode_id(first.corpus_id),
                    name="head",
                    latest_snapshot_id=_decode_id(first.id),
                ),
            )
            with patch.object(
                reader._executor, "GetCorpus", wraps=reader._executor.GetCorpus
            ) as get:
                reopened = reader.Corpus("head")
                assert reopened.id == first.id
                assert get.call_count == 1
            second = writer.Corpus("head", [p.Source("a", "second")], base=first)
            assert handle.latest().id == second.id
            assert reopened.preview()[0]["text"] == "first"


def test_saved_resource_reads_reject_a_closed_session(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        corpus = db._create_corpus("closed")
    for read in (
        lambda: db.Corpus("closed"),
        corpus.latest,
        lambda: db._snapshot(b"s" * 32),
        lambda: db._query(b"q" * 32),
        lambda: db._dataset(b"d" * 32),
        lambda: db._mix(b"m" * 32),
    ):
        with pytest.raises(ValueError, match="PremixDB is closed"):
            read()


@pytest.mark.parametrize("read_only", [False, True])
def test_unwritable_sessions_reject_recipes_before_consuming_input(
    tmp_path: Path, read_only: bool
) -> None:
    class UnusedInput:
        def __iter__(self) -> Iterator[Never]:
            raise AssertionError("recipe consumed input")

    with p.PremixDB(storage=tmp_path) as writer:
        writer.Corpus("saved", [p.Source("a", "saved text")])
    with p.PremixDB(storage=tmp_path, read_only=read_only) as db:
        snapshot = db.Corpus("saved")
        corpus = corpus_handle(snapshot)
        if not read_only:
            db.close()
        error = PermissionError if read_only else ValueError
        message = "read-only" if read_only else "PremixDB is closed"
        with patch.object(
            p.HuggingFaceSource,
            "_to_proto",
            side_effect=AssertionError("recipe resolved Hub metadata"),
        ):
            for source in (UnusedInput(), p.HuggingFaceSource("org/data")):
                with pytest.raises(error, match=message):
                    corpus.snapshot(source=source)
        with pytest.raises(error, match=message):
            snapshot.query(steps=UnusedInput())


@pytest.mark.parametrize("exists", [False, True])
def test_named_corpus_requires_a_successful_snapshot(tmp_path: Path, exists: bool) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        if exists:
            db._create_corpus("empty")
        before = db.Corpus.list()
        with pytest.raises(ValueError, match="has no snapshot; capture with db.Corpus"):
            db.Corpus("empty")
        assert db.Corpus.list() == before


@pytest.mark.parametrize(
    "kind,response",
    [
        ("Corpus", c.GetCorpusResponse(corpus=c.Corpus(id=b"x" * 16))),
        ("Snapshot", s.GetSnapshotResponse(snapshot=s.Snapshot(id=b"x" * 32))),
        ("Query", q.GetQueryResponse(query=q.Query(id=b"x" * 32))),
        ("Dataset", d.GetDatasetResponse(dataset=d.Dataset(id=b"x" * 32))),
        ("Mix", d.GetMixResponse(mix=d.Mix(id=b"x" * 32))),
    ],
)
def test_saved_resource_reads_verify_catalog_identity(
    tmp_path: Path,
    kind: Literal["Corpus", "Snapshot", "Query", "Dataset", "Mix"],
    response: Message,
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        identity = b"y" * (16 if kind == "Corpus" else 32)
        with patch.object(db._executor, "Get" + kind, return_value=response):
            with pytest.raises(ValueError, match="catalog returned a different resource"):
                db._get(kind, identity)


def test_paginated_lists_include_pending_and_failed_results(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.Corpus("many", [p.Source("a", "text")])
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
        assert len(db.Corpus.list(limit=1000)) == 131
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
            db.Corpus.list,
            corpus_handle(snapshot).list_snapshot,
            corpus_handle(snapshot).list_query,
            corpus_handle(snapshot).list_mixture,
            corpus_handle(snapshot).list_dataset,
        ):
            complete = listing(limit=1000)

            def identities(
                values: Iterable[p.DataMixture | p.Dataset | str | CorpusListing | SnapshotListing],
            ) -> list[str | CorpusListing | SnapshotListing]:
                return [
                    value.id if isinstance(value, (p.DataMixture, p.Dataset)) else value
                    for value in values
                ]

            assert len(listing()) == 5
            assert identities(listing(limit=3, offset=4)) == identities(complete[4:7])
            assert identities(listing(limit=10, offset=125)) == identities(complete[125:135])
            assert listing(limit=0) == []
            assert listing(offset=1000) == []


def test_dataset_listing_follows_all_pages_for_one_query(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.Corpus("c4", [p.Source("a", "text")])
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
        snapshot = db.Corpus("c4", [p.Source("a", "abcd")])
        query = snapshot.query().wait()
        mixture = query.mix(tokenizer=p.ByteTokenizer(), tokens=4, sequence_length=4)
        assert mixture.datasets
        snapshots = corpus_handle(snapshot).list_snapshot(limit=1000)
    with p.PremixDB(storage=tmp_path, read_only=True) as db:
        assert isinstance(db._executor, Catalog)
        with patch.object(
            db._executor._storage, "_get", side_effect=AssertionError("read bulk data")
        ):
            assert db.Corpus.list(limit=1000) == [{"id": snapshot.corpus_id, "name": "c4"}]
            source = db.Corpus("c4")
            assert corpus_handle(source).list_snapshot(limit=1000) == snapshots
            assert corpus_handle(source).list_query(limit=1000) == [query.id]
            assert {value.id for value in corpus_handle(source).list_mixture(limit=1000)} == {
                mixture.id
            }
            assert len(corpus_handle(source).list_dataset(limit=1000)) == 1


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
        snapshot = db.Corpus("documents", [p.Source(str(i), "text") for i in range(8)])
        corpus = db._create_corpus("documents")
        query = snapshot.query().wait()
        complete = corpus.list_document(limit=100)
        for listing in (corpus.list_document,):
            assert listing() == complete[:5]
            assert listing(limit=2, offset=5) == complete[5:7]
        assert [row["id"] for row in query.preview(limit=2, offset=5, max_characters=0)] == [
            row["id"] for row in complete[5:7]
        ]
        for listing in (
            db.Corpus.list,
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


def test_catalog_windows_load_only_requested_payloads(tmp_path: Path) -> None:
    from premixdb.execution import metadata

    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.Corpus("indexed", [p.Source("a", "text")])
        corpus = db._create_corpus("indexed")
        store = coordinator(db)._storage
        for i in range(260):
            query = q.Query(id=(i + 1).to_bytes(32), snapshot_ids=[_decode_id(snapshot.id)])
            store.save("query", query.id, query, suffix=".pending")
            dataset = d.Dataset(id=(i + 1).to_bytes(32), query_id=query.id)
            store.save("dataset", dataset.id, dataset, suffix=".recipe")
        with (
            patch.object(store, "list", side_effect=AssertionError("scanned payloads")),
            patch.object(metadata, "_decode", wraps=metadata._decode) as decode,
        ):
            assert len(corpus.list_query(limit=3, offset=200)) == 3
            assert decode.call_count == 0
            assert len(corpus.list_dataset(limit=3, offset=200)) == 3
            assert decode.call_count == 3
            decode.reset_mock()
            assert len(db.Corpus.list(limit=1)) == 1
            assert decode.call_count == 1
        with patch.object(store.metadata, "members", wraps=store.metadata.members) as members:
            first = db._executor.ListQuery(q.ListQueryRequest())
            second = db._executor.ListQuery(q.ListQueryRequest(page_token=first.next_page_token))
            assert len(first.queries) == len(second.queries) == 128
            assert members.call_count == 1


def test_continuation_detects_membership_changes_from_another_connection(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        catalog = Catalog(store)
        for i in range(130):
            value = d.Mix(id=(i + 1).to_bytes(32), query_id=b"q" * 32)
            store.save("mixture", value.id, value)
        first = catalog.ListMix(d.ListMixRequest(query_id=b"q" * 32))
        with ObjectStore(tmp_path) as writer:
            value = d.Mix(id=b"z" * 32, query_id=b"q" * 32)
            writer.save("mixture", value.id, value)
        with pytest.raises(ValueError, match="page token"):
            catalog.ListMix(d.ListMixRequest(query_id=b"q" * 32, page_token=first.next_page_token))
        assert len(catalog.ListMix(d.ListMixRequest(query_id=b"q" * 32)).mixtures) == 128


def test_catalog_state_precedence_is_applied_before_parent_filtering(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        catalog = Catalog(store)
        identity = b"q" * 32
        pending = q.Query(id=identity, snapshot_ids=[b"a" * 32], status=status.STATUS_PENDING)
        failed = q.Query(id=identity, snapshot_ids=[b"a" * 32], status=status.STATUS_ERROR)
        store.save("query", identity, pending, suffix=".pending")
        store.save("query", identity, failed, suffix=".failed")
        assert catalog.ListQuery(q.ListQueryRequest()).queries[0].status == status.STATUS_ERROR
        active = q.Query(id=identity, snapshot_ids=[b"a" * 32], status=status.STATUS_RUNNING)
        catalog._queries[identity] = active
        assert catalog.ListQuery(q.ListQueryRequest()).queries[0].status == status.STATUS_RUNNING
        complete = q.Query(id=identity, snapshot_ids=[b"b" * 32], status=status.STATUS_COMPLETED)
        store.save("query", identity, complete)
        assert catalog.ListQuery(q.ListQueryRequest()).queries[0].status == status.STATUS_COMPLETED
        assert not catalog.ListQuery(q.ListQueryRequest(snapshot_id=b"a" * 32)).queries
        assert catalog.ListQuery(q.ListQueryRequest(snapshot_id=b"b" * 32)).queries == [complete]


def test_old_catalog_lists_read_only_and_backfills_on_writable_open(tmp_path: Path) -> None:
    import sqlite3
    from contextlib import closing

    with p.PremixDB(storage=tmp_path) as db:
        first = db.Corpus("legacy-index", [p.Source("a", "first")])
        query = first.query().wait()
        dataset = query.mix(tokenizer=p.ByteTokenizer(), sequence_length=2)[0].wait()
        second = db.Corpus("legacy-index", [p.Source("a", "second")], base=first)
        corpus = db._create_corpus("legacy-index")
        snapshots = corpus.list_snapshot()
        queries = corpus.list_query()
        mixtures = [value.id for value in corpus.list_mixture()]
        datasets = [value.id for value in corpus.list_dataset()]
    with closing(sqlite3.connect(tmp_path / "metadata.sqlite3")) as connection, connection:
        connection.execute("DROP TABLE catalog_parents")
        connection.execute("DELETE FROM migrations WHERE name='catalog-parents-v1'")
    for read_only in (True, False, True):
        with p.PremixDB(storage=tmp_path, read_only=read_only) as db:
            corpus = p.Corpus(db, c.Corpus(id=_decode_id(first.corpus_id)))
            assert corpus.list_snapshot() == snapshots
            assert corpus.list_query() == queries
            assert [value.id for value in corpus.list_mixture()] == mixtures
            assert [value.id for value in corpus.list_dataset()] == datasets == [dataset.id]
            assert db.Corpus("legacy-index").id == second.id
            assert db._executor.ListExecutions(status.ListExecutionRequest()).events


def test_continuation_scope_keeps_the_revision_of_its_membership_snapshot(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store, ObjectStore(tmp_path) as writer:
        catalog = Catalog(store)
        for i in range(130):
            value = d.Mix(id=(i + 1).to_bytes(32), query_id=b"q" * 32)
            store.save("mixture", value.id, value)
        read_members = store.metadata.members

        def publish_after_read(
            namespace: str,
            *,
            suffixes: tuple[str, ...],
            parents: tuple[bytes, ...],
            order: Literal["id", "public", "capture", "execution"],
        ) -> list[tuple[bytes, str, int | None]]:
            members = read_members(namespace, suffixes=suffixes, parents=parents, order=order)
            value = d.Mix(id=b"z" * 32, query_id=b"q" * 32)
            writer.save("mixture", value.id, value)
            return members

        with patch.object(store.metadata, "members", side_effect=publish_after_read):
            first = catalog.ListMix(d.ListMixRequest(query_id=b"q" * 32))
        assert len(first.mixtures) == 128
        with pytest.raises(ValueError, match="page token"):
            catalog.ListMix(d.ListMixRequest(query_id=b"q" * 32, page_token=first.next_page_token))
