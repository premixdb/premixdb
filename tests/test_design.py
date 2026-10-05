"""Cross-layer compatibility, bounded retention, and reusable derivation outcomes."""

from __future__ import annotations

import gc
import tempfile
import unittest
import weakref
from collections.abc import MutableMapping, Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import _reference as direct
from _type_support import coordinator

import premixdb
from premixdb.engine import datasets as engine_datasets
from premixdb.engine import execution
from premixdb.engine import queries as engine_query
from premixdb.engine.plans import external_filter, filter
from premixdb.enrichment import DupekitIndex
from premixdb.enrichment.types import ComputedRow, Document, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.internal import derivation_pb2 as d
from premixdb.runtime import catalog, enrichment
from premixdb.runtime import environment as _runtime
from premixdb.runtime.materialization import SingleFlight
from premixdb.schemas import requests as _requests
from premixdb.schemas.ids import _decode_id
from premixdb.storage.cache import MemoryCache
from premixdb.storage.objects import ObjectStore
from premixdb.v1 import query_pb2 as query_pb
from premixdb.v1 import status_pb2 as status_pb


class CoreDesignTests(unittest.TestCase):
    def test_forged_typed_policies_fail_even_for_empty_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code = _runtime.current_code()
            snapshot = execution.Store(directory).capture("01" * 16, [], code)
            for step in (
                replace(filter("bytes", "gt", 0), encoding=b"forged"),
                replace(filter("bytes", "gt", 0), value=True),
                replace(external_filter(b"d" * 32), members=frozenset({"invalid"})),
            ):
                with self.assertRaises(ValueError):
                    execution.execute([snapshot], [step], code)

    def test_sdk_and_direct_adapters_share_policies_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sources = [
                premixdb.Source("repo/a", "same"),
                direct.Source("repo/b", "same"),
                premixdb.Source("other/c", "a longer text"),
                premixdb.Source("empty", ""),
            ]
            local_snapshot = (
                direct.PremixDB(storage=Path(directory) / "direct")
                .corpus("shared")
                .snapshot(source=sources)
            )
            packing = direct.Concat(separator=256, drop_remainder=False, pad_token=257)
            steps = [
                premixdb.where(premixdb.text.characters > 0),
                premixdb.dedupe(order_by=[direct.object.uri.asc()]),
            ]
            with premixdb.PremixDB(storage=Path(directory) / "sdk") as client:
                snapshot = client.Corpus("shared", sources)
                self.assertEqual(_decode_id(snapshot.id).hex(), local_snapshot.id)
                local_query = local_snapshot.query(steps=steps)
                query = snapshot.query(steps=steps)
                self.assertEqual(_decode_id(query.id).hex(), local_query.id)
                self.assertEqual(
                    query.profile().output_documents, local_query.summary()["output"]["documents"]
                )
                self.assertEqual(
                    query.profile().output_content_bytes, local_query.summary()["output"]["bytes"]
                )
                self.assertEqual(
                    query.profile().output_characters, local_query.summary()["output"]["characters"]
                )
                local_dataset = local_query.dataset(
                    tokenizer=direct.ByteTokenizer(), sequence_length=7, packing=packing
                )
                dataset = query.mix(
                    tokenizer=direct.ByteTokenizer(),
                    sequence_length=7,
                    packing=packing,
                    splits=premixdb.Splits(train=1, validation=0, test=0),
                )[0]
                # The persisted split policy participates in the SDK dataset identity.
                self.assertNotEqual(_decode_id(dataset.id).hex(), local_dataset.id)
                self.assertEqual([s.tokens for s in dataset], [s.tokens for s in local_dataset])
                self.assertEqual([s.mask for s in dataset], [s.mask for s in local_dataset])
                self.assertEqual(
                    dataset._reader(topology=direct.Topology()).checkpoint(),
                    dict(local_dataset.reader().checkpoint(), dataset=dataset.id),
                )

    def test_local_file_inventory_matches_between_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            (inputs / "nested").mkdir(parents=True)
            (inputs / "unused").mkdir()
            (inputs / "z.txt").write_bytes("é\r\n".encode())
            (inputs / "nested/a.txt").write_bytes(b"")
            native = direct.PremixDB(storage=root / "direct").corpus("files")
            with premixdb.PremixDB(storage=root / "sdk") as db:
                for path in (inputs, inputs / "z.txt"):
                    local_snapshot = native.snapshot(source=path)
                    snapshot = db.Corpus("files", path)
                    self.assertEqual(_decode_id(snapshot.id).hex(), local_snapshot.id)
                    self.assertEqual(
                        {row["source_key"]: row["text"] for row in snapshot.preview()},
                        {row.source_key: row.text for row in local_snapshot.query().rows()},
                    )

    def test_published_stream_matches_in_memory_packing_across_shards(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            premixdb.PremixDB(storage=directory) as client,
        ):
            query = client.Corpus(
                "stream", [premixdb.Source("a", "é\r\n" * 8), premixdb.Source("empty", "")]
            ).query()
            native = coordinator(client)._query(_decode_id(query.id))
            references = []
            build = engine_datasets.Dataset.from_query.__func__

            def stream(
                cls: type[engine_datasets.Dataset],
                query: engine_query.Query,
                length: int,
                separator: int | None,
                padding: int | None,
                tokenizer: engine_datasets.HuggingFaceTokenizer | None = None,
                *,
                stream: bool = False,
            ) -> engine_datasets.Dataset:
                result = build(cls, query, length, separator, padding, tokenizer, stream=stream)
                self.assertIsNone(result._sequences)
                references.append(weakref.ref(result))
                return result

            for padding in (None, 257):
                packing = premixdb.Concat(
                    separator=256, drop_remainder=padding is None, pad_token=padding
                )
                expected = native.dataset(7, 256, padding)
                with (
                    patch.object(engine_datasets, "TOKEN_SHARD_BYTES", 96),
                    patch.object(engine_datasets.Dataset, "from_query", classmethod(stream)),
                ):
                    actual = query.mix(
                        tokenizer=premixdb.ByteTokenizer(), sequence_length=7, packing=packing
                    )[0].wait()
                self.assertGreater(len(actual._proto.tokens), 1)
                self.assertEqual(
                    [s.tokens for s in actual], [s.tokens for s in expected.reader((0, 1, 0, 1))]
                )
                self.assertEqual(
                    [s.mask for s in actual], [s.mask for s in expected.reader((0, 1, 0, 1))]
                )
                gc.collect()
                self.assertIsNone(references[-1]())
                self.assertEqual(len(coordinator(client)._submissions), 0)
                self.assertLessEqual(
                    coordinator(client)._cache.used_bytes, coordinator(client)._cache.max_bytes
                )

    def test_zero_cache_budget_preserves_lazy_recipes_readers_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with premixdb.PremixDB(storage=directory, cache_bytes=0) as client:
                snapshot = client.Corpus(
                    "uncached", [premixdb.Source("a", "abcd"), premixdb.Source("b", "XYZ")]
                )
                query = snapshot.query()
                mix = query.mix(
                    tokenizer=premixdb.ByteTokenizer(),
                    domains=premixdb.object.uri,
                    tokens=9,
                    replacement=True,
                    sequence_length=4,
                )
                self.assertIs(mix[0].status, premixdb.ExecutionStatus.PENDING)
                expected = [s.tokens for s in mix[0]]
                id = mix[0].id
                self.assertEqual(coordinator(client)._cache.used_bytes, 0)
            with premixdb.PremixDB(storage=directory, cache_bytes=0) as client:
                self.assertEqual([s.tokens for s in client._dataset(id)], expected)
                self.assertEqual(coordinator(client)._cache.used_bytes, 0)

    def test_namespaces_share_one_lru_memory_budget(self) -> None:
        cache = MemoryCache(1800)
        first, second = cache.namespace("first"), cache.namespace("second")
        first["a"] = bytearray(500)
        first["b"] = bytearray(500)
        self.assertIsNotNone(first.get("a"))
        second["c"] = bytearray(500)
        self.assertIn("a", first)
        self.assertNotIn("b", first)
        self.assertIn("c", second)
        self.assertLessEqual(cache.used_bytes, cache.max_bytes)
        second["too-large"] = bytearray(5000)
        self.assertNotIn("too-large", second)
        first["a"].extend(b"x" * 5000)
        first.refresh("a")
        self.assertNotIn("a", first)
        self.assertLessEqual(cache.used_bytes, cache.max_bytes)

    def test_cache_views_keep_entries_during_other_namespace_eviction(self) -> None:
        for items in (False, True):
            with self.subTest(items=items):
                cache = MemoryCache(1800)
                first: MutableMapping[str, bytearray] = cache.namespace("first")
                second: MutableMapping[str, bytearray] = cache.namespace("second")
                a, b = bytearray(b"a" * 500), bytearray(b"b" * 500)
                first["a"], first["b"] = a, b
                view = iter(first.items() if items else first.values())
                self.assertEqual(next(view), ("a", a) if items else a)
                self.assertIs(first["a"], a)  # Keep a warm so the other namespace evicts b.
                second["c"] = bytearray(500)
                self.assertNotIn("b", first)
                self.assertEqual(list(view), [("b", b)] if items else [b])
                self.assertLessEqual(cache.used_bytes, cache.max_bytes)


class IndependentWorker:
    cache_scope = "document"
    definition = {"provider": "test-document-map", "version": 1}
    fields = (field("language.en"),)

    def __init__(self) -> None:
        self.computed = []

    def compute(self, docs: Sequence[FeatureDocument]) -> list[ComputedRow]:
        self.computed.extend(doc.id for doc in docs)
        return [{"id": doc.id, "language.en": None if not doc.text else 0.75} for doc in docs]


class DerivationReuseTests(unittest.TestCase):
    def test_duplicate_worker_names_fail_before_computation_or_publication(self) -> None:
        for index in (False, True):
            for empty in (False, True):
                with (
                    self.subTest(index=index, empty=empty),
                    tempfile.TemporaryDirectory() as directory,
                    premixdb.PremixDB(storage=directory) as client,
                ):
                    snapshot = client.Corpus(
                        "duplicate-schema", [] if empty else [premixdb.Source("a", "text")]
                    )
                    service = coordinator(client)
                    worker = DupekitIndex() if index else IndependentWorker()
                    schemas = worker.indexes if isinstance(worker, DupekitIndex) else worker.fields
                    spec = schemas[0]
                    request = catalog.plan(
                        [_decode_id(snapshot.id)],
                        spec.name,
                        bytes.fromhex(_runtime.current_code().commit),
                        index=index,
                    )
                    with (
                        patch.object(enrichment, "producer", return_value=worker),
                        patch.object(
                            type(worker),
                            "indexes" if index else "fields",
                            new_callable=PropertyMock,
                            return_value=(spec, spec),
                        ),
                        patch.object(type(worker), "compute") as compute,
                        patch.object(service._storage, "put", wraps=service._storage.put) as put,
                        patch.object(service._storage, "save", wraps=service._storage.save) as save,
                    ):
                        with self.assertRaisesRegex(ValueError, "duplicate enrichment"):
                            enrichment.build(service, request)
                    compute.assert_not_called()
                    put.assert_not_called()
                    save.assert_not_called()

    def test_overlapping_unions_and_differentials_only_compute_missing_documents(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            premixdb.PremixDB(storage=directory) as client,
        ):
            a = client.Corpus("a", [premixdb.Source("first", "one"), premixdb.Source("empty", "")])
            b = client.Corpus("b", [premixdb.Source("second", "two")])
            worker = IndependentWorker()
            with patch.object(enrichment, "producer", return_value=worker):
                a.query(steps=[premixdb.where(premixdb.language.en > 0.5)]).wait()
                self.assertEqual(len(worker.computed), 2)
                combined = (
                    a.union(b).query(steps=[premixdb.where(premixdb.language.en > 0.5)]).wait()
                )
                self.assertEqual(combined.profile().output_documents, 2)
                self.assertEqual(len(worker.computed), 3)
                changed = client._create_corpus("a").snapshot(
                    source=[premixdb.Source("first", "edited"), premixdb.Source("empty", "")],
                    base=a,
                )
                changed.query(steps=[premixdb.where(premixdb.language.en > 0.5)]).wait()
                self.assertEqual(len(worker.computed), 4)
                changed.union(b).query(
                    steps=[premixdb.where(premixdb.language.en.is_null())]
                ).wait()
                self.assertEqual(len(worker.computed), 4)

    def test_document_cache_survives_restart_and_checks_producer_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with premixdb.PremixDB(storage=directory) as client:
                a = client.Corpus("a", [premixdb.Source("first", "one")])
                worker = IndependentWorker()
                with patch.object(enrichment, "producer", return_value=worker):
                    a.query(steps=[premixdb.where(premixdb.language.en > 0.5)]).wait()
                aid = a.id
            with premixdb.PremixDB(storage=directory) as client:
                a = client._snapshot(aid)
                b = client.Corpus("b", [premixdb.Source("second", "two")])
                worker = IndependentWorker()
                with patch.object(enrichment, "producer", return_value=worker):
                    a.union(b).query(steps=[premixdb.where(premixdb.language.en > 0.5)]).wait()
                self.assertEqual(len(worker.computed), 1)
                worker.definition = {"provider": "test-document-map", "version": 2}
                c = client.Corpus("c", [premixdb.Source("third", "three")])
                with patch.object(enrichment, "producer", return_value=worker):
                    a.union(b, c).query(steps=[premixdb.where(premixdb.language.en > 0.5)]).wait()
                self.assertEqual(len(worker.computed), 4)

    def test_batch_sensitive_outputs_are_only_reused_for_identical_cohorts(self) -> None:
        class BatchWorker(IndependentWorker):
            cache_scope = "batch"

            def compute(self, docs: Sequence[FeatureDocument]) -> list[ComputedRow]:
                self.computed.extend(doc.id for doc in docs)
                return [{"id": doc.id, "language.en": len(docs) / 100} for doc in docs]

        with tempfile.TemporaryDirectory() as directory:

            class Service:
                pipeline = None
                _storage = ObjectStore(directory)
                _submissions = SingleFlight()

            worker = BatchWorker()
            plan = catalog.plan(
                [b"s" * 32], "language.en", bytes.fromhex(_runtime.current_code().commit)
            )
            docs = [Document("01" * 32, "one"), Document("02" * 32, "two")]
            kwargs = (Service(), plan, worker)
            first = list(enrichment.cached_rows(*kwargs, docs, worker.fields, b"definition"))
            again = list(enrichment.cached_rows(*kwargs, docs, worker.fields, b"definition"))
            self.assertEqual(first, again)
            self.assertEqual(len(worker.computed), 2)
            changed = list(
                enrichment.cached_rows(
                    *kwargs, [*docs, Document("03" * 32, "three")], worker.fields, b"definition"
                )
            )
            self.assertEqual(len(worker.computed), 5)
            assert isinstance(first[0], list) and isinstance(changed[0], list)
            self.assertNotEqual(first[0][0].number, changed[0][0].number)

    def test_corrupt_cache_receipt_fails_before_population_publication(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            premixdb.PremixDB(storage=directory) as client,
        ):
            a = client.Corpus("a", [premixdb.Source("first", "one")])
            worker = IndependentWorker()
            with patch.object(enrichment, "producer", return_value=worker):
                a.query(steps=[premixdb.where(premixdb.language.en > 0.5)]).wait()
            service = coordinator(client)
            receipts = service._storage.list("derivation", d.DerivationCache, suffix=".cache")
            self.assertEqual(len(receipts), 1)
            import sqlite3

            with sqlite3.connect(service._storage.metadata.path) as database:
                database.execute(
                    "UPDATE metadata SET payload=? WHERE namespace=? AND suffix=?",
                    (b"corrupt", "derivation", ".cache"),
                )
            b = client.Corpus("b", [premixdb.Source("second", "two")])
            with patch.object(enrichment, "producer", return_value=worker):
                query = a.union(b).query(steps=[premixdb.where(premixdb.language.en > 0.5)])
                with self.assertRaises(premixdb.ExecutionError):
                    query.wait()
            with self.assertRaises(KeyError):
                service._storage.load("query", _decode_id(query.id), query_pb.Query)


class MaterializationLifecycleTests(unittest.TestCase):
    def test_dataset_failure_reports_error_without_completion_and_can_retry(self) -> None:
        from premixdb.storage import tokens
        from premixdb.v1 import data_mixture_pb2 as pb

        with (
            tempfile.TemporaryDirectory() as directory,
            premixdb.PremixDB(storage=directory, cache_bytes=0) as client,
        ):
            query = client.Corpus("failure", [premixdb.Source("a", "abcd")]).query()
            service = coordinator(client)
            request = _requests.dataset(
                query.id, tokenizer=premixdb.ByteTokenizer(), sequence_length=2
            )
            id = service._dataset_id(service._resolve_dataset(request))
            with patch.object(tokens, "publish", side_effect=OSError("temporary shard failure")):
                with self.assertRaisesRegex(OSError, "temporary shard failure"):
                    service.CreateDataset(request)
            failed = service.GetDataset(pb.GetDatasetRequest(id=id)).dataset
            self.assertEqual(failed.status, status_pb.STATUS_ERROR)
            self.assertEqual(failed.error, "temporary shard failure")
            self.assertFalse(failed.tokens)
            listing = pb.ListDatasetRequest(query_id=_decode_id(query.id))
            self.assertEqual(service.ListDatasets(listing).datasets, [failed])
            with self.assertRaises(KeyError):
                service._storage.load("dataset", id, pb.Dataset)
            with self.assertRaisesRegex(premixdb.ExecutionError, "temporary shard failure"):
                client._dataset(id).wait()
            self.assertIsNone(service._jobs.active("dataset", id))
            self.assertEqual(service.CreateDataset(request).id, id)
            self.assertEqual(
                service.ListDatasets(listing).datasets,
                [service.GetDataset(pb.GetDatasetRequest(id=id)).dataset],
            )
            self.assertEqual(client._dataset(id)[0].tokens, list(b"ab"))
            self.assertEqual(len(service._submissions), 0)

    def test_materialization_deadline_includes_submission_and_fetch(self) -> None:
        from types import SimpleNamespace

        from premixdb import api as _resources
        from premixdb.api import base as resource_base
        from premixdb.v1 import data_mixture_pb2 as pb
        from premixdb.v1 import query_pb2 as query_pb
        from premixdb.v1 import status_pb2 as status

        for pending in (
            query_pb.Query(id=b"q" * 32, status=status.STATUS_PENDING),
            pb.Dataset(id=b"d" * 32, status=status.STATUS_PENDING),
        ):
            kind = type(pending).__name__
            with self.subTest(kind=kind):
                completed = type(pending)(id=pending.id, status=status.STATUS_COMPLETED)
                client = Mock(
                    _timeout=1.0, _poll_interval=0.1, _read_only=False, _progress_enabled=False
                )
                client._submit.return_value = SimpleNamespace(id=pending.id)
                client._get.return_value = completed
                resource = (
                    _resources.Query(client, pending)
                    if isinstance(pending, query_pb.Query)
                    else _resources.Dataset(client, pending)
                )
                with patch.object(resource_base.time, "monotonic", side_effect=[0.0, 0.8]):
                    resource.wait(timeout=1.0)
                client._submit.assert_called_once()
                self.assertEqual(client._submit.call_args.kwargs, {})
                self.assertEqual(client._get.call_args.kwargs, {})
                self.assertEqual(client._get.call_args.args, (kind, pending.id))
                self.assertIs(resource.status, premixdb.ExecutionStatus.COMPLETED)

                client.reset_mock()
                resource = (
                    _resources.Query(client, pending)
                    if isinstance(pending, query_pb.Query)
                    else _resources.Dataset(client, pending)
                )
                with patch.object(resource_base.time, "monotonic", side_effect=[0.0, 1.1]):
                    with self.assertRaisesRegex(TimeoutError, f"waiting for {kind.lower()}"):
                        resource.wait(timeout=1.0)
                client._get.assert_not_called()

    def test_failed_query_survives_zero_cache_and_restart_until_explicit_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with premixdb.PremixDB(storage=directory, cache_bytes=0) as client:
                snapshot = client.Corpus("failed-query", [premixdb.Source("a", "one")])
                worker = IndependentWorker()

                def fail_compute(docs: Sequence[FeatureDocument]) -> list[ComputedRow]:
                    raise ValueError("bad model")

                setattr(worker, "compute", fail_compute)
                with patch.object(enrichment, "producer", return_value=worker):
                    query = snapshot.query(steps=[premixdb.where(premixdb.language.en > 0.5)])
                    with self.assertRaisesRegex(premixdb.ExecutionError, "bad model"):
                        query.wait()
                id, sid = query.id, snapshot.id
            with premixdb.PremixDB(storage=directory, cache_bytes=0) as client:
                with patch.object(
                    enrichment, "producer", side_effect=AssertionError("implicit retry")
                ):
                    with self.assertRaisesRegex(premixdb.ExecutionError, "bad model"):
                        client._query(id).wait()
                with patch.object(enrichment, "producer", return_value=IndependentWorker()):
                    retried = (
                        client._snapshot(sid)
                        .query(steps=[premixdb.where(premixdb.language.en > 0.5)])
                        .wait()
                    )
                self.assertEqual(retried.id, id)
                self.assertEqual(retried.profile().output_documents, 1)
