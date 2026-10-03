"""Automatic field planning, background derivation, persisted cache and replay."""

from __future__ import annotations

import tempfile
import threading
import unittest
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from _type_support import coordinator

import premixdb
from premixdb import ContentType, Topic, content_type, language, quality, topic, where
from premixdb._ids import _decode_id
from premixdb._typing import FieldValue
from premixdb.enrichment.types import ComputedRow, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.execution import catalog
from premixdb.execution import enrichment as worker
from premixdb.internal import derivation_pb2 as d
from premixdb.internal import derivation_pb2 as derivation_pb
from premixdb.v1 import query_pb2 as q


class ControlledFields:
    definition = {"provider": "test-controlled-model", "version": 1}

    def __init__(self, policy: derivation_pb.EnrichmentProducer) -> None:
        self.policy = policy
        if policy.HasField("language"):
            self.fields = (field("language.en"), field("language.fr"))
        elif policy.model.kind == d.ModelProducer.QUALITY:
            self.fields = (field("quality.educational_value"),)
        elif policy.model.kind == d.ModelProducer.TOPIC:
            self.fields = (
                field("weborganizer.topic", width=24, classes=tuple(v.value for v in Topic)),
            )
        else:
            self.fields = (
                field(
                    "weborganizer.content_type",
                    width=24,
                    classes=tuple(v.value for v in ContentType),
                ),
            )

    def compute(self, docs: Sequence[FeatureDocument]) -> list[ComputedRow]:
        rows: list[ComputedRow] = []
        for doc in docs:
            score = None if not doc.text else 0.95 if doc.url and doc.url.endswith("/b") else 0.2
            values: dict[str, FieldValue] = {
                "language.en": score,
                "language.fr": None if score is None else 1 - score,
                "quality.educational_value": score,
                "weborganizer.topic": [10.0 if t is Topic.SCIENCE_AND_TECH else 0.0 for t in Topic],
                "weborganizer.content_type": [
                    10.0 if t is ContentType.TUTORIAL else 0.0 for t in ContentType
                ],
            }
            rows.append({"id": doc.id, **{spec.name: values[spec.name] for spec in self.fields}})
        return rows


class EnrichmentServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = premixdb.PremixDB(storage=self.temp.name)
        self.addCleanup(self.client.close)
        self.sources = [
            premixdb.Source("https://example.org/a", "same useful document"),
            premixdb.Source("https://example.org/b", "same useful document"),
            premixdb.Source("https://example.org/c", "different document"),
            premixdb.Source("https://example.org/empty", ""),
        ]
        self.snapshot = self.client.corpus("enrichment-test", self.sources)
        real = worker.producer
        self.producer = patch.object(
            worker,
            "producer",
            side_effect=lambda p: real(p) if p.HasField("dupekit") else ControlledFields(p),
        ).start()
        self.addCleanup(patch.stopall)

    def test_filter_before_indexed_dedupe_keeps_later_eligible_member(self) -> None:
        dedupe = premixdb.indexed_dedupe(order_by=[premixdb.object.uri.asc()])
        filtered = self.snapshot.query(steps=[where(language.en > 0.8), dedupe]).wait()
        self.assertEqual(filtered.profile().output_documents, 1)
        result = coordinator(self.client)._query(_decode_id(filtered.id))
        self.assertEqual(result.row(0).source_key, "https://example.org/b")
        self.assertEqual(
            self.snapshot.query(steps=[dedupe, where(language.en > 0.8)])
            .profile()
            .output_documents,
            0,
        )
        self.assertEqual(
            self.snapshot.query(steps=[where(quality.educational_value.is_null())])
            .profile()
            .output_documents,
            1,
        )
        self.assertEqual(
            self.snapshot.query(steps=[where(language.en != 0.95)]).profile().output_documents, 2
        )
        self.assertEqual(
            self.snapshot.query(
                steps=[
                    where(topic.science_and_tech > 0.9),
                    where(content_type.label == ContentType.TUTORIAL),
                ]
            )
            .profile()
            .output_documents,
            4,
        )
        self.assertGreater(
            len(filtered.dataset(tokenizer=premixdb.ByteTokenizer(), sequence_length=8)), 0
        )

    def test_query_defers_derivation_and_reuses_one_producer_across_thresholds(self) -> None:
        entered, release = threading.Event(), threading.Event()
        real_compute = ControlledFields.compute

        def compute(
            instance: ControlledFields, docs: Sequence[FeatureDocument]
        ) -> list[ComputedRow]:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test worker was not released")
            return real_compute(instance, docs)

        with patch.object(ControlledFields, "compute", compute), ThreadPoolExecutor() as pool:
            try:
                first = self.snapshot.query(steps=[where(language.en > 0.8)])
                self.assertEqual(first.status, premixdb.ExecutionStatus.PENDING)
                self.producer.assert_not_called()
                future = pool.submit(first.wait)
                self.assertTrue(entered.wait(5))
                second = self.snapshot.query(steps=[where(language.fr < 0.5)])
                self.assertEqual(self.producer.call_count, 1)
            finally:
                release.set()
            future.result()
            self.assertEqual(first.profile().output_documents, 1)
            self.assertEqual(second.profile().output_documents, 1)
        self.assertEqual(self.producer.call_count, 1)

    def test_restart_reuses_fields_without_running_models(self) -> None:
        query = self.snapshot.query(steps=[where(language.en > 0.8)]).wait()
        self.client.close()
        with (
            premixdb.PremixDB(storage=self.temp.name) as other,
            patch.object(worker, "producer", side_effect=AssertionError("recomputed model")),
        ):
            restored = other.corpus("enrichment-test")
            self.assertEqual(restored.query(steps=[where(language.en > 0.8)]).id, query.id)
            self.assertEqual(
                restored.query(steps=[where(language.en > 0.7)]).profile().output_documents, 1
            )
            self.assertGreater(
                len(
                    restored.query(steps=[where(language.en > 0.8)]).dataset(
                        tokenizer=premixdb.ByteTokenizer(), sequence_length=8
                    )
                ),
                0,
            )

    def test_union_derives_complete_population(self) -> None:
        other = self.client.corpus(
            "other", [premixdb.Source("https://other.org/b", "another document")]
        )
        query = self.snapshot.union(other).query(steps=[where(language.en > 0.8)])
        self.assertEqual(query.profile().output_documents, 2)
        self.assertEqual(
            self.snapshot.query(steps=[where(language.en > 0.8)]).profile().output_documents, 1
        )
        self.assertEqual(self.producer.call_count, 2)

    def test_pending_query_resumes_after_restart(self) -> None:
        from premixdb.execution.planner import compile_query

        pending = compile_query(premixdb.query(self.snapshot.id, steps=[where(language.en > 0.8)]))
        coordinator(self.client)._storage.save("query", pending.id, pending, suffix=".pending")
        self.client.close()
        with premixdb.PremixDB(storage=self.temp.name) as other:
            restored = other._query(pending.id)
            self.assertEqual(restored.profile().output_documents, 1)
        self.assertEqual(self.producer.call_count, 1)

    def test_bad_catalog_requests_rejected_before_running_a_model(self) -> None:
        bad = where(language.en > 0.8)
        bad.field_where.field_name = "language.unknown"
        for steps in ([bad], [premixdb.indexed_dedupe(premixdb.DedupeIndex.MINHASH_LSH)]):
            with self.assertRaises((ValueError, NotImplementedError)):
                self.snapshot.query(steps=steps)
        with self.assertRaises((ValueError, NotImplementedError)):
            from premixdb.execution.planner import compile_query

            compile_query(premixdb.query(self.snapshot.id, field_snapshot_ids=[b"x" * 32]))
        wrong = where(language.en > 0.8)
        wrong.field_where.field_snapshot_id = b"x" * 32
        with self.assertRaises((ValueError, NotImplementedError)):
            self.snapshot.query(steps=[wrong])
        for invalid_field in (q.FIELD_UNSPECIFIED, 999999, q.FIELD_TEXT_BYTES):
            unknown = where(language.en > 0.8)
            setattr(unknown.field_where, "field", invalid_field)
            with self.assertRaises((ValueError, NotImplementedError)):
                self.snapshot.query(steps=[unknown])
        self.producer.assert_not_called()

    def test_enum_requests_and_legacy_names_resolve_identically(self) -> None:
        from premixdb.execution.planner import compile_query

        current = where(language.en > 0.8)
        legacy = q.Operation.FromString(current.SerializeToString())
        legacy.field_where.ClearField("field")
        legacy.field_where.field_name = "language.en"
        canonical = compile_query(premixdb.query(self.snapshot.id, steps=[current]))
        self.assertEqual(canonical, compile_query(premixdb.query(self.snapshot.id, steps=[legacy])))
        self.assertFalse(canonical.operations[0].field_where.field_name)
        self.assertEqual(canonical.operations[0].field_where.field, q.FIELD_LANGUAGE_EN)
        legacy.field_where.field = q.FIELD_LANGUAGE_FR
        with self.assertRaisesRegex(ValueError, "disagree"):
            compile_query(premixdb.query(self.snapshot.id, steps=[legacy]))
        self.producer.assert_not_called()

    def test_failed_derivation_reports_error_and_no_completed_query(self) -> None:
        with patch.object(worker, "producer", side_effect=RuntimeError("model unavailable")):
            query = self.snapshot.query(steps=[where(language.en > 0.8)])
            with self.assertRaisesRegex(premixdb.ExecutionError, "model unavailable"):
                query.wait()
        with self.assertRaises(KeyError):
            coordinator(self.client)._storage.load("query", _decode_id(query.id), q.Query)

    def test_tampered_shard_fails_before_selection(self) -> None:
        query = self.snapshot.query(steps=[where(language.en > 0.8)]).wait()
        build, manifest = worker.load_build(
            coordinator(self.client),
            "field",
            query._proto.field_snapshot_ids[0],
            query._proto.snapshot_ids,
        )
        path = Path(self.temp.name) / "field/objects" / manifest.shards[0].blake3_digest.hex()
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(premixdb.ExecutionError, "integrity"):
            self.snapshot.query(steps=[where(language.en > 0.7)]).wait()

    def test_sharding_and_storage_location_do_not_change_query_identity(self) -> None:
        first = self.snapshot.query(steps=[where(language.en > 0.8)]).wait()
        with (
            tempfile.TemporaryDirectory() as directory,
            premixdb.PremixDB(storage=directory) as client,
            patch.object(worker, "SHARD_ROWS", 1),
        ):
            second = (
                client.corpus("enrichment-test", self.sources)
                .query(steps=[where(language.en > 0.8)])
                .wait()
            )
            self.assertEqual(first.id, second.id)
            self.assertEqual(first.profile(), second.profile())

    def test_planning_pins_models_without_constructing_workers(self) -> None:
        from premixdb.execution.planner import compile_query

        plan = compile_query(
            premixdb.query(self.snapshot.id, steps=[where(topic.science_and_tech > 0.5)])
        )
        recipe = catalog.resolve(plan)[0].producer.model
        self.assertEqual(recipe.repository, "WebOrganizer/TopicClassifier-NoURL")
        self.assertEqual(len(recipe.revision), 40)
        self.assertTrue(plan.field_snapshot_ids)
        self.producer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
