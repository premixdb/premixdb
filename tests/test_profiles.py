"""Profile aggregation and reporting before dataset materialization."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from _type_support import coordinator

import premixdb
from premixdb._ids import _decode_id, _public_dataset_profile
from premixdb._policies import Concat as ConcatPolicy
from premixdb._profiles import DistributionSummary, _describe_field
from premixdb._typing import field_value
from premixdb.engine.snapshots import StoredDocument
from premixdb.enrichment.types import field
from premixdb.execution import Coordinator
from premixdb.execution.profiles import FieldProfiler, Histogram
from premixdb.v1 import dataset_pb2 as dataset_pb
from premixdb.v1 import dataset_pb2 as pb
from premixdb.v1 import field_pb2 as f
from premixdb.v1 import profile_pb2 as p
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import snapshot_pb2, storage_pb2


class ProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = temp.name
        self.client = premixdb.PremixDB(storage=self.root)
        self.addCleanup(self.client.close)
        self.corpus = self.client._create_corpus("profiles")
        self.snapshot = self.corpus.snapshot(
            source=[
                premixdb.Source("a", "é\n🌍\r\nend"),
                premixdb.Source("b", "abc"),
                premixdb.Source("empty", ""),
            ]
        )

    def test_researcher_summary_of_lengths_and_filtered_output(self) -> None:
        captured = _describe_field(self.snapshot.profile().fields, premixdb.text.characters)
        self.assertIsInstance(captured, premixdb.DistributionSummary)
        self.assertEqual((captured.count, captured.minimum, captured.maximum), (3, 0, 8))
        self.assertEqual(captured.total, 11)
        assert captured.mean is not None
        self.assertAlmostEqual(captured.mean, 11 / 3)
        self.assertEqual(captured.quantile(0.5), premixdb.QuantileRange(3, 3))
        selected = self.snapshot.query(steps=[premixdb.where(premixdb.text.characters > 0)])
        summary = selected._describe("text.characters")
        self.assertEqual(summary.documents, 2)
        self.assertEqual(summary.mean, 5.5)
        self.assertEqual(summary.standard_deviation, 2.5)
        self.assertEqual(summary.quantile(0.9), premixdb.QuantileRange(8, 8))
        with self.assertRaises(FrozenInstanceError):
            setattr(summary, "mean", 0)
        reopened = premixdb.PremixDB(storage=self.root)
        self.addCleanup(reopened.close)
        with patch.object(
            coordinator(reopened), "_snapshot", side_effect=AssertionError("read text")
        ):
            self.assertEqual(
                _describe_field(
                    reopened._snapshot(self.snapshot.id).profile().fields, "text.characters"
                ),
                captured,
            )

    def test_missing_field_does_not_launch_inference(self) -> None:
        with patch(
            "premixdb.execution.enrichment.producer", side_effect=AssertionError("inference")
        ):
            with self.assertRaises(KeyError):
                _describe_field(self.snapshot.profile().fields, premixdb.quality.educational_value)

    def test_profiles_aggregate_and_reopen_without_loading_text(self) -> None:
        profile = self.snapshot.profile()
        self.assertIsInstance(profile, premixdb.SnapshotProfile)
        self.assertEqual((profile.documents, profile.objects, profile.spans), (3, 3, 2))
        self.assertEqual((profile.content_bytes, profile.characters, profile.newlines), (15, 11, 2))
        profile.Clear()
        self.assertEqual(self.snapshot.profile().documents, 3)
        # Object/span refs stay private to the worker.
        objects = coordinator(self.client)._objects(_decode_id(self.snapshot.id))
        self.assertEqual(sum(o.profile.content_bytes for o in objects.values()), 15)
        native = coordinator(self.client)._store.objects(_decode_id(self.snapshot.id).hex())
        for info in native.values():
            self.assertEqual(
                info["content_bytes"], sum(s["content_bytes"] for s in info["span_refs"])
            )
            self.assertEqual(info["characters"], sum(s["characters"] for s in info["span_refs"]))
        service = Coordinator(self.root)
        self.addCleanup(service.close)
        with patch.object(service, "_snapshot", side_effect=AssertionError("loaded text")):
            reopened = service.GetSnapshot(
                snapshot_pb2.GetSnapshotRequest(id=_decode_id(self.snapshot.id))
            )
        self.assertEqual(reopened.snapshot.profile, self.snapshot.profile())
        self.assertFalse(service._snapshot_handles)

    def test_query_and_snapshot_profiles_have_specific_semantics(self) -> None:
        query = self.snapshot.union(self.snapshot).query(
            steps=[premixdb.where(premixdb.text.characters > 0)]
        )
        self.assertEqual(
            (
                query.profile().snapshots,
                query.profile().input_documents,
                query.profile().output_documents,
            ),
            (1, 3, 2),
        )
        self.assertEqual(query.profile().steps[0].input_documents, 3)
        query.dataset(tokenizer=premixdb.ByteTokenizer(), sequence_length=4)
        second = self.corpus.snapshot(source=[premixdb.Source("new", "x")], base=self.snapshot)
        self.assertEqual((second.profile().documents, second.profile().content_bytes), (1, 1))

    def test_unfiltered_query_profiles_read_text_only_for_the_bounded_preview(self) -> None:
        # One document beyond the ten-document preview detects accidental full reads.
        sources = [premixdb.Source(str(i), "é🌍" * i) for i in range(11)]
        snapshot = self.corpus.snapshot(source=sources)
        captured = snapshot.profile()
        documents = coordinator(self.client)._snapshot(_decode_id(snapshot.id)).documents.values()
        preview_ids = sorted(document.id for document in documents)[:10]
        read_ids = set()
        read_text = StoredDocument.text.fget

        def preview_text(document: StoredDocument) -> str:
            self.assertIn(document.id, preview_ids, "profiling read text outside the preview")
            read_ids.add(document.id)
            return read_text(document)

        query = snapshot.query()
        self.assertEqual(query.status, premixdb.ExecutionStatus.PENDING)
        with patch.object(StoredDocument, "text", property(preview_text)):
            profile = query.profile()
        self.assertEqual(read_ids, set(preview_ids))
        self.assertEqual(profile.output_documents, captured.documents)
        self.assertEqual(profile.output_characters, captured.characters)
        self.assertEqual(profile.output_content_bytes, captured.content_bytes)
        for selector in (premixdb.text.bytes, premixdb.text.characters):
            actual = _describe_field(profile.fields, selector)
            expected = _describe_field(captured.fields, selector)
            self.assertEqual(actual.documents, expected.documents)
            self.assertEqual(actual.buckets, expected.buckets)
            self.assertEqual(actual.total, expected.total)
            assert actual.mean is not None and expected.mean is not None
            self.assertAlmostEqual(actual.mean, expected.mean)
            assert actual.standard_deviation is not None and expected.standard_deviation is not None
            self.assertAlmostEqual(actual.standard_deviation, expected.standard_deviation)
        self.assertEqual(list(profile.fields[2:]), list(captured.fields[2:]))
        with patch.object(
            coordinator(self.client), "_execute_query", side_effect=AssertionError("reran query")
        ):
            self.assertEqual(query.profile(), profile)
            self.assertEqual(snapshot.query().profile(), profile)

    def test_planned_dataset_and_mixture_profiles_match_built_output(self) -> None:
        query = self.snapshot.query()

        def profile(
            *,
            tokenizer: dataset_pb.Tokenizer,
            sequence_length: int,
            packing: dataset_pb.Packing | ConcatPolicy,
        ) -> dataset_pb.DatasetProfile:
            return _public_dataset_profile(
                coordinator(self.client)._profile_dataset(
                    premixdb.dataset(
                        query.id,
                        tokenizer=tokenizer,
                        sequence_length=sequence_length,
                        packing=packing,
                    )
                )
            )

        for packing in (
            premixdb.Concat(separator=256),
            premixdb.Concat(separator=256, drop_remainder=False, pad_token=257),
        ):
            planned = profile(
                tokenizer=premixdb.ByteTokenizer(), sequence_length=7, packing=packing
            )
            self.assertEqual(planned.planned_content_tokens, 15)
            before = len(coordinator(self.client)._storage.list("dataset", pb.Dataset))
            self.assertEqual(
                profile(tokenizer=premixdb.ByteTokenizer(), sequence_length=7, packing=packing),
                planned,
            )
            self.assertEqual(
                len(coordinator(self.client)._storage.list("dataset", pb.Dataset)), before
            )
            built = query.dataset(
                tokenizer=premixdb.ByteTokenizer(), sequence_length=7, packing=packing
            ).profile()
            self.assertEqual(built, planned)
            candidates = query.mix(
                domains=premixdb.object.uri,
                sampler=premixdb.RegMixSampler(),
                size=premixdb.Tokens(10, tokenizer=premixdb.ByteTokenizer()),
                bounds=premixdb.Bounds(max_epochs=4),
                sequence_length=7,
                packing=packing,
                seed=42,
            )
            before = len(coordinator(self.client)._storage.list("dataset", pb.Dataset))
            planned = candidates.profile(0)
            self.assertEqual(candidates[0].status, premixdb.ExecutionStatus.PENDING)
            self.assertEqual(
                len(coordinator(self.client)._storage.list("dataset", pb.Dataset)), before
            )
            self.assertEqual(sum(planned.planned_stratum_tokens.values()), 10)
            built = candidates[0].profile()
            self.assertEqual(built, planned)
            self.assertEqual(candidates[:1].profile(0), planned)

    def test_public_api_has_profiles_without_counts_or_content_inspection(self) -> None:
        self.assertFalse(hasattr(Coordinator, "GetSnapshotObjects"))
        self.assertFalse(hasattr(Coordinator, "GetQueryRows"))
        self.assertNotIn("Counts", storage_pb2.DESCRIPTOR.message_types_by_name)
        self.assertFalse(hasattr(self.snapshot, "objects"))
        self.assertFalse(hasattr(self.snapshot.query(), "rows"))


class DistributionSummaryTests(unittest.TestCase):
    def describe(self, values: Iterable[int | None]) -> DistributionSummary:
        profiler = FieldProfiler(field("datatrove.n_words", element_type=f.VALUE_INT64))
        for value in values:
            profiler.add(value)
        return _describe_field([profiler.proto()], premixdb.datatrove.n_words)

    def test_quantiles_bound_actual_ranks_and_compressed_counts_are_unknown(self) -> None:
        values = list(range(65))
        summary = self.describe(values)
        self.assertIsNone(summary.counts)
        self.assertIsNone(summary.fractions)
        for probability, expected in ((0, 0), (0.5, 32), (0.9, 58), (0.99, 64), (1, 64)):
            bounds = summary.quantile(probability)
            assert bounds is not None
            self.assertLessEqual(bounds.lower, expected)
            self.assertGreaterEqual(bounds.upper, expected)
        for invalid in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                summary.quantile(invalid)

    def test_coverage_and_empty_summaries(self) -> None:
        summary = self.describe([None, 0, 0, 3])
        self.assertEqual((summary.documents, summary.count, summary.null_documents), (4, 3, 1))
        self.assertEqual(summary.missing_fraction, 0.25)
        self.assertEqual(summary.counts, {0: 2, 3: 1})
        self.assertEqual(summary.fractions, {0: 2 / 3, 3: 1 / 3})
        for values, missing in (([], None), ([None, None], 1)):
            empty = self.describe(values)
            self.assertIsNone(empty.mean)
            self.assertIsNone(empty.minimum)
            self.assertIsNone(empty.quantile(0.5))
            self.assertEqual(empty.counts, {})
            self.assertEqual(empty.fractions, {})
            self.assertEqual(empty.missing_fraction, missing)

    def test_legacy_histograms_do_not_invent_numeric_moments(self) -> None:
        histogram = Histogram("integer", projection=p.FieldDistribution.SCALAR)
        for value in (1, 2, 3):
            histogram.add(value)
        distribution = histogram.proto()
        distribution.ClearField("numeric")
        profile = p.FieldProfile(
            field=q.FIELD_DATATROVE_N_WORDS, documents=3, distributions=[distribution]
        )
        summary = _describe_field([profile], "datatrove.n_words")
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.total)
        self.assertIsNone(summary.standard_deviation)
        self.assertEqual(summary.quantile(0.5), premixdb.QuantileRange(2, 2))

    def test_classifier_composition_and_probability_moments(self) -> None:
        spec = field("weborganizer.topic", element_type=f.VALUE_FLOAT64, width=2)
        spec.classification.CopyFrom(
            f.Classification(
                classes=["Science & Tech.", "Software"], transform=f.PROBABILITY_TRANSFORM_SOFTMAX
            )
        )
        profiler = FieldProfiler(spec)
        for logits in ([2, 0], [0, 2], [2, 0], None):
            profiler.add(field_value(logits))
        profile = profiler.proto()
        labels = _describe_field([profile], premixdb.topic.label)
        self.assertEqual(labels.counts, {"Science & Tech.": 2, "Software": 1})
        self.assertEqual(labels.fractions, {"Science & Tech.": 2 / 3, "Software": 1 / 3})
        self.assertIsNone(labels.mean)
        with self.assertRaises(TypeError):
            labels.quantile(0.5)
        probability = _describe_field([profile], premixdb.topic.software)
        self.assertEqual(probability.class_name, "Software")
        self.assertEqual(probability.count, 3)
        assert probability.mean is not None and probability.total is not None
        self.assertAlmostEqual(probability.mean, probability.total / 3)
        spec.classification.transform = f.PROBABILITY_TRANSFORM_SIGMOID
        sigmoid = FieldProfiler(spec)
        sigmoid.add([1, 2])
        with self.assertRaises(KeyError):
            _describe_field([sigmoid.proto()], premixdb.topic.label)
        self.assertIsNotNone(_describe_field([sigmoid.proto()], premixdb.topic.software).mean)
