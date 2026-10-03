"""Profiles are useful indexes, with conservative and inspectable estimates."""

from __future__ import annotations

import operator
import statistics
import tempfile
import threading
import unittest
from collections.abc import Iterable, Sequence
from unittest.mock import patch

import premixdb
from premixdb._typing import field_value
from premixdb.enrichment.types import ComputedRow, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.execution import Coordinator, catalog, compile_query, enrichment, profiles
from premixdb.v1 import field_pb2 as f
from premixdb.v1 import profile_pb2 as p
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import snapshot_pb2 as s
from premixdb.v1 import storage_pb2 as storage


class HistogramTests(unittest.TestCase):
    def scalar(
        self, values: Iterable[float | int | None], *, integer: bool = False
    ) -> p.FieldProfile:
        profiler = profiles.FieldProfiler(
            field(
                "datatrove.n_words" if integer else "language.en",
                element_type=(f.VALUE_INT64 if integer else f.VALUE_FLOAT64),
            )
        )
        for value in values:
            profiler.add(field_value(value))
        return profiler.proto()

    def selector(
        self, profile: p.FieldProfile, op: q.Comparison.Operator, value: float
    ) -> q.FieldComparison:
        return q.FieldComparison(
            field=profile.field, projection=q.FieldComparison.SCALAR, operator=op, number=value
        )

    def test_numeric_summary_uses_values_before_bucket_compaction(self) -> None:
        values = list(range(profiles.MAX_BUCKETS + 1)) + [2] * 3
        profile = self.scalar([*values, None, None], integer=True)
        summary = profile.distributions[0].numeric
        self.assertEqual(summary.documents, len(values))
        self.assertEqual(summary.total, sum(values))
        self.assertEqual(
            (summary.minimum.integer, summary.maximum.integer), (0, profiles.MAX_BUCKETS)
        )
        self.assertAlmostEqual(summary.mean, statistics.mean(values))
        self.assertAlmostEqual(summary.standard_deviation, statistics.pstdev(values))
        self.assertEqual(len(profile.distributions[0].buckets), profiles.MAX_BUCKETS)

    def test_large_integer_moments_preserve_small_variations(self) -> None:
        values = [2**60, 2**60 + 1, 2**60 + 2]
        summary = self.scalar(values, integer=True).distributions[0].numeric
        self.assertEqual(summary.minimum.integer, values[0])
        self.assertEqual(summary.maximum.integer, values[-1])
        self.assertAlmostEqual(summary.standard_deviation, statistics.pstdev(values))

    def test_empty_all_null_and_constant_moments(self) -> None:
        for values in ([], [None, None]):
            self.assertFalse(self.scalar(values).distributions[0].HasField("numeric"))
        for values in ([0.0], [0.0] * 2, [-1.25] * 2):
            summary = self.scalar(values).distributions[0].numeric
            self.assertEqual(summary.mean, values[0])
            self.assertEqual(summary.total, sum(values))
            self.assertEqual(summary.standard_deviation, 0)

    def test_nonfinite_values_fail_before_updating_histograms(self) -> None:
        histogram = profiles.Histogram("number")
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                histogram.add(value)
        self.assertEqual(histogram.buckets, [])
        self.assertFalse(histogram.proto().HasField("numeric"))

    def test_bounded_histogram_contains_true_counts_for_all_comparisons(self) -> None:
        values = [i / 10 for i in range(profiles.MAX_BUCKETS + 1)] + [2.5] * 3 + [None] * 3
        profile = self.scalar(values)
        self.assertEqual(profile.documents, len(values))
        self.assertEqual(profile.null_documents, 3)
        self.assertEqual(len(profile.distributions[0].buckets), profiles.MAX_BUCKETS)
        operations = (
            (q.Comparison.OPERATOR_EQ, operator.eq),
            (q.Comparison.OPERATOR_NE, operator.ne),
            (q.Comparison.OPERATOR_LT, operator.lt),
            (q.Comparison.OPERATOR_LE, operator.le),
            (q.Comparison.OPERATOR_GT, operator.gt),
            (q.Comparison.OPERATOR_GE, operator.ge),
        )
        for op, compare in operations:
            for threshold in (-1, 0, 2.5, 3.25, 6.4, 10):
                with self.subTest(op=op, threshold=threshold):
                    bounds = profiles.predicate_bounds(
                        profile, self.selector(profile, op, threshold)
                    )
                    assert bounds is not None
                    lo, hi = bounds
                    actual = sum(v is not None and compare(v, threshold) for v in values)
                    self.assertLessEqual(lo, actual)
                    self.assertGreaterEqual(hi, actual)

    def test_null_is_explicit_and_ordinary_not_equal_excludes_it(self) -> None:
        profile = self.scalar([None, None, 0.25, 0.75])
        self.assertEqual(
            profiles.predicate_bounds(
                profile, self.selector(profile, q.Comparison.OPERATOR_NE, 0.25)
            ),
            (1, 1),
        )
        for op in (q.Comparison.OPERATOR_EQ, q.Comparison.OPERATOR_NE):
            selector = q.FieldComparison(
                projection=q.FieldComparison.IS_NULL, operator=op, boolean=True
            )
            self.assertEqual(profiles.predicate_bounds(profile, selector), (2, 2))
        all_null = self.scalar([None, None])
        self.assertEqual(
            profiles.predicate_bounds(
                all_null, self.selector(all_null, q.Comparison.OPERATOR_NE, 1)
            ),
            (0, 0),
        )

    def test_integer_endpoints_preserve_precision_and_contradictions_are_empty(self) -> None:
        values = [2**53, 2**53 + 1, 2**53 + 2]
        profile = self.scalar(values, integer=True)
        selector = q.FieldComparison(
            field=profile.field,
            projection=q.FieldComparison.SCALAR,
            operator=q.Comparison.OPERATOR_EQ,
            integer=2**53 + 1,
        )
        self.assertEqual(profiles.predicate_bounds(profile, selector), (1, 1))
        profile = self.scalar(range(profiles.MAX_BUCKETS + 1), integer=True)
        selectors = [
            self.selector(profile, q.Comparison.OPERATOR_GT, 10),
            self.selector(profile, q.Comparison.OPERATOR_LT, 11),
        ]
        self.assertEqual(
            profiles.predicate_bounds(profile, selectors[0], selectors=selectors), (0, 0)
        )

    def test_classifier_counts_and_sigmoid_has_no_top_class(self) -> None:
        spec = field("weborganizer.topic", width=2, classes=("a", "b"))
        profiler = profiles.FieldProfiler(spec)
        for value in ([5, 0], [0, 5], [5, 0], None):
            profiler.add(field_value(value))
        profile = profiler.proto()
        distribution = profile.distributions[0]
        self.assertEqual(distribution.projection, p.FieldDistribution.TOP_CLASS)
        self.assertEqual(
            {b.lower.text: b.documents for b in distribution.buckets}, {"a": 2, "b": 1}
        )
        probability = q.FieldComparison(
            field=profile.field,
            projection=q.FieldComparison.CLASS_PROBABILITY,
            class_name="a",
            operator=q.Comparison.OPERATOR_GT,
            number=0.9,
        )
        self.assertEqual(profiles.predicate_bounds(profile, probability), (2, 2))
        spec.classification.transform = f.PROBABILITY_TRANSFORM_SIGMOID
        sigmoid = profiles.FieldProfiler(spec)
        sigmoid.add([5, 5])
        self.assertTrue(
            all(
                d.projection == p.FieldDistribution.CLASS_PROBABILITY
                for d in sigmoid.proto().distributions
            )
        )

    def test_embedding_reports_coverage_and_unprofiled_components_are_unknown(self) -> None:
        profiler = profiles.FieldProfiler(field("embedding.harrier", width=2))
        profiler.add([1.0, 2.0])
        profiler.add(None)
        profile = profiler.proto()
        self.assertEqual((profile.documents, profile.null_documents), (2, 1))
        selector = q.FieldComparison(
            field=profile.field,
            projection=q.FieldComparison.VECTOR_COMPONENT,
            component=0,
            operator=q.Comparison.OPERATOR_GT,
            number=0,
        )
        self.assertIsNone(profiles.predicate_bounds(profile, selector))


class PlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = temp.name
        self.service = Coordinator(self.root)
        self.addCleanup(self.service.close)
        self.snapshot = self.capture("profiles", ["", "ab"])

    def capture(
        self, name: str, texts: Iterable[str], *, base: s.Snapshot | None = None
    ) -> s.Snapshot:
        corpus = self.service.CreateCorpus(premixdb.corpus(name)).id
        return self.service.CreateSnapshot(
            premixdb.snapshot(
                corpus,
                source=storage.Source(
                    memory=storage.MemorySources(
                        documents=[
                            storage.MemorySource(uri=f"doc/{i}", text=text)
                            for i, text in enumerate(texts)
                        ]
                    )
                ),
                base=base,
            )
        ).snapshot

    def test_metadata_only_planning_uses_logical_uri_and_length_distributions(self) -> None:
        plan = compile_query(
            premixdb.query(self.snapshot.id, steps=[premixdb.where(premixdb.text.bytes > 1)])
        )
        with (
            patch.object(self.service, "_snapshot", side_effect=AssertionError("loaded text")),
            patch.object(
                self.service._storage, "read_object", side_effect=AssertionError("read shard")
            ),
        ):
            estimate = profiles.estimate_query(self.service, plan)
        self.assertEqual((estimate.input.lower, estimate.input.upper), (2, 2))
        self.assertEqual((estimate.output.lower, estimate.output.upper), (1, 1))
        uris = next(field for field in estimate.fields if field.field == q.FIELD_OBJECT_URI)
        self.assertEqual(
            {b.lower.text for b in uris.distributions[0].buckets},
            {
                "doc/0",
                "doc/1",
            },
        )

    def test_same_projection_combines_ranges_but_cross_field_correlation_is_unknown(self) -> None:
        plan = compile_query(
            premixdb.query(
                self.snapshot.id,
                steps=[
                    premixdb.where(premixdb.text.bytes > 0),
                    premixdb.where(premixdb.text.bytes > 1),
                ],
            )
        )
        estimate = profiles.estimate_query(self.service, plan)
        self.assertEqual((estimate.output.lower, estimate.output.upper), (1, 1))
        plan = compile_query(
            premixdb.query(
                self.snapshot.id,
                steps=[
                    premixdb.where(premixdb.text.bytes > 1),
                    premixdb.where(premixdb.text.characters > 1),
                ],
            )
        )
        estimate = profiles.estimate_query(self.service, plan)
        self.assertEqual((estimate.output.lower, estimate.output.upper), (0, 1))
        result = self.service.run_query(plan)
        self.assertEqual(result.profile.output_documents, 1)
        self.assertEqual(result.estimate, estimate)

    def test_overlapping_snapshots_are_not_summed_as_exact_population(self) -> None:
        other = self.capture("profiles", ["", "changed"], base=self.snapshot)
        plan = compile_query(premixdb.query(self.snapshot.id, other.id))
        estimate = profiles.estimate_query(self.service, plan)
        self.assertEqual((estimate.input.lower, estimate.input.upper), (2, 4))
        self.assertFalse(estimate.fields)
        actual = self.service.run_query(plan).profile.input_documents
        self.assertLessEqual(estimate.input.lower, actual)
        self.assertGreaterEqual(estimate.input.upper, actual)

    def test_line_dedupe_invalidates_original_length_distribution(self) -> None:
        plan = compile_query(
            premixdb.query(
                self.snapshot.id,
                steps=[
                    premixdb.dedupe(algorithm=premixdb.DedupeAlgorithm.EXACT_LINE),
                    premixdb.where(premixdb.text.bytes < 2),
                ],
            )
        )
        estimate = profiles.estimate_query(self.service, plan)
        self.assertIn(q.FIELD_TEXT_BYTES, estimate.unavailable_fields)
        self.assertEqual((estimate.output.lower, estimate.output.upper), (0, 2))
        self.assertFalse(estimate.output.HasField("expected"))

    def test_profiles_survive_restart_and_return_detached_estimates(self) -> None:
        plan = compile_query(
            premixdb.query(self.snapshot.id, steps=[premixdb.where(premixdb.text.bytes > 1)])
        )
        result = self.service.run_query(plan)
        other = Coordinator(self.root)
        self.addCleanup(other.close)
        with patch.object(other, "_snapshot", side_effect=AssertionError("loaded text")):
            restored = other.GetQuery(q.GetQueryRequest(id=result.id)).query
            self.assertEqual(restored.estimate, result.estimate)
            restored.estimate.Clear()
            self.assertEqual(
                other.GetQuery(q.GetQueryRequest(id=result.id)).query.estimate, result.estimate
            )
            self.assertEqual(
                other.GetSnapshot(s.GetSnapshotRequest(id=self.snapshot.id)).snapshot.profile,
                self.snapshot.profile,
            )

    def test_async_derived_profile_is_reused_without_text_or_value_reads(self) -> None:
        entered, release = threading.Event(), threading.Event()

        class Worker:
            definition = {"test": "profile-index"}
            fields = (field("language.en"),)

            def compute(self, docs: Sequence[FeatureDocument]) -> list[ComputedRow]:
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("test worker was not released")
                return [{"id": doc.id, "language.en": 0.8 if doc.text else None} for doc in docs]

        with patch.object(enrichment, "producer", return_value=Worker()) as producer:
            try:
                id = self.service.CreateQuery(
                    premixdb.query(
                        self.snapshot.id,
                        steps=[premixdb.where(premixdb.language.en > 0.5)],
                    )
                ).id
                self.assertTrue(entered.wait(5))
                pending = self.service.GetQuery(q.GetQueryRequest(id=id)).query
                self.assertIn(q.FIELD_LANGUAGE_EN, pending.estimate.unavailable_fields)
                self.assertEqual(pending.estimate.output.upper, 2)
                self.assertFalse(pending.estimate.output.HasField("expected"))
            finally:
                release.set()
            future = self.service._jobs.active("query", id)
            if future is not None:
                future.result()
            completed = self.service.GetQuery(q.GetQueryRequest(id=id)).query
            self.assertEqual(
                (completed.estimate.output.lower, completed.estimate.output.upper), (1, 1)
            )
            self.assertEqual(completed.profile.output_documents, 1)
            plan = compile_query(
                premixdb.query(
                    self.snapshot.id,
                    steps=[premixdb.where(premixdb.language.en > 0.9)],
                )
            )
            with (
                patch.object(self.service, "_snapshot", side_effect=AssertionError("loaded text")),
                patch.object(
                    self.service._storage, "read_object", side_effect=AssertionError("read shard")
                ),
            ):
                estimate = profiles.estimate_query(self.service, plan)
            self.assertEqual((estimate.output.lower, estimate.output.upper), (0, 0))
            self.assertEqual(producer.call_count, 1)
            # No scalar predicate needs evaluation when the shard profile proves
            # that every non-null value lies outside the new threshold.
            with patch.object(enrichment, "matches", side_effect=AssertionError("tested row")):
                self.assertEqual(self.service.run_query(plan).profile.output_documents, 0)

    def test_derived_histograms_are_identical_across_sharding_and_restart(self) -> None:
        snapshot = self.capture("many-values", ["", "x"])
        plan = compile_query(
            premixdb.query(snapshot.id, steps=[premixdb.where(premixdb.language.en > 0.1)])
        )
        recipe = catalog.resolve(plan)[0]

        class Worker:
            definition = {"test": "deterministic-distribution"}
            fields = (field("language.en"),)

            def compute(self, docs: Sequence[FeatureDocument]) -> list[ComputedRow]:
                return [{"id": doc.id, "language.en": len(doc.text) / 1000} for doc in docs]

        with (
            patch.object(profiles, "MAX_BUCKETS", 1),
            patch.object(enrichment, "producer", return_value=Worker()),
        ):
            with patch.object(enrichment, "SHARD_ROWS", 1):
                first = enrichment.build(self.service, recipe).fields[0].snapshot.profile
            with tempfile.TemporaryDirectory() as root:
                other = Coordinator(root)
                try:
                    # Frozen source requests reproduce the same snapshot identity.
                    other.CreateCorpus(premixdb.corpus("many-values"))
                    other.CreateSnapshot(
                        premixdb.snapshot(
                            snapshot.corpus_id,
                            source=storage.Source(
                                memory=storage.MemorySources(
                                    documents=[
                                        storage.MemorySource(uri=f"doc/{i}", text="x" * i)
                                        for i in range(2)
                                    ]
                                )
                            ),
                        )
                    )
                    with patch.object(enrichment, "SHARD_ROWS", 2):
                        second = enrichment.build(other, recipe).fields[0].snapshot.profile
                    self.assertEqual(first, second)
                    self.assertEqual(len(first.distributions[0].buckets), profiles.MAX_BUCKETS)
                finally:
                    other.close()
                reopened = Coordinator(root)
                try:
                    with (
                        patch.object(
                            reopened, "_snapshot", side_effect=AssertionError("loaded text")
                        ),
                        patch.object(
                            enrichment, "producer", side_effect=AssertionError("ran model")
                        ),
                    ):
                        estimate = profiles.estimate_query(reopened, plan)
                    self.assertEqual(
                        next(p for p in estimate.fields if p.field == first.field), first
                    )
                finally:
                    reopened.close()


if __name__ == "__main__":
    unittest.main()
