"""Integration checks for new readers, ingestion, execution history and inspection."""

from __future__ import annotations

import tempfile
import threading
import unittest
import weakref
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb.contracts import FieldValue
from premixdb.engine.identity import CodeVersion
from premixdb.engine.plans import Step
from premixdb.engine.queries import CorpusIndex, Query
from premixdb.engine.snapshots import StoredDocument
from premixdb.enrichment.types import ComputedRow, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.internal import derivation_pb2 as e
from premixdb.runtime import enrichment
from premixdb.schemas.ids import _decode_id
from premixdb.training.reader import permutation
from premixdb.v1 import data_mixture_pb2 as dataset_pb
from premixdb.v1 import snapshot_pb2 as s
from premixdb.v1 import storage_pb2 as storage


class InspectionFields:
    definition = {"provider": "test-extended-fields", "version": 1}

    def __init__(self, spec: e.EnrichmentProducer) -> None:
        if spec.model.kind == e.ModelProducer.QUALITY:
            self.fields = (field("quality.educational_value"),)
        elif spec.model.kind == e.ModelProducer.TOPIC:
            self.fields = (
                field("weborganizer.topic", width=24, classes=tuple(t.value for t in p.Topic)),
            )
        else:
            self.fields = (field("embedding.harrier", width=1024),)

    def compute(self, documents: Sequence[FeatureDocument]) -> list[ComputedRow]:
        return [
            {
                "id": d.id,
                self.fields[0].name: None
                if (d.url or "").endswith("null")
                else (0.9 if (d.url or "").endswith("b") else 0.1)
                if not self.fields[0].length
                else (
                    [10.0 if t is p.Topic.SCIENCE_AND_TECH else 0.0 for t in p.Topic]
                    if self.fields[0].name == "weborganizer.topic"
                    else [1.0, 0.0] + [0.0] * 1022
                    if d.text.startswith("same")
                    else [0.0, 1.0] + [0.0] * 1022
                ),
            }
            for d in documents
        ]


class ExtendedResourceTests(unittest.TestCase):
    def test_requested_fields_start_on_wait_and_reuse_completed_builds(self) -> None:
        started, release = threading.Event(), threading.Event()
        snapshot = self.population()
        compute = InspectionFields.compute

        def blocked(
            worker: InspectionFields, documents: Sequence[FeatureDocument]
        ) -> list[ComputedRow]:
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("field computation was not released")
            return compute(worker, documents)

        with (
            patch("premixdb.runtime.enrichment.producer", side_effect=InspectionFields) as producer,
            patch.object(InspectionFields, "compute", blocked),
            ThreadPoolExecutor() as pool,
        ):
            try:
                query = snapshot.query()._with_fields([p.topic.label, p.quality.educational_value])
                self.assertEqual(query.status, p.ExecutionStatus.PENDING)
                producer.assert_not_called()
                future = pool.submit(query.wait)
                self.assertTrue(started.wait(timeout=5))
                self.assertIn(query.status, (p.ExecutionStatus.PENDING, p.ExecutionStatus.RUNNING))
                self.assertEqual(producer.call_count, 1)
            finally:
                release.set()
            ready = future.result()
            self.assertEqual(ready.profile().output_documents, 3)
            self.assertEqual(len(ready._proto.field_snapshot_ids), 2)
        with patch("premixdb.runtime.enrichment.producer", side_effect=AssertionError("inference")):
            self.assertEqual(ready.profile().output_documents, 3)
            reordered = (
                snapshot.query()
                ._with_fields([p.quality.educational_value, p.topic.label, p.topic.label])
                .wait()
            )
            self.assertEqual(reordered.id, ready.id)
            self.assertNotEqual(snapshot.query().id, ready.id)
            coordinator(self.client)._query_handles.clear()
            mixed = self.client._query(ready.id).mix(
                tokenizer=p.ByteTokenizer(), domains=p.topic.label, tokens=4
            )
            self.assertEqual(mixed[0].wait().profile().planned_content_tokens, 4)

    def test_mixture_domains_derive_and_reuse_fields(self) -> None:
        query = self.population().query()
        with patch(
            "premixdb.runtime.enrichment.producer", side_effect=InspectionFields
        ) as producer:
            mix = query.mix(tokenizer=p.ByteTokenizer(), domains=p.topic, tokens=4)
            self.assertEqual(producer.call_count, 0)
            self.assertEqual(mix[0].profile().planned_content_tokens, 4)
            self.assertEqual(producer.call_count, 1)
        with patch("premixdb.runtime.enrichment.producer", side_effect=AssertionError("inference")):
            self.assertEqual(
                query.mix(tokenizer=p.ByteTokenizer(), domains=p.topic, tokens=4).id, mix.id
            )

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.client = p.PremixDB(storage=self.root)

    def tearDown(self) -> None:
        self.client.close()
        self.directory.cleanup()

    def population(self) -> p.Snapshot:
        return self.client.Corpus(
            "population",
            [
                p.Source("https://test/a", "same text"),
                p.Source("https://test/b", "same text"),
                p.Source("https://test/null", "other"),
            ],
        )

    def test_quality_ordering_nulls_and_derived_strata(self) -> None:
        with patch("premixdb.runtime.enrichment.producer", side_effect=InspectionFields):
            snapshot = self.population()
            query = snapshot.query(
                steps=[p.dedupe(order_by=[p.quality.educational_value.desc()])]
            ).wait()
            self.assertEqual(
                {
                    row.source_key
                    for row in coordinator(self.client)._query(_decode_id(query.id)).rows()
                },
                {"https://test/b", "https://test/null"},
            )
            sampled = snapshot.query(
                sampling=p.sample(
                    seed=3, documents=2, domains=(p.quality.educational_value, p.object.uri)
                )
            )
            self.assertEqual(sampled.wait().profile().output_documents, 2)
            mix = (
                snapshot.query()
                ._with_fields([p.quality.educational_value])
                .mix(
                    tokenizer=p.ByteTokenizer(),
                    domains=p.quality.educational_value,
                    tokens=4,
                    n_candidates=1,
                )
            )
            self.assertEqual(len(mix), 1)
            self.assertEqual(mix[0].wait().profile().planned_content_tokens, 4)

    def test_cosine_and_lsh_selection(self) -> None:
        with patch("premixdb.runtime.enrichment.producer", side_effect=InspectionFields):
            semantic = (
                self.population()
                .query(steps=[p.similarity_dedupe(embedding=p.embedding.harrier, threshold=1.0)])
                .wait()
            )
            self.assertEqual(semantic.profile().output_documents, 2)
        exact = (
            self.population()
            .query(steps=[p.indexed_dedupe(p.DedupeIndex.MINHASH_LSH, threshold=1.0)])
            .wait()
        )
        self.assertEqual(exact.profile().output_documents, 2)

    def test_cosine_streams_only_selected_embeddings(self) -> None:
        snapshot = self.client.Corpus(
            "streamed-embeddings",
            [p.Source("https://test/a", "same text"), p.Source("https://test/b", "same text")]
            + [p.Source(f"https://test/other/{i}", "different outside") for i in range(32)],
        )
        with patch.object(enrichment, "producer", InspectionFields):
            snapshot.query()._with_fields([p.embedding.harrier]).wait()

        class Vector(list[float]):
            pass

        project = enrichment.numeric_vector
        live = peak = calls = 0

        def released() -> None:
            nonlocal live
            live -= 1

        def measured(value: FieldValue) -> list[float]:
            nonlocal live, peak, calls
            vector = Vector(project(value))
            calls += 1
            live += 1
            peak = max(peak, live)
            weakref.finalize(vector, released)
            return vector

        execute = CorpusIndex.execute

        def tracked(
            index: CorpusIndex,
            plan: Iterable[Step],
            version: CodeVersion,
            fields: tuple[bytes, ...],
        ) -> Query:
            with patch.object(enrichment, "numeric_vector", side_effect=measured):
                return execute(index, plan, version, fields)

        for filtered in (False, True):
            with self.subTest(filtered=filtered):
                live = peak = calls = 0
                steps = [p.where(p.text.bytes < 10)] if filtered else []
                steps.append(p.similarity_dedupe(embedding=p.embedding.harrier, threshold=1.0))
                with patch.object(CorpusIndex, "execute", autospec=True, side_effect=tracked):
                    query = snapshot.query(steps=steps).wait()
                self.assertEqual(query.profile().output_documents, 1 if filtered else 2)
                self.assertEqual(calls, 2 if filtered else 34)
                self.assertLessEqual(peak, 2)
                self.assertEqual(live, 0)

    def test_similarity_dedupe_pins_and_reuses_derived_ordering(self) -> None:
        snapshot = self.population()
        for embedding in (None, p.embedding.harrier):
            with self.subTest(embedding=embedding):
                with patch(
                    "premixdb.runtime.enrichment.producer", side_effect=InspectionFields
                ) as producer:
                    query = snapshot.query(
                        steps=[
                            p.similarity_dedupe(
                                embedding=embedding,
                                threshold=1.0,
                                n=1,
                                order_by=[p.quality.educational_value.desc()],
                            )
                        ]
                    )
                    producer.assert_not_called()
                    ordering = query._proto.operations[0].similarity_dedupe.order_by[0].selector
                    self.assertEqual(len(ordering.field_snapshot_id), 32)
                    self.assertIn(ordering.field_snapshot_id, query._proto.field_snapshot_ids)
                    query.wait()
                self.assertEqual(
                    {
                        row.source_key
                        for row in coordinator(self.client)._query(_decode_id(query.id)).rows()
                    },
                    {"https://test/b", "https://test/null"},
                )
                with patch(
                    "premixdb.runtime.enrichment.producer",
                    side_effect=AssertionError("recomputed a completed build"),
                ):
                    self.assertEqual(self.client._query(query.id).wait().id, query.id)

    def test_shuffle_partitions_and_checkpoint_policy(self) -> None:
        for count in (1, 2, 3, 4, 5, 15, 16, 17, 63, 64, 65, 127, 128, 129):
            self.assertEqual(
                sorted(permutation(i, count, 7) for i in range(count)), list(range(count))
            )
        dataset = self.population().query().mix(tokenizer=p.ByteTokenizer(), sequence_length=1)[0]
        reader = dataset._reader(seed=7)
        first = next(reader).ordinal
        state = reader.checkpoint()
        remainder = [s.ordinal for s in reader]
        self.assertEqual([s.ordinal for s in dataset._reader(seed=7, checkpoint=state)], remainder)
        self.assertEqual(sorted([first, *remainder]), list(range(len(dataset))))
        with self.assertRaises(ValueError):
            dataset._reader(seed=8, checkpoint=state)
        parts = [
            [
                s.ordinal
                for s in dataset._reader(seed=7, topology=p.Topology(rank=rank, world_size=3))
            ]
            for rank in range(3)
        ]
        self.assertEqual(
            sorted(ordinal for part in parts for ordinal in part), list(range(len(dataset)))
        )

    def test_idempotent_capture_and_execution_history_survive_restart(self) -> None:
        corpus = self.client._create_corpus("history")
        request = s.CreateSnapshotRequest(
            request_id="capture-history",
            corpus_id=_decode_id(corpus.id),
            source=storage.Source(
                memory=p.MemorySources(documents=[p.MemorySource(uri="a", text="data")])
            ),
        )
        initial = self.client._submit(request).snapshot
        self.client.close()
        self.client = p.PremixDB(storage=self.root)
        retried = self.client._submit(request).snapshot
        self.assertEqual(initial, retried)
        events = self.client._executions(initial.id)
        self.assertEqual(len(events), 2)
        self.assertTrue(events[-1].cache_hit)
        request.source.memory.documents[0].text = "changed"
        with self.assertRaises(Exception):
            self.client._submit(request)
        self.assertTrue(any(event.status == "error" for event in self.client._executions()))

    def test_manifest_ingestion_checks_captured_bytes(self) -> None:
        ref = coordinator(self.client)._storage.put("snapshot", "pré".encode())
        # Authorize the service's storage directory for executor-side file objects.
        coordinator(self.client)._source_root = self.root
        snapshot = self.client.Corpus(
            "manifest", p.SourceSpec(manifest=p.SourceManifest(objects={"object": ref}))
        )
        self.assertEqual(snapshot.profile().characters, 3)
        ref.blake3_digest = b"x" * 32
        with self.assertRaises(Exception):
            self.client.Corpus(
                "bad", p.SourceSpec(manifest=p.SourceManifest(objects={"object": ref}))
            )

    def test_huggingface_capture_pins_revision_and_row_keys(self) -> None:
        with patch(
            "datasets.load_dataset",
            return_value=[{"key": "a", "text": "one"}, {"key": "b", "text": "two"}],
        ) as load:
            snapshot = self.client.Corpus(
                "hub",
                p.HuggingFaceDataset(
                    repository="test/repo", revision="a" * 40, split="train", key_column="key"
                ),
            )
            self.assertEqual(snapshot.profile().documents, 2)
            self.assertEqual(load.call_args.kwargs["revision"], "a" * 40)
        with patch(
            "datasets.load_dataset",
            return_value=[{"key": "a", "text": "one"}, {"key": "a", "text": "two"}],
        ):
            with self.assertRaises(Exception):
                self.client.Corpus(
                    "hub-invalid",
                    p.HuggingFaceDataset(
                        repository="test/repo", revision="a" * 40, split="train", key_column="key"
                    ),
                )

    def test_alignment_preserves_original_utf8_coordinates(self) -> None:
        snapshot = self.client.Corpus("text", [p.Source("a", "pré\n秘密\nfin")])
        reference = self.client.Corpus("ref", [p.Source("b", "秘密")])
        dataset = snapshot.query(
            decontaminate=p.decontaminate(reference, algorithm="line", granularity="span")
        ).mix(tokenizer=p.ByteTokenizer(), sequence_length=20)[0]
        regions = dataset[0].spans
        ranges = [
            (i, i + 1)
            for region in regions
            for r in region.source_ranges
            for i in range(r.start, r.end)
        ]
        self.assertEqual(ranges, [(i, i + 1) for i in [0, 1, 2, 3, 4, 11, 12, 13, 14]])

    def test_catalog_preview_and_history(self) -> None:
        snapshot = self.population()
        query = snapshot.query(steps=[p.where(p.text.bytes == 5)])
        self.assertEqual(len(self.client.Corpus.list()), 1)
        self.assertEqual(snapshot.profile().documents, 3)
        self.assertEqual(query.profile().output_documents, 1)
        self.assertEqual(len(query.preview()), 1)
        self.assertTrue(self.client._execution_events())

    def test_published_fields_reuse_pinned_values(self) -> None:
        with patch("premixdb.runtime.enrichment.producer", side_effect=InspectionFields):
            query = (
                self.population()
                .query()
                ._with_fields([p.topic.label, p.quality.educational_value])
                .wait()
            )
        with patch(
            "premixdb.runtime.enrichment.projections",
            side_effect=AssertionError("recomputed pinned fields"),
        ):
            self.assertEqual(query.profile().output_documents, 3)
            self.assertTrue(query.profile().fields)

    def test_filtered_preview_and_lineage_agree(self) -> None:
        snapshot = self.population()
        with patch("premixdb.runtime.enrichment.producer", side_effect=InspectionFields):
            query = snapshot.query(steps=[p.where(p.quality.educational_value >= 0.5)]).wait()
        self.assertEqual((snapshot.profile().documents, query.profile().output_documents), (3, 1))
        self.assertEqual(query.preview()[0]["source_key"], "https://test/b")
        lineage = query._provenance()
        self.assertEqual(
            sum(value["selection"]["kind"] == "filtered" for value in lineage.values()), 2
        )

    def test_indexed_dedupe_publishes_evidence(self) -> None:
        snapshot = self.population()
        query = snapshot.query(steps=[p.indexed_dedupe(p.DedupeIndex.EXACT_DOCUMENT)]).wait()
        self.assertTrue(query._proto.index_snapshot_ids)
        self.assertEqual(query.profile().output_documents, 2)

    def test_shared_field_projections_filter_without_decoding_text(self) -> None:
        snapshot = self.population()
        cases = [
            (p.quality.educational_value >= 0.5, {"b"}),
            (p.topic.label == p.Topic.SCIENCE_AND_TECH, {"a", "b"}),
            (p.topic.science_and_tech >= 0.9, {"a", "b"}),
            (p.quality.educational_value.is_null(), {"null"}),
            (p.embedding.harrier.component(0) == 1.0, {"a", "b"}),
        ]
        with patch("premixdb.runtime.enrichment.producer", side_effect=InspectionFields):
            snapshot.query()._with_fields(
                [p.topic.label, p.quality.educational_value, p.embedding.harrier.component(0)]
            ).wait()
        with patch.object(
            StoredDocument,
            "text",
            new_callable=PropertyMock,
            side_effect=AssertionError("decoded whole document"),
        ):
            for predicate, expected in cases:
                query = snapshot.query(steps=[p.where(predicate)]).wait()
                self.assertEqual(
                    {
                        row["source_key"].rsplit("/", 1)[-1]
                        for row in query.preview(max_characters=0)
                    },
                    expected,
                )

    def test_sampled_profiles_decode_and_project_each_selected_document_once(self) -> None:
        from premixdb.runtime import enrichment
        from premixdb.storage import profiles

        with patch("premixdb.runtime.enrichment.producer", side_effect=InspectionFields):
            query = (
                self.population()
                .query(sampling=p.sample(seed=7, documents=200, replacement=True))
                ._with_fields([p.topic.label, p.quality.educational_value])
                .wait()
            )
        service = coordinator(self.client)
        handle = service._query(_decode_id(query.id))
        self.assertEqual(len({row.id for row in handle}), 3)
        with (
            patch.object(enrichment, "decode_value", wraps=enrichment.decode_value) as decode,
            patch.object(profiles, "probabilities", wraps=profiles.probabilities) as project,
        ):
            from premixdb.runtime.profiles import output_profiles

            fields = output_profiles(service, query._proto, handle)
        self.assertEqual(decode.call_count, 6)
        self.assertEqual(project.call_count, 2)
        self.assertEqual(fields, list(query._proto.profile.fields))
        nulls = sum(row.source_key.endswith("null") for row in handle)
        self.assertGreater(nulls, 0)
        for profile in fields[4:]:
            self.assertEqual(profile.documents, 200)
            self.assertEqual(profile.null_documents, nulls)

    def test_fluent_recipes_reuse_completed_results_after_restart(self) -> None:
        path = self.root / "source.txt"
        path.write_text("first")
        snapshot = self.client.Corpus("retry", path)
        query = snapshot.query()
        dataset = query.mix(tokenizer=p.ByteTokenizer(), sequence_length=2)[0].wait()
        mix = query.mix(tokenizer=p.ByteTokenizer(), tokens=3)
        path.write_text("updated")
        self.client.close()
        self.client = p.PremixDB(storage=self.root)
        replay = self.client.Corpus("retry")
        self.assertEqual(replay.id, snapshot.id)
        self.assertEqual(replay.profile().content_bytes, 5)
        query_replay = replay.query()
        self.assertEqual(query_replay.id, query.id)
        self.assertEqual(
            query_replay.mix(tokenizer=p.ByteTokenizer(), sequence_length=2)[0].id, dataset.id
        )
        self.assertEqual(query_replay.mix(tokenizer=p.ByteTokenizer(), tokens=3).id, mix.id)

    def test_sequence_preview_matches_packed_tokens_and_profile(self) -> None:
        left = self.client.Corpus("left", [p.Source("left", "abcdef")])
        right = self.client.Corpus("right", [p.Source("right", "12345")])
        dataset = (
            left.union(right).query().mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0].wait()
        )
        page = dataset.preview(limit=100)
        self.assertEqual(len(page), len(dataset))
        self.assertEqual([row["tokens"] for row in page], [row.tokens for row in dataset])
        self.assertEqual(sum(dataset.profile().documents_per_sequence.values()), len(dataset))

    def test_process_pipeline_preserves_dataset_identity_and_boundaries(self) -> None:
        self.client.close()
        self.client = p.PremixDB(storage=self.root, process_workers=2)
        self.assertIsNotNone(coordinator(self.client).pipeline)
        pipeline = coordinator(self.client).pipeline
        assert pipeline is not None
        pipeline.packing_shard_sequences = 1
        snapshot = self.population()
        query = snapshot.query(steps=[p.dedupe()])
        parallel = query.mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]

        def sequences(
            dataset: p.Dataset,
        ) -> list[tuple[list[int], list[bool], list[bool], list[dataset_pb.TokenRegion]]]:
            return [(seq.tokens, seq.mask, seq.attention_mask, seq.spans) for seq in dataset]

        expected = sequences(parallel)
        expected_summary = parallel.profile()
        expected_provenance = query._provenance()
        self.client.close()
        self.client = p.PremixDB(storage=self.root / "sequential")
        sequential_query = self.population().query(steps=[p.dedupe()])
        sequential = sequential_query.mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]
        self.assertEqual(parallel.id, sequential.id)
        self.assertEqual(expected, sequences(sequential))
        self.assertEqual(expected_summary, sequential.profile())
        self.assertEqual(expected_provenance, sequential_query._provenance())
        self.assertTrue(list((self.root / "partitions" / "receipts").glob("*")))

    @pytest.mark.integration
    def test_process_reference_and_index_producers(self) -> None:
        self.client.close()
        self.client = p.PremixDB(storage=self.root, process_workers=2)
        reference = self.client.Corpus("evaluation", [p.Source("ref", "same text")])
        query = (
            self.population()
            .query(
                steps=[p.indexed_dedupe(p.DedupeIndex.EXACT_DOCUMENT)],
                decontaminate=p.decontaminate(reference, algorithm="ngram", n=2),
            )
            .wait()
        )
        self.assertEqual(query.profile().output_documents, 1)
        self.assertEqual(query.profile().decontamination.matched_documents, 1)
        self.assertTrue(list((self.root / "partitions" / "receipts").glob("*")))
        fields = self.population().query(steps=[p.where(p.datatrove.n_words >= 1)]).wait()
        self.assertEqual(fields.profile().output_documents, 3)
        self.assertTrue(fields.profile().fields)

    def test_lazy_snapshots_keep_frames_instead_of_captured_text(self) -> None:
        from premixdb.engine.snapshots import Store, StoredDocument
        from premixdb.runtime import environment as _runtime

        store = Store(self.root / "captured")
        text = "pré 🌍\n" * 2
        snapshot = store.capture_inputs(
            "01" * 16, iter([("a", text), ("b", "")]), [], _runtime.current_code(), stream=True
        )
        self.assertTrue(all(isinstance(doc, StoredDocument) for doc in snapshot.documents.values()))
        self.assertTrue(all("text" not in doc.__dict__ for doc in snapshot.documents.values()))
        lazy = store.load(snapshot.id, lazy=True)
        reference = store.load(snapshot.id)
        self.assertEqual(lazy.summary(), reference.summary())
        self.assertEqual(
            [d.id for d in lazy.documents.values()], [d.id for d in reference.documents.values()]
        )
        self.assertEqual(lazy.documents["a"].text, text)
        self.assertNotIn("text", lazy.documents["a"].__dict__)


if __name__ == "__main__":
    unittest.main()
