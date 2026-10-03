"""Integration checks for new readers, ingestion, execution history and inspection."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import coordinator

import premixdb as p
import premixdb as sdk
from premixdb._ids import _decode_id
from premixdb._reader import permutation
from premixdb.enrichment.types import ComputedRow, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.internal import derivation_pb2 as derivation_pb
from premixdb.internal import derivation_pb2 as e
from premixdb.v1 import dataset_pb2 as dataset_pb
from premixdb.v1 import snapshot_pb2 as s
from premixdb.v1 import storage_pb2 as storage


class InspectionFields:
    definition = {"provider": "test-extended-fields", "version": 1}

    def __init__(self, spec: derivation_pb.EnrichmentProducer) -> None:
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
            patch(
                "premixdb.execution.enrichment.producer", side_effect=InspectionFields
            ) as producer,
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
        from premixdb.execution.inspection import matrix

        with patch(
            "premixdb.execution.enrichment.producer", side_effect=AssertionError("inference")
        ):
            self.assertEqual(
                sum(cell["documents"] for cell in matrix(coordinator(self.client), ready.id)), 3
            )
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

    def test_inspection_requires_fields_but_mixture_domains_derive_them(self) -> None:
        from premixdb.execution.inspection import matrix

        query = self.population().query()
        with patch(
            "premixdb.execution.enrichment.producer", side_effect=AssertionError("inference")
        ):
            with self.assertRaisesRegex(ValueError, "requested by the query"):
                matrix(coordinator(self.client), query.id)
        with patch(
            "premixdb.execution.enrichment.producer", side_effect=InspectionFields
        ) as producer:
            mix = query.mix(tokenizer=p.ByteTokenizer(), domains=p.topic, tokens=4)
            self.assertEqual(producer.call_count, 1)
            self.assertEqual(mix[0].profile().planned_content_tokens, 4)
        with patch(
            "premixdb.execution.enrichment.producer", side_effect=AssertionError("inference")
        ):
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

    def population(self) -> sdk.Snapshot:
        return self.client.corpus(
            "population",
            [
                p.Source("https://test/a", "same text"),
                p.Source("https://test/b", "same text"),
                p.Source("https://test/null", "other"),
            ],
        )

    def test_quality_ordering_nulls_and_derived_strata(self) -> None:
        with patch("premixdb.execution.enrichment.producer", side_effect=InspectionFields):
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
        with patch("premixdb.execution.enrichment.producer", side_effect=InspectionFields):
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

    def test_shuffle_partitions_and_checkpoint_policy(self) -> None:
        for count in (1, 2, 3, 4, 5, 15, 16, 17, 63, 64, 65, 127, 128, 129):
            self.assertEqual(
                sorted(permutation(i, count, 7) for i in range(count)), list(range(count))
            )
        dataset = self.population().query().dataset(tokenizer=p.ByteTokenizer(), sequence_length=1)
        reader = dataset.reader(seed=7)
        first = next(reader).ordinal
        state = reader.checkpoint()
        remainder = [s.ordinal for s in reader]
        self.assertEqual([s.ordinal for s in dataset.reader(seed=7, checkpoint=state)], remainder)
        self.assertEqual(sorted([first, *remainder]), list(range(len(dataset))))
        with self.assertRaises(ValueError):
            dataset.reader(seed=8, checkpoint=state)
        parts = [
            [
                s.ordinal
                for s in dataset.reader(seed=7, topology=p.Topology(rank=rank, world_size=3))
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
        snapshot = self.client.corpus(
            "manifest", p.SourceSpec(manifest=p.SourceManifest(objects={"object": ref}))
        )
        self.assertEqual(snapshot.profile().characters, 3)
        ref.blake3_digest = b"x" * 32
        with self.assertRaises(Exception):
            self.client.corpus(
                "bad", p.SourceSpec(manifest=p.SourceManifest(objects={"object": ref}))
            )

    def test_huggingface_capture_pins_revision_and_row_keys(self) -> None:
        with patch(
            "datasets.load_dataset",
            return_value=[{"key": "a", "text": "one"}, {"key": "b", "text": "two"}],
        ) as load:
            snapshot = self.client.corpus(
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
                self.client.corpus(
                    "hub-invalid",
                    p.HuggingFaceDataset(
                        repository="test/repo", revision="a" * 40, split="train", key_column="key"
                    ),
                )

    def test_alignment_preserves_original_utf8_coordinates(self) -> None:
        snapshot = self.client.corpus("text", [p.Source("a", "pré\n秘密\nfin")])
        reference = self.client.corpus("ref", [p.Source("b", "秘密")])
        dataset = snapshot.query(
            decontaminate=p.decontaminate(reference, algorithm="line", granularity="span")
        ).dataset(tokenizer=p.ByteTokenizer(), sequence_length=20)
        regions = dataset[0].spans
        ranges = [
            (i, i + 1)
            for region in regions
            for r in region.source_ranges
            for i in range(r.start, r.end)
        ]
        self.assertEqual(ranges, [(i, i + 1) for i in [0, 1, 2, 3, 4, 11, 12, 13, 14]])

    def test_python_inspection_catalog_rows_and_history(self) -> None:
        from premixdb.execution.inspection import catalog, history, rows

        query = self.population().query()
        service = coordinator(self.client)
        self.assertEqual(len(catalog(service)["snapshot"]), 1)
        self.assertEqual(rows(service, "query", query.id, {})["total"], 3)
        parameters = dict(
            field=["FIELD_TEXT_BYTES"],
            lower=[json.dumps({"count": "5"})],
            upper=[json.dumps({"count": "5"})],
        )
        self.assertEqual(rows(service, "query", query.id, parameters)["total"], 1)
        self.assertEqual(len(history(service, self.client._create_corpus("population").id)), 1)

    def test_python_inspection_matrix_reuses_query_pinned_fields(self) -> None:
        from premixdb.execution.inspection import matrix

        with patch("premixdb.execution.enrichment.producer", side_effect=InspectionFields):
            query = (
                self.population()
                .query(
                    sampling=p.sample(
                        seed=3, documents=3, domains=(p.topic.label, p.quality.educational_value)
                    )
                )
                .wait()
            )
        with patch(
            "premixdb.execution.enrichment.projections",
            side_effect=AssertionError("recomputed pinned fields"),
        ):
            cells = matrix(coordinator(self.client), query.id)
        self.assertEqual(sum(cell["documents"] for cell in cells), 3)
        self.assertTrue(any(cell["topic"] == "unknown" for cell in cells))

    def test_python_inspection_input_drilldown_and_field_coverage(self) -> None:
        from premixdb.execution.inspection import coverage, rows

        snapshot = self.population()
        with patch("premixdb.execution.enrichment.producer", side_effect=InspectionFields):
            query = snapshot.query(steps=[p.where(p.quality.educational_value >= 0.5)]).wait()
        service = coordinator(self.client)
        inputs = rows(service, "query", query.id, {"population": ["input"]})
        outputs = rows(service, "query", query.id, {})
        self.assertEqual((inputs["total"], outputs["total"]), (3, 1))
        removed = rows(
            service,
            "query",
            query.id,
            {"population": ["input"], "stage": ["0"], "outcome": ["removed"]},
        )
        self.assertEqual(removed["total"], 2)
        self.assertTrue(
            all(
                row["provenance"] is not None
                and row["provenance"]["selection"]["kind"] == "filtered"
                for row in removed["rows"]
            )
        )
        available = coverage(service, "snapshot", snapshot.id)
        self.assertEqual(len(available["fields"]), 1)
        quality = available["fields"][0]
        self.assertEqual(quality["profile"]["null_documents"], "1")
        self.assertTrue(coverage(service, "query", query.id)["fields"][0]["pinned"])
        parameters = dict(
            build=[quality["id"]],
            field=["FIELD_QUALITY_EDUCATIONAL_VALUE"],
            lower=[json.dumps({"number": 0.5})],
            upper=[json.dumps({"number": 1.0})],
        )
        example = rows(service, "snapshot", snapshot.id, parameters)
        self.assertEqual(example["rows"][0]["source"], "https://test/b")

    def test_python_inspection_exact_and_candidate_index_statistics(self) -> None:
        from premixdb.execution.inspection import coverage, index_statistics

        snapshot = self.population()
        query = snapshot.query(steps=[p.indexed_dedupe(p.DedupeIndex.EXACT_DOCUMENT)]).wait()
        available = coverage(coordinator(self.client), "snapshot", snapshot.id)
        index = next(
            index for index in available["indexes"] if index["name"] == "dupekit.exact_candidates"
        )
        statistics = index_statistics(coordinator(self.client), index["id"], {"size": ["2"]})
        self.assertEqual(statistics["group_sizes"], {"2": "1"})
        self.assertEqual(statistics["duplicate_documents"], "1")
        self.assertEqual(
            {r["source"] for r in statistics["examples"]}, {"https://test/a", "https://test/b"}
        )
        self.assertEqual(query.profile().output_documents, 2)

    def test_fluent_recipes_reuse_completed_results_after_restart(self) -> None:
        path = self.root / "source.txt"
        path.write_text("first")
        snapshot = self.client.corpus("retry", path)
        query = snapshot.query()
        dataset = query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=2).wait()
        mix = query.mix(tokenizer=p.ByteTokenizer(), tokens=3)
        path.write_text("updated")
        self.client.close()
        self.client = p.PremixDB(storage=self.root)
        replay = self.client.corpus("retry")
        self.assertEqual(replay.id, snapshot.id)
        self.assertEqual(replay.profile().content_bytes, 5)
        query_replay = replay.query()
        self.assertEqual(query_replay.id, query.id)
        self.assertEqual(
            query_replay.dataset(tokenizer=p.ByteTokenizer(), sequence_length=2).id, dataset.id
        )
        self.assertEqual(query_replay.mix(tokenizer=p.ByteTokenizer(), tokens=3).id, mix.id)

    def test_python_inspection_sequence_geometry_source_and_stratum_drilldown(self) -> None:
        from premixdb.execution.inspection import sequences

        left = self.client.corpus("left", [p.Source("left", "abcdef")])
        right = self.client.corpus("right", [p.Source("right", "12345")])
        query = left.union(right).query()
        dataset = query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=4).wait()
        service = coordinator(self.client)
        page = sequences(service, dataset.id, {})
        self.assertEqual(page["total"], len(dataset))
        self.assertEqual([r["tokens"] for r in page["rows"]], [r.tokens for r in dataset])
        crossing = sequences(service, dataset.id, {"crossing": ["true"]})
        self.assertEqual(crossing["total"], dataset.profile().boundary_crossing_sequences)
        for count, total in dataset.profile().documents_per_sequence.items():
            self.assertEqual(
                sequences(service, dataset.id, {"documents": [str(count)]})["total"], total
            )
        source_page = sequences(
            service, dataset.id, {"source": [self.client._create_corpus("left").id]}
        )
        self.assertTrue(source_page["rows"])
        self.assertTrue(
            all(97 in row["tokens"] or 101 in row["tokens"] for row in source_page["rows"])
        )
        mixed = query.mix(
            domains=p.source.corpus_id,
            tokens=8,
            n_candidates=1,
            sampler=p.RegMixSampler(seed=1),
        )[0].wait()
        source = self.client._create_corpus("left").id
        strata_page = sequences(service, mixed.id, {"stratum": [source]})
        source_page = sequences(service, mixed.id, {"source": [source]})
        self.assertEqual(strata_page, source_page)
        with self.assertRaises(ValueError):
            sequences(service, dataset.id, {"offset": ["-1"]})

    @pytest.mark.integration
    def test_process_pipeline_preserves_dataset_identity_and_boundaries(self) -> None:
        self.client.close()
        self.client = p.PremixDB(storage=self.root, process_workers=2)
        self.assertIsNotNone(coordinator(self.client).pipeline)
        pipeline = coordinator(self.client).pipeline
        assert pipeline is not None
        pipeline.packing_shard_sequences = 1
        snapshot = self.population()
        query = snapshot.query(steps=[p.dedupe()])
        parallel = query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=4)

        def sequences(
            dataset: sdk.Dataset,
        ) -> list[tuple[list[int], list[bool], list[bool], list[dataset_pb.TokenRegion]]]:
            return [(seq.tokens, seq.mask, seq.attention_mask, seq.spans) for seq in dataset]

        expected = sequences(parallel)
        expected_summary = parallel._summary()
        expected_provenance = query._provenance()
        self.client.close()
        self.client = p.PremixDB(storage=self.root / "sequential")
        sequential_query = self.population().query(steps=[p.dedupe()])
        sequential = sequential_query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=4)
        self.assertEqual(parallel.id, sequential.id)
        self.assertEqual(expected, sequences(sequential))
        self.assertEqual(expected_summary, sequential._summary())
        self.assertEqual(expected_provenance, sequential_query._provenance())
        self.assertTrue(list((self.root / "partitions" / "receipts").glob("*")))

    @pytest.mark.integration
    def test_process_reference_and_index_producers(self) -> None:
        self.client.close()
        self.client = p.PremixDB(storage=self.root, process_workers=2)
        reference = self.client.corpus("evaluation", [p.Source("ref", "same text")])
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
        from premixdb import _runtime
        from premixdb.engine.snapshots import Store, StoredDocument

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
