"""Embedding fields share automatic query derivation, caching and replay."""

from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from _type_support import Vectors, coordinator

import premixdb
from premixdb import EmbeddingModel, embedding, where
from premixdb._ids import _decode_id
from premixdb.execution import catalog
from premixdb.execution import enrichment as worker
from premixdb.execution.planner import compile_query


class EmbeddingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = premixdb.PremixDB(storage=self.temp.name)
        self.addCleanup(self.client.close)
        self.snapshot = self.client.corpus(
            "embedding-test", [premixdb.Source("a", "physics"), premixdb.Source("b", "food")]
        )
        self.models, self.calls = [], []
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.addCleanup(self.release.set)
        fixture = self

        class Encoder:
            def __init__(self, repository: str, *, revision: str, device: str) -> None:
                fixture.models.append((repository, revision, device))
                self.width = 1024
                self.max_seq_length = None

            def get_embedding_dimension(self) -> int:
                return self.width

            def encode(
                self,
                texts: list[str],
                *,
                batch_size: int,
                normalize_embeddings: bool,
                show_progress_bar: bool,
                convert_to_numpy: bool,
                convert_to_tensor: bool,
                prompt: str | None = None,
                prompt_name: str | None = None,
            ) -> Vectors:
                options = dict(
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=convert_to_numpy,
                    convert_to_tensor=convert_to_tensor,
                    prompt=prompt,
                    prompt_name=prompt_name,
                )
                fixture.calls.append((texts, options, self.max_seq_length))
                fixture.entered.set()
                if not fixture.release.wait(10):
                    raise TimeoutError("test embedding worker was not released")
                vectors = [
                    [1.0 if text == "physics" else -1.0] + [0.0] * (self.width - 1)
                    for text in texts
                ]
                return cast(Vectors, SimpleNamespace(tolist=lambda: vectors))

        self.encoder = Encoder
        self.encoder_patch = patch.dict(
            "sys.modules", {"sentence_transformers": SimpleNamespace(SentenceTransformer=Encoder)}
        )
        self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        # Controlled inference needs no downloaded weights.
        versions = patch(
            "premixdb.enrichment.models.package_versions",
            side_effect=lambda *names: {name: "test" for name in names},
        )
        versions.start()
        self.addCleanup(versions.stop)

    def test_planning_and_unrelated_queries_do_not_load_embedding_models(self) -> None:
        for model in EmbeddingModel:
            with self.subTest(model=model):
                request = premixdb.query(
                    self.snapshot.id, steps=[where(embedding[model].component(0) > 0.0)]
                )
                plan = compile_query(request)
                recipe = catalog.resolve(plan)[0].producer.model
                kind, repository, revision = catalog.MODELS[embedding[model].name]
                self.assertEqual(
                    (recipe.kind, recipe.repository, recipe.revision), (kind, repository, revision)
                )
                self.assertTrue(plan.field_snapshot_ids)
        self.assertEqual(self.snapshot.query().profile().output_documents, 2)
        self.assertEqual(self.models, [])
        self.assertEqual(self.calls, [])

    def test_consumption_starts_embeddings_and_components_reuse_vectors(self) -> None:
        for model in EmbeddingModel:
            with self.subTest(model=model):
                self.entered.clear()
                self.release.clear()
                before = len(self.calls)
                with ThreadPoolExecutor() as pool:
                    try:
                        first = self.snapshot.query(
                            steps=[where(embedding[model].component(0) > 0.0)]
                        )
                        self.assertIs(first.status, premixdb.ExecutionStatus.PENDING)
                        self.assertEqual(len(self.calls), before)
                        future = pool.submit(first.wait)
                        self.assertTrue(self.entered.wait(5))
                        self.assertIs(
                            self.client._query(first.id).status, premixdb.ExecutionStatus.RUNNING
                        )
                        second = self.snapshot.query(
                            steps=[where(embedding[model].component(1) >= 0.0)]
                        )
                        self.assertIs(second.status, premixdb.ExecutionStatus.PENDING)
                        self.assertEqual(len(self.calls), before + 1)
                    finally:
                        self.release.set()
                    future.result()
                # Reading results waits for completion without an explicit execution call.
                self.assertEqual(first._summary()["output"]["documents"], 1)
                self.assertEqual(second.profile().output_documents, 2)
                stricter = self.snapshot.query(steps=[where(embedding[model].component(0) > 1.0)])
                self.assertEqual(stricter.profile().output_documents, 0)
                self.assertEqual(len(self.calls), before + 1)
                self.assertEqual(len(self.models), before + 1)
                texts, options, max_length = self.calls[-1]
                self.assertCountEqual(texts, ["physics", "food"])
                self.assertEqual(options["prompt"], "")
                self.assertTrue(options["normalize_embeddings"])
                self.assertEqual(max_length, 8192)
                result = coordinator(self.client)._query(_decode_id(first.id))
                self.assertEqual(result.row(0).source_key, "a")
                self.assertGreater(len(first.dataset(sequence_length=8)), 0)

    def test_restart_reuses_embedding_vectors_across_thresholds(self) -> None:
        ids = {}
        for model in EmbeddingModel:
            ids[model] = (
                self.snapshot.query(steps=[where(embedding[model].component(0) > 0.0)]).wait().id
            )
        self.client.close()
        with (
            premixdb.PremixDB(storage=self.temp.name) as other,
            patch.object(worker, "producer", side_effect=AssertionError("recomputed embeddings")),
        ):
            restored = other.corpus("embedding-test")
            for model in EmbeddingModel:
                with self.subTest(model=model):
                    replay = restored.query(steps=[where(embedding[model].component(0) > 0.0)])
                    self.assertEqual(replay.id, ids[model])
                    self.assertEqual(replay.profile().output_documents, 1)
                    changed = restored.query(steps=[where(embedding[model].component(0) > -0.5)])
                    self.assertEqual(changed.profile().output_documents, 1)
        self.assertEqual(len(self.calls), 1)

    def test_invalid_components_are_rejected_before_loading_models(self) -> None:
        self.assertFalse(hasattr(embedding, "qwen"))
        self.assertEqual(list(EmbeddingModel), [EmbeddingModel.HARRIER])
        compile_query(
            premixdb.query(self.snapshot.id, steps=[where(embedding.harrier.component(1023) > 0.0)])
        )
        for model, width in ((EmbeddingModel.HARRIER, 1024),):
            with self.subTest(model=model), self.assertRaises((ValueError, NotImplementedError)):
                self.snapshot.query(steps=[where(embedding[model].component(width) > 0.0)])
        self.assertEqual(self.models, [])
        self.assertEqual(self.calls, [])

    def test_embedding_failures_surface_through_query_results(self) -> None:
        with patch.object(
            self.encoder, "encode", side_effect=RuntimeError("embedding unavailable")
        ):
            for model in EmbeddingModel:
                with self.subTest(model=model):
                    query = self.snapshot.query(steps=[where(embedding[model].component(0) > 0.0)])
                    with self.assertRaisesRegex(premixdb.ExecutionError, "embedding unavailable"):
                        query._summary()


if __name__ == "__main__":
    unittest.main()
