"""End-to-end checks for curation extensions and persisted training output."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path

import pytest
from _type_support import coordinator
from blake3 import blake3

import premixdb as p
import premixdb as sdk
from premixdb.execution.enrichment import numeric_vector


class ExtendedCurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.client = p.PremixDB(storage=self.root)

    def tearDown(self) -> None:
        self.client.close()
        self.directory.cleanup()

    def snapshot(self, name: str, texts: Iterable[str]) -> sdk.Snapshot:
        return self.client.corpus(name, [p.Source(str(i), text) for i, text in enumerate(texts)])

    def test_decontamination_trims_utf8_ranges_and_persists_witnesses(self) -> None:
        target = self.snapshot("target", ["pré\n秘密\nfin", "safe"])
        reference = self.snapshot("reference", ["秘密"])
        query = target.query(
            decontaminate=p.decontaminate(reference, algorithm="line", granularity="span")
        )
        self.assertEqual(sorted(d["text"] for d in query.preview()), ["pré\n\nfin", "safe"])
        lineage = json.loads(
            self.client._object_reader.read(
                p.SpanRef(
                    object=query._proto.lineage,
                    end=query._proto.lineage.size_bytes,
                    blake3_digest=query._proto.lineage.blake3_digest,
                )
            )
        )
        match = next(record for record in lineage.values() if "contamination" in record)
        self.assertEqual(match["contamination"][0]["start"], 5)
        self.assertEqual(match["contamination"][0]["end"], 11)
        self.assertEqual(
            query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=4)
            .profile()
            .source_content_bytes,
            13,
        )

    def test_ngram_document_removal_precedes_sampling(self) -> None:
        target = self.snapshot("target", ["one two three", "clean", "also clean"])
        reference = self.snapshot("reference", ["xx one two yy"])
        query = target.query(
            decontaminate=p.decontaminate(reference, algorithm="ngram", n=2),
            sampling=p.sample(seed=7, documents=2),
        )
        self.assertEqual(query.profile().output_documents, 2)
        self.assertNotIn("one two three", [d.text for d in query._proto.preview.documents])

    def test_sampling_intrinsic_strata_use_retained_text_lengths(self) -> None:
        from premixdb.execution.inspection import rows

        target = self.snapshot("target", ["ab\n秘密\ncd", "uvwxyz"])
        reference = self.snapshot("reference", ["秘密"])
        query = target.query(
            decontaminate=p.decontaminate(reference, algorithm="line", granularity="span"),
            sampling=p.sample(seed=3, documents=2, domains=p.text.bytes, weights={"[6]": 1.0}),
        )
        self.assertEqual(dict(query.profile().sampling.realized_domains), {"[6]": 2})
        self.assertEqual(
            rows(coordinator(self.client), "query", query.id, {"stratum": ["[6]"]})["total"], 2
        )

    def test_cosine_spill_covers_block_boundaries_and_extreme_finite_scales(self) -> None:
        from premixdb.engine.curation import cosine_edges
        from premixdb.engine.value_cache import ValueCache

        vectors = ValueCache(
            (f"{i:064x}", [1e200, 0] if i % 3 == 0 else [1e-200, 0] if i % 3 == 1 else [0, 1])
            for i in range(65)
        )
        vectors[f"{65:064x}"] = None
        vectors[f"{66:064x}"] = [0, 0]
        self.assertEqual(
            sum(
                1
                for _ in cosine_edges(
                    {
                        key: numeric_vector(value) if value is not None else None
                        for key, value in vectors.items()
                    },
                    1.0,
                )
            ),
            44 * 43 // 2 + 21 * 20 // 2,
        )
        selected = {f"{i:064x}" for i in (0, 1, 2, 65, 66)}
        self.assertEqual(
            list(
                cosine_edges(
                    {
                        key: numeric_vector(value) if value is not None else None
                        for key, value in vectors.items()
                    },
                    1.0,
                    selected=selected,
                )
            ),
            [(f"{0:064x}", f"{1:064x}")],
        )
        self.assertEqual(list(cosine_edges({"a": [1, 0], "b": [0.8, 0.6]}, 0.8)), [("a", "b")])

    def test_sampling_is_seeded_whole_document_and_reports_overshoot(self) -> None:
        snapshot = self.snapshot("population", ["aaaa", "bbbb", "cccc"])
        a = snapshot.query(sampling=p.sample(seed=8, bytes=5))
        b = snapshot.query(sampling=p.sample(seed=8, bytes=5))
        self.assertEqual(a.id, b.id)
        self.assertEqual(a.profile().output_content_bytes, 8)
        repeated = snapshot.query(sampling=p.sample(seed=8, documents=7, replacement=True))
        self.assertEqual(repeated.profile().output_documents, 7)
        profile = repeated.dataset(tokenizer=p.ByteTokenizer(), sequence_length=4).profile()
        self.assertEqual(profile.document_occurrences, 7)
        self.assertEqual(profile.source_documents, repeated.profile().sampling.unique_documents)
        self.assertEqual(sum(profile.source_tokens.values()), profile.content_tokens)
        self.assertEqual(sum(profile.documents_per_sequence.values()), profile.sequences)
        with self.assertRaises(Exception):
            snapshot.query(sampling=p.sample(seed=8, documents=4)).wait()

    def test_jaccard_chain_is_not_an_equivalence_class(self) -> None:
        snapshot = self.snapshot("population", ["a b", "a b c", "b c"])
        query = snapshot.query(
            steps=[
                p.similarity_dedupe(
                    n=1, threshold=0.6, order_by=[p.text.bytes.asc(), p.object.uri.asc()]
                )
            ]
        )
        self.assertEqual(sorted(d["text"] for d in query.preview()), ["a b", "b c"])

    def test_model_tokenizer_runs_through_service_and_reopens(self) -> None:
        asset = Path(__file__).parent / "fixtures" / "wordpiece.json"
        tokenizer = p.hugging_face_tokenizer(asset, digest=blake3(asset.read_bytes()).hexdigest())
        query = self.snapshot("population", ["hello world"]).query()
        dataset = query.dataset(
            tokenizer=tokenizer,
            sequence_length=2,
            packing=p.concat(drop_remainder=False, pad_token=0),
        )
        self.assertEqual(dataset.profile().planned_content_tokens, 2)
        self.assertEqual(len(dataset[0].tokens), 2)
        self.assertEqual(self.client._dataset(dataset.id)[0].tokens, dataset[0].tokens)
        sampled = self.client._snapshot(query._proto.snapshot_ids[0])
        selected = sampled.query(sampling=p.sample(seed=4, tokens=1, tokenizer=tokenizer))
        self.assertEqual(selected.profile().sampling.realized, 2)
        self.assertEqual(selected.profile().sampling.overshoot, 1)

    @pytest.mark.integration
    def test_large_tokenizer_is_uploaded_once_and_recipes_are_compact(self) -> None:
        asset = self.root / "large-tokenizer.json"
        data = (
            b" " * (1024 * 1024 + 1)
            + (Path(__file__).parent / "fixtures" / "wordpiece.json").read_bytes()
        )
        asset.write_bytes(data)
        tokenizer = p.hugging_face_tokenizer(asset, digest=blake3(data).hexdigest())
        snapshot = self.snapshot("population", ["hello world"])
        query = snapshot.query(sampling=p.sample(seed=3, tokens=1, tokenizer=tokenizer))
        self.assertFalse(query._proto.sampling.tokenizer_json)
        dataset = query.dataset(tokenizer=tokenizer, sequence_length=2)
        self.assertLess(dataset._proto.ByteSize(), 32 * 1024)
        self.assertFalse(dataset._proto.tokenizer.hugging_face.json)
        mix = query.mix(tokens=4, tokenizer=tokenizer, sequence_length=2, n_candidates=1)
        self.assertEqual(mix[0].wait().profile().content_tokens, 4)
        self.client.close()
        self.client = p.PremixDB(storage=self.root, process_workers=2)
        built = (
            self.client._snapshot(snapshot.id)
            .query(sampling=p.sample(seed=4, tokens=1, tokenizer=tokenizer))
            .dataset(tokenizer=tokenizer, sequence_length=1)
        )
        self.assertEqual(built.profile().content_tokens, 2)


if __name__ == "__main__":
    unittest.main()
