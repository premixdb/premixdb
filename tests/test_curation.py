"""Python curation contracts, using local assets and hand-derived expectations."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Unpack

from _type_support import (
    SnapshotOptions,
    TokenizerOptions,
    invalid_call,
)
from blake3 import blake3

from premixdb import local
from premixdb.local import (
    ByteTokenizer,
    Concat,
    HuggingFaceTokenizer,
    PremixDB,
    Source,
    SourceGroup,
    dedupe,
    object,
    where,
)

FIXTURES = Path(__file__).parent / "fixtures"


class CurationTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.client = PremixDB(storage=self.root / "store")

    def capture(
        self,
        rows: Iterable[tuple[str, str]],
        corpus: str = "test",
        **kwargs: Unpack[SnapshotOptions],
    ) -> local.Snapshot:
        return self.client.corpus(corpus).snapshot(
            source=[Source(key, value) for key, value in rows], **kwargs
        )

    def tokenizer(
        self, name: str = "wordpiece", **kwargs: Unpack[TokenizerOptions]
    ) -> local.HuggingFaceTokenizer:
        path = FIXTURES / f"{name}.json"
        return HuggingFaceTokenizer(path, digest=blake3(path.read_bytes()).hexdigest(), **kwargs)

    def test_line_witnesses_are_utf8_ranges_and_decisions_are_simultaneous(self) -> None:
        snapshot = self.capture(
            [
                ("a", "é\n"),
                ("b", "other\né\nshared"),
                ("c", "shared"),
                ("cr", "é\r\n"),
                ("empty", "\n\n"),
                ("repeat", "unique\nunique"),
            ]
        )
        query = snapshot.query(steps=[dedupe(comparison="line", order_by=[object.uri.asc()])])
        self.assertEqual({r.source_key for r in query.rows()}, {"a", "cr", "empty", "repeat"})
        rows = {r.source_key: r for r in snapshot.query().rows()}
        provenance = query.provenance()
        self.assertEqual(
            provenance[rows["b"].id]["selection"],
            {
                "kind": "duplicate_unit",
                "step": 0,
                "matched": {"document": rows["b"].id, "start": 6, "end": 8},
                "kept": {"document": rows["a"].id, "start": 0, "end": 2},
            },
        )
        # c loses to b even though b loses to a in this same step.
        self.assertEqual(
            provenance[rows["c"].id]["selection"]["kept"],
            {
                "document": rows["b"].id,
                "start": 9,
                "end": 15,
            },
        )
        for entry in provenance.values():
            selection = entry["selection"]
            if selection["kind"] == "duplicate_unit":
                texts = {row.id: row.text.encode() for row in rows.values()}
                matched, kept = selection["matched"], selection["kept"]
                assert isinstance(kept, dict)
                self.assertEqual(
                    texts[matched["document"]][matched["start"] : matched["end"]],
                    texts[kept["document"]][kept["start"] : kept["end"]],
                )
        self.assertEqual(json.loads(json.dumps(provenance)), provenance)
        self.assertEqual(query.summary()["steps"][0]["after"]["documents"], 4)

    def test_source_groups_remove_siblings_but_are_scoped_to_corpus(self) -> None:
        snapshot = self.capture(
            [
                ("a/first", "shared"),
                ("b/match", "shared"),
                ("b/sibling", "unique"),
            ]
        )
        other = self.capture([("b/safe", "different")], corpus="other")
        policy = dedupe(removal=SourceGroup(), order_by=[object.uri.asc()])
        query = snapshot.union(other).query(steps=[policy])
        self.assertEqual({r.source_key for r in query.rows()}, {"a/first", "b/safe"})
        entries = {p["source_key"]: p for p in query.provenance().values()}
        self.assertEqual(entries["b/match"]["selection"], entries["b/sibling"]["selection"])
        self.assertEqual(query.id, other.union(snapshot, snapshot).query(steps=[policy]).id)
        before = snapshot.query(steps=[where(object.uri != "b/match"), policy])
        after = snapshot.query(steps=[policy, where(object.uri != "b/match")])
        self.assertEqual({r.source_key for r in before.rows()}, {"a/first", "b/sibling"})
        self.assertEqual({r.source_key for r in after.rows()}, {"a/first"})

    def test_group_ranking_unicode_separator_and_differential_capture(self) -> None:
        rows = [("a→best", "no match"), ("a→z", "shared\ntail"), ("b→x", "shared")]
        initial = self.capture(rows)
        policy = dedupe(comparison="line", removal=SourceGroup("→"), order_by=[object.uri.asc()])
        query = initial.query(steps=[policy])
        self.assertEqual({r.source_key for r in query.rows()}, {"a→best", "a→z"})
        changed = rows + [("c→x", "new")]
        incremental = self.capture(changed, base=initial).query(steps=[policy])
        full = self.capture(reversed(changed)).query(steps=[policy])
        self.assertEqual([r.id for r in incremental.rows()], [r.id for r in full.rows()])
        self.assertEqual(incremental.summary(), full.summary())
        default = initial.query(steps=[dedupe()])
        self.assertEqual(default.id, initial.query(steps=[dedupe(comparison="document")]).id)
        self.assertNotEqual(default.id, initial.query(steps=[dedupe(comparison="line")]).id)

    def test_invalid_dedupe_policies_are_rejected_early(self) -> None:
        for separator in ("", "//", "\0", "\ud800"):
            with self.subTest(separator=repr(separator)), self.assertRaises(ValueError):
                SourceGroup(separator)
        with self.assertRaises(TypeError):
            invalid_call(SourceGroup, 1)
        with self.assertRaises(ValueError):
            invalid_call(dedupe, comparison="paragraph")
        with self.assertRaises(TypeError):
            invalid_call(dedupe, removal="corpus")
        with self.assertRaises(TypeError):
            invalid_call(dedupe, order_by=[object.uri])

    def test_model_tokenizer_golden_vectors_and_identity(self) -> None:
        tokenizer = self.tokenizer()
        self.assertEqual(
            tokenizer.asset_digest,
            "e2f3f0814e9c45a1dd889a9ca0ba4c431d5f828dc672ae78a51b9d38975f2ef8",
        )
        self.assertEqual(
            tokenizer.definition, "6d87dc458f151ba153cf5ef906a921223704e0a7651f41e4a164afc3b2fa7637"
        )
        for text, tokens in [
            ("", []),
            ("Hello world!", [4, 5, 8]),
            ("playing", [6, 7]),
            ("CAFÉ cafe\u0301", [9, 9]),
            ("中文 🌍", [10, 11, 0]),
            ("[CLS]hello[SEP]", [1, 4, 2]),
        ]:
            self.assertEqual(tokenizer.encode(text), tokens)
        self.assertEqual(self.tokenizer("bpe").encode("abab ab é🙂"), [4, 3, 5, 6])
        self.assertEqual(tokenizer.token_to_id("[SEP]"), 2)
        self.assertIsNone(tokenizer.token_to_id("missing"))
        with self.assertRaises(FrozenInstanceError):
            setattr(tokenizer, "_handle", None)

    def test_model_tokens_pack_with_provenance_and_checkpoint_resume(self) -> None:
        tokenizer = self.tokenizer()
        query = self.capture([("a", "HELLO world"), ("b", "playing"), ("c", "")]).query()
        expected_tokens = {"a": [4, 5], "b": [6, 7], "c": []}
        flat = [t for row in query.rows() for t in [*expected_tokens[row.source_key], 2]]
        for length in (1, 4, 8):
            for drop in (True, False):
                dataset = query.dataset(
                    tokenizer=tokenizer,
                    sequence_length=length,
                    packing=Concat(separator=2, drop_remainder=drop, pad_token=None if drop else 3),
                )
                tail = len(flat) % length
                padding = (length - tail) if tail and not drop else 0
                expected = flat[: len(flat) - tail] if drop and tail else flat + [3] * padding
                self.assertEqual([t for seq in dataset for t in seq.tokens], expected)
                self.assertEqual(
                    [m for seq in dataset for m in seq.mask],
                    [True] * (len(expected) - padding) + [False] * padding,
                )
                self.assertEqual(dataset.tokenizer_definition, tokenizer.definition)
                self.assertEqual(dataset.summary()["content_tokens"], 4)
                self.assertEqual(dataset.summary()["separator_tokens"], 3)
                self.assertEqual(
                    [o["tokens"] for o in dataset.occurrences()],
                    [len(expected_tokens[row.source_key]) for row in query.rows()],
                )
                reader = dataset.reader()
                next(reader, None)
                state = json.loads(json.dumps(reader.checkpoint()))
                self.assertEqual(
                    [s.tokens for s in dataset.reader(checkpoint=state)],
                    [dataset[i].tokens for i in range(1, len(dataset))],
                )
        byte_dataset = query.dataset(tokenizer=ByteTokenizer(), sequence_length=4)
        self.assertNotEqual(byte_dataset.tokenizer_definition, tokenizer.definition)

    def test_asset_capture_pins_and_admission_limits(self) -> None:
        data = (FIXTURES / "wordpiece.json").read_bytes()
        path = self.root / "tokenizer.json"
        path.write_bytes(data)
        tokenizer = HuggingFaceTokenizer(path, digest=blake3(data).hexdigest())
        self.assertEqual(tokenizer.definition, self.tokenizer().definition)
        path.write_bytes(data + b"\n")
        repackaged = HuggingFaceTokenizer(path, digest=blake3(data + b"\n").hexdigest())
        self.assertNotEqual(repackaged.definition, tokenizer.definition)
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            HuggingFaceTokenizer(path, digest=blake3(data).hexdigest())
        path.unlink()
        self.assertEqual(tokenizer.encode("hello"), [4])
        query = self.capture([("a", "hello")]).query()
        first = query.dataset(tokenizer=tokenizer, sequence_length=1)
        second = query.dataset(tokenizer=repackaged, sequence_length=1)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first[0].tokens, second[0].tokens)
        limited = self.tokenizer(max_document_bytes=4)
        self.assertEqual(limited.definition, tokenizer.definition)
        with self.assertRaisesRegex(NotImplementedError, "limit"):
            limited.encode("ééé")
        with self.assertRaisesRegex(NotImplementedError, "limit"):
            query.dataset(tokenizer=limited, sequence_length=1)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                invalid_call(self.tokenizer, max_document_bytes=value)
        with self.assertRaises(ValueError):
            HuggingFaceTokenizer(FIXTURES / "wordpiece.json", digest="invalid")

    def test_unsupported_tokenizer_policies_fail_instead_of_changing_tokens(self) -> None:
        wordpiece = json.loads((FIXTURES / "wordpiece.json").read_bytes())
        bpe = json.loads((FIXTURES / "bpe.json").read_bytes())
        bpe["model"]["dropout"] = 0.5
        cases = [
            wordpiece | {"truncation": {"max_length": 2, "strategy": "LongestFirst", "stride": 0}},
            wordpiece
            | {
                "padding": {
                    "strategy": {"Fixed": 8},
                    "direction": "Right",
                    "pad_to_multiple_of": None,
                    "pad_id": 3,
                    "pad_type_id": 0,
                    "pad_token": "[PAD]",
                }
            },
            bpe,
        ]
        path = self.root / "invalid-policy.json"
        for case in cases:
            data = json.dumps(case).encode()
            path.write_bytes(data)
            with self.assertRaises(NotImplementedError):
                HuggingFaceTokenizer(path, digest=blake3(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
