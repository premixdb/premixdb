"""Capture, curation, packing, and resume through the direct local API."""

from __future__ import annotations

import gc
import json
import operator
import subprocess
import tempfile
import unittest
from collections.abc import Iterable, Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Unpack
from unittest.mock import patch

from _type_support import (
    PackingOptions,
    SnapshotOptions,
    invalid_call,
)

from premixdb import _runtime, local
from premixdb.engine import execution
from premixdb.local import (
    ByteTokenizer,
    Concat,
    PremixDB,
    Source,
    Topology,
    dedupe,
    object,
    text,
    where,
)

ROOT = Path(__file__).resolve().parents[1]
CODE = execution.CodeVersion("local://test", "a" * 40, "09" * 32)


class PremixDBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = PremixDB(storage=self.root / "store")
        self.corpus = self.client.corpus("test")

    def capture(
        self, rows: Iterable[tuple[str, str]], **kwargs: Unpack[SnapshotOptions]
    ) -> local.Snapshot:
        return self.corpus.snapshot(source=[Source(key, value) for key, value in rows], **kwargs)

    def pack(
        self, query: local.Query, length: int = 4, **kwargs: Unpack[PackingOptions]
    ) -> local.Dataset:
        return query.dataset(
            tokenizer=ByteTokenizer(), sequence_length=length, packing=Concat(**kwargs)
        )

    def test_capture_reopen_differential_and_storage_independent_identity(self) -> None:
        initial = [("a", "hello 🌍\r\n"), ("b", "gone"), ("empty", "")]
        first = self.capture(initial)
        reordered = self.capture(reversed(initial))
        self.assertEqual(first.id, reordered.id)
        other_store = PremixDB(storage=self.root / "other")
        copied = other_store.corpus("test").snapshot(source=[Source(k, v) for k, v in initial])
        self.assertEqual(first.id, copied.id)
        reopened = PremixDB(storage=self.root / "store").snapshot(first.id)
        self.assertEqual(reopened.summary(), {"documents": 3, "bytes": 16, "characters": 13})
        rows = [("a", initial[0][1]), ("empty", "changed"), ("new", "new")]
        next_snapshot = self.capture(rows, base=reopened)
        self.assertEqual(
            next_snapshot.changes(), dict(added=1, removed=1, changed=1, unchanged=1, reused=1)
        )
        full = self.capture(rows)
        self.assertEqual(
            [(r.id, r.text) for r in next_snapshot.query().rows()],
            [(r.id, r.text) for r in full.query().rows()],
        )
        self.assertNotEqual(next_snapshot.id, full.id)
        with self.assertRaises(ValueError):
            self.client.corpus("other").snapshot(source=[], base=first)
        consumed = []

        def sources() -> Iterator[Source]:
            consumed.append("read")
            yield Source("a", "unused")

        with self.assertRaisesRegex(ValueError, "base belongs to another corpus"):
            self.client.corpus("other").snapshot(source=sources(), base=first)
        self.assertEqual(consumed, [])

    def test_file_inputs_preserve_newlines_and_unicode(self) -> None:
        inputs = self.root / "inputs"
        (inputs / "nested").mkdir(parents=True)
        (inputs / "a.txt").write_bytes("é\r\n".encode())
        (inputs / "nested/empty.txt").write_bytes(b"")
        snap = self.corpus.snapshot(source=inputs)
        self.assertEqual(
            {r.source_key: r.text for r in snap.query().rows()},
            {"a.txt": "é\r\n", "nested/empty.txt": ""},
        )
        self.assertEqual(
            self.corpus.snapshot(source=inputs / "a.txt").summary(),
            dict(documents=1, bytes=4, characters=3),
        )
        with self.assertRaises(ValueError):
            self.corpus.snapshot(source=inputs / "missing.txt")
        (inputs / "invalid.txt").write_bytes(b"\xff")
        with self.assertRaises(ValueError):
            self.corpus.snapshot(source=inputs)

    def test_noop_capture_preserves_snapshot_and_edits_create_new_versions(self) -> None:
        path = self.root / "article.txt"
        path.write_bytes(b"original\r\n")
        original = self.corpus.snapshot(source=path)
        original_id = original.id
        original_summary = original.summary()
        with self.assertRaises(FrozenInstanceError):
            setattr(original, "_handle", None)
        with self.assertRaises(AttributeError):
            setattr(original._handle, "id", "changed")

        path.write_bytes(b"edited\r\n")
        edited = self.corpus.snapshot(source=path, base=original)
        self.assertNotEqual(edited.id, original_id)
        self.assertEqual(edited.changes()["changed"], 1)
        self.assertEqual(original.query().rows()[0].text, "original\r\n")
        unchanged = self.corpus.snapshot(source=path, base=edited)
        self.assertIs(unchanged, edited)
        self.assertEqual(unchanged.id, edited.id)
        self.assertEqual(unchanged.changes(), edited.changes())
        self.assertEqual(original.id, original_id)
        self.assertEqual(original.summary(), original_summary)
        self.assertEqual(self.client.snapshot(original_id).query().rows()[0].text, "original\r\n")

        # A new engine affects queries, not an existing snapshot's captured state.
        new_engine = execution.CodeVersion("premixdb://repository", "b" * 40, "00" * 32)
        with patch.object(_runtime, "current_code", return_value=new_engine):
            reopened = PremixDB(storage=self.root / "store").snapshot(edited.id)
            same = self.corpus.snapshot(source=path, base=reopened)
            self.assertEqual(same.id, edited.id)
            self.assertEqual(same.changes(), edited.changes())
        empty = self.capture([], base=edited)
        self.assertNotEqual(empty.id, edited.id)
        self.assertEqual(self.capture([], base=empty).id, empty.id)

    def test_query_freezes_execution_identity_and_plan_at_creation(self) -> None:
        with patch.object(_runtime, "current_code", side_effect=AssertionError("too early")):
            client = PremixDB(storage=self.root / "store")
        snapshot = self.capture([("a", "keep"), ("empty", "")])
        with patch.object(_runtime, "current_code", side_effect=AssertionError("too early")):
            union = client.snapshot(snapshot.id).union(snapshot)
        steps = [where(text.characters > 0)]
        first = union.query(steps=steps)
        dataset = self.pack(first)
        steps.clear()
        first_id = first.id
        new_engine = execution.CodeVersion("premixdb://repository", "b" * 40, "00" * 32)
        with patch.object(_runtime, "current_code", return_value=new_engine):
            second = union.query(steps=[where(text.characters > 0)])
            self.assertNotEqual(second.id, first_id)
            self.assertNotEqual(self.pack(second).id, dataset.id)
            self.assertEqual(self.pack(first).id, dataset.id)
        self.assertEqual(first.id, first_id)
        self.assertEqual([row.source_key for row in first.rows()], ["a"])
        with self.assertRaises(FrozenInstanceError):
            setattr(first, "_handle", second._handle)
        with self.assertRaises(TypeError):
            invalid_call(PremixDB, storage=self.root / "store", code=CODE)

    def test_runtime_identity_uses_repository_commit_and_is_captured_once(self) -> None:
        code = _runtime.current_code()
        self.assertEqual(code.repository, "premixdb://repository")
        self.assertEqual(code.commit, _runtime._capture_commit())
        with patch.object(
            subprocess, "check_output", side_effect=AssertionError("revision must be frozen")
        ):
            self.assertEqual(_runtime.current_code().commit, code.commit)
        with patch.dict(_runtime.os.environ, {"PREMIXDB_GIT_COMMIT": "a" * 40}):
            self.assertEqual(_runtime._capture_commit(), "a" * 40)
            self.assertEqual(_runtime.current_code().commit, code.commit)
        with patch.dict(_runtime.os.environ, {"PREMIXDB_GIT_COMMIT": "main"}):
            with self.assertRaisesRegex(ValueError, "full lowercase Git commit"):
                _runtime._capture_commit()
        with self.assertRaises(AttributeError):
            setattr(code, "commit", "c" * 40)

    def test_union_dedupe_order_and_complete_provenance(self) -> None:
        first = self.capture([("a", "same"), ("z", "same"), ("empty", "")])
        second = self.client.corpus("other").snapshot(
            source=[Source("b", "same"), Source("unique", "🌍")]
        )
        union = first.union(second).union(first)
        steps = [
            where(text.characters > 0),
            dedupe(order_by=[object.uri.asc()]),
            where(object.uri != "a"),
        ]
        query = union.query(steps=steps)
        self.assertEqual([r.source_key for r in query.rows()], ["unique"])
        self.assertEqual(query.id, second.union(first).query(steps=steps).id)
        self.assertEqual(query.summary()["input"]["documents"], 5)
        self.assertEqual([s["after"]["documents"] for s in query.summary()["steps"]], [4, 2, 1])
        provenance = {p["source_key"]: (id, p) for id, p in query.provenance().items()}
        self.assertEqual(
            provenance["z"][1]["selection"], dict(kind="duplicate", step=1, kept=provenance["a"][0])
        )
        self.assertEqual(provenance["a"][1]["selection"], dict(kind="filtered", step=2))
        self.assertEqual(provenance["empty"][1]["snapshots"], [first.id])
        descending = union.query(steps=[dedupe(order_by=[object.uri.desc()])])
        self.assertEqual({r.source_key for r in descending.rows()}, {"z", "unique", "empty"})
        self.assertEqual(
            first.query(steps=[where(text.bytes >= 4)]).summary()["output"]["documents"], 2
        )

    def test_packing_matches_flat_byte_stream_and_retains_provenance(self) -> None:
        query = self.capture([("a", "é🌍"), ("b", ""), ("c", "last\n")]).query()
        rows = query.rows()
        for length in (1, 4, 9, 32):
            for separator in (None, 256, 0):
                flat = []
                for row in rows:
                    flat.extend(row.text.encode())
                    if separator is not None:
                        flat.append(separator)
                for drop in (True, False):
                    with self.subTest(length=length, separator=separator, drop=drop):
                        dataset = self.pack(
                            query,
                            length,
                            separator=separator,
                            drop_remainder=drop,
                            pad_token=None if drop else 0,
                        )
                        tail = len(flat) % length
                        padding = (length - tail) if tail and not drop else 0
                        expected = (
                            flat[: len(flat) - tail] if drop and tail else flat + [0] * padding
                        )
                        self.assertEqual(
                            [token for seq in dataset for token in seq.tokens], expected
                        )
                        self.assertEqual(
                            [mask for seq in dataset for mask in seq.mask],
                            [True] * (len(expected) - padding) + [False] * padding,
                        )
                        self.assertEqual(dataset.summary()["output_tokens"], len(expected))
                        self.assertEqual(dataset.summary()["padding_tokens"], padding)
                        self.assertEqual(
                            [o["source"]["source_key"] for o in dataset.occurrences()],
                            [r.source_key for r in rows],
                        )
                        for seq in dataset:
                            self.assertEqual(seq.spans[0]["start"], 0)
                            self.assertEqual(seq.spans[-1]["end"], length)
        dataset = self.pack(query, 2)
        sequence = dataset[0]
        expected = sequence.tokens
        sequence.tokens.append(999)  # Returned lists never mutate engine data.
        del query, dataset
        gc.collect()
        self.assertEqual(sequence.tokens, expected)

    def test_reader_partition_json_resume_and_ownership(self) -> None:
        dataset = self.pack(self.capture([("a", "abcdefghijklmnopqrstuvwxyz")]).query(), 2)
        seen = []
        for rank in range(2):
            for worker in range(3):
                topology = Topology(rank, 2, worker, 3)
                reader = dataset.reader(topology=topology)
                first = next(reader, None)
                checkpoint = json.loads(json.dumps(reader.checkpoint()))
                remaining = [
                    s.ordinal for s in dataset.reader(topology=topology, checkpoint=checkpoint)
                ]
                expected = list(range(rank * 3 + worker, len(dataset), 6))
                self.assertEqual(([first.ordinal] if first else []) + remaining, expected)
                seen.extend(expected)
        self.assertEqual(sorted(seen), list(range(len(dataset))))
        reader = dataset.reader()
        self.assertIs(iter(reader), reader)
        list(reader)
        self.assertEqual(list(dataset.reader(checkpoint=reader.checkpoint())), [])
        self.assertEqual(dataset[-1].ordinal, len(dataset) - 1)
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset)]
        checkpoint = dataset.reader().checkpoint()
        for change in ({"dataset": "00" * 32}, {"next_ordinal": len(dataset)}, {"version": 9}):
            with self.assertRaises((ValueError, NotImplementedError)):
                invalid_call(dataset.reader, checkpoint=checkpoint | change)
        with self.assertRaises(ValueError):
            dataset.reader(topology=Topology(world_size=2), checkpoint=checkpoint)
        with self.assertRaises(ValueError):
            dataset.reader(topology=Topology(world_size=0))
        reader = dataset.reader()
        expected = dataset[0].tokens
        del dataset
        gc.collect()
        self.assertEqual(next(reader).tokens, expected)

    def test_empty_inputs_and_invalid_plans_fail_explicitly(self) -> None:
        query = self.capture([]).query()
        self.assertEqual(len(self.pack(query)), 0)
        self.assertEqual(list(self.pack(query).reader()), [])
        with self.assertRaises(ValueError):
            self.capture([("same", "a"), ("same", "b")])
        for value in (-1, 1.5, "3", True):
            with (
                self.subTest(value=value),
                self.assertRaises((TypeError, ValueError, OverflowError)),
            ):
                invalid_call(where, invalid_call(operator.gt, text.characters, value))
        with self.assertRaises(TypeError):
            invalid_call(where, object.uri == 1)
        with self.assertRaises(TypeError):
            where(0 < text.characters < 10)
        with self.assertRaises(TypeError):
            invalid_call(where, True)
        for kwargs in (
            {"drop_remainder": False},
            {"pad_token": 0},
            {"separator": -1},
            {"separator": True},
        ):
            with self.assertRaises(ValueError):
                invalid_call(Concat, **kwargs)
        with self.assertRaises(ValueError):
            self.pack(query, 0)
        with self.assertRaises(NotImplementedError):
            invalid_call(query.dataset, tokenizer="minhash", sequence_length=4)
        with self.assertRaises(NotImplementedError):
            PremixDB(storage="s3://bucket/premixdb")
        with self.assertRaises(ValueError):
            execution.CodeVersion("repo", "main", "00" * 32)
        with self.assertRaises(ValueError):
            execution.CodeVersion("repo", "a" * 40, "bad-digest")
        with self.assertRaises(TypeError):
            query = invalid_call(self.capture([]).query, sample=10)

    def test_execution_identity_matches_premixdb_golden_vectors(self) -> None:
        # Fixed corpus ID, source, code pins and byte-packing policy.
        store = execution.Store(self.root / "golden")
        snap = store.capture("01" * 16, [Source("a", "hello 🌍\n")], CODE)
        self.assertEqual(
            snap.id, "726d2924fe85585ccd97fe92cbbddf1e45a69fc15100e94d4d90e9c287ba379f"
        )
        query = execution.execute([], [where(text.characters >= 2)], CODE)
        self.assertEqual(
            query.id, "7263f1ad3bf14d52de14c124c533ba4f3363a3fc8d5d13121f32297326819c5e"
        )
        dataset = execution.execute([], [], CODE).dataset(4, 256, 257)
        self.assertEqual(
            dataset.id, "b1b29586b675a13b75ac9450f12483774ef8082b701943d9e774f9fdc65f51f9"
        )


if __name__ == "__main__":
    unittest.main()
