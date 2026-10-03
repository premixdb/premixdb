"""Bounded S2ORC demo preparation, cache integrity and publication failures."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

from premixdb import Source
from premixdb._typing import JSON

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_s2orc", ROOT / "scripts/prepare_s2orc.py")
assert SPEC is not None and SPEC.loader is not None
assert SPEC is not None and SPEC.loader is not None
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


class S2ORCDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "papers.jsonl"
        self.rows = [
            dict(id="paper/123", text="First paper é", source="s2orc/valid"),
            dict(id="paper/456", text="Second paper", source="s2orc/valid"),
        ]

    def response(self, rows: Sequence[Mapping[str, JSON]] | None = None) -> io.BytesIO:
        data = "".join(json.dumps(row) + "\n" for row in (rows or self.rows)).encode()
        # Trailing malformed input must not be consumed past the requested prefix.
        return io.BytesIO(gzip.compress(data + b"invalid trailing record\n"))

    def test_prefix_preserves_paper_ids_and_reuses_verified_cache(self) -> None:
        with patch.object(DEMO, "urlopen", return_value=self.response()) as request:
            DEMO.prepare(self.output, limit=2)
            request.assert_called_once_with(DEMO.URL, timeout=60)
        self.assertEqual(
            list(Source.read_jsonl(self.output, key_column="id")),
            [Source("paper/123", "First paper é"), Source("paper/456", "Second paper")],
        )
        with patch.object(DEMO, "urlopen", side_effect=AssertionError("downloaded again")):
            self.assertEqual(DEMO.prepare(self.output, limit=2), self.output)

    def test_changed_cache_is_prepared_again(self) -> None:
        with patch.object(DEMO, "urlopen", return_value=self.response()):
            DEMO.prepare(self.output, limit=2)
        self.output.write_text("corrupt cached data")
        with patch.object(DEMO, "urlopen", return_value=self.response()) as request:
            DEMO.prepare(self.output, limit=2)
            self.assertEqual(request.call_count, 1)
        self.assertEqual(len(list(Source.read_jsonl(self.output))), 2)

    def test_training_prefix_checks_split_and_cannot_reuse_validation_receipt(self) -> None:
        with patch.object(DEMO, "urlopen", return_value=self.response()):
            DEMO.prepare(self.output, limit=2)
        training = [dict(row, source="s2orc/train") for row in self.rows]
        with patch.object(DEMO, "urlopen", return_value=self.response(training)) as request:
            DEMO.prepare(self.output, limit=2, split="train")
            request.assert_called_once_with(DEMO.TRAIN_URL, timeout=60)
        receipt = json.loads(self.output.with_suffix(".receipt.json").read_text())
        self.assertEqual(receipt["url"], DEMO.TRAIN_URL)
        self.assertEqual(len(list(Source.read_jsonl(self.output, key_column="id"))), 2)
        with patch.object(DEMO, "urlopen", return_value=self.response()):
            with self.assertRaisesRegex(ValueError, "full-text S2ORC train"):
                DEMO.prepare(self.output, limit=1, split="train")

    def test_default_paths_keep_train_and_validation_separate(self) -> None:
        validation = self.output.with_name("validation.jsonl")
        training = self.output.with_name("train.jsonl")
        train_rows = [dict(row, source="s2orc/train") for row in self.rows]
        with (
            patch.object(DEMO, "OUTPUT", validation),
            patch.object(DEMO, "TRAIN_OUTPUT", training),
            patch.object(DEMO, "urlopen", side_effect=[self.response(), self.response(train_rows)]),
        ):
            self.assertEqual(DEMO.prepare(limit=2), validation)
            self.assertEqual(DEMO.prepare(limit=2, split="train"), training)
        self.assertTrue(validation.is_file())
        self.assertTrue(training.is_file())

    def test_failure_does_not_replace_existing_demo_or_leave_temporary_files(self) -> None:
        self.output.write_text("previous demo")
        for rows in (
            [dict(self.rows[0], source="s2ag/valid")],
            [self.rows[0], self.rows[0]],
            [dict(self.rows[0], text=None)],
        ):
            with (
                self.subTest(rows=rows),
                patch.object(DEMO, "urlopen", return_value=self.response(rows)),
            ):
                with self.assertRaises(ValueError):
                    DEMO.prepare(self.output, limit=2)
            self.assertEqual(self.output.read_text(), "previous demo")
            self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_unbounded_or_empty_preparation_is_rejected_before_network_access(self) -> None:
        with patch.object(DEMO, "urlopen", side_effect=AssertionError("opened network")):
            for limit in (0, -1, None):
                with self.assertRaises(ValueError):
                    DEMO.prepare(self.output, limit=limit)
