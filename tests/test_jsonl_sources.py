"""Row-level JSONL ingestion and reproducible C4 preparation, without network."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _type_support import invalid_call

from premixdb import PremixDB, Source
from premixdb.local import PremixDB as LocalDB

ROOT = Path(__file__).resolve().parents[1]


class JsonlSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_compression_relocation_and_duplicate_occurrences(self) -> None:
        rows = [{"text": "hello 🌍\r\n", "url": "same"}] * 2 + [{"text": ""}]
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        plain = self.root / "shard.jsonl"
        plain.write_bytes(payload)
        compressed = self.root / "elsewhere/shard.jsonl.gz"
        compressed.parent.mkdir()
        compressed.write_bytes(gzip.compress(payload))
        expected = [Source(f"shard.jsonl/{i:08d}", row["text"]) for i, row in enumerate(rows)]
        self.assertEqual(list(Source.read_jsonl(plain)), expected)
        self.assertEqual(list(Source.read_jsonl(compressed)), expected)
        # Both APIs capture separate row occurrences and identical content identity.
        for factory in (PremixDB, LocalDB):
            db = factory(storage=self.root / factory.__module__)
            if isinstance(db, PremixDB):
                self.addCleanup(db.close)
            if isinstance(db, PremixDB):
                first = db.corpus("jsonl", Source.read_jsonl(plain))
                same = db.corpus("jsonl", Source.read_jsonl(compressed), base=first)
                documents = first.profile().documents
            else:
                assert isinstance(db, LocalDB)
                corpus = db.corpus("jsonl")
                first = corpus.snapshot(source=Source.read_jsonl(plain))
                same = corpus.snapshot(source=Source.read_jsonl(compressed), base=first)
                documents = first.summary()["documents"]
            self.assertEqual(first.id, same.id)
            self.assertEqual(documents, 3)

    def test_limit_does_not_parse_later_rows_and_columns_are_configurable(self) -> None:
        path = self.root / "input.jsonl"
        path.write_text('{"id":"a", "body":"é"}\nnot json\n', encoding="utf-8")
        self.assertEqual(
            list(Source.read_jsonl(path, limit=1, text_column="body", key_column="id")),
            [Source("a", "é")],
        )
        self.assertEqual(
            list(Source.read_jsonl(path, limit=1, text_column="body", key_prefix="c4/en/shard")),
            [Source("c4/en/shard/00000000", "é")],
        )
        self.assertEqual(list(Source.read_jsonl(path, limit=0)), [])
        with self.assertRaisesRegex(ValueError, "row 2"):
            list(Source.read_jsonl(path, text_column="body"))
        for limit in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                invalid_call(list, invalid_call(Source.read_jsonl, path, limit=limit))

    def test_invalid_rows_and_duplicate_explicit_ids(self) -> None:
        path = self.root / "input.jsonl"
        for payload in (
            "[]",
            "{}",
            '{"text":null}',
            '{"text":42}',
            "broken",
            '{"text":"ok", "id":0}',
        ):
            path.write_text(payload + "\n")
            with self.assertRaisesRegex(ValueError, "row 1"):
                list(Source.read_jsonl(path, key_column="id" if '"id"' in payload else None))
        path.write_text('{"text":"a", "id":"duplicate"}\n' * 2)
        with self.assertRaisesRegex(ValueError, "row 2: duplicate"):
            list(Source.read_jsonl(path, key_column="id"))

    def test_preparation_verifies_cache_and_publishes_only_complete_downloads(self) -> None:
        spec = importlib.util.spec_from_file_location("prepare_c4", ROOT / "scripts/prepare_c4.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = gzip.compress(b'{"text":"hello"}\n')
        output = self.root / "download/shard.json.gz"
        with patch.object(module, "SHA256", hashlib.sha256(payload).hexdigest()):
            with patch.object(module, "urlopen", return_value=io.BytesIO(payload)):
                module.prepare(output)
            self.assertEqual(output.read_bytes(), payload)
            with patch.object(module, "urlopen", side_effect=AssertionError("cached")):
                module.prepare(output)
            output.write_bytes(b"old invalid cache")
            with patch.object(module, "urlopen", return_value=io.BytesIO(b"bad download")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    module.prepare(output)
            self.assertEqual(output.read_bytes(), b"old invalid cache")
            with patch.object(module, "urlopen", side_effect=OSError("connection lost")):
                with self.assertRaises(OSError):
                    module.prepare(output)
            self.assertEqual(list(output.parent.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
