"""Storage publication, immutable references and verified token-range reads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import premixdb
from premixdb.execution.storage import ObjectStore


class StorageTests(unittest.TestCase):
    def test_local_reader_requires_root_and_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectStore(directory)
            obj = store.put("dataset", b"abcd")
            span = premixdb.SpanRef(object=obj, end=4, blake3_digest=obj.blake3_digest)
            with self.assertRaisesRegex(ValueError, "explicit local"):
                premixdb.RangeReader().read(span)
            reader = premixdb.RangeReader(local_root=directory)
            self.assertEqual(reader.read(span), b"abcd")
            path = Path(directory) / "dataset/objects" / obj.blake3_digest.hex()
            path.write_bytes(b"abce")
            with self.assertRaisesRegex(ValueError, "integrity"):
                reader.read(span)
            with self.assertRaisesRegex(ValueError, "conflicting immutable"):
                store.put("dataset", b"abcd")


if __name__ == "__main__":
    unittest.main()
