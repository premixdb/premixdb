"""SQLite publication, legacy migration, durable statistics, and backup recovery."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from _type_support import coordinator

import premixdb
from premixdb._ids import _public_dataset_profile
from premixdb.execution.metadata import MetadataStore
from premixdb.execution.planner import copy_fields
from premixdb.execution.storage import ObjectStore
from premixdb.v1 import corpus_pb2 as c
from premixdb.v1 import dataset_pb2 as d
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import snapshot_pb2 as s
from premixdb.v1 import status_pb2 as status
from premixdb.v1.storage_pb2 import SpanRef


class MetadataTests(unittest.TestCase):
    def test_resources_and_uint64_profiles_survive_database_only_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObjectStore(root / "original")
            resource = s.Snapshot(id=b"s" * 32, corpus_id=b"c" * 16)
            resource.profile.documents = 2**64 - 1
            store.save("snapshot", resource.id, resource)
            self.assertFalse(list(store.root.rglob("*.ref")))
            store.close()
            recovered_path = root / "recovered.sqlite3"
            shutil.copyfile(store.metadata.path, recovered_path)
            shutil.rmtree(store.root)
            recovered = ObjectStore(root / "replacement", metadata_path=recovered_path)
            self.addCleanup(recovered.close)
            self.assertEqual(recovered.load("snapshot", resource.id, s.Snapshot), resource)
            self.assertEqual(recovered.list("snapshot", s.Snapshot), [resource])
            detached = recovered.load("snapshot", resource.id, s.Snapshot)
            detached.profile.Clear()
            self.assertEqual(recovered.load("snapshot", resource.id, s.Snapshot), resource)

    def test_concurrent_immutable_publication_and_failure_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = ObjectStore(directory), ObjectStore(directory)
            self.addCleanup(first.close)
            self.addCleanup(second.close)
            corpus = c.Corpus(id=b"c" * 16, name="stable")
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(
                    pool.map(
                        lambda store: store.save("corpus", corpus.id, corpus), [first, second] * 8
                    )
                )
            with self.assertRaisesRegex(ValueError, "conflicting immutable"):
                second.save("corpus", corpus.id, c.Corpus(id=corpus.id, name="different"))
            failed = q.Query(id=b"q" * 32, status=status.STATUS_ERROR, error="first failure")
            first.save("query", failed.id, failed, suffix=".failed", failure=True)
            failed.error = "retry failed"
            second.save("query", failed.id, failed, suffix=".failed", failure=True)
            self.assertEqual(first.load("query", failed.id, q.Query, suffix=".failed"), failed)
            with self.assertRaisesRegex(ValueError, "only failed"):
                first.save("query", failed.id, failed, failure=True)

    def test_legacy_resources_are_imported_and_originals_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old = ObjectStore(directory)
            resources = [
                ("corpus", "", c.Corpus(id=b"c" * 16, name="legacy")),
                ("query", ".pending", q.Query(id=b"q" * 32)),
                ("dataset", ".recipe", d.Dataset(id=b"d" * 32)),
            ]
            originals = []
            for namespace, suffix, resource in resources:
                payload = resource.SerializeToString(deterministic=True)
                obj = old.put(namespace, payload)
                ref = SpanRef(object=obj, end=len(payload), blake3_digest=obj.blake3_digest)
                relative = f"{namespace}/{resource.id.hex()}{suffix}.ref"
                old._put(relative, ref.SerializeToString(deterministic=True))
                originals.append(old.root / relative)
            old.close()
            old.metadata.path.unlink()
            for _ in range(2):
                migrated = ObjectStore(directory)
                try:
                    for namespace, suffix, resource in resources:
                        self.assertEqual(
                            migrated.load(namespace, resource.id, type(resource), suffix=suffix),
                            resource,
                        )
                    self.assertTrue(all(path.exists() for path in originals))
                finally:
                    migrated.close()

    def test_integrity_and_message_type_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectStore(directory)
            self.addCleanup(store.close)
            corpus = c.Corpus(id=b"c" * 16, name="stable")
            store.save("corpus", corpus.id, corpus)
            with self.assertRaisesRegex(ValueError, "message type"):
                store.load("corpus", corpus.id, q.Query)
            with self.assertRaisesRegex(ValueError, "message type"):
                store.list("corpus", q.Query)
            with sqlite3.connect(store.metadata.path) as database:
                database.execute("UPDATE metadata SET payload=?", (b"corrupt",))
            with self.assertRaisesRegex(ValueError, "integrity"):
                store.list("corpus", c.Corpus)

    def test_lists_filter_namespace_and_suffix_and_return_sorted_detached_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectStore(directory)
            self.addCleanup(store.close)
            resources = [q.Query(id=bytes([i]) * 32) for i in (3, 1, 2)]
            for resource in resources:
                store.save("query", resource.id, resource, suffix=".pending")
            completed = q.Query(id=resources[0].id, status=status.STATUS_COMPLETED)
            store.save("query", completed.id, completed)
            dataset = d.Dataset(id=resources[1].id)
            store.save("dataset", dataset.id, dataset, suffix=".pending")

            pending = store.list("query", q.Query, suffix=".pending")
            self.assertEqual(pending, sorted(resources, key=lambda resource: resource.id))
            self.assertEqual(store.list("query", q.Query), [completed])
            self.assertEqual(store.list("dataset", d.Dataset, suffix=".pending"), [dataset])
            pending[0].Clear()
            self.assertEqual(
                store.list("query", q.Query, suffix=".pending"),
                sorted(resources, key=lambda resource: resource.id),
            )

    def test_unknown_schema_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.sqlite3"
            with sqlite3.connect(path) as database:
                database.execute("PRAGMA user_version=999")
            with self.assertRaisesRegex(ValueError, "schema version"):
                MetadataStore(path)

    def test_dataset_statistics_are_reused_after_restart_with_no_memory_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.sqlite3"
            storage_path = root / "objects"
            with premixdb.PremixDB(
                storage=storage_path, metadata_path=metadata_path, cache_bytes=0
            ) as client:
                query = client.corpus("stats", [premixdb.Source("a", "abcdef")]).query()
                mix = query.mix(tokenizer=premixdb.ByteTokenizer(), sequence_length=4)
                candidate = mix[0]
                expected = candidate.profile()
                recipe = copy_fields(candidate._proto, d.CreateDatasetRequest())
                self.assertTrue(
                    coordinator(client)._storage.metadata.ids("dataset", suffix=".profile")
                )
            with premixdb.PremixDB(
                storage=storage_path, metadata_path=metadata_path, cache_bytes=0
            ) as reopened:
                with patch.object(
                    coordinator(reopened), "_query", side_effect=AssertionError("recompute")
                ):
                    actual = _public_dataset_profile(
                        coordinator(reopened)._profile_dataset(recipe), corpus_strata=True
                    )
                self.assertEqual(actual, expected)

    def test_sqlite_snapshot_restores_profiles_after_original_storage_is_lost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObjectStore(root / "original")
            resource = d.Dataset(id=b"d" * 32, status=status.STATUS_COMPLETED)
            resource.profile.planned_content_tokens = 2**63 + 17
            store.save("dataset", resource.id, resource)
            later = s.Snapshot(id=b"s" * 32, corpus_id=b"c" * 16)
            later.profile.documents = 2**64 - 1
            store.save("snapshot", later.id, later)
            checkpoint = root / "catalog.sqlite3"
            store.metadata.backup(checkpoint)
            store.close()
            shutil.rmtree(store.root)
            restored = ObjectStore(root / "replacement", metadata_path=checkpoint)
            self.addCleanup(restored.close)
            self.assertEqual(restored.load("dataset", resource.id, d.Dataset), resource)
            self.assertEqual(restored.load("snapshot", later.id, s.Snapshot), later)


if __name__ == "__main__":
    unittest.main()
