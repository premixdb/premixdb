"""Persistence acceptance tests independent of the former PremixDB extension."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from premixdb.engine import snapshots
from premixdb.engine import snapshots as engine_snapshots
from premixdb.engine.contracts import Manifest
from premixdb.engine.identity import CodeVersion
from premixdb.engine.queries import CorpusIndex

CODE = CodeVersion("local://test", "a" * 40, "09" * 32)
CORPUS = "01" * 16


class SnapshotEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = snapshots.Store(self.root)

    def capture(
        self, sources: Iterable[tuple[str, str]], base: engine_snapshots.Snapshot | None = None
    ) -> engine_snapshots.Snapshot:
        return self.store.capture_inputs(CORPUS, sources, [], CODE.as_tuple(), base)

    def rewrite(self, id: str, transform: Callable[[Manifest], None]) -> None:
        manifest = self.store._manifest(id)
        transform(manifest)
        value = self.store._put(snapshots.encode(manifest))
        (self.root / "snapshots" / id).write_bytes(snapshots.COMMIT_HEADER + bytes(value))

    def test_checked_in_snapshot_loads(self) -> None:
        fixture = json.loads((Path(__file__).parent / "fixtures/snapshot-v2.json").read_text())
        for name, data in fixture["files"].items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes.fromhex(data))
        snapshot = self.store.load(fixture["id"])
        self.assertEqual(snapshot.documents["a"].text, "hello 🌍\n")
        self.assertEqual(snapshot.documents["empty"].text, "")
        self.assertEqual(snapshot._summary, dict(documents=2, bytes=11, characters=8))
        query = CorpusIndex([snapshot]).execute([], CODE)
        self.assertEqual(query.row_count, 2)
        self.assertEqual(query.dataset(4, 256, 257)._summary["content_tokens"], 11)

    def test_legacy_inline_manifest_without_profiles_loads(self) -> None:
        snapshot = self.capture([("a", "é\r\n"), ("empty", "")])

        def inline(manifest: Manifest) -> None:
            records = self.store._records(manifest)
            for record in records:
                for frame in record["frames"]:
                    frame.pop("profile")
            manifest.update(version=1, documents=records)
            manifest.pop("pages")

        self.rewrite(snapshot.id, inline)
        self.assertEqual(self.store.load(snapshot.id).documents["a"].text, "é\r\n")
        with self.assertRaisesRegex(RuntimeError, "profile"):
            self.store.objects(snapshot.id)

    def test_layout_and_inventory_order_do_not_change_identity(self) -> None:
        sources = [("a", "é🌍\r\n" * 2), ("empty", "")]
        snapshot = snapshots.Snapshot(CORPUS, sources, CODE)
        self.store.save(snapshot, frame_bytes=4)
        with tempfile.TemporaryDirectory() as directory:
            other = snapshots.Store(directory)
            rebuilt = snapshots.Snapshot(CORPUS, reversed(sources), CODE)
            other.save(rebuilt, frame_bytes=9)
            self.assertEqual(snapshot.id, rebuilt.id)
            first = self.store.load(snapshot.id)
            second = other.load(rebuilt.id)
            self.assertEqual(first._summary, second._summary)
            for store in (self.store, other):
                for record in store._records(store._manifest(snapshot.id)):
                    for frame in record["frames"]:
                        store._frame(frame).decode()
            a = CorpusIndex([first]).execute([], CODE).dataset(7, 256, 257)
            b = CorpusIndex([second]).execute([], CODE).dataset(7, 256, 257)
            self.assertEqual(a.id, b.id)
            assert a._sequences is not None and b._sequences is not None
            self.assertEqual([s.tokens for s in a._sequences], [s.tokens for s in b._sequences])

    def test_restart_and_differential_load_require_no_ancestors(self) -> None:
        base = self.capture([("a", "unchanged"), ("b", "old"), ("removed", "x")])
        changed = self.capture([("a", "unchanged"), ("b", "new"), ("added", "é")], base)
        (self.root / "snapshots" / base.id).unlink()
        reopened = snapshots.Store(self.root).load(changed.id)
        self.assertEqual(
            reopened._changes, dict(added=1, changed=1, removed=1, unchanged=1, reused=1)
        )
        self.assertEqual(reopened.documents["b"].text, "new")
        self.assertEqual(
            self.capture([(k, d.text) for k, d in reopened.documents.items()], reopened).id,
            changed.id,
        )

    def test_noop_capture_publishes_base_in_another_store(self) -> None:
        base = self.capture([("a", "unchanged")])
        with tempfile.TemporaryDirectory() as directory:
            other = snapshots.Store(directory)
            result = other.capture_inputs(CORPUS, [("a", "unchanged")], [], CODE.as_tuple(), base)
            self.assertEqual(result.id, base.id)
            self.assertEqual(other.load(base.id)._changes, base._changes)

    def test_reference_joins_choose_canonical_witness_without_cartesian_expansion(self) -> None:
        target = self.capture([("target", "é\nunique")])
        reference = snapshots.Snapshot(
            "02" * 16, [("a", "é\nreference"), ("b", "é\nreference")], CODE
        )
        index = CorpusIndex([target, reference])
        matches = index.reference_matches([target.id], (reference.id,), "Line")
        self.assertEqual(len(matches), 1)
        matched, kept = matches[0]
        self.assertEqual(matched[1:], (0, 2))
        self.assertEqual(kept[1:], (0, 2))
        self.assertEqual(kept[0], min(d.id for d in reference.documents.values()))
        self.assertEqual(index.reference_matches([target.id], [reference.id]), [])
        self.assertEqual(len(index.reference_matches([target.id], [target.id])), 1)
        for targets, references, unit in (
            ([], [target.id], "Line"),
            (["00" * 32], [target.id], "Line"),
            ([target.id], [target.id], "Unknown"),
        ):
            with self.assertRaises(ValueError):
                index.reference_matches(targets, references, unit)

    def test_corrupt_missing_and_oversized_committed_objects_fail(self) -> None:
        snapshot = self.capture([("a", "payload")])
        record = self.store._records(self.store._manifest(snapshot.id))[0]
        path = self.root / "objects" / bytes(record["frames"][0]["digest"]).hex()
        original = path.read_bytes()
        for data in (b"wrong", original + b"x"):
            path.write_bytes(data)
            with self.assertRaises(RuntimeError):
                self.store.load(snapshot.id)
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "committed object"):
            self.store.load(snapshot.id)

    def test_corrupt_metadata_with_valid_checksums_fails(self) -> None:
        for mutation in (
            lambda m: m["summary"].update(characters=999),
            lambda m: m["changes"].update(reused=999),
            lambda m: m.update(inventory=[0] * 32),
            lambda m: m.update(membership=[0] * 32),
            lambda m: m.update(version=9),
        ):
            with self.subTest(mutation=mutation):
                snapshot = self.capture([("a", "text")])
                self.rewrite(snapshot.id, mutation)
                with self.assertRaises(RuntimeError):
                    self.store.load(snapshot.id)
                (self.root / "snapshots" / snapshot.id).unlink()

    def test_pages_validate_sorted_keys_sizes_and_profiles(self) -> None:
        snapshot = self.capture([("a", "text"), ("b", "other")])
        original = self.store._manifest(snapshot.id)
        records = self.store._records(original)
        for mutate in (
            lambda r: r.reverse(),
            lambda r: r[1].update(key="a"),
            lambda r: r[0].update(bytes=100),
            lambda r: r[0]["frames"][0]["profile"].update(characters=99),
        ):
            rows = json.loads(json.dumps(records))
            mutate(rows)
            raw = snapshots.encode(rows)

            def replace_page(m: Manifest) -> None:
                m["pages"] = [dict(digest=self.store._put(raw), bytes=len(raw))]

            self.rewrite(snapshot.id, replace_page)
            with self.assertRaises(RuntimeError):
                self.store.load(snapshot.id)

    def test_metadata_reports_do_not_read_text_frames(self) -> None:
        snapshot = self.capture([("a", "payload"), ("empty", "")])
        frames = {
            bytes(f["digest"])
            for r in self.store._records(self.store._manifest(snapshot.id))
            for f in r["frames"]
        }
        read = self.store._object

        def metadata_only(value: list[int], limit: int = snapshots.MANIFEST_BYTES) -> bytes:
            self.assertNotIn(bytes(value), frames)
            return read(value, limit)

        with patch.object(self.store, "_object", side_effect=metadata_only):
            rows = self.store.objects(snapshot.id)
            self.assertEqual(
                {k: rows["a"][k] for k in ("spans", "content_bytes", "characters", "newlines")},
                dict(spans=1, content_bytes=7, characters=7, newlines=0),
            )
            self.assertEqual(len(rows), 2)

    def test_parallel_publication_preserves_first_valid_layout(self) -> None:
        snapshot = snapshots.Snapshot(CORPUS, [("a", "é🌍" * 2)], CODE)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda size: self.store.save(snapshot, size), (4, 9, 20, 37)))
        self.assertEqual(self.store.load(snapshot.id).documents["a"].text, "é🌍" * 2)
        self.assertEqual(list(self.root.rglob(".tmp-*")), [])

    def test_interrupted_publication_is_retryable_and_not_visible(self) -> None:
        snapshot = snapshots.Snapshot(CORPUS, [("a", "payload")], CODE)
        publish = snapshots._publish

        def interrupt(path: Path, data: bytes) -> None:
            if path.parent.name == "snapshots":
                raise OSError("interrupted before commit")
            publish(path, data)

        with patch.object(snapshots, "_publish", side_effect=interrupt):
            with self.assertRaises(OSError):
                self.store.save(snapshot)
        with self.assertRaisesRegex(ValueError, "not committed"):
            self.store.load(snapshot.id)
        self.store.save(snapshot)
        self.assertEqual(self.store.load(snapshot.id).id, snapshot.id)


if __name__ == "__main__":
    unittest.main()
