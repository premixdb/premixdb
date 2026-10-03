"""Cold snapshot loads verify each object once; profiles need only metadata."""

from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb._ids import _decode_id
from premixdb.engine.snapshots import COMMIT_HEADER


def capture(root: Path) -> bytes:
    with p.PremixDB(storage=root) as db:
        snapshot = db.corpus("load", [p.Source("unicode", "pré 🌍\n"), p.Source("empty", "")])
        return _decode_id(snapshot.id)


def test_cold_snapshot_load_reads_each_committed_object_once(tmp_path: Path) -> None:
    identity = capture(tmp_path)
    with p.PremixDB(storage=tmp_path) as db:
        service = coordinator(db)
        store = service._store
        manifest = store._manifest(identity.hex())
        records = store._records(manifest)
        commit = (store.root / "snapshots" / identity.hex()).read_bytes()
        digests = [commit[len(COMMIT_HEADER) :].hex()]
        digests.extend(bytes(page["digest"]).hex() for page in manifest.get("pages", []))
        digests.extend(bytes(frame["digest"]).hex() for row in records for frame in row["frames"])
        with (
            patch.object(store, "_object", wraps=store._object) as engine_reads,
            patch.object(service._storage, "_get", wraps=service._storage._get) as storage_reads,
        ):
            snapshot = service._snapshot(identity)
        assert snapshot.id == identity.hex()
        assert Counter(
            bytes(call.args[0]).hex() for call in engine_reads.call_args_list
        ) == Counter(digests)
        storage_reads.assert_not_called()
        assert snapshot.documents["unicode"].text == "pré 🌍\n"


@pytest.mark.parametrize("kind", ["manifest", "page", "text"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_snapshot_load_rejects_missing_or_corrupt_graph_objects(
    tmp_path: Path, kind: str, damage: str
) -> None:
    identity = capture(tmp_path)
    with p.PremixDB(storage=tmp_path) as db:
        service = coordinator(db)
        store = service._store
        manifest = store._manifest(identity.hex())
        records = store._records(manifest)
        commit = (store.root / "snapshots" / identity.hex()).read_bytes()
        digest = (
            commit[len(COMMIT_HEADER) :]
            if kind == "manifest"
            else bytes(manifest["pages"][0]["digest"])
            if kind == "page"
            else bytes(next(row for row in records if row["frames"])["frames"][0]["digest"])
        )
        path = store.root / "objects" / digest.hex()
        if damage == "missing":
            path.unlink()
        else:
            path.write_bytes(b"corrupt")
        with pytest.raises(RuntimeError, match="committed object|checksum"):
            service._snapshot(identity)
        assert identity not in service._snapshot_handles
        if kind == "text":
            # Profiles still work when text storage is unavailable.
            profile = service._snapshot_resource(identity).profile
            assert profile.documents == 2
            assert profile.content_bytes == len("pré 🌍\n".encode())
        else:
            with pytest.raises((KeyError, ValueError), match="objects|integrity"):
                service._snapshot_resource(identity)
