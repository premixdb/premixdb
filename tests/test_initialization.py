"""Invalid configuration and failed startup leave no live resources behind."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import invalid_call

import premixdb as p
from premixdb.runtime import Coordinator
from premixdb.storage.objects import ObjectStore


@pytest.mark.parametrize("factory", [p.PremixDB, Coordinator])
@pytest.mark.parametrize(
    "option,value",
    [
        ("workers", 0),
        ("workers", True),
        ("workers", 1.5),
        ("process_workers", -1),
        ("process_workers", True),
        ("process_workers", 1.5),
        ("cache_bytes", -1),
        ("cache_bytes", True),
        ("cache_bytes", 1.5),
    ],
)
def test_invalid_execution_options_do_not_create_storage(
    tmp_path: Path, factory: type[p.PremixDB] | type[Coordinator], option: str, value: object
) -> None:
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match=option):
        if factory is p.PremixDB:
            invalid_call(factory, storage=root, **{option: value})
        else:
            invalid_call(factory, root, **{option: value})
    assert not root.exists()


def test_missing_read_only_catalog_does_not_create_directories(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    with pytest.raises(sqlite3.OperationalError):
        p.PremixDB(storage=root, read_only=True)
    assert not root.exists()


@pytest.mark.parametrize("read_only", [False, True])
def test_pathlike_storage_uses_the_filesystem_protocol(tmp_path: Path, read_only: bool) -> None:
    root = tmp_path / "database"

    class StoragePath:
        def __fspath__(self) -> str:
            return str(root)

        def __str__(self) -> str:
            raise AssertionError("storage paths must use __fspath__")

    with p.PremixDB(storage=root) as db:
        expected = db.Corpus("paths", [p.Source("a", "captured text")])
    with p.PremixDB(storage=StoragePath(), read_only=read_only) as db:
        snapshot = db.Corpus("paths")
        assert snapshot.id == expected.id
        assert snapshot.preview()[0]["text"] == "captured text"


@pytest.mark.parametrize(
    "failure",
    [
        "premixdb.runtime.coordinator.execution.Store",
        "premixdb.runtime.pipeline.PartitionPipeline",
    ],
)
def test_failed_startup_closes_owned_metadata_and_workers(tmp_path: Path, failure: str) -> None:
    opened: list[ObjectStore] = []
    pools: list[ThreadPoolExecutor] = []

    class TrackingStore(ObjectStore):
        def __init__(self, root: str | Path, *, metadata_path: str | Path | None = None) -> None:
            super().__init__(root, metadata_path=metadata_path)
            opened.append(self)

    class TrackingPool(ThreadPoolExecutor):
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
            pools.append(self)

    with (
        patch("premixdb.runtime.coordinator.ObjectStore", TrackingStore),
        patch("premixdb.runtime.materialization.ThreadPoolExecutor", TrackingPool),
        patch(failure, side_effect=OSError("startup failed")),
    ):
        with pytest.raises(OSError, match="startup failed"):
            Coordinator(tmp_path, process_workers=1)
    assert len(opened) == 1
    assert len(pools) == 2
    try:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            opened[0].metadata.contains("corpus", b"c" * 16)
        for pool in pools:
            with pytest.raises(RuntimeError, match="cannot schedule"):
                pool.submit(lambda: None)
    finally:
        opened[0].close()
        for pool in pools:
            pool.shutdown(wait=True)


INVALID_DURATIONS = (
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinite"),
    pytest.param(True, id="boolean"),
    pytest.param("1", id="string"),
    pytest.param(10**400, id="overflow"),
)


@pytest.mark.parametrize("option", ["timeout", "poll_interval"])
@pytest.mark.parametrize("value", [*INVALID_DURATIONS, pytest.param(None, id="none")])
def test_invalid_session_durations_fail_before_creating_storage(
    tmp_path: Path, option: str, value: object
) -> None:
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match=f"{option} must be positive and finite"):
        invalid_call(p.PremixDB, storage=root, **{option: value})
    assert not root.exists()


@pytest.mark.parametrize("value", INVALID_DURATIONS)
def test_invalid_wait_duration_does_not_execute_a_pending_recipe(
    tmp_path: Path, value: object
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        query = db.Corpus("pending", [p.Source("a", "one")]).query()
        assert query.status is p.ExecutionStatus.PENDING
        with patch.object(db, "_submit", side_effect=AssertionError("executed recipe")):
            with pytest.raises(ValueError, match="timeout must be positive and finite"):
                invalid_call(query.wait, timeout=value)
        assert query.status is p.ExecutionStatus.PENDING


def test_failed_worker_setup_preserves_caller_supplied_store(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        with patch(
            "premixdb.runtime.pipeline.PartitionPipeline", side_effect=OSError("no workers")
        ):
            with pytest.raises(OSError, match="no workers"):
                Coordinator(store, process_workers=1)
        assert store.metadata.contains("corpus", b"c" * 16) is False


def test_storage_creates_blob_namespaces_only_when_published(tmp_path: Path) -> None:
    from premixdb.storage.objects import ObjectStore

    with ObjectStore(tmp_path) as store:
        assert {path.name for path in tmp_path.iterdir()} <= {
            "metadata.sqlite3",
            "metadata.sqlite3-wal",
            "metadata.sqlite3-shm",
        }
        store.put("query", b"published")
        assert (tmp_path / "query/objects").is_dir()
        assert not (tmp_path / "execution").exists()
        assert not (tmp_path / "submission").exists()
    with ObjectStore(tmp_path, read_only=True):
        assert not (tmp_path / "dataset").exists()

    root = tmp_path / "session"
    with p.PremixDB(storage=root) as db:
        assert not (root / "snapshot").exists()
        db.Corpus("lazy", [p.Source("a", "published")])
        assert (root / "snapshot/objects").is_dir()
        assert (root / "snapshot/snapshots").is_dir()
