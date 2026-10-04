"""Publication preserves the winner and cleans staging files after failures."""

import os
import stat
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from premixdb.storage import publication as _files
from premixdb.storage.objects import ObjectStore


def test_publication_leaves_existing_bytes_untouched(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "object"
    assert _files.publish(path, b"first")
    assert not _files.publish(path, b"first")
    assert not _files.publish(path, b"second")
    assert path.read_bytes() == b"first"
    assert list(path.parent.iterdir()) == [path]


def test_concurrent_writers_publish_one_complete_winner(tmp_path: Path) -> None:
    path = tmp_path / "object"
    payloads = [bytes([value]) * 100_000 for value in range(8)]
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda data: _files.publish(path, data), payloads))
    assert sum(results) == 1
    assert path.read_bytes() == payloads[results.index(True)]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("phase", ["file", "directory"])
@pytest.mark.parametrize("stream", [False, True])
def test_sync_failure_cleans_staging_and_allows_retry(
    tmp_path: Path, phase: str, stream: bool
) -> None:
    path = tmp_path / "object"
    sync = os.fsync
    directories: list[int] = []

    def fail(descriptor: int) -> None:
        directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if directory:
            directories.append(descriptor)
        if directory == (phase == "directory"):
            raise OSError("sync failed")
        sync(descriptor)

    with patch.object(_files.os, "fsync", side_effect=fail):
        with pytest.raises(OSError, match="sync failed"):
            _files.publish(path, iter((b"com", b"plete")) if stream else b"complete")
    assert path.exists() == (phase == "directory")
    assert list(tmp_path.glob(".tmp-*")) == []
    for descriptor in directories:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert _files.publish(path, b"complete") == (phase == "file")
    assert path.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [path]


def test_object_storage_syncs_data_before_published_directory(tmp_path: Path) -> None:
    events: list[str] = []
    sync = os.fsync
    with ObjectStore(tmp_path) as store:

        def record(descriptor: int) -> None:
            directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
            if directory:
                assert [path.read_bytes() for path in (tmp_path / "dataset/objects").iterdir()] == [
                    b"complete",
                    b"complete",
                ]
            events.append("directory" if directory else "file")
            sync(descriptor)

        with patch.object(_files.os, "fsync", side_effect=record):
            obj = store.put("dataset", b"complete")
        assert events == ["file", "directory"]
        path = tmp_path / "dataset/objects" / obj.blake3_digest.hex()
        assert list(path.parent.iterdir()) == [path]


def test_interrupted_stream_cleans_staging_while_the_traceback_remains_alive(
    tmp_path: Path,
) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"partial"
        raise RuntimeError("stream interrupted")

    path = tmp_path / "object"
    with pytest.raises(RuntimeError, match="stream interrupted") as failure:
        _files.publish(path, chunks())
    assert failure.value.__traceback__ is not None
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
