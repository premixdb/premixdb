"""Storage publication, immutable references and verified token-range reads."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import BinaryIO
from unittest.mock import patch

import pytest

import premixdb
from premixdb.execution.storage import ObjectStore


def test_object_reads_bound_growth_and_keep_missing_errors(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        relative = "dataset/objects/small"
        path = tmp_path / relative
        path.write_bytes(b"abcd")
        assert store._get(relative, 4) == b"abcd"
        with pytest.raises(ValueError, match="size limit"):
            store._get(relative, 3)

        def grow_before_open(target: Path, mode: str = "r") -> BinaryIO:
            assert target == path and mode == "rb"
            with open(target, "ab") as writer:
                writer.write(b"growth")
            return open(target, "rb")

        with patch.object(Path, "open", autospec=True, side_effect=grow_before_open):
            with pytest.raises(ValueError, match="size limit"):
                store._get(relative, 4)
        path.unlink()
        with pytest.raises(KeyError) as missing:
            store._get(relative)
        assert missing.value.args == (relative,)


def test_range_size_check_uses_the_opened_file(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store, premixdb.RangeReader(local_root=tmp_path) as reader:
        object = store.put("dataset", b"abcd")
        span = premixdb.SpanRef(object=object, end=4, blake3_digest=object.blake3_digest)
        path = tmp_path / "dataset/objects" / object.blake3_digest.hex()
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"different size")

        def replace_after_open(target: Path, mode: str = "r") -> BinaryIO:
            assert target == path and mode == "rb"
            stream = open(target, "rb")
            replacement.replace(target)
            return stream

        with patch.object(Path, "open", autospec=True, side_effect=replace_after_open):
            assert reader.read(span) == b"abcd"
        with pytest.raises(ValueError, match="object size changed"):
            reader.read(span)


def test_local_reader_requires_root_and_detects_corruption(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store, premixdb.RangeReader(local_root=tmp_path) as reader:
        obj = store.put("dataset", b"abcd")
        span = premixdb.SpanRef(object=obj, end=4, blake3_digest=obj.blake3_digest)
        with pytest.raises(ValueError, match="explicit local"):
            premixdb.RangeReader().read(span)
        assert reader.read(span) == b"abcd"
        path = tmp_path / "dataset/objects" / obj.blake3_digest.hex()
        path.write_bytes(b"abce")
        with pytest.raises(ValueError, match="integrity"):
            reader.read(span)
        with pytest.raises(ValueError, match="conflicting immutable"):
            store.put("dataset", b"abcd")


def test_reader_context_releases_threads_and_pickle_keeps_only_configuration(
    tmp_path: Path,
) -> None:
    with ObjectStore(tmp_path) as store:
        objects = [store.put("dataset", data) for data in (b"first", b"second")]
    spans = [
        premixdb.SpanRef(object=obj, end=obj.size_bytes, blake3_digest=obj.blake3_digest)
        for obj in objects
    ]
    with premixdb.RangeReader(local_root=tmp_path) as reader:
        assert reader.read_many(spans) == [b"first", b"second"]
        pool = reader._pool
        assert pool is not None
        with pickle.loads(pickle.dumps(reader)) as restored:
            assert restored.local_root == reader.local_root
            assert restored._pool is None
            assert restored.read_many(spans) == [b"first", b"second"]
            assert restored._pool is not pool
    assert reader._pool is None
    with pytest.raises(RuntimeError, match="cannot schedule"):
        pool.submit(lambda: None)


def test_close_waits_for_in_flight_verified_reads(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        objects = [store.put("dataset", data) for data in (b"first", b"second")]
    spans = [
        premixdb.SpanRef(object=obj, end=obj.size_bytes, blake3_digest=obj.blake3_digest)
        for obj in objects
    ]
    entered, release, closing, closed = Event(), Event(), Event(), Event()
    with premixdb.RangeReader(local_root=tmp_path) as reader:
        original = reader._read_bytes

        def blocked_read(span: premixdb.SpanRef) -> bytes:
            entered.set()
            if not release.wait(5):
                raise TimeoutError("read gate was not released")
            return original(span)

        def close_reader() -> None:
            closing.set()
            reader.close()
            closed.set()

        with patch.object(reader, "_read_bytes", side_effect=blocked_read):
            with ThreadPoolExecutor(max_workers=2) as workers:
                reading = workers.submit(reader.read_many, spans)
                try:
                    assert entered.wait(5)
                    shutdown = workers.submit(close_reader)
                    assert closing.wait(5)
                    assert not closed.wait(0.01)
                finally:
                    release.set()
                assert reading.result(timeout=5) == [b"first", b"second"]
                shutdown.result(timeout=5)
                assert closed.is_set()
                assert reader._pool is None


def test_coordinator_releases_storage_when_shutdown_fails(tmp_path: Path) -> None:
    from premixdb.execution import Coordinator

    service = Coordinator(tmp_path, process_workers=1)
    assert service.pipeline is not None
    try:
        with patch.object(service._storage, "close", wraps=service._storage.close) as close:
            with patch.object(
                service.pipeline, "close", side_effect=OSError("worker close failed")
            ):
                with pytest.raises(OSError, match="worker close failed"):
                    service.close()
            close.assert_called_once()
    finally:
        service.close()
