"""Object references and verified byte-range reads, independent of the execution engine."""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Self, TypedDict
from urllib.parse import unquote, urlsplit

from blake3 import blake3

from .v1.storage_pb2 import SpanRef


class _ReaderState(TypedDict):
    local_root: Path | None


class RangeReader:
    """Read verified local file ranges within an explicit storage root."""

    def __init__(self, *, local_root: str | Path | None = None) -> None:
        self.local_root = Path(local_root).resolve() if local_root is not None else None
        self._pid = os.getpid()
        self._lock = RLock()
        self._pool: ThreadPoolExecutor | None = None

    def __getstate__(self) -> _ReaderState:
        return _ReaderState(local_root=self.local_root)

    def __setstate__(self, state: _ReaderState) -> None:
        RangeReader.__init__(self, local_root=state["local_root"])

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _after_fork(self) -> None:
        if os.getpid() != self._pid:
            self._pool: ThreadPoolExecutor | None = None
            self._lock = RLock()
            self._pid = os.getpid()

    @staticmethod
    def _validate(span: SpanRef) -> None:
        if (
            not 0 <= span.start <= span.end <= span.object.size_bytes
            or span.end - span.start > 64 * 1024 * 1024
        ):
            raise ValueError("invalid or oversized object range")
        if len(span.blake3_digest) != 32 or len(span.object.blake3_digest) != 32:
            raise ValueError("object and span require BLAKE3 digests")

    def _read_bytes(self, span: SpanRef) -> bytes:
        object = span.object
        size = span.end - span.start
        location = urlsplit(object.uri)
        if size == 0:
            data = b""
        elif location.scheme == "file":
            if location.netloc not in ("", "localhost") or self.local_root is None:
                raise ValueError("file ranges require an explicit local storage root")
            path = Path(unquote(location.path)).resolve()
            if not path.is_relative_to(self.local_root):
                raise ValueError("object range escapes the local storage root")
            with path.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size != object.size_bytes:
                    raise ValueError("object size changed")
                stream.seek(span.start)
                data = stream.read(size)
        else:
            raise ValueError("unsupported object URI")
        if len(data) != size:
            raise ValueError("span integrity check failed")
        return data

    def read(self, span: SpanRef) -> bytes:
        """Read and verify the bytes covered by an immutable storage span."""
        self._validate(span)
        return self._verified(span, self._read_bytes(span))

    @staticmethod
    def _verified(span: SpanRef, data: bytes) -> bytes:
        digest = blake3(data).digest()
        if digest != span.blake3_digest:
            raise ValueError("span integrity check failed")
        if span.start == 0 and span.end == span.object.size_bytes:
            if digest != span.object.blake3_digest:
                raise ValueError("object integrity check failed")
        return data

    def read_many(self, spans: Iterable[SpanRef], *, max_gap_bytes: int = 65536) -> list[bytes]:
        """Coalesce nearby ranges, fetch distinct objects concurrently, verify every span."""
        if type(max_gap_bytes) is not int or not 0 <= max_gap_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_gap_bytes must be a bounded nonnegative integer")
        spans = list(spans)
        groups = {}
        for index, span in enumerate(spans):
            self._validate(span)
            key = span.object.SerializeToString(deterministic=True)
            groups.setdefault(key, []).append((index, span))
        tasks = []
        for group in groups.values():
            ordered = sorted(group, key=lambda item: (item[1].start, item[1].end))
            chunk, start, end = [], 0, 0
            for item in ordered:
                span = item[1]
                if chunk and (
                    span.start > end + max_gap_bytes
                    or max(end, span.end) - start > 64 * 1024 * 1024
                ):
                    tasks.append((start, end, chunk))
                    chunk = []
                if not chunk:
                    start, end = span.start, span.end
                else:
                    end = max(end, span.end)
                chunk.append(item)
            if chunk:
                tasks.append((start, end, chunk))

        def fetch(task: tuple[int, int, list[tuple[int, SpanRef]]]) -> list[tuple[int, bytes]]:
            start, end, group = task
            merged = SpanRef(object=group[0][1].object, start=start, end=end)
            data = self._read_bytes(merged)
            values = []
            verified: dict[tuple[int, int, bytes], bytes] = {}
            for index, span in group:
                key = span.start, span.end, span.blake3_digest
                value = verified.get(key)
                if value is None:
                    value = self._verified(span, data[span.start - start : span.end - start])
                    verified[key] = value
                values.append((index, value))
            return values

        if len(tasks) > 1:
            # Recreate thread pools after fork/spawn.
            self._after_fork()
            with self._lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(
                        max_workers=4, thread_name_prefix="premixdb-reads"
                    )
                fetched = self._pool.map(fetch, tasks)
        else:
            fetched = map(fetch, tasks)
        result = [b""] * len(spans)
        for values in fetched:
            for index, value in values:
                result[index] = value
        return result

    def close(self) -> None:
        """Release optional IO threads; stored references remain independently readable."""
        self._after_fork()
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
