"""Transactional protobuf metadata and portable SQLite snapshots."""

from __future__ import annotations

import builtins
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Callable

from blake3 import blake3
from google.protobuf.message import Message

from ..v1.status_pb2 import ExecutionEvent


class MetadataStore:
    """Keep complete typed profiles without losing uint64 precision to JSON/SQL."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._closed = False
        self.before_save: Callable[[str, bytes, bytes, str, bool], None] | None = None
        self.after_save: Callable[[], None] | None = None
        self.read_only = read_only
        self._db = sqlite3.connect(
            self.path.as_uri() + "?mode=ro" if read_only else self.path,
            uri=read_only,
            timeout=30,
            check_same_thread=False,
        )
        try:
            if not read_only:
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA synchronous=FULL")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"unsupported metadata schema version: {version}")
            if read_only:
                if version != 1:
                    raise ValueError("read-only metadata requires a published catalog")
                return
            with self._db:
                self._db.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        namespace TEXT NOT NULL,
                        id BLOB NOT NULL,
                        suffix TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        payload BLOB NOT NULL,
                        digest BLOB NOT NULL,
                        PRIMARY KEY (namespace, id, suffix)
                    ) WITHOUT ROWID
                """)
                self._db.execute("""
                    CREATE TABLE IF NOT EXISTS migrations (name TEXT PRIMARY KEY)
                """)
                self._db.execute("PRAGMA user_version=1")
        except BaseException:
            self._db.close()
            raise

    def begin_execution(
        self, operation: str, request_digest: bytes = b"", resource_id: bytes = b""
    ) -> ExecutionEvent:
        import time
        from uuid import uuid4

        from ..v1.status_pb2 import ExecutionEvent

        event = ExecutionEvent(
            id=uuid4().hex,
            operation=operation,
            resource_id=resource_id,
            status="running",
            started_ns=time.time_ns(),
            request_digest=request_digest,
        )
        self.save("execution", event.id.encode(), event, mutable=True)
        return event

    def end_execution(
        self,
        event: ExecutionEvent,
        *,
        resource_id: bytes = b"",
        error: str = "",
        cache_hit: bool = False,
    ) -> None:
        import time

        event.resource_id = resource_id or event.resource_id
        event.error = error
        event.status = "error" if error else "completed"
        event.cache_hit = cache_hit
        event.ended_ns = time.time_ns()
        self.save("execution", event.id.encode(), event, mutable=True)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True

    def contains(self, namespace: str, id: bytes, *, suffix: str = "") -> bool:
        with self._lock:
            return (
                self._db.execute(
                    "SELECT 1 FROM metadata WHERE namespace=? AND id=? AND suffix=?",
                    (namespace, id, suffix),
                ).fetchone()
                is not None
            )

    def save(
        self,
        namespace: str,
        id: bytes,
        message: Message,
        *,
        suffix: str = "",
        mutable: bool = False,
    ) -> None:
        assert message.DESCRIPTOR is not None
        self.save_bytes(
            namespace,
            id,
            message.SerializeToString(deterministic=True),
            suffix=suffix,
            message_type=message.DESCRIPTOR.full_name,
            mutable=mutable,
        )

    def save_bytes(
        self,
        namespace: str,
        id: bytes,
        payload: bytes,
        *,
        suffix: str = "",
        message_type: str = "",
        mutable: bool = False,
    ) -> None:
        if self.read_only:
            raise PermissionError("catalog is read-only")
        if self.before_save is not None:
            self.before_save(namespace, id, payload, suffix, mutable)
        with self._lock, self._db:
            # Serialize concurrent publishers across processes as well as threads.
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT payload FROM metadata WHERE namespace=? AND id=? AND suffix=?",
                (namespace, id, suffix),
            ).fetchone()
            if row is not None and not mutable:
                if row[0] != payload:
                    raise ValueError("conflicting immutable storage object")
                return
            self._db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, ?, ?, ?)",
                (namespace, id, suffix, message_type, payload, blake3(payload).digest()),
            )
        if self.after_save is not None:
            self.after_save()

    def backup(self, path: Path) -> None:
        with self._lock, sqlite3.connect(path) as target:
            self._db.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("catalog snapshot failed SQLite integrity check")

    def load[T: Message](
        self, namespace: str, id: bytes, message_type: type[T], *, suffix: str = ""
    ) -> T:
        with self._lock:
            row = self._db.execute(
                "SELECT message_type, payload, digest FROM metadata "
                "WHERE namespace=? AND id=? AND suffix=?",
                (namespace, id, suffix),
            ).fetchone()
        if row is None:
            raise KeyError((namespace, id, suffix))
        return _decode(row, message_type)

    def list[T: Message](
        self, namespace: str, message_type: type[T], *, suffix: str = ""
    ) -> builtins.list[T]:
        with self._lock:
            rows = self._db.execute(
                "SELECT message_type, payload, digest FROM metadata "
                "WHERE namespace=? AND suffix=? ORDER BY id",
                (namespace, suffix),
            ).fetchall()
        return [_decode(row, message_type) for row in rows]

    def ids(self, namespace: str, *, suffix: str = "") -> builtins.list[bytes]:
        with self._lock:
            return [
                row[0]
                for row in self._db.execute(
                    "SELECT id FROM metadata WHERE namespace=? AND suffix=? ORDER BY id",
                    (namespace, suffix),
                )
            ]

    def migrated(self, name: str) -> bool:
        with self._lock:
            return (
                self._db.execute(
                    "SELECT 1 FROM migrations WHERE name=?",
                    (name,),
                ).fetchone()
                is not None
            )

    def mark_migrated(self, name: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO migrations VALUES (?)", (name,))


def _decode[T: Message](row: tuple[str, bytes, bytes], message_type: type[T]) -> T:
    stored_type, payload, digest = row
    if blake3(payload).digest() != digest:
        raise ValueError("stored resource integrity check failed")
    descriptor = message_type.DESCRIPTOR
    if descriptor is None or stored_type and stored_type != descriptor.full_name:
        raise ValueError("stored resource message type mismatch")
    result = message_type()
    result.ParseFromString(payload)
    return result
