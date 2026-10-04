"""Transactional protobuf metadata and portable SQLite snapshots."""

from __future__ import annotations

import builtins
import sqlite3
from contextlib import closing
from pathlib import Path
from threading import RLock
from time import time_ns
from typing import Literal, cast
from uuid import uuid4

from blake3 import blake3
from google.protobuf.message import Message

from ..v1 import corpus_pb2 as c
from ..v1 import data_mixture_pb2 as d
from ..v1 import query_pb2 as q
from ..v1 import snapshot_pb2 as s
from ..v1.status_pb2 import ExecutionEvent

type CatalogValue = c.Corpus | d.Dataset | d.Mix | q.Query | s.Snapshot | ExecutionEvent


class MetadataStore:
    """Keep complete typed profiles without losing uint64 precision to JSON/SQL."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path).absolute()
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._closed = False
        self.read_only = read_only
        self._revision = 0
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
            self._has_catalog_index = bool(
                self._db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='catalog_parents'"
                ).fetchone()
            )
            if read_only:
                if version != 1:
                    raise ValueError("read-only metadata requires a published catalog")
                return
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
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
                self._db.execute("""
                    CREATE TABLE IF NOT EXISTS catalog_parents (
                        namespace TEXT NOT NULL,
                        id BLOB NOT NULL,
                        suffix TEXT NOT NULL,
                        parent BLOB NOT NULL,
                        public_id TEXT NOT NULL,
                        started_ns BLOB NOT NULL,
                        captured_ns BLOB,
                        PRIMARY KEY (namespace, id, suffix, parent)
                    ) WITHOUT ROWID
                """)
                self._db.execute(
                    "CREATE INDEX IF NOT EXISTS catalog_by_parent ON catalog_parents(namespace, parent, public_id)"
                )
                if (
                    not self._has_catalog_index
                    or not self._db.execute(
                        "SELECT 1 FROM migrations WHERE name='catalog-parents-v1'"
                    ).fetchone()
                ):
                    for namespace, id, suffix, kind, payload, digest in self._db.execute(
                        "SELECT namespace, id, suffix, message_type, payload, digest FROM metadata"
                    ).fetchall():
                        _verified_payload((kind, payload, digest), "")
                        self._index(namespace, id, suffix, kind, payload)
                    self._db.execute(
                        "INSERT OR IGNORE INTO migrations VALUES ('catalog-parents-v1')"
                    )
                self._has_catalog_index = True
                self._db.execute("PRAGMA user_version=1")
        except BaseException:
            self._db.close()
            raise

    def begin_execution(
        self, operation: str, request_digest: bytes = b"", resource_id: bytes = b""
    ) -> ExecutionEvent:
        event = ExecutionEvent(
            id=uuid4().hex,
            operation=operation,
            resource_id=resource_id,
            status="running",
            started_ns=time_ns(),
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
        event.resource_id = resource_id or event.resource_id
        event.error = error
        event.status = "error" if error else "completed"
        event.cache_hit = cache_hit
        event.ended_ns = time_ns()
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
        with self._lock, self._db:
            # Serialize concurrent publishers across processes as well as threads.
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT message_type, payload, digest FROM metadata "
                "WHERE namespace=? AND id=? AND suffix=?",
                (namespace, id, suffix),
            ).fetchone()
            if row is not None and not mutable:
                if _verified_payload(row, message_type) != payload:
                    raise ValueError("conflicting immutable storage object")
                return
            self._db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, ?, ?, ?)",
                (namespace, id, suffix, message_type, payload, blake3(payload).digest()),
            )
            self._index(namespace, id, suffix, message_type, payload)
            self._revision += 1

    @property
    def revision(self) -> tuple[int, int]:
        """Invalidate listing membership after writes here or on another connection."""
        with self._lock:
            return self._revision, int(self._db.execute("PRAGMA data_version").fetchone()[0])

    def _index(self, namespace: str, id: bytes, suffix: str, kind: str, payload: bytes) -> None:
        message_type = _catalog_type(namespace, suffix)
        if message_type is None:
            return
        assert message_type.DESCRIPTOR is not None
        if kind and kind != message_type.DESCRIPTOR.full_name:
            return
        value = message_type.FromString(payload)
        from .._ids import _encode_id

        parents = tuple(dict.fromkeys(_parents(value))) or (b"",)
        public_id = id.decode() if isinstance(value, ExecutionEvent) else _encode_id(id)
        started = (value.started_ns if isinstance(value, ExecutionEvent) else 0).to_bytes(8, "big")
        captured = (
            value.ended_ns.to_bytes(8, "big")
            if isinstance(value, ExecutionEvent)
            and value.operation == "CreateSnapshot"
            and value.status == "completed"
            and value.ended_ns
            else None
        )
        self._db.execute(
            "DELETE FROM catalog_parents WHERE namespace=? AND id=? AND suffix=?",
            (namespace, id, suffix),
        )
        self._db.executemany(
            "INSERT INTO catalog_parents VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(namespace, id, suffix, parent, public_id, started, captured) for parent in parents],
        )

    def members(
        self,
        namespace: str,
        *,
        suffixes: tuple[str, ...] = ("",),
        parents: tuple[bytes, ...] = (),
        order: Literal["id", "public", "capture", "execution"] = "id",
        limit: int | None = None,
        offset: int = 0,
    ) -> builtins.list[tuple[bytes, str, int | None]]:
        """Select parents, state precedence and a window without decoding payloads.

        The suffix order defines durable state precedence. Old read-only stores
        remain readable; opening them writable creates the derived parent index.
        """
        if offset >= 2**63:
            return []
        if not self._has_catalog_index:
            return self._legacy_members(namespace, suffixes, parents, order, limit, offset)
        placeholders = ",".join("?" for _ in suffixes)
        precedence = (
            "CASE p.suffix " + " ".join(f"WHEN ? THEN {i}" for i in range(len(suffixes))) + " END"
        )
        where_parent = ""
        arguments: list[str | bytes | int] = [*suffixes, namespace, *suffixes, namespace]
        if parents:
            where_parent = " AND p.parent IN (" + ",".join("?" for _ in parents) + ")"
            arguments.extend(parents)
        ordering = {
            "id": "id",
            "public": "public_id",
            "capture": "captured_ns IS NULL, captured_ns, id",
            "execution": "started_ns, id",
        }[order]
        # Pick one state before parent filtering so a lower-priority recipe never
        # changes membership when a completed resource is published.
        capture = (
            "(SELECT MIN(e.captured_ns) FROM catalog_parents e WHERE e.namespace='execution' AND e.parent=p.id)"
            if order == "capture"
            else "NULL"
        )
        sql = f"""
            WITH ranked AS (
                SELECT p.*, ROW_NUMBER() OVER (PARTITION BY p.id ORDER BY {precedence}) AS rank
                FROM catalog_parents p WHERE p.namespace=? AND p.suffix IN ({placeholders})
            ), chosen AS (
                SELECT DISTINCT p.id, p.suffix, p.public_id, p.started_ns,
                    {capture} AS captured_ns
                FROM catalog_parents p JOIN ranked r ON r.id=p.id AND r.suffix=p.suffix AND r.rank=1
                WHERE p.namespace=? {where_parent}
            )
            SELECT id, suffix, captured_ns FROM chosen ORDER BY {ordering}
        """
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            arguments.extend((limit, offset))
        with self._lock:
            rows = cast(
                builtins.list[tuple[bytes, str, bytes | None]],
                self._db.execute(sql, arguments).fetchall(),
            )
        return [
            (id, suffix, int.from_bytes(ns, "big") if ns is not None else None)
            for id, suffix, ns in rows
        ]

    def _legacy_members(
        self,
        namespace: str,
        suffixes: tuple[str, ...],
        parents: tuple[bytes, ...],
        order: str,
        limit: int | None,
        offset: int,
    ) -> builtins.list[tuple[bytes, str, int | None]]:
        from .._ids import _encode_id

        selected = {}
        for suffix in reversed(suffixes):
            message_type = _catalog_type(namespace, suffix)
            if message_type is not None:
                selected.update(
                    (
                        value.id.encode() if isinstance(value, ExecutionEvent) else value.id,
                        (suffix, value),
                    )
                    for value in self.list(namespace, message_type, suffix=suffix)
                )
        ids = [
            id
            for id, (_, value) in selected.items()
            if not parents or set(_parents(value)).intersection(parents)
        ]
        times = {}
        if order == "capture":
            for event in self.list("execution", ExecutionEvent):
                if (
                    event.operation == "CreateSnapshot"
                    and event.status == "completed"
                    and event.ended_ns
                ):
                    times[event.resource_id] = min(
                        event.ended_ns, times.get(event.resource_id, event.ended_ns)
                    )
            ids.sort(key=lambda id: (id not in times, times.get(id, 0), id))
        elif order == "public":
            ids.sort(key=_encode_id)
        elif order == "execution":

            def started(id: bytes) -> tuple[int, bytes]:
                event = selected[id][1]
                assert isinstance(event, ExecutionEvent)
                return event.started_ns, id

            ids.sort(key=started)
        else:
            ids.sort()
        return [
            (id, selected[id][0], times.get(id))
            for id in ids[offset : None if limit is None else offset + limit]
        ]

    def backup(self, path: Path) -> None:
        with self._lock, closing(sqlite3.connect(path)) as target:
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


def _verified_payload(row: tuple[str, bytes, bytes], message_type: str) -> bytes:
    stored_type, payload, digest = row
    if blake3(payload).digest() != digest:
        raise ValueError("stored resource integrity check failed")
    if stored_type and message_type and stored_type != message_type:
        raise ValueError("stored resource message type mismatch")
    return payload


def _decode[T: Message](row: tuple[str, bytes, bytes], message_type: type[T]) -> T:
    descriptor = message_type.DESCRIPTOR
    if descriptor is None:
        raise ValueError("stored resource message type mismatch")
    result = message_type()
    result.ParseFromString(_verified_payload(row, descriptor.full_name))
    return result


def _catalog_type(namespace: str, suffix: str) -> type[CatalogValue] | None:

    if suffix not in {"", ".pending", ".failed", ".recipe"}:
        return None
    return {
        "corpus": c.Corpus,
        "snapshot": s.Snapshot,
        "query": q.Query,
        "dataset": d.Dataset,
        "mixture": d.Mix,
        "execution": ExecutionEvent,
    }.get(namespace)


def _parents(value: Message) -> tuple[bytes, ...]:

    if isinstance(value, s.Snapshot):
        return (value.corpus_id,)
    if isinstance(value, q.Query):
        return tuple(value.snapshot_ids)
    if isinstance(value, (d.Dataset, d.Mix)):
        return (value.query_id,)
    if isinstance(value, ExecutionEvent):
        return (value.resource_id,)
    return ()
