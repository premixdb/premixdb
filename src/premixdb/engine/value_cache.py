"""Ephemeral projected columns with a fixed SQLite cache instead of resident vectors."""

from __future__ import annotations

import math
from collections.abc import ItemsView, MutableMapping
from contextlib import ExitStack
from threading import RLock
from typing import Iterable, Iterator

from premixdb.contracts import FieldValue, json_string
from premixdb.engine.spill import _database
from premixdb.internal import transport_pb2 as t
from premixdb.schemas.binary import decode_value, encode_value, parse
from premixdb.schemas.protobuf import wire


class ValueCache(MutableMapping[str, FieldValue]):
    def __init__(self, rows: Iterable[tuple[str, FieldValue]] = ()) -> None:
        self._lock = RLock()
        with ExitStack() as startup:
            self._database = startup.enter_context(_database("values", check_same_thread=False))
            self._database.execute(
                "CREATE TABLE values_index (id TEXT PRIMARY KEY,kind TEXT,value) WITHOUT ROWID"
            )
            self._database.executemany(
                "INSERT INTO values_index VALUES (?,?,?)",
                ((key, *self._encode(value)) for key, value in rows),
            )
            self._database.commit()
            self._lifetime = startup.pop_all()

    @staticmethod
    def _encode(value: FieldValue) -> tuple[str, str | int | float | bytes | None]:
        if value is None:
            return "null", None
        if type(value) is bool:
            return "bool", int(value)
        if isinstance(value, int) and -(1 << 63) <= value < 1 << 63:
            return "int", value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("field requires finite numeric values")
            if value == 0.0 and math.copysign(1, value) < 0:
                return "protobuf", wire(encode_value(value))
            return "float", value
        if isinstance(value, str):
            return "str", value
        return "protobuf", wire(encode_value(value))

    @staticmethod
    def _decode(kind: str, value: str | int | float | bytes | None) -> FieldValue:
        if kind == "null" and value is None:
            return None
        if kind == "bool" and isinstance(value, int) and value in (0, 1):
            return bool(value)
        if kind == "int" and isinstance(value, int):
            return value
        if kind == "float" and isinstance(value, float) and math.isfinite(value):
            return value
        if kind == "str" and isinstance(value, str):
            return value
        if kind == "protobuf" and isinstance(value, bytes):
            return decode_value(parse(value, t.TransportValue()))
        raise ValueError("invalid cached value")

    def __getitem__(self, key: str) -> FieldValue:
        with self._lock:
            row = self._database.execute(
                "SELECT kind,value FROM values_index WHERE id=?", (key,)
            ).fetchone()
        if row is None:
            raise KeyError(key)
        return self._decode(*row)

    def __setitem__(self, key: str, value: FieldValue) -> None:
        with self._lock:
            self._database.execute(
                "INSERT INTO values_index VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,value=excluded.value",
                (key, *self._encode(value)),
            )

    def __delitem__(self, key: str) -> None:
        with self._lock:
            cursor = self._database.execute("DELETE FROM values_index WHERE id=?", (key,))
        if not cursor.rowcount:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            cursor = self._database.execute("SELECT id FROM values_index ORDER BY id")
        while True:
            with self._lock:
                batch = cursor.fetchmany(64)
            if not batch:
                break
            yield from (json_string(row[0]) for row in batch)

    def items(self) -> ItemsView[str, FieldValue]:
        return _ValueItems(self)

    def _items(self) -> Iterator[tuple[str, FieldValue]]:
        with self._lock:
            cursor = self._database.execute("SELECT id,kind,value FROM values_index ORDER BY id")
        while True:
            with self._lock:
                batch = cursor.fetchmany(64)
            if not batch:
                break
            yield from ((json_string(key), self._decode(kind, value)) for key, kind, value in batch)

    def __len__(self) -> int:
        with self._lock:
            return int(self._database.execute("SELECT COUNT(*) FROM values_index").fetchone()[0])

    def close(self) -> None:
        """Release temporary storage; repeated calls are safe."""
        with self._lock:
            self._lifetime.close()

    def __del__(self) -> None:
        if hasattr(self, "_lifetime"):
            self.close()


class _ValueItems(ItemsView[str, FieldValue]):
    def __iter__(self) -> Iterator[tuple[str, FieldValue]]:
        assert isinstance(self._mapping, ValueCache)
        return self._mapping._items()
