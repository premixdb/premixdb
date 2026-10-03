"""Ephemeral projected columns with a fixed SQLite cache instead of resident vectors."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import ItemsView, MutableMapping
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Iterable, Iterator

from .._typing import FieldValue, field_value, json_string, load_json


class ValueCache(MutableMapping[str, FieldValue]):
    def __init__(self, rows: Iterable[tuple[str, FieldValue]] = ()) -> None:
        self._directory = TemporaryDirectory(prefix="premixdb-values-")
        self._lock = RLock()
        self._database = sqlite3.connect(
            Path(self._directory.name) / "values.sqlite3", check_same_thread=False
        )
        self._database.execute("PRAGMA cache_size=-8192")
        self._database.execute("PRAGMA temp_store=FILE")
        self._database.execute(
            "CREATE TABLE values_index (id TEXT PRIMARY KEY,value TEXT) WITHOUT ROWID"
        )
        self._database.executemany(
            "INSERT INTO values_index VALUES (?,?)",
            ((key, self._encode(value)) for key, value in rows),
        )
        self._database.commit()

    @staticmethod
    def _encode(value: FieldValue) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def __getitem__(self, key: str) -> FieldValue:
        with self._lock:
            row = self._database.execute(
                "SELECT value FROM values_index WHERE id=?", (key,)
            ).fetchone()
        if row is None:
            raise KeyError(key)
        return field_value(load_json(row[0]))

    def __setitem__(self, key: str, value: FieldValue) -> None:
        with self._lock:
            self._database.execute(
                "INSERT INTO values_index VALUES (?,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value",
                (key, self._encode(value)),
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
            cursor = self._database.execute("SELECT id,value FROM values_index ORDER BY id")
        while True:
            with self._lock:
                batch = cursor.fetchmany(64)
            if not batch:
                break
            yield from ((json_string(key), field_value(load_json(value))) for key, value in batch)

    def __len__(self) -> int:
        with self._lock:
            return int(self._database.execute("SELECT COUNT(*) FROM values_index").fetchone()[0])

    def __del__(self) -> None:
        database = getattr(self, "_database", None)
        if database is not None:
            database.close()
        directory = getattr(self, "_directory", None)
        if directory is not None:
            directory.cleanup()


class _ValueItems(ItemsView[str, FieldValue]):
    def __iter__(self) -> Iterator[tuple[str, FieldValue]]:
        assert isinstance(self._mapping, ValueCache)
        return self._mapping._items()
