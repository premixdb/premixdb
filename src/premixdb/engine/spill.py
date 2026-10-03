"""Disk-backed exact evidence for large populations, independent of physical layout."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, ContextManager, Iterable, Iterator, Mapping, Sequence, cast

from .._typing import Edge, EvidenceRange
from ..v1 import query_pb2 as q

if TYPE_CHECKING:
    from .curation import SelectedDocument, Unit
    from .queries import CorpusIndex
    from .snapshots import Document


def unit_key(value: bytes | tuple[str, ...] | tuple[int, int]) -> bytes:
    import json

    return (
        b"b" + value
        if isinstance(value, bytes)
        else b"n" + json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    )


class ReferenceLookup:
    def __init__(self, database: sqlite3.Connection) -> None:
        self.database = database

    def get(self, value: Unit) -> EvidenceRange | None:
        return cast(
            EvidenceRange | None,
            self.database.execute(
                "SELECT id,start,end FROM references_index WHERE value=?", (unit_key(value),)
            ).fetchone(),
        )


@contextmanager
def reference_rows(rows: Iterable[tuple[bytes, str, int, int]]) -> Iterator[ReferenceLookup]:
    with TemporaryDirectory(prefix="premixdb-reference-") as directory:
        database = sqlite3.connect(Path(directory) / "reference.sqlite3")
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                "CREATE TABLE references_index (value BLOB PRIMARY KEY, id TEXT, start INTEGER, end INTEGER) WITHOUT ROWID"
            )
            for value, document, start, end in rows:
                database.execute(
                    "INSERT INTO references_index VALUES (?,?,?,?) ON CONFLICT(value) DO UPDATE SET id=excluded.id,start=excluded.start,end=excluded.end WHERE (excluded.id,excluded.start,excluded.end)<(id,start,end)",
                    (value, document, start, end),
                )
            yield ReferenceLookup(database)
        finally:
            database.close()


def references(
    documents: Sequence[Document], policy: q.Decontaminate
) -> ContextManager[ReferenceLookup]:
    from .curation import units

    return reference_rows(
        (unit_key(value), doc.id, start, end)
        for doc in documents
        for value, start, end in units(doc, policy.algorithm, policy.n)
    )


@contextmanager
def edge_database(edges: Iterable[Edge]) -> Iterator[sqlite3.Connection]:
    with TemporaryDirectory(prefix="premixdb-similarity-") as directory:
        database = sqlite3.connect(Path(directory) / "edges.sqlite3")
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute("CREATE TABLE edges (a TEXT,b TEXT,PRIMARY KEY(a,b)) WITHOUT ROWID")
            database.execute("CREATE INDEX reverse_edges ON edges(b,a)")
            database.execute("CREATE TABLE kept (id TEXT PRIMARY KEY,rank INTEGER) WITHOUT ROWID")
            database.executemany(
                "INSERT OR IGNORE INTO edges VALUES (?,?)",
                (tuple(sorted((a, b))) for a, b in edges),
            )
            yield database
        finally:
            database.close()


def candidate_pairs(rows: Iterable[tuple[str, Sequence[int]]]) -> Iterator[Edge]:
    """Join LSH bands on disk without materializing a quadratic pair set."""
    with TemporaryDirectory(prefix="premixdb-lsh-") as directory:
        database = sqlite3.connect(Path(directory) / "bands.sqlite3")
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                "CREATE TABLE postings (value BLOB,id TEXT,PRIMARY KEY(value,id)) WITHOUT ROWID"
            )
            for identity, bands in rows:
                database.executemany(
                    "INSERT OR IGNORE INTO postings VALUES (?,?)",
                    ((unit_key((band, bucket)), identity) for band, bucket in enumerate(bands)),
                )
            yield from (
                cast(Edge, row)
                for row in database.execute(
                    "SELECT DISTINCT a.id,b.id FROM postings a JOIN postings b ON a.value=b.value WHERE a.id<b.id ORDER BY a.id,b.id"
                )
            )
        finally:
            database.close()


def cosine_edges(
    vectors: Mapping[str, Sequence[float] | None],
    threshold: float,
    selected: set[str] | None = None,
) -> Iterator[Edge]:
    """Exhaustive, verified pairs over bounded disk-backed vector blocks."""
    import math
    import struct

    with TemporaryDirectory(prefix="premixdb-cosine-") as directory:
        database = sqlite3.connect(Path(directory) / "vectors.sqlite3")
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                "CREATE TABLE vectors (id TEXT PRIMARY KEY,vector BLOB,norm REAL) WITHOUT ROWID"
            )
            width = None
            for id, vector in vectors.items():
                if vector is None or selected is not None and id not in selected:
                    continue
                if width is None:
                    width = len(vector)
                if len(vector) != width or not all(math.isfinite(value) for value in vector):
                    raise ValueError("cosine vectors require a fixed finite width")
                scale = max((abs(value) for value in vector), default=0)
                if not scale:
                    continue
                vector = [value / scale for value in vector]
                norm = math.fsum(value * value for value in vector)
                database.execute(
                    "INSERT INTO vectors VALUES (?,?,?)",
                    (id, struct.pack(f"<{width}d", *vector), norm),
                )
            if width is None:
                return
            unpack = struct.Struct(f"<{width}d").unpack
            cursor = database.execute("SELECT id,vector,norm FROM vectors ORDER BY id")
            while raw := cursor.fetchmany(32):
                left = [(id, unpack(vector), norm) for id, vector, norm in raw]
                following = database.execute(
                    "SELECT id,vector,norm FROM vectors WHERE id>? ORDER BY id", (left[0][0],)
                )
                while batch := following.fetchmany(64):
                    for right_id, vector, right_norm in batch:
                        right = unpack(vector)
                        for left_id, vector, left_norm in left:
                            if left_id >= right_id:
                                continue
                            score = math.fsum(a * b for a, b in zip(vector, right)) / math.sqrt(
                                left_norm * right_norm
                            )
                            if score >= threshold:
                                yield str(left_id), str(right_id)
        finally:
            database.close()


def jaccard_edges(
    documents: list[SelectedDocument],
    n: int,
    threshold: float,
    candidates: Iterable[Edge] | None = None,
) -> Iterator[Edge]:
    from .curation import word_ranges

    by_id = {doc.id: doc for doc in documents}
    with TemporaryDirectory(prefix="premixdb-jaccard-") as directory:
        database = sqlite3.connect(Path(directory) / "grams.sqlite3")
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                "CREATE TABLE postings (value BLOB,id TEXT,PRIMARY KEY(value,id)) WITHOUT ROWID"
            )
            database.execute("CREATE INDEX documents ON postings(id,value)")
            database.execute("CREATE TABLE sizes (id TEXT PRIMARY KEY,size INTEGER) WITHOUT ROWID")
            for doc in documents:
                database.executemany(
                    "INSERT OR IGNORE INTO postings VALUES (?,?)",
                    ((unit_key(value), doc.id) for value, _, _ in word_ranges(doc.text, n)),
                )
                database.execute(
                    "INSERT INTO sizes VALUES (?,(SELECT COUNT(*) FROM postings WHERE id=?))",
                    (doc.id, doc.id),
                )
            if candidates is None:
                candidates = (
                    cast(Edge, row)
                    for row in database.execute(
                        "SELECT a.id,b.id FROM sizes a JOIN sizes b ON a.id<b.id ORDER BY a.id,b.id"
                        if threshold == 0
                        else "SELECT a.id,b.id FROM postings a JOIN postings b ON a.value=b.value WHERE a.id<b.id UNION SELECT a.id,b.id FROM sizes a JOIN sizes b ON a.id<b.id WHERE a.size=0 AND b.size=0 ORDER BY 1,2"
                    )
                )
            for a, b in candidates:
                if a not in by_id or b not in by_id:
                    continue
                intersection = database.execute(
                    "SELECT COUNT(*) FROM postings a JOIN postings b ON a.value=b.value WHERE a.id=? AND b.id=?",
                    (a, b),
                ).fetchone()[0]
                count = database.execute(
                    "SELECT SUM(size) FROM sizes WHERE id IN (?,?)", (a, b)
                ).fetchone()[0]
                score = (
                    intersection / (count - intersection)
                    if count
                    else float(by_id[a].text == by_id[b].text)
                )
                if score >= threshold:
                    yield a, b
        finally:
            database.close()


class Group:
    def __init__(self, database: sqlite3.Connection, value: bytes) -> None:
        self.database, self.value = database, value

    def __iter__(self) -> Iterator[EvidenceRange]:
        return (
            cast(EvidenceRange, row)
            for row in (
                self.database.execute(
                    "SELECT id,start,end FROM evidence WHERE value=? ORDER BY id,start,end",
                    (self.value,),
                )
            )
        )


def classes(index: CorpusIndex, unit: str) -> Iterator[Group]:
    from .curation import units

    with TemporaryDirectory(prefix="premixdb-index-") as directory:
        database = sqlite3.connect(Path(directory) / "evidence.sqlite3")
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                "CREATE TABLE evidence (value BLOB, id TEXT, start INTEGER, end INTEGER)"
            )
            for document in index.documents.values():
                for value, start, end in units(document, 1 if unit == "Document" else 2):
                    database.execute(
                        "INSERT INTO evidence VALUES (?,?,?,?)", (value, document.id, start, end)
                    )
            database.execute("CREATE INDEX groups ON evidence(value,id,start,end)")
            for (value,) in database.execute("SELECT DISTINCT value FROM evidence ORDER BY value"):
                yield Group(database, value)
        finally:
            database.close()
