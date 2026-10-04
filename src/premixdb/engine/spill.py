"""Disk-backed exact evidence for large populations, independent of physical layout."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    TYPE_CHECKING,
    ContextManager,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    cast,
)

from premixdb.contracts import Edge, EvidenceRange
from premixdb.v1 import query_pb2 as q

if TYPE_CHECKING:
    from premixdb.engine.curation import SelectedDocument, Unit
    from premixdb.engine.queries import CorpusIndex
    from premixdb.engine.snapshots import Document


@contextmanager
def _database(kind: str, *, check_same_thread: bool = True) -> Iterator[sqlite3.Connection]:
    """Bound SQLite memory and release temporary evidence on every exit."""
    with TemporaryDirectory(prefix=f"premixdb-{kind}-") as directory:
        database = sqlite3.connect(
            Path(directory) / "evidence.sqlite3", check_same_thread=check_same_thread
        )
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            yield database
        finally:
            database.close()


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
    with _database("reference") as database:
        database.execute(
            "CREATE TABLE references_index (value BLOB PRIMARY KEY, id TEXT, start INTEGER, end INTEGER) WITHOUT ROWID"
        )
        for value, document, start, end in rows:
            database.execute(
                "INSERT INTO references_index VALUES (?,?,?,?) ON CONFLICT(value) DO UPDATE SET id=excluded.id,start=excluded.start,end=excluded.end WHERE (excluded.id,excluded.start,excluded.end)<(id,start,end)",
                (value, document, start, end),
            )
        yield ReferenceLookup(database)


def references(
    documents: Sequence[Document], policy: q.Decontaminate
) -> ContextManager[ReferenceLookup]:
    from premixdb.engine.curation import units

    return reference_rows(
        (unit_key(value), doc.id, start, end)
        for doc in documents
        for value, start, end in units(doc, policy.algorithm, policy.n)
    )


@contextmanager
def edge_database(edges: Iterable[Edge]) -> Iterator[sqlite3.Connection]:
    with _database("similarity") as database:
        database.execute("CREATE TABLE edges (a TEXT,b TEXT,PRIMARY KEY(a,b)) WITHOUT ROWID")
        database.execute("CREATE INDEX reverse_edges ON edges(b,a)")
        database.execute("CREATE TABLE kept (id TEXT PRIMARY KEY,rank INTEGER) WITHOUT ROWID")
        database.executemany(
            "INSERT OR IGNORE INTO edges VALUES (?,?)",
            (tuple(sorted((a, b))) for a, b in edges),
        )
        yield database


def candidate_pairs(rows: Iterable[tuple[str, Sequence[int]]]) -> Iterator[Edge]:
    """Join LSH bands on disk without materializing a quadratic pair set."""
    with _database("lsh") as database:
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


def cosine_edges(
    vectors: Mapping[str, Sequence[float] | None] | Iterable[tuple[str, Sequence[float] | None]],
    threshold: float,
    selected: set[str] | None = None,
) -> Iterator[Edge]:
    """Exhaustive, verified pairs over bounded disk-backed vector blocks."""
    import math
    import struct

    with _database("cosine") as database:
        database.execute(
            "CREATE TABLE vectors (id TEXT PRIMARY KEY,vector BLOB,norm REAL) WITHOUT ROWID"
        )
        width = None
        rows = vectors.items() if isinstance(vectors, Mapping) else vectors
        for id, vector in rows:
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


def jaccard_edges(
    documents: list[SelectedDocument],
    n: int,
    threshold: float,
    candidates: Iterable[Edge] | None = None,
) -> Iterator[Edge]:
    from premixdb.engine.curation import word_ranges

    by_id = {doc.id: doc for doc in documents}
    with _database("jaccard") as database:
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


class Group:
    def __init__(self, database: sqlite3.Connection, value: bytes) -> None:
        self.database, self.value = database, value

    def __iter__(self) -> Iterator[EvidenceRange]:
        return cast(
            Iterator[EvidenceRange],
            iter(
                self.database.execute(
                    "SELECT id,start,end FROM evidence WHERE value=? ORDER BY id,start,end",
                    (self.value,),
                )
            ),
        )


def evidence_groups(rows: Iterable[tuple[bytes, str, int, int]]) -> Generator[Group, None, None]:
    """Stream evidence to disk and expose repeatable groups in canonical order."""
    with _database("index") as database:
        database.execute("CREATE TABLE evidence (value BLOB, id TEXT, start INTEGER, end INTEGER)")
        database.executemany("INSERT INTO evidence VALUES (?,?,?,?)", rows)
        database.execute("CREATE INDEX groups ON evidence(value,id,start,end)")
        for (value,) in database.execute("SELECT DISTINCT value FROM evidence ORDER BY value"):
            yield Group(database, value)


def classes(index: CorpusIndex, unit: str) -> Generator[Group, None, None]:
    from premixdb.engine.curation import units

    # Document and line units are already byte keys; retain them without copying.
    return evidence_groups(
        (cast(bytes, value), document.id, start, end)
        for document in index.documents.values()
        for value, start, end in units(document, 1 if unit == "Document" else 2)
    )
