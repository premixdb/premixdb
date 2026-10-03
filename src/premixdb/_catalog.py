"""Browse saved resources without reading documents or starting recipes."""

from __future__ import annotations

import builtins
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

from ._ids import _encode_id
from ._types import CorpusListing, DocumentListing, SnapshotListing
from .v1 import corpus_pb2 as c
from .v1 import dataset_pb2 as d
from .v1 import query_pb2 as q
from .v1 import snapshot_pb2 as s

if TYPE_CHECKING:
    from ._resources import Dataset, Mix, PremixDB, Query, Snapshot, SourceInput


class PageRequest(Protocol):
    page_token: bytes


class PageResponse(Protocol):
    @property
    def next_page_token(self) -> bytes: ...


class PageMethod[Request, Response](Protocol):
    def __call__(self, request: Request, /, *, timeout: float | None = None) -> Response: ...


def _pages[Request: PageRequest, Response: PageResponse, Item](
    db: PremixDB,
    method: PageMethod[Request, Response],
    request: Request,
    values: Callable[[Response], Iterable[Item]],
) -> list[Item]:
    if db._closed:
        raise ValueError("PremixDB is closed")
    rows: list[Item] = []
    seen: set[bytes] = set()
    while True:
        response = method(request, timeout=db._timeout)
        rows.extend(values(response))
        page = response.next_page_token
        if not page:
            return rows
        if page in seen:
            raise ValueError("catalog repeated a continuation token")
        seen.add(page)
        request.page_token = page


def _timestamp(ns: int) -> str:
    return datetime.fromtimestamp(ns // 1_000_000_000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _window(db: PremixDB, *, limit: int, offset: int) -> tuple[int, int]:
    from ._requests import _uint

    if db._closed:
        raise ValueError("PremixDB is closed")
    start = _uint(offset, 64, "offset")
    return start, start + _uint(limit, 32, "limit")


def _list_documents(
    resource: Snapshot | Query, *, limit: int, offset: int
) -> list[DocumentListing]:
    return [
        dict(
            id=row["id"],
            source_key=row["source_key"],
            corpus_id=row["corpus_id"],
            ordinal=row["ordinal"],
        )
        for row in resource.preview(limit=limit, offset=offset, max_characters=0)
    ]


class CorpusCollection:
    """Call to capture or reopen a corpus. Use list() to browse their names."""

    def __init__(self, db: PremixDB) -> None:
        self._db = db

    def __call__(
        self,
        name: str,
        source: SourceInput | None = None,
        *,
        limit: int | None = None,
        base: Snapshot | None = None,
    ) -> Snapshot:
        """Capture sources under name, or open its latest snapshot when source is omitted."""
        return self._db._corpus(
            name,
            source,
            limit=limit,
            base=base,
        )

    def list(self, *, limit: int = 5, offset: int = 0) -> builtins.list[CorpusListing]:
        """List a page of saved corpus IDs and names, ordered by ID."""
        start, end = _window(self._db, limit=limit, offset=offset)
        if start == end:
            return []
        rows: builtins.list[CorpusListing] = [
            dict(id=_encode_id(value.id), name=value.name)
            for value in sorted(
                _pages(
                    self._db,
                    self._db._executor.ListCorpus,
                    c.ListCorpusRequest(),
                    lambda response: response.corpora,
                ),
                key=lambda value: _encode_id(value.id),
            )
        ]
        return rows[start:end]


class _CorpusListings:
    """Corpus-wide listings available on both corpus and snapshot handles."""

    _db: PremixDB
    _resource: c.Corpus | s.Snapshot

    def _corpus_snapshots(self) -> list[s.Snapshot]:
        corpus_id = (
            self._resource.corpus_id
            if isinstance(self._resource, s.Snapshot)
            else self._resource.id
        )
        return _pages(
            self._db,
            self._db._executor.ListSnapshot,
            s.ListSnapshotRequest(corpus_id=corpus_id),
            lambda response: response.snapshots,
        )

    def _corpus_queries(self) -> list[q.Query]:
        snapshots = {value.id for value in self._corpus_snapshots()}
        if not snapshots:
            return []
        return [
            value
            for value in _pages(
                self._db,
                self._db._executor.ListQuery,
                q.ListQueryRequest(),
                lambda response: response.queries,
            )
            if snapshots.intersection(value.snapshot_ids)
        ]

    def list_snapshot(self, *, limit: int = 5, offset: int = 0) -> list[SnapshotListing]:
        """List a page of snapshot IDs and UTC times, ordered by first capture.

        Older imports without execution history have timestamp=None.
        Recapturing an unchanged snapshot keeps its original timestamp.
        """
        start, end = _window(self._db, limit=limit, offset=offset)
        if start == end:
            return []
        snapshots = self._corpus_snapshots()
        if not snapshots:
            return []
        times = {}
        for event in self._db._execution_events():
            if (
                event.operation != "CreateSnapshot"
                or event.status != "completed"
                or not event.ended_ns
            ):
                continue
            ns = event.ended_ns
            times[event.resource_id] = min(ns, times.get(event.resource_id, ns))
        rows: list[SnapshotListing] = [
            dict(
                id=_encode_id(value.id),
                timestamp=_timestamp(times[value.id]) if value.id in times else None,
            )
            for value in sorted(
                snapshots,
                key=lambda value: (value.id not in times, times.get(value.id, 0), value.id),
            )
        ]
        return rows[start:end]

    def list_query(self, *, limit: int = 5, offset: int = 0) -> list[str]:
        """List a page of query IDs across this corpus's snapshots, ordered by ID."""
        start, end = _window(self._db, limit=limit, offset=offset)
        if start == end:
            return []
        return sorted(_encode_id(value.id) for value in self._corpus_queries())[start:end]

    def list_mixture(self, *, limit: int = 5, offset: int = 0) -> list[Mix]:
        """List a page of mixture handles by ID without packing candidates."""
        from ._resources import Mix

        start, end = _window(self._db, limit=limit, offset=offset)
        if start == end:
            return []
        queries = {value.id for value in self._corpus_queries()}
        if not queries:
            return []
        values = [
            value
            for value in _pages(
                self._db,
                self._db._executor.ListMix,
                d.ListMixRequest(),
                lambda response: response.mixtures,
            )
            if value.query_id in queries
        ]
        return [
            Mix(self._db, value)
            for value in sorted(values, key=lambda value: _encode_id(value.id))[start:end]
        ]

    def list_dataset(self, *, limit: int = 5, offset: int = 0) -> list[Dataset]:
        """List a page of dataset handles by ID, including pending candidates."""
        from ._resources import Dataset

        start, end = _window(self._db, limit=limit, offset=offset)
        if start == end:
            return []
        values = {}
        for query in self._corpus_queries():
            for value in _pages(
                self._db,
                self._db._executor.ListDatasets,
                d.ListDatasetRequest(query_id=query.id),
                lambda response: response.datasets,
            ):
                values[value.id] = value
        return [Dataset(self._db, values[id]) for id in sorted(values, key=_encode_id)[start:end]]
