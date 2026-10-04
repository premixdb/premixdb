"""Browse saved resources without reading documents or starting recipes."""

from __future__ import annotations

import builtins
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

from ._ids import _encode_id
from ._types import CorpusListing, DocumentListing, SnapshotListing
from .v1 import corpus_pb2 as c
from .v1 import dataset_pb2 as d

if TYPE_CHECKING:
    from ._inputs import SourceInput
    from ._resources import Dataset, Mix, PremixDB, Query, Snapshot


class PageRequest(Protocol):
    page_token: bytes


class PageResponse(Protocol):
    @property
    def next_page_token(self) -> bytes: ...


class PageMethod[Request, Response](Protocol):
    def __call__(self, request: Request, /) -> Response: ...


def _pages[Request: PageRequest, Response: PageResponse, Item](
    db: PremixDB,
    method: PageMethod[Request, Response],
    request: Request,
    values: Callable[[Response], Iterable[Item]],
) -> list[Item]:
    db._require_open()
    rows: list[Item] = []
    seen: set[bytes] = set()
    while True:
        response = method(request)
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

    db._require_open()
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
        values = self._db._executor.browse("corpus", c.Corpus, limit=end - start, offset=start)
        return [dict(id=_encode_id(value.id), name=value.name) for value in values]


class _CorpusListings:
    """Browse the history and recipes of a named corpus."""

    _db: PremixDB
    _resource: c.Corpus

    def _snapshot_ids(self) -> tuple[bytes, ...]:
        return tuple(
            id for id, _, _ in self._db._executor._members("snapshot", (self._resource.id,))
        )

    def _query_ids(self) -> tuple[bytes, ...]:
        snapshots = self._snapshot_ids()
        if not snapshots:
            return ()
        return tuple(id for id, _, _ in self._db._executor._members("query", snapshots))

    def list_snapshot(self, *, limit: int = 5, offset: int = 0) -> list[SnapshotListing]:
        """List snapshots by first capture time; older imports have timestamp=None."""
        start, end = _window(self._db, limit=limit, offset=offset)
        if start == end:
            return []
        members = self._db._executor._storage.metadata.members(
            "snapshot",
            parents=(self._resource.id,),
            order="capture",
            limit=end - start,
            offset=start,
        )
        return [
            dict(id=_encode_id(id), timestamp=_timestamp(ns) if ns is not None else None)
            for id, _, ns in members
        ]

    def list_query(self, *, limit: int = 5, offset: int = 0) -> list[str]:
        """List query IDs across this corpus's snapshots, ordered by public ID."""
        start, end = _window(self._db, limit=limit, offset=offset)
        snapshots = self._snapshot_ids() if start != end else ()
        if not snapshots:
            return []
        members = self._db._executor._storage.metadata.members(
            "query",
            suffixes=("", ".failed", ".pending"),
            parents=snapshots,
            order="public",
            limit=end - start,
            offset=start,
        )
        return [_encode_id(id) for id, _, _ in members]

    def list_mixture(self, *, limit: int = 5, offset: int = 0) -> list[Mix]:
        """List mixture handles by public ID without packing candidates."""
        from ._resources import Mix

        start, end = _window(self._db, limit=limit, offset=offset)
        queries = self._query_ids() if start != end else ()
        if not queries:
            return []
        return [
            Mix(self._db, value)
            for value in self._db._executor.browse(
                "mixture", d.Mix, parents=queries, limit=end - start, offset=start
            )
        ]

    def list_dataset(self, *, limit: int = 5, offset: int = 0) -> list[Dataset]:
        """List dataset handles by public ID, including pending candidates."""
        from ._resources import Dataset

        start, end = _window(self._db, limit=limit, offset=offset)
        queries = self._query_ids() if start != end else ()
        if not queries:
            return []
        return [
            Dataset(self._db, value)
            for value in self._db._executor.browse(
                "dataset", d.Dataset, parents=queries, limit=end - start, offset=start
            )
        ]
