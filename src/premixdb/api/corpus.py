"""Named corpus handles and immutable snapshot capture."""

from __future__ import annotations

from premixdb.api.base import _Resource
from premixdb.api.collections import (
    _CorpusListings,
    _list_documents,
)
from premixdb.api.progress import report_progress
from premixdb.api.snapshot import Snapshot
from premixdb.contracts import (
    DocumentListing,
)
from premixdb.engine.sources import SourceInput, source_proto
from premixdb.schemas import requests as _requests
from premixdb.v1 import corpus_pb2 as corpora


class Corpus(_Resource[corpora.Corpus, corpora.CreateCorpusRequest], _CorpusListings):
    def list_document(self, *, limit: int = 5, offset: int = 0) -> list[DocumentListing]:
        """List IDs and source keys from the latest snapshot, up to limit rows."""
        return _list_documents(self.latest(), limit=limit, offset=offset)

    @property
    def name(self) -> str:
        """Return the saved corpus name."""
        return self._resource.name

    def latest(self) -> Snapshot:
        """Load the last successful capture; the returned snapshot is immutable."""
        resource = self._db._get("Corpus", self._resource.id)
        if not resource.latest_snapshot_id:
            raise ValueError(
                f"corpus {self.name!r} has no snapshot; capture with corpus(source=...) first"
            )
        return self._db._snapshot(resource.latest_snapshot_id)

    @report_progress("Capturing snapshot")
    def snapshot(
        self,
        *,
        source: SourceInput,
        limit: int | None = None,
        base: Snapshot | None = None,
    ) -> Snapshot:
        """Capture sources into a new immutable snapshot, optionally reusing a base."""
        self._db._require_writable("capture snapshots")
        if limit is not None:
            _requests._uint(limit, 64, "limit")
        if base is not None:
            self._same_session(base)
            if base._resource.corpus_id != self._resource.id:
                raise ValueError("base snapshot must belong to the same corpus")
            base = base.wait()
        request = _requests.snapshot(
            self._resource,
            source=source_proto(source, limit=limit),
            base=base._proto if base else None,
        )
        result = self._db._submit(request).snapshot
        if base is not None and result.id == base._resource.id:
            return base
        return Snapshot(self._db, result, request)

    def _same_session(self, other: Snapshot) -> None:
        if other._db is not self._db:
            raise ValueError("resources belong to different PremixDB sessions; reopen by ID")
