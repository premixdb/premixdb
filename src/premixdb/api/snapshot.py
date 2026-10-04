"""Immutable captures and lazy query planning."""

from __future__ import annotations

from typing import (
    Iterable,
)

from premixdb.api.base import _Execution
from premixdb.api.query import Query
from premixdb.api.unions import _SnapshotOperations
from premixdb.contracts import (
    PreviewDocument,
)
from premixdb.schemas import requests as _requests
from premixdb.schemas.ids import _encode_id
from premixdb.schemas.protobuf import copy_message
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import snapshot_pb2 as snapshots


class Snapshot(
    _Execution[snapshots.Snapshot, snapshots.CreateSnapshotRequest], _SnapshotOperations
):
    def profile(self) -> snapshots.SnapshotProfile:
        """Return capture totals and change counts, displayed in at most 15 lines."""
        return copy_message(self.wait()._resource.profile)

    def preview(
        self, *, limit: int = 3, offset: int = 0, max_characters: int = 1024
    ) -> list[PreviewDocument]:
        """Browse up to three captured documents by default, starting at offset."""
        return self._preview(limit=limit, offset=offset, max_characters=max_characters)

    @property
    def corpus_id(self) -> str:
        """Return the parent corpus identity as an unpadded base64url string."""
        return _encode_id(self._resource.corpus_id)

    def _query_union(
        self,
        snapshots: Iterable[Snapshot],
        *,
        steps: Iterable[queries.Operation] = (),
        decontaminate: queries.Decontaminate | None = None,
        sampling: queries.QuerySampling | None = None,
    ) -> Query:
        self._db._require_writable("plan queries")
        request = _requests.query(
            *(s._resource.id for s in snapshots),
            steps=steps,
            decontaminate=decontaminate,
            sampling=sampling,
        )
        from premixdb.runtime import Coordinator

        assert isinstance(self._db._executor, Coordinator)
        return Query(self._db, self._db._executor._plan_query(request), request)
