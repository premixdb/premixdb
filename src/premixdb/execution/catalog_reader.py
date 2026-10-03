"""Metadata-only catalog access. No engine, model, or scheduler imports."""

from __future__ import annotations

from collections.abc import MutableMapping
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Callable, Concatenate

from blake3 import blake3
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from google.protobuf.message import Message

from .. import _requests
from .._protobuf import copy_message, descriptor_name
from .._wire import reject_unknown
from ..v1 import corpus_pb2 as corpora
from ..v1 import dataset_pb2 as datasets
from ..v1 import query_pb2 as queries
from ..v1 import snapshot_pb2 as snapshots
from ..v1 import status_pb2 as status
from .storage import ObjectStore

type ListRequest = (
    corpora.ListCorpusRequest
    | snapshots.ListSnapshotRequest
    | queries.ListQueryRequest
    | datasets.ListDatasetRequest
    | datasets.ListMixRequest
    | status.ListExecutionRequest
)
type ListResponse = (
    corpora.ListCorpusResponse
    | snapshots.ListSnapshotResponse
    | queries.ListQueryResponse
    | datasets.ListDatasetResponse
    | datasets.ListMixResponse
    | status.ListExecutionResponse
)
type ListedResource = (
    corpora.Corpus
    | snapshots.Snapshot
    | queries.Query
    | datasets.Dataset
    | datasets.Mix
    | status.ExecutionEvent
)


def _read[Request: Message, Response, **P](
    method: Callable[Concatenate[Catalog, Request, P], Response],
) -> Callable[Concatenate[Catalog, Request, P], Response]:
    @wraps(method)
    def call(self: Catalog, request: Request, *args: P.args, **kwargs: P.kwargs) -> Response:
        reject_unknown(request)
        return method(self, copy_message(request), *args, **kwargs)

    return call


class Catalog:
    def __init__(self, store: ObjectStore | str | Path) -> None:
        self._storage = (
            store if isinstance(store, ObjectStore) else ObjectStore(store, read_only=True)
        )
        self._lock = RLock()
        self._queries: MutableMapping[bytes, queries.Query] = {}
        self._datasets: MutableMapping[bytes, datasets.Dataset] = {}
        self._mixes: MutableMapping[bytes, datasets.Mix] = {}

    def close(self) -> None:
        self._storage.close()

    def _resource[T: Message](
        self, kind: str, identity: bytes, message_type: type[T], pending: str = ""
    ) -> T:
        _requests._id(identity, 32)
        for suffix in ("", ".failed", pending):
            try:
                return self._storage.load(kind, identity, message_type, suffix=suffix)
            except KeyError:
                pass
        raise KeyError((kind, identity.hex()))

    @_read
    def GetSnapshot(
        self, request: snapshots.GetSnapshotRequest, *, timeout: float | None = None
    ) -> snapshots.GetSnapshotResponse:
        return snapshots.GetSnapshotResponse(
            snapshot=self._resource("snapshot", request.id, snapshots.Snapshot)
        )

    @_read
    def GetQuery(
        self, request: queries.GetQueryRequest, *, timeout: float | None = None
    ) -> queries.GetQueryResponse:
        return queries.GetQueryResponse(
            query=self._resource("query", request.id, queries.Query, ".pending")
        )

    @_read
    def GetDataset(
        self, request: datasets.GetDatasetRequest, *, timeout: float | None = None
    ) -> datasets.GetDatasetResponse:
        return datasets.GetDatasetResponse(
            dataset=self._dataset_profile(
                self._resource("dataset", request.id, datasets.Dataset, ".recipe")
            )
        )

    def _dataset_profile(self, resource: datasets.Dataset) -> datasets.Dataset:
        if not resource.HasField("profile"):
            try:
                profile = self._storage.load(
                    "dataset", resource.id, datasets.DatasetProfile, suffix=".planned-profile"
                )
            except KeyError:
                pass
            else:
                resource.profile.CopyFrom(profile)
        return resource

    @_read
    def Preview(
        self, request: queries.PreviewRequest, *, timeout: float | None = None
    ) -> queries.PreviewResponse:
        from .previewing import preview

        return preview(self, request)

    @_read
    def ListExecutions(
        self, request: status.ListExecutionRequest, *, timeout: float | None = None
    ) -> status.ListExecutionResponse:
        if request.resource_id and len(request.resource_id) not in (16, 32):
            raise ValueError("execution resource ID must be 16 or 32 bytes")
        events: list[status.ExecutionEvent] = self._storage.metadata.list(
            "execution", status.ExecutionEvent
        )
        ordered = sorted(
            [
                event
                for event in events
                if not request.resource_id or event.resource_id == request.resource_id
            ],
            key=lambda e: (e.started_ns, e.id),
        )
        return self._listing(
            request,
            ordered,
            status.ListExecutionResponse(),
            lambda response: response.events,
            ordered=True,
        )

    @_read
    def GetCorpus(
        self, request: corpora.GetCorpusRequest, *, timeout: float | None = None
    ) -> corpora.GetCorpusResponse:
        _requests._id(request.id, 16)
        with self._lock:
            try:
                resource = self._storage.load(
                    "corpus", request.id, corpora.Corpus, suffix=".latest"
                )
            except KeyError:
                resource = self._storage.load("corpus", request.id, corpora.Corpus)
            return corpora.GetCorpusResponse(corpus=resource)

    def _listing[Request: ListRequest, Response: ListResponse, Item: ListedResource](
        self,
        request: Request,
        resources: list[Item],
        response: Response,
        values: Callable[[Response], RepeatedCompositeFieldContainer[Item]],
        *,
        ordered: bool = False,
    ) -> Response:
        # Bind continuation tokens to this listing and its immutable membership.
        if not ordered:
            resources = sorted(resources, key=lambda value: value.id)
        selector = copy_message(request)
        selector.ClearField("page_token")
        scope = blake3(
            descriptor_name(request).encode()
            + b"\0"
            + selector.SerializeToString(deterministic=True)
            + b"".join(
                value.id.encode() if isinstance(value.id, str) else value.id for value in resources
            )
        ).digest()
        start, size, cap = _page(scope, request.page_token, len(resources))
        rows = values(response)
        for resource in resources[start : start + size]:
            rows.add().CopyFrom(resource)
            if response.ByteSize() > cap:
                del rows[-1]
                if not rows:
                    raise ValueError("one resource exceeds the response byte limit")
                break
        end = start + len(rows)
        if end < len(resources):
            response.next_page_token = scope + end.to_bytes(8, "big")
        return response

    @_read
    def ListCorpus(
        self, request: corpora.ListCorpusRequest, *, timeout: float | None = None
    ) -> corpora.ListCorpusResponse:
        with self._lock:
            return self._listing(
                request,
                self._storage.list("corpus", corpora.Corpus),
                corpora.ListCorpusResponse(),
                lambda response: response.corpora,
            )

    @_read
    def ListSnapshot(
        self, request: snapshots.ListSnapshotRequest, *, timeout: float | None = None
    ) -> snapshots.ListSnapshotResponse:
        _requests._id(request.corpus_id, 16)
        with self._lock:
            return self._listing(
                request,
                [
                    s
                    for s in self._storage.list("snapshot", snapshots.Snapshot)
                    if s.corpus_id == request.corpus_id
                ],
                snapshots.ListSnapshotResponse(),
                lambda response: response.snapshots,
            )

    @_read
    def ListQuery(
        self, request: queries.ListQueryRequest, *, timeout: float | None = None
    ) -> queries.ListQueryResponse:
        if request.snapshot_id:
            _requests._id(request.snapshot_id, 32)
        with self._lock:
            resources = self._resources("query", queries.Query, self._queries, ".pending")
            return self._listing(
                request,
                [
                    q
                    for q in resources
                    if not request.snapshot_id or request.snapshot_id in q.snapshot_ids
                ],
                queries.ListQueryResponse(),
                lambda response: response.queries,
            )

    @_read
    def ListDatasets(
        self, request: datasets.ListDatasetRequest, *, timeout: float | None = None
    ) -> datasets.ListDatasetResponse:
        _requests._id(request.query_id, 32)
        with self._lock:
            return self._listing(
                request,
                [
                    d
                    for d in self._resources("dataset", datasets.Dataset, self._datasets, ".recipe")
                    if d.query_id == request.query_id
                ],
                datasets.ListDatasetResponse(),
                lambda response: response.datasets,
            )

    def _resources[T: queries.Query | datasets.Dataset](
        self, kind: str, message_type: type[T], cache: MutableMapping[bytes, T], recipe_suffix: str
    ) -> list[T]:
        # Durable completion takes priority over active, failed, and pending state.
        resources = {}
        for group in (
            self._storage.list(kind, message_type, suffix=recipe_suffix),
            self._storage.list(kind, message_type, suffix=".failed"),
            cache.values(),
            self._storage.list(kind, message_type),
        ):
            resources.update((resource.id, resource) for resource in group)
        return list(resources.values())

    @_read
    def GetMix(
        self, request: datasets.GetMixRequest, *, timeout: float | None = None
    ) -> datasets.GetMixResponse:
        _requests._id(request.id, 32)
        with self._lock:
            return datasets.GetMixResponse(
                mix=self._mixes.get(request.id)
                or self._storage.load("mixture", request.id, datasets.Mix)
            )

    @_read
    def ListMix(
        self, request: datasets.ListMixRequest, *, timeout: float | None = None
    ) -> datasets.ListMixResponse:
        if request.query_id:
            _requests._id(request.query_id, 32)
        with self._lock:
            resources = {value.id: value for value in self._storage.list("mixture", datasets.Mix)}
            resources.update(self._mixes)
            return self._listing(
                request,
                [
                    value
                    for value in resources.values()
                    if not request.query_id or value.query_id == request.query_id
                ],
                datasets.ListMixResponse(),
                lambda response: response.mixtures,
            )


def _page(id: bytes, page_token: bytes, count: int) -> tuple[int, int, int]:
    _requests._id(id, 32)
    start = 0
    if page_token:
        if len(page_token) != 40 or page_token[:32] != id:
            raise ValueError("invalid page token for this resource")
        start = int.from_bytes(page_token[32:], "big")
    if start > count:
        raise ValueError("page starts beyond the resource")
    return start, 128, 1024 * 1024 - 128
