"""Metadata-only catalog access. No engine, model, or scheduler imports."""

from __future__ import annotations

from collections.abc import MutableMapping
from functools import cached_property, wraps
from pathlib import Path
from threading import RLock
from typing import Callable, Concatenate, Literal

from blake3 import blake3
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from google.protobuf.message import Message

from .. import _requests
from .._protobuf import copy_message, descriptor_name
from .._wire import reject_unknown
from ..v1 import corpus_pb2 as corpora
from ..v1 import data_mixture_pb2 as datasets
from ..v1 import query_pb2 as queries
from ..v1 import snapshot_pb2 as snapshots
from ..v1 import status_pb2 as status
from .cache import MemoryCache
from .metadata import CatalogValue as ListedResource
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
type Membership = list[tuple[bytes, str, int | None]]
type ListingSnapshot = tuple[tuple[int, int], Membership]


def _read[Host: Catalog, Request: Message, Response, **P](
    method: Callable[Concatenate[Host, Request, P], Response],
) -> Callable[Concatenate[Host, Request, P], Response]:
    @wraps(method)
    def call(self: Host, request: Request, *args: P.args, **kwargs: P.kwargs) -> Response:
        reject_unknown(request)
        return method(self, copy_message(request), *args, **kwargs)

    return call


class Catalog:
    def __init__(self, store: ObjectStore | str | Path) -> None:
        self._storage = (
            store if isinstance(store, ObjectStore) else ObjectStore(store, read_only=True)
        )
        self._cache = MemoryCache(8 * 1024 * 1024)
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
    def GetSnapshot(self, request: snapshots.GetSnapshotRequest) -> snapshots.GetSnapshotResponse:
        return snapshots.GetSnapshotResponse(
            snapshot=self._resource("snapshot", request.id, snapshots.Snapshot)
        )

    @_read
    def GetQuery(self, request: queries.GetQueryRequest) -> queries.GetQueryResponse:
        resource = self._resource("query", request.id, queries.Query, ".pending")
        if not resource.HasField("profile") and not resource.HasField("estimate"):
            from .profiles import estimate_query

            resource.estimate.CopyFrom(estimate_query(self, resource))
        return queries.GetQueryResponse(query=resource)

    @_read
    def GetDataset(self, request: datasets.GetDatasetRequest) -> datasets.GetDatasetResponse:
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
    def Preview(self, request: queries.PreviewRequest) -> queries.PreviewResponse:
        from .previewing import preview

        return preview(self, request)

    @_read
    def ListExecutions(self, request: status.ListExecutionRequest) -> status.ListExecutionResponse:
        if request.resource_id and len(request.resource_id) not in (16, 32):
            raise ValueError("execution resource ID must be 16 or 32 bytes")
        return self._listing(
            request,
            "execution",
            status.ExecutionEvent,
            status.ListExecutionResponse(),
            lambda response: response.events,
            parents=(request.resource_id,) if request.resource_id else (),
            order="execution",
        )

    @_read
    def GetCorpus(self, request: corpora.GetCorpusRequest) -> corpora.GetCorpusResponse:
        _requests._id(request.id, 16)
        with self._lock:
            try:
                resource = self._storage.load(
                    "corpus", request.id, corpora.Corpus, suffix=".latest"
                )
            except KeyError:
                resource = self._storage.load("corpus", request.id, corpora.Corpus)
            return corpora.GetCorpusResponse(corpus=resource)

    @cached_property
    def _membership_cache(
        self,
    ) -> MutableMapping[tuple[str, tuple[bytes, ...], str], ListingSnapshot]:
        return self._cache.namespace("catalog_members")

    @cached_property
    def _scope_cache(self) -> MutableMapping[bytes, tuple[tuple[int, int], bytes]]:
        return self._cache.namespace("catalog_scopes")

    def _membership_snapshot(
        self,
        kind: str,
        parents: tuple[bytes, ...] = (),
        *,
        order: Literal["id", "public", "capture", "execution"] = "id",
    ) -> ListingSnapshot:
        metadata = self._storage.metadata
        key, revision = (kind, parents, order), metadata.revision
        cached = self._membership_cache.get(key)
        if cached is None or cached[0] != revision:
            suffixes = _suffixes(kind)
            members = metadata.members(kind, suffixes=suffixes, parents=parents, order=order)
            cached = revision, members
            self._membership_cache[key] = cached
        return cached

    def _members(
        self,
        kind: str,
        parents: tuple[bytes, ...] = (),
        *,
        order: Literal["id", "public", "capture", "execution"] = "id",
    ) -> Membership:
        return self._membership_snapshot(kind, parents, order=order)[1]

    def _listed[T: ListedResource](
        self, kind: str, id: bytes, suffix: str, message_type: type[T]
    ) -> T:
        # Completion wins; otherwise an active handle wins over failed/pending state.
        if kind == "mixture":
            resource = self.GetMix(datasets.GetMixRequest(id=id)).mix
            assert isinstance(resource, message_type)
            return resource
        if suffix:
            cache = self._queries if kind == "query" else self._datasets
            active = cache.get(id)
            if isinstance(active, message_type):
                return active
        elif kind == "mixture":
            active = self._mixes.get(id)
            if isinstance(active, message_type):
                return active
        return self._storage.load(kind, id, message_type, suffix=suffix)

    def browse[T: ListedResource](
        self,
        kind: str,
        message_type: type[T],
        *,
        parents: tuple[bytes, ...] = (),
        limit: int,
        offset: int,
    ) -> list[T]:
        """Public-ID order, filtering and a window before loading resource payloads."""
        suffixes = _suffixes(kind)
        members = self._storage.metadata.members(
            kind, suffixes=suffixes, parents=parents, order="public", limit=limit, offset=offset
        )
        return [self._listed(kind, id, suffix, message_type) for id, suffix, _ in members]

    def _listing[Request: ListRequest, Response: ListResponse, Item: ListedResource](
        self,
        request: Request,
        kind: str,
        message_type: type[Item],
        response: Response,
        values: Callable[[Response], RepeatedCompositeFieldContainer[Item]],
        *,
        parents: tuple[bytes, ...] = (),
        order: Literal["id", "execution"] = "id",
    ) -> Response:
        selector = copy_message(request)
        selector.ClearField("page_token")
        revision, members = self._membership_snapshot(kind, parents, order=order)
        key = (
            descriptor_name(request).encode()
            + b"\0"
            + selector.SerializeToString(deterministic=True)
        )
        cached = self._scope_cache.get(key)
        if cached is None or cached[0] != revision:
            digest = blake3(key)
            for id, _, _ in members:
                digest.update(id)
            scope = digest.digest()
            self._scope_cache[key] = revision, scope
        else:
            scope = cached[1]
        start, size, cap = _page(scope, request.page_token, len(members))
        rows = values(response)
        for id, suffix, _ in members[start : start + size]:
            rows.add().CopyFrom(self._listed(kind, id, suffix, message_type))
            if response.ByteSize() > cap:
                del rows[-1]
                if not rows:
                    raise ValueError("one resource exceeds the response byte limit")
                break
        end = start + len(rows)
        if end < len(members):
            response.next_page_token = scope + end.to_bytes(8, "big")
        return response

    @_read
    def ListCorpus(self, request: corpora.ListCorpusRequest) -> corpora.ListCorpusResponse:
        with self._lock:
            return self._listing(
                request,
                "corpus",
                corpora.Corpus,
                corpora.ListCorpusResponse(),
                lambda response: response.corpora,
            )

    @_read
    def ListSnapshot(
        self, request: snapshots.ListSnapshotRequest
    ) -> snapshots.ListSnapshotResponse:
        _requests._id(request.corpus_id, 16)
        with self._lock:
            return self._listing(
                request,
                "snapshot",
                snapshots.Snapshot,
                snapshots.ListSnapshotResponse(),
                lambda response: response.snapshots,
                parents=(request.corpus_id,),
            )

    @_read
    def ListQuery(self, request: queries.ListQueryRequest) -> queries.ListQueryResponse:
        if request.snapshot_id:
            _requests._id(request.snapshot_id, 32)
        with self._lock:
            return self._listing(
                request,
                "query",
                queries.Query,
                queries.ListQueryResponse(),
                lambda response: response.queries,
                parents=(request.snapshot_id,) if request.snapshot_id else (),
            )

    @_read
    def ListDatasets(self, request: datasets.ListDatasetRequest) -> datasets.ListDatasetResponse:
        _requests._id(request.query_id, 32)
        with self._lock:
            return self._listing(
                request,
                "dataset",
                datasets.Dataset,
                datasets.ListDatasetResponse(),
                lambda response: response.datasets,
                parents=(request.query_id,),
            )

    @_read
    def GetMix(self, request: datasets.GetMixRequest) -> datasets.GetMixResponse:
        _requests._id(request.id, 32)
        with self._lock:
            resource = (
                copy_message(self._mixes[request.id])
                if request.id in self._mixes
                else self._resource("mixture", request.id, datasets.Mix, ".recipe")
            )
            if not resource.HasField("profile"):
                try:
                    resource.profile.CopyFrom(
                        self._storage.load(
                            "mixture", request.id, datasets.MixProfile, suffix=".profile"
                        )
                    )
                except KeyError:
                    pass
            return datasets.GetMixResponse(mix=resource)

    @_read
    def ListMix(self, request: datasets.ListMixRequest) -> datasets.ListMixResponse:
        if request.query_id:
            _requests._id(request.query_id, 32)
        with self._lock:
            return self._listing(
                request,
                "mixture",
                datasets.Mix,
                datasets.ListMixResponse(),
                lambda response: response.mixtures,
                parents=(request.query_id,) if request.query_id else (),
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


def _suffixes(kind: str) -> tuple[str, ...]:
    return (
        ("", ".failed", ".pending")
        if kind == "query"
        else ("", ".failed", ".recipe")
        if kind in ("dataset", "mixture")
        else ("",)
    )
