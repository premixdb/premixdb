"""PremixDB sessions and saved-resource access."""

from __future__ import annotations

import os
from functools import cached_property
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import (
    Literal,
    overload,
)
from urllib.parse import urlsplit

from premixdb.api.base import _duration, _ResourceKind, _ResourceValue
from premixdb.api.collections import (
    CorpusCollection,
    _pages,
    _timestamp,
)
from premixdb.api.corpus import Corpus as CorpusHandle
from premixdb.api.dataset import Dataset
from premixdb.api.mixture import DataMixture
from premixdb.api.progress import report_progress
from premixdb.api.query import Query
from premixdb.api.snapshot import Snapshot
from premixdb.contracts import (
    ExecutionRecord,
)
from premixdb.engine.sources import SourceInput
from premixdb.schemas import requests as _requests
from premixdb.schemas.ids import _decode_id, _encode_id
from premixdb.storage.ranges import RangeReader
from premixdb.v1 import corpus_pb2 as corpora
from premixdb.v1 import data_mixture_pb2 as mix_pb
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import snapshot_pb2 as snapshots
from premixdb.v1 import status_pb2 as status


class PremixDB:
    """Open local storage for reusable capture, curation, and training recipes."""

    @property
    def version(self) -> str:
        """Return the installed premixdb package version."""
        from premixdb.version import __version__

        return __version__

    def __init__(
        self,
        *,
        storage: str | PathLike[str] | None = None,
        read_only: bool | None = None,
        timeout: float = 3600.0,
        poll_interval: float = 0.1,
        workers: int = 1,
        cache_bytes: int = 128 * 1024 * 1024,
        object_reader: RangeReader | None = None,
        metadata_path: str | PathLike[str] | None = None,
        process_workers: int = 0,
        progress: bool = True,
    ) -> None:
        storage = (
            storage
            if storage is not None
            else os.environ.get(
                "PREMIXDB_STORAGE",
                str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "premixdb"),
            )
        )
        storage = os.fspath(storage)
        if urlsplit(storage).scheme:
            raise ValueError("storage must be a local filesystem path")
        read_only = False if read_only is None else read_only
        if type(read_only) is not bool:
            raise TypeError("read_only must be boolean")
        if type(progress) is not bool:
            raise TypeError("progress must be boolean")
        if read_only and process_workers:
            raise ValueError("read-only sessions cannot configure compute workers")
        self._timeout = _duration(timeout, "timeout")
        self._poll_interval = _duration(poll_interval, "poll_interval")
        self._storage, self._read_only = storage, read_only
        self._closed = False
        self._progress_enabled = progress
        self._owns_object_reader = object_reader is None
        self._object_reader = (
            RangeReader(local_root=storage) if object_reader is None else object_reader
        )
        if read_only:
            from premixdb.storage.catalog import Catalog
            from premixdb.storage.objects import ObjectStore

            self._executor = Catalog(
                ObjectStore(
                    storage,
                    metadata_path=Path(metadata_path) if metadata_path is not None else None,
                    read_only=True,
                )
            )
        else:
            from premixdb.runtime import Coordinator

            self._executor = Coordinator(
                storage,
                allow_local_files=True,
                workers=workers,
                cache_bytes=cache_bytes,
                metadata_path=Path(metadata_path) if metadata_path is not None else None,
                process_workers=process_workers,
            )

    @report_progress("Waiting for background operations")
    def close(self) -> None:
        """Wait for local work and release session resources, keeping saved data."""
        if not self._closed:
            self._closed = True
            try:
                self._executor.close()
            finally:
                if self._owns_object_reader:
                    self._object_reader.close()

    def __enter__(self) -> PremixDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("PremixDB is closed")

    def _require_writable(self, action: str = "execute recipes") -> None:
        self._require_open()
        if self._read_only:
            raise PermissionError(f"read-only session: reopen with read_only=False to {action}")

    @overload
    def _submit(self, request: corpora.CreateCorpusRequest) -> corpora.CreateCorpusResponse: ...
    @overload
    def _submit(
        self, request: snapshots.CreateSnapshotRequest
    ) -> snapshots.CreateSnapshotResponse: ...
    @overload
    def _submit(self, request: queries.CreateQueryRequest) -> queries.CreateQueryResponse: ...
    @overload
    def _submit(self, request: mix_pb.CreateDatasetRequest) -> mix_pb.CreateDatasetResponse: ...
    @overload
    def _submit(self, request: mix_pb.CreateMixRequest) -> mix_pb.CreateMixResponse: ...
    @report_progress("Submitting operation")
    def _submit(
        self,
        request: corpora.CreateCorpusRequest
        | snapshots.CreateSnapshotRequest
        | queries.CreateQueryRequest
        | mix_pb.CreateDatasetRequest
        | mix_pb.CreateMixRequest,
    ) -> (
        corpora.CreateCorpusResponse
        | snapshots.CreateSnapshotResponse
        | queries.CreateQueryResponse
        | mix_pb.CreateDatasetResponse
        | mix_pb.CreateMixResponse
    ):
        self._require_writable()
        from premixdb.runtime.coordinator import Coordinator

        assert isinstance(self._executor, Coordinator)
        if isinstance(request, corpora.CreateCorpusRequest):
            return self._executor.CreateCorpus(request)
        if isinstance(request, snapshots.CreateSnapshotRequest):
            return self._executor.CreateSnapshot(request)
        if isinstance(request, queries.CreateQueryRequest):
            return self._executor.CreateQuery(request)
        if isinstance(request, mix_pb.CreateDatasetRequest):
            return self._executor.CreateDataset(request)
        if isinstance(request, mix_pb.CreateMixRequest):
            return self._executor.CreateMix(request)
        raise TypeError("expected a corpus, snapshot, query, dataset or mix create request")

    @overload
    def _get(self, kind: Literal["Corpus"], id: bytes) -> corpora.Corpus: ...
    @overload
    def _get(self, kind: Literal["Snapshot"], id: bytes) -> snapshots.Snapshot: ...
    @overload
    def _get(self, kind: Literal["Query"], id: bytes) -> queries.Query: ...
    @overload
    def _get(self, kind: Literal["Dataset"], id: bytes) -> mix_pb.Dataset: ...
    @overload
    def _get(self, kind: Literal["Mix"], id: bytes) -> mix_pb.Mix: ...
    @overload
    def _get(
        self,
        kind: _ResourceKind,
        id: bytes,
    ) -> _ResourceValue: ...
    def _get(
        self,
        kind: _ResourceKind,
        id: bytes,
    ) -> _ResourceValue:
        self._require_open()
        if kind == "Corpus":
            resource = self._executor.GetCorpus(corpora.GetCorpusRequest(id=id)).corpus
        elif kind == "Snapshot":
            resource = self._executor.GetSnapshot(snapshots.GetSnapshotRequest(id=id)).snapshot
        elif kind == "Query":
            resource = self._executor.GetQuery(queries.GetQueryRequest(id=id)).query
        elif kind == "Dataset":
            resource = self._executor.GetDataset(mix_pb.GetDatasetRequest(id=id)).dataset
        elif kind == "Mix":
            resource = self._executor.GetMix(mix_pb.GetMixRequest(id=id)).mix
        else:
            raise ValueError("unknown resource kind")
        if resource.id != id:
            raise ValueError("catalog returned a different resource")
        return resource

    @cached_property
    def Corpus(self) -> CorpusCollection:
        """Capture or reopen with Corpus(...), or browse with Corpus.list()."""
        return CorpusCollection(self)

    def _corpus(
        self,
        name: str,
        source: SourceInput | None = None,
        *,
        limit: int | None = None,
        base: Snapshot | None = None,
    ) -> Snapshot:
        """Capture a named corpus, or reopen its latest immutable snapshot."""
        if source is None and limit is not None:
            raise ValueError("limit requires a source")
        if source is None and base is not None:
            raise ValueError("base requires a source")
        if limit is not None:
            _requests._uint(limit, 64, "limit")
        if source is None:
            from premixdb.engine.names import corpus_id as _corpus_id

            _requests.corpus(name)
            try:
                resource = self._get("Corpus", _corpus_id(name))
            except KeyError:
                resource = None
            if resource is None or not resource.latest_snapshot_id:
                raise ValueError(
                    f"corpus {name!r} has no snapshot; capture with db.Corpus(name, source=...)"
                ) from None
            return self._snapshot(resource.latest_snapshot_id)
        handle = self._create_corpus(name)
        return handle.snapshot(source=source, limit=limit, base=base)

    def _create_corpus(self, name: str, *, request_id: str = "") -> CorpusHandle:
        """Create/reopen the mutable named handle for explicit snapshot management."""
        request = _requests.corpus(name, request_id=request_id)
        id = self._submit(request).id
        return CorpusHandle(self, self._get("Corpus", id), request)

    def _executions(self, resource_id: bytes | str | None = None) -> list[ExecutionRecord]:
        """List history with base64url IDs and UTC timestamps to whole seconds.

        resource_id identifies the resource created or materialized by the operation.
        request_digest fingerprints the request for idempotency checks.
        """
        return [
            ExecutionRecord(
                id=_encode_id(bytes.fromhex(event.id)),
                operation=event.operation,
                resource_id=_encode_id(event.resource_id),
                status=event.status,
                started_at=_timestamp(event.started_ns) if event.started_ns else None,
                ended_at=_timestamp(event.ended_ns) if event.ended_ns else None,
                request_digest=_encode_id(event.request_digest),
                error=event.error,
                cache_hit=event.cache_hit,
            )
            for event in self._execution_events(resource_id)
        ]

    def _execution_events(
        self, resource_id: bytes | str | None = None
    ) -> list[status.ExecutionEvent]:
        self._require_open()
        from premixdb.v1.status_pb2 import ListExecutionRequest

        identity = (
            b""
            if resource_id is None
            else _decode_id(resource_id)
            if isinstance(resource_id, str)
            else bytes(resource_id)
        )
        if identity and len(identity) not in (16, 32):
            raise ValueError("execution resource ID must be 16 or 32 bytes")
        return _pages(
            self,
            self._executor.ListExecutions,
            ListExecutionRequest(resource_id=identity),
            lambda response: response.events,
        )

    def _snapshot(self, id: bytes | str) -> Snapshot:
        """Open an immutable saved snapshot by its base64url, hexadecimal, or byte ID."""
        return Snapshot(self, self._get("Snapshot", _requests._id(id, 32)))

    def _query(self, id: bytes | str) -> Query:
        """Open a saved query by its base64url, hexadecimal, or byte ID."""
        return Query(self, self._get("Query", _requests._id(id, 32)))

    def _dataset(self, id: bytes | str) -> Dataset:
        """Open a saved training dataset by its base64url, hexadecimal, or byte ID."""
        return Dataset(self, self._get("Dataset", _requests._id(id, 32)))

    def _mix(self, id: bytes | str) -> DataMixture:
        """Open a saved mixture collection without packing its candidates."""
        return DataMixture(self, self._get("Mix", _requests._id(id, 32)))
