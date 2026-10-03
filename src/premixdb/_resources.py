"""Fluent recipes and resource handles for local data."""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, Mapping, Self, cast, overload
from urllib.parse import urlsplit

from google.protobuf.message import Message

from . import _requests
from ._catalog import CorpusCollection, _CorpusListings, _list_documents, _pages, _timestamp
from ._enums import ExecutionStatus
from ._field_expr import FieldProjection
from ._ids import _decode_id, _encode_id, _public_dataset_profile, _public_lineage
from ._inputs import HuggingFaceSource
from ._inputs import Source as Source
from ._mixing import Bounds, RegMixSampler, Tokens
from ._policies import ByteTokenizer as BytePolicy
from ._policies import Concat as ConcatPolicy
from ._profiles import DistributionSummary, ProfileSelector, _MixProfiles
from ._progress import report_progress
from ._protobuf import copy_message, parse
from ._reader import Reader as Reader
from ._reader import Topology as Topology
from ._requests import _Field
from ._storage import RangeReader
from ._types import (
    Checkpoint,
    DatasetSummary,
    DocumentListing,
    ExecutionRecord,
    PreviewDocument,
    PreviewSequence,
    QueryPopulationSummary,
    QueryStepSummary,
    QuerySummary,
    SnapshotSummary,
)
from ._typing import load_json
from ._unions import SnapshotUnion as SnapshotUnion
from ._unions import _SnapshotOperations
from .engine.contracts import Provenance
from .fields import ContentType, ContentTypeFields, Language, LanguageFields, Topic, TopicFields
from .v1 import corpus_pb2 as corpora
from .v1 import dataset_pb2 as datasets
from .v1 import profile_pb2 as profiles
from .v1 import query_pb2 as queries
from .v1 import snapshot_pb2 as snapshots
from .v1 import status_pb2 as status
from .v1 import storage_pb2 as source_types

if TYPE_CHECKING:
    from ._torch import StreamingDataset, TorchDataset

SourceInput = (
    str
    | PathLike[str]
    | source_types.Source
    | source_types.HuggingFaceDataset
    | HuggingFaceSource
    | Iterable[Source]
)

DomainInput = (
    type[Topic]
    | type[ContentType]
    | type[Language]
    | _Field[int]
    | _Field[str]
    | FieldProjection
    | LanguageFields
    | TopicFields
    | ContentTypeFields
    | Iterable[_Field[int] | _Field[str] | FieldProjection]
    | datasets.Domains
    | Mapping[str, str]
)


def _counts(totals: snapshots.SnapshotProfile) -> SnapshotSummary:
    return SnapshotSummary(
        documents=totals.documents, bytes=totals.content_bytes, characters=totals.characters
    )


def _query_counts(
    profile: queries.QueryProfile | queries.QueryStepProfile, prefix: str
) -> QueryPopulationSummary:
    return QueryPopulationSummary(
        documents=getattr(profile, f"{prefix}_documents"),
        bytes=getattr(profile, f"{prefix}_content_bytes"),
        characters=getattr(profile, f"{prefix}_characters"),
    )


class ExecutionError(RuntimeError):
    pass


class PremixDB:
    """Open a local demo database and execute recipes in this Python process."""

    @property
    def version(self) -> str:
        """Return the installed premixdb package version."""
        from ._version import __version__

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
        if urlsplit(str(storage)).scheme:
            raise ValueError("storage must be a local filesystem path")
        read_only = False if read_only is None else read_only
        if type(read_only) is not bool:
            raise TypeError("read_only must be boolean")
        if type(progress) is not bool:
            raise TypeError("progress must be boolean")
        if read_only and process_workers:
            raise ValueError("read-only sessions cannot configure compute workers")
        if not 0 < timeout < float("inf") or not 0 < poll_interval < float("inf"):
            raise ValueError("timeout and poll_interval must be positive and finite")
        self._storage, self._read_only = str(storage), read_only
        self._timeout, self._poll_interval = timeout, poll_interval
        self._closed = False
        self._progress_enabled = progress
        self._object_reader = object_reader or RangeReader(local_root=Path(storage))
        if read_only:
            from .execution.catalog_reader import Catalog
            from .execution.storage import ObjectStore

            self._executor = Catalog(
                ObjectStore(
                    Path(storage),
                    metadata_path=Path(metadata_path) if metadata_path is not None else None,
                    read_only=True,
                )
            )
        else:
            from .execution import Coordinator

            self._executor = Coordinator(
                Path(storage),
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
            self._executor.close()

    def __enter__(self) -> PremixDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @overload
    def _submit(
        self, request: corpora.CreateCorpusRequest, *, timeout: float | None = None
    ) -> corpora.CreateCorpusResponse: ...
    @overload
    def _submit(
        self, request: snapshots.CreateSnapshotRequest, *, timeout: float | None = None
    ) -> snapshots.CreateSnapshotResponse: ...
    @overload
    def _submit(
        self, request: queries.CreateQueryRequest, *, timeout: float | None = None
    ) -> queries.CreateQueryResponse: ...
    @overload
    def _submit(
        self, request: datasets.CreateDatasetRequest, *, timeout: float | None = None
    ) -> datasets.CreateDatasetResponse: ...
    @overload
    def _submit(
        self, request: datasets.CreateMixRequest, *, timeout: float | None = None
    ) -> datasets.CreateMixResponse: ...
    @report_progress("Submitting operation")
    def _submit(
        self,
        request: corpora.CreateCorpusRequest
        | snapshots.CreateSnapshotRequest
        | queries.CreateQueryRequest
        | datasets.CreateDatasetRequest
        | datasets.CreateMixRequest,
        *,
        timeout: float | None = None,
    ) -> (
        corpora.CreateCorpusResponse
        | snapshots.CreateSnapshotResponse
        | queries.CreateQueryResponse
        | datasets.CreateDatasetResponse
        | datasets.CreateMixResponse
    ):
        if self._closed:
            raise ValueError("PremixDB is closed")
        if self._read_only:
            raise PermissionError(
                "read-only session: reopen with read_only=False to execute recipes"
            )
        from .execution.coordinator import Coordinator

        assert isinstance(self._executor, Coordinator)
        if isinstance(request, corpora.CreateCorpusRequest):
            return self._executor.CreateCorpus(request)
        if isinstance(request, snapshots.CreateSnapshotRequest):
            return self._executor.CreateSnapshot(request)
        if isinstance(request, queries.CreateQueryRequest):
            return self._executor.CreateQuery(request)
        if isinstance(request, datasets.CreateDatasetRequest):
            return self._executor.CreateDataset(request)
        if isinstance(request, datasets.CreateMixRequest):
            return self._executor.CreateMix(request)
        raise TypeError("expected a corpus, snapshot, query, dataset or mix create request")

    @overload
    def _get(
        self, kind: Literal["Snapshot"], id: bytes, *, timeout: float | None = None
    ) -> snapshots.Snapshot: ...
    @overload
    def _get(
        self, kind: Literal["Query"], id: bytes, *, timeout: float | None = None
    ) -> queries.Query: ...
    @overload
    def _get(
        self, kind: Literal["Dataset"], id: bytes, *, timeout: float | None = None
    ) -> datasets.Dataset: ...
    @overload
    def _get(
        self,
        kind: Literal["Snapshot", "Query", "Dataset"],
        id: bytes,
        *,
        timeout: float | None = None,
    ) -> snapshots.Snapshot | queries.Query | datasets.Dataset: ...
    def _get(
        self,
        kind: Literal["Snapshot", "Query", "Dataset"],
        id: bytes,
        *,
        timeout: float | None = None,
    ) -> snapshots.Snapshot | queries.Query | datasets.Dataset:
        duration = self._timeout if timeout is None else timeout
        if kind == "Snapshot":
            resource = self._executor.GetSnapshot(
                snapshots.GetSnapshotRequest(id=id), timeout=duration
            ).snapshot
        elif kind == "Query":
            resource = self._executor.GetQuery(
                queries.GetQueryRequest(id=id), timeout=duration
            ).query
        elif kind == "Dataset":
            resource = self._executor.GetDataset(
                datasets.GetDatasetRequest(id=id), timeout=duration
            ).dataset
        else:
            raise ValueError("unknown resource kind")
        if resource.id != id:
            raise ValueError("catalog returned a different resource")
        return resource

    @cached_property
    def corpus(self) -> CorpusCollection:
        """Capture or reopen a corpus with corpus(...), or browse with corpus.list()."""
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
            from ._identity import corpus_id as _corpus_id

            _requests.corpus(name)
            try:
                resource = self._executor.GetCorpus(
                    corpora.GetCorpusRequest(id=_corpus_id(name)),
                    timeout=self._timeout,
                ).corpus
            except KeyError:
                raise ValueError(
                    f"corpus {name!r} has no snapshot; capture with db.corpus(name, source=...)"
                ) from None
            return Corpus(self, resource).latest()
        handle = self._create_corpus(name)
        return handle.snapshot(source=source, limit=limit, base=base)

    def _create_corpus(self, name: str, *, request_id: str = "") -> Corpus:
        """Create/reopen the mutable named handle for explicit snapshot management."""
        request = _requests.corpus(name, request_id=request_id)
        id = self._submit(request).id
        resource = self._executor.GetCorpus(
            corpora.GetCorpusRequest(id=id), timeout=self._timeout
        ).corpus
        return Corpus(self, resource, request)

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
        if self._closed:
            raise ValueError("PremixDB is closed")
        from .v1.status_pb2 import ListExecutionRequest

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

    def _datasets(self, id: bytes | str) -> Datasets:
        """Open a saved mixture collection without packing its candidates."""
        id = _requests._id(id, 32)
        response = self._executor.GetMix(
            datasets.GetMixRequest(id=id),
            timeout=self._timeout,
        )
        if response.mix.id != id:
            raise ValueError("catalog returned a different mixture collection")
        return Mix(self, response.mix)

    def _source(self, value: SourceInput, *, limit: int | None = None) -> source_types.Source:
        if isinstance(value, HuggingFaceSource):
            value = value._to_proto()
        if isinstance(value, source_types.HuggingFaceDataset):
            value = source_types.Source(hugging_face=value)
        if isinstance(value, source_types.Source):
            value = copy_message(value)
            if limit is not None:
                value.limit = limit
            return value
        if isinstance(value, (str, PathLike)):
            path = str(value)
            location = urlsplit(path)
            if location.scheme:
                raise ValueError("source paths must be local files")
            return source_types.Source(
                limit=limit,
                files=source_types.FileSources(
                    documents=[source_types.FileSource(path=str(Path(path).resolve()))]
                ),
            )
        return source_types.Source(
            limit=limit,
            memory=source_types.MemorySources(
                documents=[
                    source_types.MemorySource(uri=item.key, text=item.text)
                    for item in islice(value, limit)
                ]
            ),
        )


class _Resource[
    ResourceT: corpora.Corpus
    | snapshots.Snapshot
    | queries.Query
    | datasets.Dataset
    | datasets.Mix,
    RequestT: Message,
]:
    def __init__(
        self, client: PremixDB, resource: ResourceT, request: RequestT | None = None
    ) -> None:
        self._db, self._resource = client, copy_message(resource)
        self._creation_request = copy_message(request) if request is not None else None

    def __repr__(self) -> str:
        from ._display import _resource_repr

        if not isinstance(self, (Corpus, Snapshot, Query, Mix, Dataset)):
            raise TypeError("unsupported resource handle")
        return _resource_repr(self)

    @property
    def id(self) -> str:
        """Return the resource identity as an unpadded base64url string."""
        return _encode_id(self._resource.id)

    @property
    def _proto(self) -> ResourceT:
        """A detached protobuf view; editing it cannot mutate this handle."""
        return copy_message(self._resource)

    @property
    def _request(self) -> RequestT | None:
        """Return a detached creation request, or None for a reopened handle."""
        return copy_message(self._creation_request) if self._creation_request is not None else None


class Corpus(_Resource[corpora.Corpus, corpora.CreateCorpusRequest], _CorpusListings):
    def list_document(self, *, limit: int = 5, offset: int = 0) -> list[DocumentListing]:
        """List IDs and source keys from the latest snapshot, up to limit rows."""
        return _list_documents(self.latest(), limit=limit, offset=offset)

    @property
    def name(self) -> str:
        """Return the saved corpus name."""
        assert isinstance(self._resource, corpora.Corpus)
        return self._resource.name

    def latest(self) -> Snapshot:
        """Load the last successful capture; the returned snapshot is immutable."""
        resource = self._db._executor.GetCorpus(
            corpora.GetCorpusRequest(id=self._resource.id),
            timeout=self._db._timeout,
        ).corpus
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
        if limit is not None:
            _requests._uint(limit, 64, "limit")
        if base is not None:
            self._same_session(base)
            base = base.wait()
        request = _requests.snapshot(
            cast(corpora.Corpus, self._resource),
            source=self._db._source(source, limit=limit),
            base=base._proto if base else None,
        )
        result = self._db._submit(request).snapshot
        if base is not None and result.id == base._resource.id:
            return base
        return Snapshot(self._db, result, request)

    def _same_session(self, other: Snapshot) -> None:
        if other._db is not self._db:
            raise ValueError("resources belong to different PremixDB sessions; reopen by ID")


class _Execution[
    ResourceT: snapshots.Snapshot | queries.Query | datasets.Dataset,
    RequestT: Message,
](_Resource[ResourceT, RequestT]):
    @property
    def status(self) -> ExecutionStatus:
        """Return the current handle status; wait() refreshes execution progress."""
        return ExecutionStatus(
            status.Status.Name(self._resource.status).removeprefix("STATUS_").lower()
        )

    @report_progress("Waiting for {kind} {id}")
    def wait(self, *, timeout: float | None = None) -> Self:
        """Return a completed handle; propagate failure and enforce one total deadline."""
        budget = self._db._timeout if timeout is None else timeout
        if not 0 < budget < float("inf"):
            raise ValueError("timeout must be positive and finite")
        deadline = time.monotonic() + budget
        value = self._resource
        if self._db._read_only and value.status != status.STATUS_COMPLETED:
            raise ExecutionError(
                "resource is not complete; materialize it in a writable session first"
            )
        if isinstance(self, (Query, Dataset)) and value.status == status.STATUS_PENDING:
            if isinstance(self, Query):
                from ._wire import copy_fields

                recipe = copy_fields(value, queries.CreateQueryRequest())
            else:
                recipe = self._recipe
            response = self._db._submit(
                recipe,
                timeout=budget,
            )
            if response.id != value.id:
                raise ExecutionError("materialization returned a different resource")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for dataset {self.id}")
            value = cast(
                ResourceT,
                self._db._get(
                    "Query" if isinstance(self, Query) else "Dataset",
                    response.id,
                    timeout=remaining,
                ),
            )
        while value.status != status.STATUS_COMPLETED:
            if value.status == status.STATUS_ERROR:
                raise ExecutionError(
                    f"{type(self).__name__} {self.id} failed: {getattr(value, 'error', '')}"
                )
            if value.status not in (status.STATUS_PENDING, status.STATUS_RUNNING):
                raise ExecutionError("resource has an unspecified execution status")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {type(self).__name__.lower()} {self.id}")
            time.sleep(min(self._db._poll_interval, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {self.id}")
            value = cast(
                ResourceT,
                self._db._get(
                    "Snapshot"
                    if isinstance(self, Snapshot)
                    else "Query"
                    if isinstance(self, Query)
                    else "Dataset",
                    value.id,
                    timeout=remaining,
                ),
            )
        self._resource = value
        return self

    def profile(self) -> snapshots.SnapshotProfile | queries.QueryProfile | datasets.DatasetProfile:
        """Return a detached typed profile, waiting for background execution."""
        return copy_message(self.wait()._resource.profile)

    def _preview(
        self, *, limit: int = 3, offset: int = 0, max_characters: int = 1024
    ) -> list[PreviewDocument]:
        """Browse selected documents in deterministic order, with bounded text."""
        request = queries.PreviewRequest(
            limit=_requests._uint(limit, 32, "limit"),
            offset=_requests._uint(offset, 64, "offset"),
            max_characters=_requests._uint(max_characters, 32, "max_characters"),
        )
        ready = self.wait()._resource
        if isinstance(ready, snapshots.Snapshot):
            request.snapshot_id = ready.id
        else:
            request.query_id = ready.id
        response = self._db._executor.Preview(
            request,
            timeout=self._db._timeout,
        )
        return [
            dict(
                id=_encode_id(doc.id),
                text=doc.text,
                truncated=doc.truncated,
                source_key=doc.source_key,
                corpus_id=_encode_id(doc.corpus_id),
                ordinal=doc.ordinal,
            )
            for doc in response.preview.documents
        ]

    @overload
    def _summary(self: _Execution[snapshots.Snapshot, RequestT]) -> SnapshotSummary: ...
    @overload
    def _summary(self: _Execution[queries.Query, RequestT]) -> QuerySummary: ...
    @overload
    def _summary(self: _Execution[datasets.Dataset, RequestT]) -> DatasetSummary: ...
    def _summary(self) -> SnapshotSummary | QuerySummary | DatasetSummary:
        """Return selection or packing counts for this completed resource."""
        ready = self.wait()._resource
        if not ready.HasField("profile"):
            raise ExecutionError("completed resource has no published profile")
        if isinstance(ready, snapshots.Snapshot):
            return _counts(ready.profile)
        if isinstance(ready, queries.Query):
            p = ready.profile
            return QuerySummary(
                input=_query_counts(p, "input"),
                output=_query_counts(p, "output"),
                steps=[
                    QueryStepSummary(
                        before=_query_counts(op, "input"), after=_query_counts(op, "output")
                    )
                    for op in p.steps
                ],
            )
        assert isinstance(ready, datasets.Dataset)
        p = ready.profile
        return DatasetSummary(
            content_tokens=p.content_tokens,
            separator_tokens=p.separator_tokens,
            padding_tokens=p.padding_tokens,
            dropped_tokens=p.dropped_tokens,
            sequences=p.sequences,
            output_tokens=p.sequences * ready.sequence_length,
        )


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
        request = _requests.query(
            *(s._resource.id for s in snapshots),
            steps=steps,
            decontaminate=decontaminate,
            sampling=sampling,
        )
        if self._db._closed:
            raise ValueError("PremixDB is closed")
        if self._db._read_only:
            raise PermissionError("query planning requires a writable session")
        from .execution import Coordinator

        assert isinstance(self._db._executor, Coordinator)
        return Query(self._db, self._db._executor._plan_query(request), request)


class Query(_Execution[queries.Query, queries.CreateQueryRequest]):
    def profile(self) -> queries.QueryProfile:
        """Execute the full query if needed, then return cached selection statistics.

        Even a query with no steps materializes its selection on the first call.
        Use snapshot.profile() for existing capture statistics without a query.
        """
        return copy_message(self.wait()._resource.profile)

    def preview(
        self, *, limit: int = 3, offset: int = 0, max_characters: int = 1024
    ) -> list[PreviewDocument]:
        """Wait for results and browse up to three selected documents, starting at offset."""
        return self._preview(limit=limit, offset=offset, max_characters=max_characters)

    def _list_document(self, *, limit: int = 5, offset: int = 0) -> list[DocumentListing]:
        """List selected document IDs and source keys without fetching text.

        Wait for completion, then return at most limit rows (maximum 1000)
        starting at offset. Repeated sampled occurrences keep separate ordinals.
        """
        from ._catalog import _list_documents

        return _list_documents(self, limit=limit, offset=offset)

    def _with_fields(self, fields: Iterable[ProfileSelector]) -> Query:
        request = _requests.query(
            *self._resource.snapshot_ids,
            steps=self._resource.operations,
            fields=fields,
            decontaminate=self._resource.decontaminate
            if self._resource.HasField("decontaminate")
            else None,
            sampling=self._resource.sampling if self._resource.HasField("sampling") else None,
            git_commit=self._resource.git_commit,
        )
        if self._db._read_only:
            from .execution.planner import compile_query

            return self._db._query(compile_query(request).id)
        from .execution import Coordinator

        assert isinstance(self._db._executor, Coordinator)
        return Query(self._db, self._db._executor._plan_query(request), request)

    def _describe(self, field: ProfileSelector) -> DistributionSummary:
        """Summarize a field, computing a missing derived projection when needed."""
        from ._profiles import _describe_field

        try:
            return _describe_field(self.profile().fields, field)
        except KeyError:
            return _describe_field(self._with_fields([field]).profile().fields, field)

    def _provenance(self) -> dict[str, Provenance]:
        """Trace each selected document to its source and query decisions."""
        import json

        from .execution.selections import decode_lineage

        resource = self.wait()._resource
        ref = resource.lineage
        public = _public_lineage(
            load_json(
                self._db._object_reader.read(
                    source_types.SpanRef(
                        object=ref, end=ref.size_bytes, blake3_digest=ref.blake3_digest
                    )
                )
            )
        )
        return decode_lineage(json.dumps(public).encode())

    @property
    def _estimate(self) -> profiles.QueryEstimate:
        """Refresh cardinality bounds and field distributions without waiting."""
        return copy_message(self._db._get("Query", self._resource.id).estimate)

    def dataset(
        self,
        *,
        tokenizer: datasets.Tokenizer | BytePolicy | None = None,
        sequence_length: int = 2048,
        packing: datasets.Packing | ConcatPolicy | None = None,
    ) -> Dataset:
        """Plan a lazy dataset; profiling, reading and torch() consume its recipe."""
        request = _requests.dataset(
            self._resource.id,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            packing=packing,
        )
        if self._db._closed:
            raise ValueError("PremixDB is closed")
        if self._db._read_only:
            raise PermissionError("dataset planning requires a writable session")
        from .execution import Coordinator

        assert isinstance(self._db._executor, Coordinator)
        return Dataset(self._db, self._db._executor._plan_dataset(request), request)

    @report_progress("Creating mixture from query {id}")
    def mix(
        self,
        *,
        domains: DomainInput | None = None,
        sampler: RegMixSampler | None = None,
        size: Tokens | None = None,
        tokens: int | None = None,
        tokenizer: datasets.Tokenizer | BytePolicy | None = None,
        sequence_length: int = 2048,
        packing: datasets.Packing | ConcatPolicy | None = None,
        bounds: Bounds | None = None,
        n_candidates: int = 3,
        replacement: bool = True,
        seed: int = 0,
    ) -> Datasets:
        """Register three lazy datasets by default, with reproducible sampling."""
        request = _requests.mix(
            self.wait()._proto,
            domains=domains,
            sampler=sampler,
            size=size,
            tokens=tokens,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            packing=packing,
            bounds=bounds,
            n_candidates=n_candidates,
            replacement=replacement,
            seed=seed,
        )
        id = self._db._submit(request).id
        response = self._db._executor.GetMix(
            datasets.GetMixRequest(id=id),
            timeout=self._db._timeout,
        )
        return Mix(self._db, response.mix, request)


class Mix(_Resource[datasets.Mix, datasets.CreateMixRequest]):
    """An ordered collection of registered dataset recipes, with lazy token packing."""

    def __init__(
        self,
        client: PremixDB,
        resource: datasets.Mix,
        request: datasets.CreateMixRequest | None = None,
        *,
        indices: Iterable[int] | None = None,
        cache: dict[int, Dataset] | None = None,
    ) -> None:
        super().__init__(client, resource, request)
        self._indices = (
            tuple(range(len(resource.dataset_ids))) if indices is None else tuple(indices)
        )
        self._cache = {} if cache is None else cache

    @property
    def weights(self) -> list[dict[str, float]]:
        """Return a detached domain-weight mapping for each selected candidate."""
        values = []
        for dataset in self:
            sampling = dataset._resource.sampling
            weights = dict(sampling.weights)
            if sampling.domains.field == queries.FIELD_SOURCE_CORPUS_ID:
                weights = {_encode_id(_decode_id(key)): weight for key, weight in weights.items()}
            values.append(weights)
        return values

    @property
    def _configs(self) -> list[datasets.CreateDatasetRequest]:
        """Return detached dataset recipes for the selected candidates."""
        return [dataset._recipe for dataset in self]

    @overload
    def profile(self, index: None = None) -> list[datasets.DatasetProfile]: ...
    @overload
    def profile(self, index: int) -> datasets.DatasetProfile: ...
    @report_progress("Profiling mixture {id}")
    def profile(
        self, index: int | None = None
    ) -> list[datasets.DatasetProfile] | datasets.DatasetProfile:
        """Return candidate profiles without packing, displayed in at most 15 lines."""
        return (
            _MixProfiles(dataset.profile() for dataset in self)
            if index is None
            else self[index].profile()
        )

    def __len__(self) -> int:
        return len(self._indices)

    def _index(self, index: int) -> int:
        if type(index) is not int:
            raise TypeError("candidate index must be an integer")
        return self._indices[index]

    @overload
    def __getitem__(self, index: int) -> Dataset: ...
    @overload
    def __getitem__(self, index: slice) -> Mix: ...
    def __getitem__(self, index: int | slice) -> Dataset | Mix:
        if isinstance(index, slice):
            return Mix(
                self._db,
                self._resource,
                self._creation_request,
                indices=self._indices[index],
                cache=self._cache,
            )
        original = self._index(index)
        if original not in self._cache:
            self._cache[original] = self._db._dataset(self._resource.dataset_ids[original])
        return self._cache[original]

    def __iter__(self) -> Iterator[Dataset]:
        for index in range(len(self)):
            yield self[index]


Datasets = Mix


@dataclass(frozen=True)
class Sequence:
    _value: datasets.Sequence
    _reader: RangeReader

    def __repr__(self) -> str:
        return f"Sequence(ordinal={self.ordinal}, documents={self.document_ids()!r})"

    @property
    def ordinal(self) -> int:
        """Return the zero-based position of this sequence in its dataset."""
        return self._value.ordinal

    @cached_property
    def _tokens(self) -> tuple[int, ...]:
        data = self._reader.read(self._value.tokens)
        if self._value.tokens.start % 4 or len(data) % 4:
            raise ValueError("token range is not uint32 aligned")
        return tuple(int(value) for value in struct.unpack(f"<{len(data) // 4}I", data))

    @property
    def tokens(self) -> list[int]:
        """Read the token IDs for this packed sequence."""
        return list(self._tokens)

    @cached_property
    def _mask(self) -> tuple[bool, ...]:
        data = self._reader.read(self._value.loss_mask)
        if any(value > 1 for value in data):
            raise ValueError("invalid stored token mask")
        return tuple(bool(value) for value in data)

    @property
    def mask(self) -> list[bool]:
        """Mark content and separator tokens as True and padding tokens as False."""
        return list(self._mask)

    @cached_property
    def attention_mask(self) -> list[bool]:
        """Return the sequence mask used to exclude padding from attention."""
        data = self._reader.read(self._value.attention_mask)
        if any(value > 1 for value in data):
            raise ValueError("invalid stored attention mask")
        return [bool(value) for value in data]

    @property
    def spans(self) -> list[datasets.TokenRegion]:
        """Return source, separator, and padding regions within this sequence."""
        return [copy_message(region) for region in self._value.regions]

    def document_ids(self) -> list[str]:
        """Unique source IDs in first-content order, excluding separators and padding."""
        return list(
            dict.fromkeys(
                _encode_id(region.document_id)
                for region in self._value.regions
                if region.kind == datasets.TokenRegion.KIND_CONTENT
            )
        )


class Dataset(_Execution[datasets.Dataset, datasets.CreateDatasetRequest]):
    @report_progress("Previewing dataset {id}")
    def preview(
        self, *, limit: int = 3, offset: int = 0, max_characters: int = 1024
    ) -> list[PreviewSequence]:
        """Materialize if needed and show up to three packed sequences by default.

        Each example includes decoded text, the first 256 token IDs and masks,
        source document IDs, and a truncation flag. Offset is a sequence ordinal.
        """
        from ._dataset_preview import preview

        return preview(self, limit=limit, offset=offset, max_characters=max_characters)

    @property
    def _tokenizer_definition(self) -> str:
        """Return the versioned identity of the tokenizer used for packing."""
        return self._resource.tokenizer.definition_digest.hex()

    @property
    def _recipe(self) -> datasets.CreateDatasetRequest:
        """Return a detached request that reproduces this dataset."""
        from ._wire import copy_fields

        return copy_fields(self._resource, datasets.CreateDatasetRequest())

    @report_progress("Profiling dataset {id}")
    def profile(self) -> datasets.DatasetProfile:
        """Compute the planned profile if needed, without packing candidate tokens."""
        if not self._resource.HasField("profile"):
            if self._db._read_only:
                raise ExecutionError(
                    "dataset profile is not computed; use a writable session first"
                )
            from .execution import Coordinator

            assert isinstance(self._db._executor, Coordinator)
            self._resource.profile.CopyFrom(
                self._db._executor._planned_dataset_profile(self._resource)
            )
        return _public_dataset_profile(
            copy_message(self._resource.profile),
            corpus_strata=self._resource.sampling.domains.field == queries.FIELD_SOURCE_CORPUS_ID,
        )

    @overload
    def torch(
        self,
        *,
        streaming: Literal[False] = False,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> TorchDataset: ...
    @overload
    def torch(
        self,
        *,
        streaming: Literal[True],
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> StreamingDataset: ...
    @overload
    def torch(
        self,
        *,
        streaming: bool,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> TorchDataset | StreamingDataset: ...
    def torch(
        self,
        *,
        streaming: bool = False,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> TorchDataset | StreamingDataset:
        """Return batched or streaming PyTorch data with input IDs, masks and labels.

        Materializes this candidate once. DataLoader workers then read verified
        ranges directly; no catalog or execution handle crosses processes.
        """
        if type(streaming) is not bool:
            raise TypeError("streaming must be boolean")
        if not streaming and (
            seed != 0 or epoch != 0 or rank is not None or world_size is not None
        ):
            raise ValueError(
                "seed, epoch, rank and world_size require streaming=True; use a sampler for map-style data"
            )
        from . import _torch

        if streaming:
            rank, world_size = _torch.streaming_topology(
                seed=seed, epoch=epoch, rank=rank, world_size=world_size
            )
        self.wait()
        if streaming:
            return _torch.StreamingDataset(
                self._proto,
                self._db._object_reader,
                seed=seed,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
            )
        return _torch.TorchDataset(self._proto, self._db._object_reader)

    def __len__(self) -> int:
        return self.wait()._resource.profile.sequences

    def __iter__(self) -> Reader[Sequence]:
        return self.reader()

    def _page(self, ordinal: int, size: int = 128) -> list[Sequence]:
        self.wait()
        return _read_page(self._resource, self._db._object_reader, ordinal, size)

    def __getitem__(self, index: int) -> Sequence:
        if type(index) is not int:
            raise TypeError("sequence index must be an integer")
        count = len(self)
        index = index + count if index < 0 else index
        if not 0 <= index < count:
            raise IndexError("sequence index out of range")
        return self._page(index, 1)[0]

    def reader(
        self,
        *,
        topology: Topology | None = None,
        checkpoint: Checkpoint | None = None,
        seed: int | None = None,
    ) -> Reader[Sequence]:
        """Iterate packed sequences with optional shuffling, partitions and resume.

        Save reader.checkpoint() after processing a sequence, then pass it back
        with the same seed and topology to resume from the next sequence.
        """
        return Reader(self, Topology() if topology is None else topology, checkpoint, seed)


def _read_page(
    resource: datasets.Dataset, reader: RangeReader, ordinal: int, size: int = 128
) -> list[Sequence]:
    page = ordinal // 128
    sequences = _sequence_page(resource, reader.read(resource.sequences[page]), page)
    return [Sequence(seq, reader) for seq in sequences if ordinal <= seq.ordinal < ordinal + size]


def _sequence_page(
    resource: datasets.Dataset, data: bytes, page: int
) -> tuple[datasets.Sequence, ...]:
    sequences = parse(datasets.SequenceBatch, data).sequences
    expected = min(128, resource.profile.sequences - page * 128)
    if len(sequences) != expected or [seq.ordinal for seq in sequences] != list(
        range(page * 128, page * 128 + expected)
    ):
        raise ExecutionError("stored sequence index is incomplete or out of order")
    return tuple(sequences)
