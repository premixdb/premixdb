"""Fluent recipes and resource handles for local data."""

from __future__ import annotations

import os
import time
from functools import cached_property
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
from ._ids import _decode_id, _encode_id, _public_dataset_profile
from ._inputs import SourceInput, source_proto
from ._lineage import decode_lineage
from ._mixing import Bounds, RegMixSampler, Tokens
from ._policies import ByteTokenizer as BytePolicy
from ._policies import Concat as ConcatPolicy
from ._profiles import ProfileSelector, _MixProfiles
from ._progress import report_progress
from ._protobuf import copy_message
from ._reader import Reader as Reader
from ._reader import Topology as Topology
from ._sequences import Sequence, read_page
from ._storage import RangeReader
from ._types import (
    Checkpoint,
    DocumentListing,
    ExecutionError,
    ExecutionRecord,
    PreviewDocument,
    PreviewSequence,
)
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

type _ResourceKind = Literal["Corpus", "Snapshot", "Query", "Dataset", "Mix"]
type _ResourceValue = (
    corpora.Corpus | snapshots.Snapshot | queries.Query | datasets.Dataset | datasets.Mix
)

DomainInput = (
    type[Topic]
    | type[ContentType]
    | type[Language]
    | FieldProjection
    | LanguageFields
    | TopicFields
    | ContentTypeFields
    | Iterable[FieldProjection]
    | datasets.Domains
    | Mapping[str, str]
)


def _duration(value: float, name: str) -> float:
    message = f"{name} must be positive and finite"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        seconds = float(value)
    except OverflowError:
        raise ValueError(message) from None
    if not 0 < seconds < float("inf"):
        raise ValueError(message)
    return seconds


class PremixDB:
    """Open local storage for reusable capture, curation, and training recipes."""

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
            from .execution.catalog_reader import Catalog
            from .execution.storage import ObjectStore

            self._executor = Catalog(
                ObjectStore(
                    storage,
                    metadata_path=Path(metadata_path) if metadata_path is not None else None,
                    read_only=True,
                )
            )
        else:
            from .execution import Coordinator

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
    def _submit(self, request: datasets.CreateDatasetRequest) -> datasets.CreateDatasetResponse: ...
    @overload
    def _submit(self, request: datasets.CreateMixRequest) -> datasets.CreateMixResponse: ...
    @report_progress("Submitting operation")
    def _submit(
        self,
        request: corpora.CreateCorpusRequest
        | snapshots.CreateSnapshotRequest
        | queries.CreateQueryRequest
        | datasets.CreateDatasetRequest
        | datasets.CreateMixRequest,
    ) -> (
        corpora.CreateCorpusResponse
        | snapshots.CreateSnapshotResponse
        | queries.CreateQueryResponse
        | datasets.CreateDatasetResponse
        | datasets.CreateMixResponse
    ):
        self._require_writable()
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
    def _get(self, kind: Literal["Corpus"], id: bytes) -> corpora.Corpus: ...
    @overload
    def _get(self, kind: Literal["Snapshot"], id: bytes) -> snapshots.Snapshot: ...
    @overload
    def _get(self, kind: Literal["Query"], id: bytes) -> queries.Query: ...
    @overload
    def _get(self, kind: Literal["Dataset"], id: bytes) -> datasets.Dataset: ...
    @overload
    def _get(self, kind: Literal["Mix"], id: bytes) -> datasets.Mix: ...
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
            resource = self._executor.GetDataset(datasets.GetDatasetRequest(id=id)).dataset
        elif kind == "Mix":
            resource = self._executor.GetMix(datasets.GetMixRequest(id=id)).mix
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
                resource = self._get("Corpus", _corpus_id(name))
            except KeyError:
                resource = None
            if resource is None or not resource.latest_snapshot_id:
                raise ValueError(
                    f"corpus {name!r} has no snapshot; capture with db.corpus(name, source=...)"
                ) from None
            return self._snapshot(resource.latest_snapshot_id)
        handle = self._create_corpus(name)
        return handle.snapshot(source=source, limit=limit, base=base)

    def _create_corpus(self, name: str, *, request_id: str = "") -> Corpus:
        """Create/reopen the mutable named handle for explicit snapshot management."""
        request = _requests.corpus(name, request_id=request_id)
        id = self._submit(request).id
        return Corpus(self, self._get("Corpus", id), request)

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

    def _mix(self, id: bytes | str) -> Mix:
        """Open a saved mixture collection without packing its candidates."""
        return Mix(self, self._get("Mix", _requests._id(id, 32)))


class _Resource[ResourceT: _ResourceValue, RequestT: Message]:
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
        budget = self._db._timeout if timeout is None else _duration(timeout, "timeout")
        deadline = time.monotonic() + budget
        kind: Literal["Snapshot", "Query", "Dataset"] = (
            "Snapshot"
            if isinstance(self, Snapshot)
            else "Query"
            if isinstance(self, Query)
            else "Dataset"
        )

        def remaining() -> float:
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError(f"timed out waiting for {kind.lower()} {self.id}")
            return seconds

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
            )
            if response.id != value.id:
                raise ExecutionError("materialization returned a different resource")
            remaining()
            value = cast(
                ResourceT,
                self._db._get(kind, response.id),
            )
        while value.status != status.STATUS_COMPLETED:
            if value.status == status.STATUS_ERROR:
                raise ExecutionError(
                    f"{type(self).__name__} {self.id} failed: {getattr(value, 'error', '')}"
                )
            if value.status not in (status.STATUS_PENDING, status.STATUS_RUNNING):
                raise ExecutionError("resource has an unspecified execution status")
            waited = False
            if isinstance(self, (Query, Dataset)):
                from .execution.coordinator import Coordinator

                assert isinstance(self._db._executor, Coordinator)
                waited = self._db._executor._wait_for_materialization(
                    "query" if isinstance(self, Query) else "dataset", value.id, remaining()
                )
                if not waited:
                    # A job can finish and leave the registry before this handle
                    # refreshes its running state. Check publication before sleeping.
                    remaining()
                    value = cast(ResourceT, self._db._get(kind, value.id))
                    if value.status not in (status.STATUS_PENDING, status.STATUS_RUNNING):
                        continue
            if not waited:
                time.sleep(min(self._db._poll_interval, remaining()))
            remaining()
            value = cast(
                ResourceT,
                self._db._get(kind, value.id),
            )
        self._resource = value
        return self

    def _preview(
        self, *, limit: int = 3, offset: int = 0, max_characters: int = 1024
    ) -> list[PreviewDocument]:
        """Browse selected documents in deterministic order, with bounded text."""
        limit, offset, max_characters = _requests._preview_options(
            limit, offset, max_characters, unit="documents"
        )
        if not limit:
            return []
        request = queries.PreviewRequest(limit=limit, offset=offset, max_characters=max_characters)
        ready = self.wait()._resource
        if isinstance(ready, snapshots.Snapshot):
            request.snapshot_id = ready.id
        else:
            request.query_id = ready.id
        response = self._db._executor.Preview(
            request,
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

    def _with_fields(self, fields: Iterable[ProfileSelector]) -> Query:
        self._db._require_open()
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

    def _provenance(self) -> dict[str, Provenance]:
        """Trace each selected document to its source and query decisions."""
        resource = self.wait()._resource
        ref = resource.lineage
        return decode_lineage(
            self._db._object_reader.read(
                source_types.SpanRef(
                    object=ref, end=ref.size_bytes, blake3_digest=ref.blake3_digest
                )
            ),
            public=True,
        )

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
        self._db._require_writable("plan datasets")
        request = _requests.dataset(
            self._resource.id,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            packing=packing,
        )
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
    ) -> Mix:
        """Register three lazy datasets by default, with reproducible sampling."""
        self._db._require_writable("plan mixtures")
        request = _requests.mix(
            self._resource.id,
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
        self.wait()
        id = self._db._submit(request).id
        response = self._db._executor.GetMix(
            datasets.GetMixRequest(id=id),
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


Datasets = Mix  # Public compatibility spelling; internal code uses Mix.


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
            self._db._require_open()
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

    def _page(self, ordinal: int, size: int | None = None) -> list[Sequence]:
        self.wait()
        return read_page(self._resource, self._db._object_reader, ordinal, size)

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
