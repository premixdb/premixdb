"""Shared resource identity, execution waiting, and document preview behavior."""

from __future__ import annotations

import time
from typing import (
    TYPE_CHECKING,
    Literal,
    Self,
    cast,
)

from google.protobuf.message import Message

from premixdb.api.progress import report_progress
from premixdb.contracts import (
    ExecutionError,
    PreviewDocument,
)
from premixdb.schemas import requests as _requests
from premixdb.schemas.enums import ExecutionStatus
from premixdb.schemas.ids import _encode_id
from premixdb.schemas.protobuf import copy_message
from premixdb.v1 import corpus_pb2 as corpora
from premixdb.v1 import data_mixture_pb2 as mix_pb
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import snapshot_pb2 as snapshots
from premixdb.v1 import status_pb2 as status

if TYPE_CHECKING:
    from premixdb.api.database import PremixDB


type _ResourceKind = Literal["Corpus", "Snapshot", "Query", "Dataset", "Mix"]


type _ResourceValue = (
    corpora.Corpus | snapshots.Snapshot | queries.Query | mix_pb.Dataset | mix_pb.Mix
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


class _Resource[ResourceT: _ResourceValue, RequestT: Message]:
    def __init__(
        self, client: PremixDB, resource: ResourceT, request: RequestT | None = None
    ) -> None:
        self._db, self._resource = client, copy_message(resource)
        self._creation_request = copy_message(request) if request is not None else None

    def __repr__(self) -> str:
        from premixdb.api import Corpus, DataMixture, Dataset, Query, Snapshot
        from premixdb.api.display import _resource_repr

        if not isinstance(self, (Corpus, Snapshot, Query, DataMixture, Dataset)):
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


class _Execution[
    ResourceT: snapshots.Snapshot | queries.Query | mix_pb.Dataset,
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
        from premixdb.api import Dataset, Query, Snapshot

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
                from premixdb.schemas.messages import copy_fields

                recipe = copy_fields(value, queries.CreateQueryRequest())
            else:
                recipe = self._recipe
            from premixdb.runtime.coordinator import Coordinator

            self._db._require_writable()
            assert isinstance(self._db._executor, Coordinator)
            admission = self._db._executor._submit_async(recipe)
            seconds = remaining()
            try:
                response = admission.result(timeout=seconds)
            except TimeoutError:
                remaining()  # Give expired waits the resource-specific error message.
                raise
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
                from premixdb.runtime.coordinator import Coordinator

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
        remaining()
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
        self._db._require_open()
        if isinstance(self._resource, snapshots.Snapshot):
            request.snapshot_id = self._resource.id
        else:
            request.query_id = self._resource.id
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
