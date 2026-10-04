"""Independent engine harness for differential and engine behavior tests."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from os import PathLike, fspath
from pathlib import Path
from typing import Iterable, Iterator, Literal, cast

from premixdb.contracts import Checkpoint
from premixdb.engine import execution
from premixdb.engine.contracts import (
    Changes,
    Counts,
    Occurrence,
    PackingSummary,
    Provenance,
    QuerySummary,
)
from premixdb.engine.execution import Reader, Row, Sequence, Source
from premixdb.engine.names import corpus_id
from premixdb.engine.policies import ByteTokenizer as ByteTokenizer
from premixdb.engine.policies import Concat as Concat
from premixdb.engine.sources import source_files
from premixdb.fields.expressions import FieldPredicate
from premixdb.runtime import environment as _runtime
from premixdb.schemas import requests as _requests
from premixdb.schemas.enums import DedupeAlgorithm, RemovalUnit

# The legacy direct API adapts the same expression/policy vocabulary as the SDK.
from premixdb.schemas.requests import SourceGroup as SourceGroup
from premixdb.schemas.requests import object as object
from premixdb.schemas.requests import text as text
from premixdb.training.reader import Topology as Topology
from premixdb.v1 import query_pb2 as q


@dataclass(frozen=True, init=False)
class HuggingFaceTokenizer:
    """Capture a local tokenizer.json verified against an expected BLAKE3.

    The asset and engine define identity; the path does not. Encoding inserts
    no special tokens. Truncation, padding, and stochastic dropout are rejected.
    The byte limit bounds each whole-document encoding, not total dataset RAM.
    """

    _handle: execution.HuggingFaceTokenizer

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        digest: str,
        max_document_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        from premixdb.engine import execution

        if "://" in fspath(path):
            raise NotImplementedError("only local filesystem paths are supported")
        if type(max_document_bytes) is not int or max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be a positive integer")
        builtins.object.__setattr__(
            self,
            "_handle",
            execution.HuggingFaceTokenizer(Path(path), digest, max_document_bytes),
        )

    @property
    def definition(self) -> str:
        """Versioned identity of the captured asset and encoding engine."""
        return self._handle.definition

    @property
    def asset_digest(self) -> str:
        """Return the captured tokenizer asset BLAKE3 digest in hexadecimal."""
        return self._handle.asset_digest

    def encode(self, text: str) -> list[int]:
        """Encode text without inserting implicit special tokens."""
        return self._handle.encode(text)

    def token_to_id(self, token: str) -> int | None:
        """Look up an explicit separator or padding token; absent tokens return None."""
        return self._handle.token_to_id(token)


def where(
    predicate: FieldPredicate | _requests._DocumentPredicate,
) -> execution.Step:
    """Describe an intrinsic-field comparison for a direct engine query."""
    return _steps([_requests.where(predicate)])[0]


def dedupe(
    *,
    comparison: Literal["document", "line"] = "document",
    removal: Literal["document"] | SourceGroup = "document",
    order_by: Iterable[q.OrderBy] = (),
    algorithm: DedupeAlgorithm | None = None,
) -> execution.Step:
    """Describe duplicate removal with explicit winner ordering."""
    if comparison not in ("document", "line"):
        raise ValueError("comparison must be 'document' or 'line'")
    if removal != "document" and not isinstance(removal, SourceGroup):
        raise TypeError("removal must be 'document' or SourceGroup")
    operation = _requests.dedupe(
        algorithm=algorithm
        if algorithm is not None
        else DedupeAlgorithm.EXACT_DOCUMENT
        if comparison == "document"
        else DedupeAlgorithm.EXACT_LINE,
        removal=RemovalUnit.DOCUMENT if removal == "document" else removal,
        order_by=order_by,
    )
    return _steps([operation])[0]


def _steps(steps: Iterable[execution.Step | q.Operation]) -> list[execution.Step]:
    from premixdb.runtime.planner import execution_steps
    from premixdb.v1.query_pb2 import CreateQueryRequest, Operation

    result = []
    for step in steps:
        result.extend(
            execution_steps(CreateQueryRequest(operations=[step]))
            if isinstance(step, Operation)
            else [step]
        )
    return result


def _local_path(value: str | PathLike[str]) -> Path:
    if "://" in fspath(value):
        raise NotImplementedError("only local filesystem paths are supported")
    return Path(value)


class _SnapshotOperations:
    def union(self, *others: Snapshot) -> SnapshotUnion:
        return SnapshotUnion((cast("Snapshot", self), *others))

    def query(self, *, steps: Iterable[execution.Step | q.Operation] = ()) -> Query:
        return self.union().query(steps=steps)


@dataclass(frozen=True)
class SnapshotUnion:
    _snapshots: tuple[Snapshot, ...]

    def union(self, *others: Snapshot) -> SnapshotUnion:
        return SnapshotUnion((*self._snapshots, *others))

    def query(self, *, steps: Iterable[execution.Step | q.Operation] = ()) -> Query:
        return self._snapshots[0]._query_union(self._snapshots, steps=steps)


class PremixDB:
    """Open a local snapshot store. PremixDB manages execution identity automatically."""

    def __init__(self, *, storage: str | PathLike[str]) -> None:
        self._store = execution.Store(_local_path(storage))

    def corpus(self, name: str) -> Corpus:
        """Names map to stable UUIDv5 IDs, independent of the storage path."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("corpus name must be a nonempty string")
        return Corpus(self, name)

    def snapshot(self, id: str) -> Snapshot:
        """Verify and load a complete snapshot by its hexadecimal ID."""
        return Snapshot(self._store.load(id))


@dataclass(frozen=True)
class Corpus:
    _db: PremixDB
    name: str

    @property
    def id(self) -> str:
        """Return the resource identity as a hexadecimal string."""
        return corpus_id(self.name).hex()

    def snapshot(
        self,
        *,
        source: str | PathLike[str] | Iterable[Source],
        base: Snapshot | None = None,
    ) -> Snapshot:
        """Capture and persist the full desired inventory, optionally reusing a base.

        A file is one document keyed by its filename. A directory recursively
        captures its files using relative POSIX paths as keys. An iterable of
        Source objects supplies explicit keys and already captured text.
        If the inventory matches the base, return that snapshot unchanged,
        including its original capture statistics. Edits create a new snapshot.
        """
        if base is not None and base.corpus_id != self.id:
            raise ValueError("base belongs to another corpus")
        sources: Iterable[Source]
        if isinstance(source, (str, PathLike)):
            sources = (Source.read(key, file) for key, file in source_files(_local_path(source)))
        else:
            sources = source
        handle = self._db._store.capture(
            self.id,
            sources,
            _runtime.current_code(),
            None if base is None else base._handle,
        )
        if base is not None and handle.id == base.id:
            return base
        return Snapshot(handle)


@dataclass(frozen=True)
class Snapshot(_SnapshotOperations):
    """Immutable captured state, independent of clients and later executions."""

    _handle: execution.Snapshot

    @property
    def id(self) -> str:
        """Return the resource identity as a hexadecimal string."""
        return self._handle.id

    @property
    def corpus_id(self) -> str:
        """Return the parent corpus identity as a hexadecimal string."""
        return self._handle.corpus_id

    def summary(self) -> Counts:
        """Return selection or packing counts for this completed resource."""
        return self._handle.summary()

    def changes(self) -> Changes:
        """Count added, changed, removed, unchanged, and reused documents."""
        return self._handle.changes()

    def _query_union(
        self, snapshots: Iterable[Snapshot], *, steps: Iterable[execution.Step | q.Operation] = ()
    ) -> Query:
        return Query(
            execution.execute(
                [snapshot._handle for snapshot in snapshots], _steps(steps), _runtime.current_code()
            )
        )


@dataclass(frozen=True)
class Query:
    """An immutable execution; dataset builds inherit its frozen engine identity."""

    _handle: execution.Query

    @property
    def id(self) -> str:
        """Return the resource identity as a hexadecimal string."""
        return self._handle.id

    @property
    def elapsed_seconds(self) -> float:
        """Return the execution time in seconds."""
        return self._handle.elapsed_seconds

    def rows(self) -> list[Row]:
        """Return selected documents in their stored ordinal order."""
        return self._handle.rows()

    def summary(self) -> QuerySummary:
        """Return selection or packing counts for this completed resource."""
        return self._handle.summary()

    def provenance(self) -> dict[str, Provenance]:
        """Selection outcomes and logical lineage for every input document."""
        return self._handle.provenance()

    def dataset(
        self,
        *,
        tokenizer: ByteTokenizer | HuggingFaceTokenizer,
        sequence_length: int,
        packing: Concat = Concat(),
    ) -> Dataset:
        """Pack selected documents using the given tokenizer and sequence policy."""
        if not isinstance(tokenizer, (ByteTokenizer, HuggingFaceTokenizer)):
            raise NotImplementedError("tokenizer must be ByteTokenizer or HuggingFaceTokenizer")
        if not isinstance(packing, Concat):
            raise TypeError("packing must be a Concat policy")
        spec = _requests.dataset(
            self.id,
            tokenizer=ByteTokenizer()._to_proto(),
            sequence_length=sequence_length,
            packing=packing,
        )
        p = spec.packing.concat
        return Dataset(
            self._handle.dataset(
                spec.sequence_length,
                p.separator_token_id if p.HasField("separator_token_id") else None,
                p.pad_token_id if p.HasField("pad_token_id") else None,
                tokenizer._handle if isinstance(tokenizer, HuggingFaceTokenizer) else None,
            )
        )


@dataclass(frozen=True)
class Dataset:
    """An in-memory dataset. Reading copies only the requested token/mask lists."""

    _handle: execution.Dataset

    @property
    def id(self) -> str:
        """Return the resource identity as a hexadecimal string."""
        return self._handle.id

    @property
    def elapsed_seconds(self) -> float:
        """Return the execution time in seconds."""
        return self._handle.elapsed_seconds

    @property
    def tokenizer_definition(self) -> str:
        """Identity of the tokenizer used to build this dataset."""
        return self._handle.tokenizer_definition

    def summary(self) -> PackingSummary:
        """Return selection or packing counts for this completed resource."""
        return self._handle.summary()

    def occurrences(self) -> list[Occurrence]:
        """List sampled document occurrences and their token ranges."""
        return self._handle.occurrences()

    def __len__(self) -> int:
        return len(self._handle)

    def __getitem__(self, index: int) -> Sequence:
        return self._handle[index]

    def __iter__(self) -> Iterator[Sequence]:
        return self.reader()

    def reader(
        self,
        *,
        topology: Topology = Topology(),
        checkpoint: Checkpoint | None = None,
        seed: int | None = None,
    ) -> Reader[Sequence]:
        """Resume from a JSON-compatible checkpoint bound to dataset and topology.

        Save reader.checkpoint() alongside training state after processing the
        returned sequences. An exhausted checkpoint remains exhausted.
        """
        return self._handle.reader(topology, checkpoint, seed)
