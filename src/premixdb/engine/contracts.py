"""Concrete records passed between selection, packing and persistence."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from .._types import Changes as Changes
from .._types import SnapshotSummary as Counts
from .._typing import Interval


class EncodedTokens(TypedDict):
    byte_tokens: NotRequired[str]
    byte_ranges: NotRequired[list[list[int]]]
    tokens: NotRequired[list[int]]
    ranges: NotRequired[list[list[list[int]]]]


class TextProfile(TypedDict):
    content_bytes: int
    characters: int
    newlines: int


class Frame(TypedDict):
    digest: list[int]
    bytes: int
    profile: NotRequired[TextProfile]


class DocumentRecord(TypedDict):
    key: str
    content: list[int]
    bytes: int
    frames: list[Frame]


class StoredCode(TypedDict):
    repository: str
    commit: str
    environment: list[int]


class Manifest(TypedDict):
    version: int
    id: list[int]
    corpus: list[int]
    code: StoredCode
    base: list[int] | None
    inventory: list[int]
    membership: list[int]
    summary: Counts
    changes: Changes
    documents: NotRequired[list[DocumentRecord]]
    pages: NotRequired[list[Frame]]


class SnapshotDescription(TypedDict):
    corpus_id: str
    parent_snapshot_id: str
    changes: Changes


class ObjectSpan(TextProfile):
    start: int
    end: int
    content_digest: str


class ObjectRecord(TextProfile):
    spans: int
    content_digest: str
    span_refs: list[ObjectSpan]


class Range(TypedDict):
    document: str
    start: int
    end: int


class Contamination(TypedDict):
    start: int
    end: int
    reference: str
    reference_start: int
    reference_end: int


class Selection(TypedDict):
    kind: Literal[
        "retained", "filtered", "duplicate", "duplicate_unit", "contaminated", "not_sampled"
    ]
    ordinal: NotRequired[int]
    step: NotRequired[int]
    kept: NotRequired[str | Range]
    matched: NotRequired[Range]
    references: NotRequired[list[Contamination]]


class Provenance(TypedDict):
    corpus_id: str
    source_key: str
    content: str
    snapshots: list[str]
    selection: Selection
    contamination: NotRequired[list[Contamination]]
    retained_ranges: NotRequired[list[Interval]]
    occurrences: NotRequired[list[int]]


class SamplingStatistics(TypedDict):
    unit: str
    requested: int
    realized: int
    overshoot: int
    unique_documents: int
    document_occurrences: int
    requested_domains: NotRequired[dict[str, int]]
    realized_domains: NotRequired[dict[str, int]]


class DecontaminationStatistics(TypedDict):
    matched_documents: int
    removed_documents: int
    removed_spans: int
    removed_bytes: int
    reference_snapshot_ids: list[bytes]


class StepSummary(TypedDict):
    before: Counts
    after: Counts


class QuerySummary(TypedDict):
    input: Counts
    output: Counts
    steps: list[StepSummary]
    decontamination: NotRequired[DecontaminationStatistics]
    sampling: NotRequired[SamplingStatistics]


class PackingProfile(TypedDict):
    source_documents: int
    source_content_bytes: int
    source_characters: int
    document_occurrences: int
    planned_content_tokens: int
    content_tokens: int
    separator_tokens: int
    dropped_tokens: int
    padding_tokens: int
    sequences: int
    output_tokens: int
    stratum_tokens: dict[str, int]


class PackingGeometry(TypedDict):
    source_tokens: dict[str, int]
    documents_per_sequence: dict[int, int]
    boundary_crossing_sequences: int


class PackingSummary(TypedDict):
    input: Counts
    content_tokens: int
    separator_tokens: int
    dropped_content_tokens: int
    dropped_separator_tokens: int
    padding_tokens: int
    sequences: int
    output_tokens: int


class Span(TypedDict):
    start: int
    end: int
    kind: Literal["content", "separator", "padding"]
    occurrence: NotRequired[int]
    offset: NotRequired[int]


class TokenRange(TypedDict):
    token: int
    occurrence: int
    start: int
    end: int


class SourceRange(TokenRange):
    token_end: int


class Occurrence(TypedDict):
    ordinal: int
    document: str
    source: Provenance
    tokens: int
