"""JSON-compatible typed summaries and reader checkpoints."""

from dataclasses import dataclass
from typing import NotRequired, TypedDict


@dataclass(frozen=True)
class ExecutionRecord:
    """Execution history with base64url identities and whole-second UTC times."""

    id: str
    operation: str
    resource_id: str
    status: str
    started_at: str | None
    ended_at: str | None
    request_digest: str
    error: str = ""
    cache_hit: bool = False


class CorpusListing(TypedDict):
    id: str
    name: str


class SnapshotListing(TypedDict):
    id: str
    timestamp: str | None


class DocumentListing(TypedDict):
    id: str
    source_key: str
    corpus_id: str
    ordinal: int


class PreviewDocument(TypedDict):
    id: str
    text: str
    truncated: bool
    source_key: str
    corpus_id: str
    ordinal: int


class PreviewSequence(TypedDict):
    ordinal: int
    text: str
    tokens: list[int]
    mask: list[bool]
    attention_mask: list[bool]
    document_ids: list[str]
    truncated: bool


class SnapshotSummary(TypedDict):
    documents: int
    bytes: int
    characters: int


class Changes(TypedDict):
    added: int
    changed: int
    removed: int
    unchanged: int
    reused: int


class QueryPopulationSummary(TypedDict):
    documents: int
    bytes: int
    characters: int


class QueryStepSummary(TypedDict):
    before: QueryPopulationSummary
    after: QueryPopulationSummary


class QuerySummary(TypedDict):
    input: QueryPopulationSummary
    output: QueryPopulationSummary
    steps: list[QueryStepSummary]


class DatasetSummary(TypedDict):
    content_tokens: int
    separator_tokens: int
    padding_tokens: int
    dropped_tokens: int
    sequences: int
    output_tokens: int


class Checkpoint(TypedDict):
    shuffle_seed: NotRequired[int]
    version: int
    dataset: str
    topology: list[int]
    next_ordinal: int | None
