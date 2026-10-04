"""Public errors, typed summaries, and reader checkpoints."""

from dataclasses import dataclass
from typing import NotRequired, TypedDict


class ExecutionError(RuntimeError):
    """A recipe failed to execute or its published result is incomplete."""


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


class Changes(TypedDict):
    added: int
    changed: int
    removed: int
    unchanged: int
    reused: int


class Checkpoint(TypedDict):
    shuffle_seed: NotRequired[int]
    version: int
    dataset: str
    topology: list[int]
    next_ordinal: int | None
