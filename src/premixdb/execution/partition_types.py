"""Validated wire records exchanged by partition workers."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from ..engine.contracts import Span, TokenRange


class InputRow(TypedDict):
    id: str
    text: str
    ordinal: int
    ranges: list[list[int]]


class FeatureInput(TypedDict):
    id: str
    text: str
    url: str | None


class FeatureRequest(TypedDict):
    version: int
    producer: str
    definition: str
    rows: list[FeatureInput]


class TokenRequest(TypedDict):
    version: int
    tokenizer: str
    rows: list[InputRow]


class EvidenceRequest(TypedDict):
    version: int
    algorithm: int
    n: NotRequired[int]
    rows: list[InputRow]


class PackingRequest(TypedDict):
    version: int
    first: int
    length: int
    sequences: int
    total_sequences: int
    separator: int | None
    padding: int | None
    prefixes: list[int]


class EncodedTokens(TypedDict):
    byte_tokens: NotRequired[str]
    byte_ranges: NotRequired[list[list[int]]]
    tokens: NotRequired[list[int]]
    ranges: NotRequired[list[list[list[int]]]]


class TokenRow(EncodedTokens):
    id: str
    ordinal: int


class PackedRow(TypedDict):
    ordinal: int
    tokens: list[int]
    spans: list[Span]
    alignment: list[TokenRange]


class EvidenceRow(TypedDict):
    id: str
    start: int
    end: int
    value: str


class IndexRow(TypedDict):
    id: str
    exact_hash: str
    minhash: list[int] | None
    lsh_buckets: list[int] | None
