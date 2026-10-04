"""Validated JSON boundaries and shared scalar values."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import UnionType
from typing import (
    Callable,
    Literal,
    NotRequired,
    Required,
    TypeAliasType,
    TypedDict,
    TypeGuard,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

type Scalar = str | int | float | bool | None
type JSON = Scalar | list[JSON] | dict[str, JSON]
type FieldValue = Scalar | list[Scalar]
type Metadata = Scalar | Sequence[Metadata] | Mapping[str, Metadata]


def is_json(value: object) -> TypeGuard[JSON]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(is_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json(item) for key, item in value.items())
    return False


def load_json(value: str | bytes) -> JSON:
    """The default decoder produces only JSON scalars, arrays and string-keyed objects."""
    return cast(JSON, json.loads(value))


def json_object(value: JSON) -> dict[str, JSON]:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def json_list(value: JSON) -> list[JSON]:
    if not isinstance(value, list):
        raise ValueError("expected a JSON array")
    return value


def json_string(value: JSON) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a JSON string")
    return value


def json_integer(value: JSON) -> int:
    if type(value) is not int:
        raise ValueError("expected a JSON integer")
    return value


def json_integers(value: JSON) -> list[int]:
    return [json_integer(item) for item in json_list(value)]


def field_value(value: object) -> FieldValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    ):
        return [scalar(item) for item in value]
    raise ValueError("expected a scalar or scalar vector")


def scalar(value: object) -> Scalar:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("expected a scalar")


type FieldValues = Mapping[str, Mapping[str, FieldValue]]
type Orders = Sequence[tuple[str, bool]]
type Interval = tuple[int, int]
type EvidenceRange = tuple[str, int, int]
type Edge = tuple[str, str]


def checked_record[T](value: object, schema: type[T]) -> T:
    """Validate a record at an untyped JSON/Arrow boundary, including nested fields."""
    if not _validator(schema)(value):
        raise ValueError(f"invalid {schema.__name__} record")
    return cast(T, value)


@cache
def _validator(annotation: object) -> Callable[[object], bool]:
    """Compile static record shapes once; only values vary between rows."""
    if annotation is float:
        return lambda value: type(value) in (float, int)
    if annotation in (int, bool, str, bytes, type(None)):
        return lambda value: type(value) is annotation
    if isinstance(annotation, TypeAliasType):
        return lambda value: _validator(annotation.__value__)(value)
    if is_typeddict(annotation):
        schema = cast(type[object], annotation)
        hints = cast(dict[str, object], get_type_hints(schema, include_extras=True))
        required = set(cast(frozenset[str], getattr(schema, "__required_keys__")))
        for key, hint in hints.items():
            if get_origin(hint) is NotRequired:
                required.discard(key)
            elif get_origin(hint) is Required:
                required.add(key)

        def record(value: object) -> bool:
            return (
                isinstance(value, dict)
                and required <= value.keys()
                and value.keys() <= hints.keys()
                and all(_validator(hints[key])(item) for key, item in value.items())
            )

        return record
    origin = cast(object, get_origin(annotation))
    args = cast(tuple[object, ...], get_args(annotation))
    if origin in (Required, NotRequired):
        return lambda value: _validator(args[0])(value)
    if origin in (Union, UnionType):
        return lambda value: any(_validator(member)(value) for member in args)
    if origin is Literal:
        return lambda value: any(type(value) is type(member) and value == member for member in args)
    if origin is list:
        member = args[0]
        if member is float:
            return lambda value: (
                isinstance(value, list) and all(type(item) in (float, int) for item in value)
            )
        if member in (int, bool, str, bytes, type(None)):
            return lambda value: (
                isinstance(value, list) and all(type(item) is member for item in value)
            )
        element = _validator(member)
        return lambda value: isinstance(value, list) and all(element(item) for item in value)
    if origin is dict:
        key_check, item_check = _validator(args[0]), _validator(args[1])
        return lambda value: (
            isinstance(value, dict)
            and all(key_check(key) and item_check(item) for key, item in value.items())
        )
    if origin is tuple:

        def fields(value: object) -> bool:
            if not isinstance(value, tuple):
                return False
            if len(args) == 2 and args[1] is Ellipsis:
                return all(_validator(args[0])(item) for item in value)
            return len(value) == len(args) and all(
                _validator(member)(item) for item, member in zip(value, args, strict=True)
            )

        return fields
    if isinstance(annotation, type):
        return lambda value: isinstance(value, annotation)
    raise TypeError(f"unsupported record annotation: {annotation!r}")


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
