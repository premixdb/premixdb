"""Concrete fixture contracts shared by the offline test suite."""

from __future__ import annotations

from collections.abc import Callable
from typing import NotRequired, Protocol, TypedDict

from _reference import Snapshot
from blake3 import blake3

from premixdb import PremixDB, Source
from premixdb.contracts import FieldValue
from premixdb.runtime import Coordinator
from premixdb.v1.data_mixture_pb2 import Packing, Tokenizer

SHELL_SOURCES = (
    Source("speech/0000", "First Citizen:\nLet us speak together.\n"),
    Source("speech/0001", "Second Citizen:\nWe are listening.\n"),
)


class SnapshotOptions(TypedDict):
    base: NotRequired[Snapshot | None]


class PackingOptions(TypedDict):
    separator: NotRequired[int | None]
    pad_token: NotRequired[int | None]
    drop_remainder: NotRequired[bool]


class TokenizerOptions(TypedDict):
    max_document_bytes: NotRequired[int]


class PreviewOptions(TypedDict, total=False):
    limit: int
    offset: int
    max_characters: int


class EncodeOptions(TypedDict, total=False):
    batch_size: int
    normalize_embeddings: bool
    show_progress_bar: bool
    convert_to_numpy: bool
    convert_to_tensor: bool
    prompt: str | None
    prompt_name: str | None


class Benchmark(Protocol):
    def __call__[**P, R](
        self, function: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R: ...


class Vectors(Protocol):
    def tolist(self) -> list[list[float]]: ...


def coordinator(db: PremixDB) -> Coordinator:
    assert isinstance(db._executor, Coordinator)
    return db._executor


def wordpiece_tokenizer() -> Tokenizer:
    """Use a small real model when a test needs decoding but not GPT-2 defaults."""
    from pathlib import Path

    from premixdb import hugging_face_tokenizer

    path = Path(__file__).parent / "fixtures/wordpiece.json"
    return hugging_face_tokenizer(path, digest=blake3(path.read_bytes()).digest())


def tokenizer_packing(tokenizer: Tokenizer) -> Packing:
    """Explicit special tokens for the small WordPiece and byte test policies."""
    from premixdb import Concat

    model = tokenizer.HasField("hugging_face")
    return Concat(
        separator=2 if model else 256, drop_remainder=False, pad_token=3 if model else 257
    )


def numeric(value: FieldValue) -> float:
    assert isinstance(value, (int, float))
    return float(value)


def invalid_call(function: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Exercise runtime rejection of input that intentionally violates the API type."""
    return function(*args, **kwargs)
