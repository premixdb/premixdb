"""Shared immutable tokenizer and packing values, with protobuf adapters."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from os import PathLike, fspath
from pathlib import Path
from typing import TYPE_CHECKING

from .v1.dataset_pb2 import Packing, Tokenizer

if TYPE_CHECKING:
    from .engine import execution


@dataclass(frozen=True)
class ByteTokenizer:
    """Unchanged UTF-8 bytes map to IDs 0–255, with no implicit special tokens."""

    def _to_proto(self) -> Tokenizer:
        from .v1.dataset_pb2 import ByteTokenizer as BytePolicy
        from .v1.dataset_pb2 import Tokenizer

        return Tokenizer(byte=BytePolicy())


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
        from .engine import execution

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


@dataclass(frozen=True)
class Concat:
    """Append a separator after every document, then drop or pad the final tail."""

    separator: int | None = None
    drop_remainder: bool = True
    pad_token: int | None = None

    def __post_init__(self) -> None:
        if type(self.drop_remainder) is not bool:
            raise TypeError("drop_remainder must be a bool")
        if self.drop_remainder == (self.pad_token is not None):
            raise ValueError("pad_token is required exactly when drop_remainder=False")
        from .engine.dataset_plan import PackingPlan

        PackingPlan(1, self.separator, self.pad_token)

    def _to_proto(self) -> Packing:
        from .v1.dataset_pb2 import Concat as ConcatPolicy
        from .v1.dataset_pb2 import Packing

        policy = ConcatPolicy(drop_remainder=self.drop_remainder)
        if self.separator is not None:
            policy.separator_token_id = self.separator
        if self.pad_token is not None:
            policy.pad_token_id = self.pad_token
        return Packing(concat=policy)


@dataclass(frozen=True)
class DecontaminateDefault:
    """Use 13-word overlap and whole-document removal when references are supplied.

    With no reference snapshots, there is no contamination evidence to remove.
    """

    def _to_proto(self) -> None:
        return None


@dataclass(frozen=True)
class SamplerDefault:
    """Keep every selected document once, in its existing deterministic order."""

    def _to_proto(self) -> None:
        return None
