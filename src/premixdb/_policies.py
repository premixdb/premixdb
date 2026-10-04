"""Shared immutable tokenizer and packing values, with protobuf adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .v1.dataset_pb2 import Packing, Tokenizer

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ByteTokenizer:
    """Unchanged UTF-8 bytes map to IDs 0–255, with no implicit special tokens."""

    def _to_proto(self) -> Tokenizer:
        from .v1.dataset_pb2 import ByteTokenizer as BytePolicy
        from .v1.dataset_pb2 import Tokenizer

        return Tokenizer(byte=BytePolicy())


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


@dataclass(frozen=True)
class SamplerDefault:
    """Keep every selected document once, in its existing deterministic order."""
