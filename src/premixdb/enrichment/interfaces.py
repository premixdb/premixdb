"""Narrow typed boundaries for third-party inference interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, Protocol, Self, overload

if TYPE_CHECKING:
    from tokenizers import Encoding, Tokenizer
    from torch import Tensor, device


class Tokenization(Protocol):
    encodings: list[Encoding]


class ModelTokenizer(Protocol):
    pad_token_id: int | None
    pad_token: str | None
    eos_token: str | None
    backend_tokenizer: Tokenizer

    @overload
    def __call__(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> Tokenization: ...
    @overload
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: Literal["pt"],
    ) -> Mapping[str, Tensor]: ...
    def pad(
        self, windows: list[dict[str, list[int]]], *, padding: bool, return_tensors: Literal["pt"]
    ) -> Mapping[str, Tensor]: ...
    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...


class ModelConfig(Protocol):
    pad_token_id: int | None
    num_labels: int
    id2label: Mapping[int | str, str]


class LogitOutput(Protocol):
    logits: Tensor


class SequenceModel(Protocol):
    config: ModelConfig
    device: device | str

    def to(self, device: str) -> Self: ...
    def eval(self) -> Self: ...
    def __call__(self, **inputs: Tensor) -> LogitOutput: ...
