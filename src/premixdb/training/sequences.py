"""Verified sequence metadata and detached values, independent of recipe execution."""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterable
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from functools import cached_property, partial
from typing import cast

from premixdb.contracts import ExecutionError
from premixdb.schemas.ids import _encode_id
from premixdb.schemas.protobuf import copy_message, parse
from premixdb.storage.ranges import RangeReader
from premixdb.v1 import data_mixture_pb2 as datasets
from premixdb.v1.storage_pb2 import SpanRef

# The stored index format groups consecutive sequences into fixed-size pages.
INDEX_PAGE_SIZE = 128


def _validate_mask(data: bytes, kind: str) -> None:
    # Deleting valid zero/one bytes leaves only invalid values, without a Python loop.
    if data.translate(None, b"\x00\x01"):
        raise ValueError(f"invalid stored {kind} mask")


def preview_decoder(
    tokenizer: datasets.Tokenizer, reader: RangeReader
) -> Callable[[list[int]], str] | None:
    """Read a verified tokenizer asset for decoding completed sequence examples."""
    if not tokenizer.HasField("hugging_face"):
        return None
    from tokenizers import Tokenizer

    asset = tokenizer.hugging_face.asset
    data = reader.read(
        SpanRef(object=asset, end=asset.size_bytes, blake3_digest=asset.blake3_digest)
    )
    decoder = Tokenizer.from_str(data.decode())
    return cast(Callable[[list[int]], str], partial(decoder.decode, skip_special_tokens=False))


def decode_preview(
    tokens: SequenceABC[int],
    regions: Iterable[datasets.TokenRegion],
    decode: Callable[[list[int]], str] | None = None,
) -> str:
    """Decode bounded token examples, labeling byte separators and padding."""
    if decode is not None:
        return decode(list(tokens))
    text = []
    for region in regions:
        values = tokens[region.start : region.end]
        if not values:
            continue
        if region.kind == datasets.TokenRegion.KIND_CONTENT:
            text.append(bytes(values).decode("utf-8", errors="replace"))
        else:
            label = "separator" if region.kind == datasets.TokenRegion.KIND_SEPARATOR else "padding"
            text.append(f"<{label}:{values[0]} ×{len(values)}>")
    return "".join(text)


@dataclass(frozen=True)
class Sequence:
    _value: datasets.Sequence
    _reader: RangeReader

    def __repr__(self) -> str:
        return f"Sequence(ordinal={self.ordinal}, documents={self.document_ids()!r})"

    @property
    def ordinal(self) -> int:
        """Return the zero-based position of this sequence in its dataset."""
        return self._value.ordinal

    @cached_property
    def _token_data(self) -> bytes:
        data = self._reader.read(self._value.tokens)
        if self._value.tokens.start % 4 or len(data) % 4:
            raise ValueError("token range is not uint32 aligned")
        return data

    def _token_values(self, limit: int | None = None) -> list[int]:
        data = self._token_data
        count = len(data) // 4 if limit is None else min(limit, len(data) // 4)
        return [int(value) for value in struct.unpack_from(f"<{count}I", data)]

    @cached_property
    def _tokens(self) -> tuple[int, ...]:
        return tuple(self._token_values())

    @property
    def tokens(self) -> list[int]:
        """Read the token IDs for this packed sequence."""
        return list(self._tokens)

    @cached_property
    def _mask_data(self) -> bytes:
        return self._read_mask(self._value.loss_mask, "token")

    def _read_mask(self, span: SpanRef, kind: str) -> bytes:
        data = self._reader.read(span)
        _validate_mask(data, kind)
        return data

    @cached_property
    def _mask(self) -> tuple[bool, ...]:
        return tuple(bool(value) for value in self._mask_data)

    @property
    def mask(self) -> list[bool]:
        """Mark content and separator tokens as True and padding tokens as False."""
        return list(self._mask)

    @cached_property
    def _attention_mask_data(self) -> bytes:
        if self._value.attention_mask == self._value.loss_mask:
            return self._mask_data
        return self._read_mask(self._value.attention_mask, "attention")

    @cached_property
    def _attention_mask(self) -> tuple[bool, ...]:
        if self._value.attention_mask == self._value.loss_mask:
            return self._mask
        return tuple(bool(value) for value in self._attention_mask_data)

    @property
    def attention_mask(self) -> list[bool]:
        """Return the sequence mask used to exclude padding from attention."""
        return list(self._attention_mask)

    def _preview(self, limit: int) -> tuple[list[int], list[bool], list[bool]]:
        return (
            self._token_values(limit),
            [bool(value) for value in self._mask_data[:limit]],
            [bool(value) for value in self._attention_mask_data[:limit]],
        )

    @property
    def spans(self) -> list[datasets.TokenRegion]:
        """Return source, separator, and padding regions within this sequence."""
        return [copy_message(region) for region in self._value.regions]

    def document_ids(self) -> list[str]:
        """Unique source IDs in first-content order, excluding separators and padding."""
        return list(
            dict.fromkeys(
                _encode_id(region.document_id)
                for region in self._value.regions
                if region.kind == datasets.TokenRegion.KIND_CONTENT
            )
        )


def read_page(
    resource: datasets.Dataset, reader: RangeReader, ordinal: int, size: int | None = None
) -> list[Sequence]:
    """Read the containing index page, optionally selecting an ordinal window."""
    page = ordinal // INDEX_PAGE_SIZE
    sequences = sequence_page(resource, reader.read(resource.sequences[page]), page)
    return [
        Sequence(seq, reader)
        for seq in sequences
        if size is None or ordinal <= seq.ordinal < ordinal + size
    ]


def sequence_page(
    resource: datasets.Dataset, data: bytes, page: int
) -> tuple[datasets.Sequence, ...]:
    sequences = parse(datasets.SequenceBatch, data).sequences
    expected = min(INDEX_PAGE_SIZE, resource.profile.sequences - page * INDEX_PAGE_SIZE)
    if len(sequences) != expected or [seq.ordinal for seq in sequences] != list(
        range(page * INDEX_PAGE_SIZE, page * INDEX_PAGE_SIZE + expected)
    ):
        raise ExecutionError("stored sequence index is incomplete or out of order")
    for sequence in sequences:
        if sequence.tokens.start % 4:
            raise ValueError("token range is not uint32 aligned")
        if (
            sequence.tokens.end - sequence.tokens.start != resource.sequence_length * 4
            or sequence.attention_mask.end - sequence.attention_mask.start
            != resource.sequence_length
            or sequence.loss_mask.end - sequence.loss_mask.start != resource.sequence_length
        ):
            raise ValueError("stored sequence has an invalid length")
    return tuple(sequences)
