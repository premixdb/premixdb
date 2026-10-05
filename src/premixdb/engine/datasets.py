"""Deterministic tokenization, packing, provenance, and mixture draws."""

from __future__ import annotations

import json
import os
import time
from array import array
from bisect import bisect_right
from collections.abc import Sequence as SequenceABC
from contextlib import ExitStack
from copy import deepcopy
from itertools import accumulate, chain, islice, repeat
from pathlib import Path
from sys import byteorder
from typing import TYPE_CHECKING, Generator, Iterable, Iterator, Literal, SupportsIndex, overload

from premixdb.contracts import Checkpoint, Interval
from premixdb.engine.contracts import (
    Counts,
    Occurrence,
    PackingSummary,
    SourceRange,
    Span,
    TokenRange,
)
from premixdb.training.reader import Reader, Topology

if TYPE_CHECKING:
    from premixdb.engine.queries import Query, Row
    from premixdb.engine.token_cache import TokenCache

from blake3 import blake3

from premixdb.engine.dataset_plan import BYTE_DEFINITION as BYTE_DEFINITION
from premixdb.engine.dataset_plan import DatasetPlan, PackingPlan
from premixdb.engine.identity import Canonical, digest, unsigned

TOKEN_SHARD_BYTES = 8 * 1024 * 1024


def _tokenizer_definition(asset_digest: str) -> str:
    """Historical v1 codec identity, not the installed tokenizers package version.

    Preserve this label so saved tokenizer and dataset IDs remain reusable.
    The execution CodeVersion separately fingerprints installed package versions.
    """
    return (
        Canonical("tokenizer")
        .string("huggingface/tokenizers/0.22.2/python/onig/encode-no-special-tokens/v1")
        .fixed(digest(asset_digest))
        .finish()
        .hex()
    )


class HuggingFaceTokenizer:
    def __init__(self, path: str | Path, expected: str, max_document_bytes: int) -> None:
        self._set_limit(max_document_bytes)
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise ValueError(str(exc)) from exc
        self._load(data, expected)

    @classmethod
    def from_bytes(
        cls, data: bytes, expected: str, max_document_bytes: int
    ) -> HuggingFaceTokenizer:
        """Load an admitted asset directly, without a temporary file."""
        tokenizer = cls.__new__(cls)
        tokenizer._set_limit(max_document_bytes)
        tokenizer._load(data, expected)
        return tokenizer

    def _set_limit(self, max_document_bytes: int) -> None:
        if not unsigned(max_document_bytes):
            raise ValueError("max_document_bytes must be positive")
        self.max_document_bytes = max_document_bytes

    def _load(self, data: bytes, expected: str) -> None:
        from tokenizers import Tokenizer

        self.asset_bytes = data
        self.asset_digest = blake3(data).hexdigest()
        if digest(expected).hex() != self.asset_digest:
            raise RuntimeError("tokenizer asset digest mismatch")
        try:
            asset = json.loads(data)
            self._inner = Tokenizer.from_str(data.decode())
        except Exception as exc:
            raise ValueError("invalid tokenizer JSON") from exc
        if asset.get("truncation") or asset.get("padding"):
            raise NotImplementedError(
                "tokenizer truncation and padding must be disabled; use dataset packing"
            )
        if asset["model"].get("dropout"):
            raise NotImplementedError(
                "tokenizer BPE dropout must be disabled for deterministic encoding"
            )
        self.definition = _tokenizer_definition(self.asset_digest)

    def encode(self, text: str) -> list[int]:
        if len(text.encode()) > self.max_document_bytes:
            raise NotImplementedError("tokenizer input exceeds whole-document limit")
        return self._inner.encode(text, add_special_tokens=False).ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[Interval]]:
        if len(text.encode()) > self.max_document_bytes:
            raise NotImplementedError("tokenizer input exceeds whole-document limit")
        encoded = self._inner.encode(text, add_special_tokens=False)
        boundaries, cursor = [0], 0
        for character in text:
            cursor += len(character.encode())
            boundaries.append(cursor)
        return encoded.ids, [(boundaries[a], boundaries[b]) for a, b in encoded.offsets]

    def token_to_id(self, token: str) -> int | None:
        return self._inner.token_to_id(token)

    def decode(self, tokens: list[int]) -> str:
        return self._inner.decode(tokens, skip_special_tokens=False)


class TokenList(list[int]):
    def __init__(self, tokens: Iterable[int], ranges: list[list[Interval]]) -> None:
        super().__init__(tokens)
        self.ranges = ranges

    @overload
    def __getitem__(self, index: SupportsIndex) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> TokenList: ...
    def __getitem__(self, index: SupportsIndex | slice) -> int | TokenList:
        if isinstance(index, slice):
            return TokenList(super().__getitem__(index), self.ranges[index])
        return super().__getitem__(index)


def _source_intervals(
    sources: SequenceABC[Interval], start: int, end: int, first: int = 0, cursor: int = 0
) -> Iterator[Interval]:
    for index in range(first, len(sources)):
        if cursor >= end:
            break
        a, b = sources[index]
        left, right = max(start, cursor), min(end, cursor + b - a)
        if left < right:
            yield a + left - cursor, a + right - cursor
        cursor += b - a


def _token_ranges(
    sources: SequenceABC[Interval], offsets: Iterable[Interval]
) -> list[list[Interval]]:
    """Locate each token in concatenated retained bytes, including repeated offsets."""
    ends = list(accumulate(b - a for a, b in sources))
    mapped = []
    for start, end in offsets:
        first = bisect_right(ends, start)
        cursor = ends[first - 1] if first else 0
        mapped.append(list(_source_intervals(sources, start, end, first, cursor)))
    return mapped


class ByteRanges:
    def __init__(
        self, sources: SequenceABC[Interval], start: int = 0, end: int | None = None
    ) -> None:
        self.sources = sources
        self.start = start
        self.end = sum(b - a for a, b in sources) if end is None else end

    def intervals(self) -> Iterator[Interval]:
        return _source_intervals(self.sources, self.start, self.end)

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self) -> Iterator[list[Interval]]:
        for start, end in self.intervals():
            for position in range(start, end):
                yield [(position, position + 1)]

    @overload
    def __getitem__(self, index: int) -> list[Interval]: ...
    @overload
    def __getitem__(self, index: slice) -> ByteRanges | list[list[Interval]]: ...
    def __getitem__(self, index: int | slice) -> list[Interval] | ByteRanges | list[list[Interval]]:
        if isinstance(index, slice):
            start, end, step = index.indices(len(self))
            if step != 1:
                return [ranges for ranges in self][index]
            return ByteRanges(self.sources, self.start + start, self.start + max(start, end))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return next(iter(self[index : index + 1]))


class ByteTokens:
    def __init__(self, data: bytes, ranges: ByteRanges | list[list[Interval]]) -> None:
        self.data, self.ranges = data, ranges

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[int]:
        return iter(self.data)

    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> ByteTokens: ...
    def __getitem__(self, index: int | slice) -> int | ByteTokens:
        if isinstance(index, slice):
            return ByteTokens(self.data[index], self.ranges[index])
        return self.data[index]


def encoded_tokens(
    row: Row, tokenizer: HuggingFaceTokenizer | None = None
) -> ByteTokens | TokenList:
    text = row.text
    from premixdb.engine.curation import RetainedDocument

    if not tokenizer:
        data = text.encode()
        source = (
            row.document.source_ranges
            if isinstance(row.document, RetainedDocument)
            else ((0, len(data)),)
        )
        return ByteTokens(data, ByteRanges(source))
    tokens, offsets = tokenizer.encode_with_offsets(text)
    source = (
        row.document.source_ranges
        if isinstance(row.document, RetainedDocument)
        else ((0, row.document.size),)
    )
    return TokenList(tokens, _token_ranges(source, offsets))


def compact_ranges(ranges: Iterable[TokenRange]) -> list[SourceRange]:
    result: list[SourceRange] = []
    for record in ranges:
        if (
            result
            and result[-1]["occurrence"] == record["occurrence"]
            and result[-1]["token_end"] == record["token"]
            and result[-1]["end"] == record["start"]
            and record["end"] - record["start"] == 1
            and result[-1]["end"] - result[-1]["start"]
            == result[-1]["token_end"] - result[-1]["token"]
        ):
            result[-1]["end"] = record["end"]
            result[-1]["token_end"] = record["token"] + 1
        else:
            result.append(SourceRange(**record, token_end=record["token"] + 1))
    return result


class Sequence:
    def __init__(
        self,
        ordinal: int,
        tokens: Iterable[int],
        spans: list[Span],
        source_ranges: Iterable[SourceRange] = (),
    ) -> None:
        self.ordinal = ordinal
        self._tokens = array("I", tokens)
        self._spans = spans
        self.source_ranges = tuple(source_ranges)

    @property
    def tokens(self) -> list[int]:
        return list(self._tokens)

    def _token_bytes(self) -> bytes:
        tokens = self._tokens
        if byteorder != "little":
            tokens = tokens[:]
            tokens.byteswap()
        return tokens.tobytes()

    @property
    def mask(self) -> list[bool]:
        return list(self._mask_values())

    def _mask_values(self) -> Iterator[bool]:
        return chain.from_iterable(
            repeat(span["kind"] != "padding", span["end"] - span["start"]) for span in self._spans
        )

    def _preview(self, limit: int) -> tuple[list[int], list[int]]:
        return list(self._tokens[:limit]), [int(v) for v in islice(self._mask_values(), limit)]

    @property
    def spans(self) -> list[Span]:
        return deepcopy(self._spans)


class Dataset:
    @classmethod
    def from_query(
        cls,
        query: Query,
        length: int,
        separator: int | None,
        padding: int | None,
        tokenizer: HuggingFaceTokenizer | None = None,
        *,
        stream: bool = False,
    ) -> Dataset:
        start = time.monotonic()
        plan = DatasetPlan(
            query.id,
            tokenizer.definition if tokenizer else BYTE_DEFINITION,
            query.code,
            PackingPlan(length, separator, padding),
        )
        pool = None
        with ExitStack() as startup:
            if tokenizer:
                from premixdb.engine.token_cache import token_pool

                pool = token_pool(query, tokenizer)
                startup.callback(pool.close)
                lengths = [pool.length(row.id) for row in query]
                encoded = ((row, pool[row.id]) for row in query)
            else:
                encoded = ((r, encoded_tokens(r)) for r in query)
                lengths = query.lengths()
            dataset = cls(
                query,
                plan,
                query.source_counts(),
                encoded,
                start,
                lengths,
                stream=stream,
                _token_pool=pool,
            )
            startup.pop_all()
            return dataset

    def __init__(
        self,
        query: Query,
        plan: DatasetPlan,
        input_counts: Counts,
        encoded: Iterable[tuple[Row, list[int] | ByteTokens | TokenList]],
        start: float,
        lengths: Iterable[int],
        *,
        stream: bool = False,
        _token_pool: TokenCache | None = None,
    ) -> None:
        self._token_pool = _token_pool
        self.plan = plan
        self._id = self.plan.id
        self.query_id = query.id
        self.tokenizer_definition = plan.definition
        self._lengths = tuple(lengths)
        self._totals = self.plan.packing.measure(self._lengths)
        self._input_counts = input_counts.copy()
        self._query = query
        self._encoded = iter(encoded)
        self._occurrences: list[Occurrence] = []
        self._consumed = False
        self._sequences = None if stream else list(self._pack())
        self.elapsed_seconds = time.monotonic() - start

    def _pack(self) -> Iterator[Sequence]:
        try:
            yield from self._pack_sequences()
        finally:
            self.close()

    def close(self) -> None:
        """Release only the token pool owned by this packing stream."""
        if self._token_pool is not None:
            self._token_pool.close()
            self._token_pool = None

    def __del__(self) -> None:
        if hasattr(self, "_token_pool"):
            self.close()

    def _pack_sequences(self) -> Iterator[Sequence]:
        if self._consumed:
            raise RuntimeError("packing stream has already been consumed")
        self._consumed = True

        def validated() -> Iterator[tuple[Row, list[int] | ByteTokens | TokenList]]:
            for ordinal, (row, tokens) in enumerate(self._encoded):
                if ordinal >= len(self._lengths) or len(tokens) != self._lengths[ordinal]:
                    raise ValueError("encoded occurrence does not match its packing plan")
                self._occurrences.append(
                    dict(
                        ordinal=ordinal,
                        document=row.id,
                        source=self._query.provenance_for(row),
                        tokens=len(tokens),
                    )
                )
                yield row, tokens
            if len(self._occurrences) != len(self._lengths):
                raise ValueError("incomplete encoded occurrence coverage")

        count = 0
        for sequence in pack_sequences(validated(), self.plan.packing):
            count += 1
            yield sequence
        if count != len(self):
            raise RuntimeError("packed sequence count does not match its plan")
        self._encoded = iter(())

    @property
    def id(self) -> str:
        return self._id

    @property
    def occurrence_count(self) -> int:
        return len(self._occurrences)

    def occurrence_document(self, ordinal: int) -> str:
        if not 0 <= ordinal < self.occurrence_count:
            raise IndexError("occurrence index out of range")
        return self._occurrences[ordinal]["document"]

    def __len__(self) -> int:
        return self._totals.sequences

    def __getitem__(self, index: int) -> Sequence:
        if type(index) is not int:
            raise TypeError("sequence index must be an integer")
        index = index + len(self) if index < 0 else index
        if not 0 <= index < len(self):
            raise IndexError("sequence index out of range")
        if self._sequences is None:
            raise TypeError("a packing stream has no random access; read its published shards")
        return self._sequences[index]

    def summary(self) -> PackingSummary:
        return self._totals.summary(self._input_counts)

    def occurrences(self) -> list[Occurrence]:
        return deepcopy(self._occurrences)

    def sequence(self, index: int) -> Sequence:
        return self[index]

    def _page(self, ordinal: int) -> list[Sequence]:
        return [self[ordinal]]

    def reader(
        self,
        topology: Iterable[int] | Topology,
        checkpoint: Checkpoint | None = None,
        seed: int | None = None,
    ) -> Reader[Sequence]:
        from premixdb.training.reader import Reader, Topology

        return Reader(
            self,
            topology if isinstance(topology, Topology) else Topology(*topology),
            checkpoint,
            seed,
        )

    def iter_sequences(self) -> Iterator[Sequence]:
        return iter(self._sequences) if self._sequences is not None else self._pack()

    def write_token_shards(
        self, directory: Path
    ) -> Iterator[tuple[int, Path, Path, list[tuple[str, str]], list[Sequence]]]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        per_shard = max(1, TOKEN_SHARD_BYTES // self.plan.packing.length // 4)
        sequences = self.iter_sequences()
        first = 0
        while chunk := list(islice(sequences, per_shard)):
            token_path, mask_path = directory / "tokens", directory / "mask"
            digests = []
            with token_path.open("wb") as tokens, mask_path.open("wb") as masks:
                for sequence in chunk:
                    data = sequence._token_bytes()
                    mask = bytes(sequence._mask_values())
                    tokens.write(data)
                    masks.write(mask)
                    digests.append((blake3(data).hexdigest(), blake3(mask).hexdigest()))
                for output in (tokens, masks):
                    output.flush()
                    os.fsync(output.fileno())
            yield first, token_path, mask_path, digests, chunk
            first += len(chunk)


def pack_sequences(
    encoded: Iterable[tuple[Row, list[int] | ByteTokens | TokenList]],
    packing: PackingPlan,
    *,
    ordinal_start: int = 0,
    occurrence_start: int = 0,
) -> Generator[Sequence, None, None]:
    """Pack incrementally; closing the iterator stops consuming source occurrences."""
    pending: list[int] = []
    spans: list[Span] = []
    alignment: list[TokenRange] = []
    sequence_ordinal = ordinal_start

    def append(
        tokens: list[int] | ByteTokens | TokenList,
        kind: Literal["content", "separator", "padding"],
        occurrence: int | None = None,
    ) -> Iterator[Sequence]:
        nonlocal sequence_ordinal
        offset = 0
        while offset < len(tokens):
            take = min(packing.length - len(pending), len(tokens) - offset)
            begin = len(pending)
            pending.extend(tokens[offset : offset + take])
            span: Span = Span(start=begin, end=len(pending), kind=kind)
            if occurrence is not None:
                span["occurrence"] = occurrence
            if kind == "content":
                span["offset"] = offset
            if kind == "content" and isinstance(tokens, (ByteTokens, TokenList)):
                assert occurrence is not None
                alignment.extend(
                    dict(token=begin + i, occurrence=occurrence, start=a, end=b)
                    for i, ranges in enumerate(tokens.ranges[offset : offset + take])
                    for a, b in ranges
                )
            spans.append(span)
            offset += take
            if len(pending) == packing.length:
                yield Sequence(sequence_ordinal, pending, spans.copy(), compact_ranges(alignment))
                sequence_ordinal += 1
                pending.clear()
                spans.clear()
                alignment.clear()

    for ordinal, (_, tokens) in enumerate(encoded, start=occurrence_start):
        yield from append(tokens, "content", ordinal)
        if packing.separator is not None:
            yield from append([packing.separator], "separator", ordinal)
    if pending and packing.padding is not None:
        yield from append([packing.padding] * (packing.length - len(pending)), "padding")
