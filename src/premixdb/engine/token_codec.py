"""Compact token values and source-byte alignments shared by pools and partitions."""

from __future__ import annotations

import base64

from premixdb.engine.contracts import EncodedTokens
from premixdb.engine.datasets import ByteRanges, ByteTokens, TokenList
from premixdb.internal import transport_pb2 as t
from premixdb.schemas.binary import CODEC_VERSION, known, parse
from premixdb.schemas.protobuf import wire


def byte_interval(pair: list[int]) -> tuple[int, int]:
    if len(pair) != 2 or not 0 <= pair[0] <= pair[1]:
        raise ValueError("expected a byte interval")
    return pair[0], pair[1]


def _covered_length(tokens: int, ranges: int) -> int:
    if tokens != ranges:
        raise ValueError("token alignment coverage differs")
    return tokens


def encode_tokens(tokens: list[int] | ByteTokens | TokenList) -> EncodedTokens:
    if isinstance(tokens, (ByteTokens, TokenList)):
        _covered_length(len(tokens), len(tokens.ranges))
    if isinstance(tokens, ByteTokens) and isinstance(tokens.ranges, ByteRanges):
        return EncodedTokens(
            byte_tokens=base64.b64encode(tokens.data).decode(),
            byte_ranges=[[a, b] for a, b in tokens.ranges.intervals()],
        )
    return EncodedTokens(
        tokens=list(tokens),
        ranges=[[[a, b] for a, b in ranges] for ranges in tokens.ranges]
        if isinstance(tokens, (TokenList, ByteTokens))
        else [[] for _ in tokens],
    )


def token_length(row: EncodedTokens) -> int:
    """Verify alignment while measuring without retaining decoded token/range lists."""
    if "byte_tokens" in row:
        tokens = len(base64.b64decode(row["byte_tokens"], validate=True))
        ranges = sum(b - a for a, b in map(byte_interval, row["byte_ranges"]))
    else:
        tokens, ranges = len(row["tokens"]), len(row["ranges"])
        for intervals in row["ranges"]:
            for pair in intervals:
                byte_interval(pair)
    return _covered_length(tokens, ranges)


def decode_tokens(row: EncodedTokens) -> ByteTokens | TokenList:
    if "byte_tokens" in row:
        result: ByteTokens | TokenList = ByteTokens(
            base64.b64decode(row["byte_tokens"], validate=True),
            ByteRanges([byte_interval(pair) for pair in row["byte_ranges"]]),
        )
    else:
        result = TokenList(
            row["tokens"], [[byte_interval(pair) for pair in ranges] for ranges in row["ranges"]]
        )
    _covered_length(len(result), len(result.ranges))
    return result


def packed_tokens(tokens: list[int] | ByteTokens | TokenList) -> t.PackedTokens:
    """Encode alignment with packed arrays, retaining compact byte encodings."""
    result = t.PackedTokens(version=CODEC_VERSION)
    if isinstance(tokens, (ByteTokens, TokenList)):
        _covered_length(len(tokens), len(tokens.ranges))
    if isinstance(tokens, ByteTokens) and isinstance(tokens.ranges, ByteRanges):
        result.byte.tokens = tokens.data
        for start, end in tokens.ranges.intervals():
            result.byte.starts.append(start)
            result.byte.ends.append(end)
    else:
        result.numeric.tokens.extend(tokens)
        offsets = [0]
        starts, ends = [], []
        ranges = (
            tokens.ranges if isinstance(tokens, (TokenList, ByteTokens)) else [[] for _ in tokens]
        )
        for intervals in ranges:
            for start, end in intervals:
                starts.append(start)
                ends.append(end)
            offsets.append(len(starts))
        result.numeric.interval_offsets.extend(offsets)
        result.numeric.starts.extend(starts)
        result.numeric.ends.extend(ends)
    packed_length(result)
    return result


def packed_length(row: t.PackedTokens) -> int:
    """Check domain validity independently of protobuf's wire type validation."""
    known(row)
    if row.version != CODEC_VERSION:
        raise ValueError("unsupported packed token version")
    kind = row.WhichOneof("representation")
    if kind == "numeric":
        values = row.numeric
        known(values)
        count, intervals = len(values.tokens), len(values.starts)
        offsets = values.interval_offsets
        if (
            len(offsets) != count + 1
            or offsets[0] != 0
            or offsets[-1] != intervals
            or any(a > b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("invalid token interval offsets")
    elif kind == "byte":
        values = row.byte
        known(values)
        count = len(values.tokens)
    else:
        raise ValueError("missing token representation")
    if len(values.starts) != len(values.ends) or any(
        start > end for start, end in zip(values.starts, values.ends)
    ):
        raise ValueError("invalid token byte intervals")
    if kind == "byte" and sum(b - a for a, b in zip(values.starts, values.ends)) != count:
        raise ValueError("token alignment coverage differs")
    return count


def unpack_tokens(row: t.PackedTokens) -> ByteTokens | TokenList:
    packed_length(row)
    if row.WhichOneof("representation") == "byte":
        return ByteTokens(row.byte.tokens, ByteRanges(list(zip(row.byte.starts, row.byte.ends))))
    values = row.numeric
    intervals = list(zip(values.starts, values.ends))
    offsets = values.interval_offsets
    return TokenList(values.tokens, [intervals[a:b] for a, b in zip(offsets, offsets[1:])])


def dump_tokens(tokens: list[int] | ByteTokens | TokenList) -> bytes:
    return wire(packed_tokens(tokens))


def load_tokens(data: bytes) -> ByteTokens | TokenList:
    return unpack_tokens(parse(data, t.PackedTokens()))
