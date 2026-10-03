"""Compact token values and source-byte alignments shared by pools and partitions."""

from __future__ import annotations

import base64

from .contracts import EncodedTokens
from .datasets import ByteRanges, ByteTokens, TokenList


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
