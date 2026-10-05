"""Pools and partitions preserve token values and compact byte alignment alike."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from contextlib import closing
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from premixdb.contracts import JSON, Interval
from premixdb.engine.contracts import EncodedTokens
from premixdb.engine.datasets import ByteRanges, ByteTokens, TokenList
from premixdb.engine.token_cache import TokenCache
from premixdb.engine.token_codec import decode_tokens, encode_tokens, token_length


def test_model_tokens_keep_the_partition_wire_format_and_detached_pool_reads() -> None:
    tokens = TokenList([0, 2**32 - 1, 3], [[], [(2, 4)], [(4, 5), (8, 10)]])
    record: EncodedTokens = {
        "tokens": [0, 2**32 - 1, 3],
        "ranges": [[], [[2, 4]], [[4, 5], [8, 10]]],
    }
    assert encode_tokens(tokens) == record
    assert token_length(record) == len(tokens)
    with closing(TokenCache()) as cache:
        cache["model"] = tokens
        tokens[0] = 5
        tokens.ranges[1].clear()
        first = cache["model"]
        assert isinstance(first, TokenList)
        assert cache.length("model") == 3
        assert encode_tokens(first) == record
        first[0] = 9
        first.ranges[2].clear()
        assert encode_tokens(cache["model"]) == record


@pytest.mark.parametrize(
    "index", [slice(None), slice(2, 5), slice(None, None, -1), slice(None, None, 2), slice(0, 0)]
)
def test_byte_tokens_round_trip_through_the_pool_without_losing_source_ranges(index: slice) -> None:
    original = ByteTokens(b"abcdefg", ByteRanges(((5, 8), (11, 15))))
    tokens = original[index]
    expected = list(tokens), list(tokens.ranges)
    assert token_length(encode_tokens(tokens)) == len(tokens)
    with closing(TokenCache()) as cache:
        cache["bytes"] = tokens
        result = cache["bytes"]
        assert (list(result), list(result.ranges)) == expected
        assert cache.length("bytes") == len(tokens)


def test_large_byte_pool_keeps_lazy_ranges_and_compact_storage() -> None:
    data = b"x" * 100_000
    tokens = ByteTokens(data, ByteRanges(((5, 50_005), (70_000, 120_000))))
    with (
        closing(TokenCache()) as cache,
        patch.object(ByteRanges, "__iter__", side_effect=AssertionError("expanded byte ranges")),
    ):
        cache["large"] = tokens
        result = cache["large"]
        assert isinstance(result, ByteTokens)
        assert result.data == data
        assert isinstance(result.ranges, ByteRanges)
        assert list(result.ranges.intervals()) == [(5, 50_005), (70_000, 120_000)]
        assert cache.length("large") == len(data)
        assert cache.database.execute("SELECT length(data) FROM tokens").fetchone()[0] < 2 * len(
            data
        )


TOKEN_RANGES = st.lists(
    st.tuples(st.integers(0, 100), st.integers(0, 100)).map(lambda pair: tuple(sorted(pair))),
    max_size=4,
)


@given(st.lists(st.tuples(st.integers(0, 2**32 - 1), TOKEN_RANGES), max_size=40))
def test_model_codec_round_trips_repeated_empty_and_disjoint_alignment(
    rows: list[tuple[int, list[Interval]]],
) -> None:
    tokens = TokenList([token for token, _ in rows], [ranges for _, ranges in rows])
    assert token_length(encode_tokens(tokens)) == len(tokens)
    result = decode_tokens(encode_tokens(tokens))
    assert list(result) == list(tokens)
    assert list(result.ranges) == tokens.ranges


@pytest.mark.parametrize(
    "tokens",
    [TokenList([1], []), ByteTokens(b"x", ByteRanges(()))],
)
def test_alignment_coverage_is_checked_before_writing_to_the_pool(
    tokens: ByteTokens | TokenList,
) -> None:
    with closing(TokenCache()) as cache:
        with pytest.raises(ValueError, match="alignment coverage"):
            cache["invalid"] = tokens
        assert "invalid" not in cache


@pytest.mark.parametrize(
    "record",
    [
        {"tokens": [1], "ranges": []},
        {"tokens": [1], "ranges": [[[0]]]},
        {"tokens": [1], "ranges": [[[0, 1, 2]]]},
        {"byte_tokens": "AQ==", "byte_ranges": []},
        {"byte_tokens": "?", "byte_ranges": [[0, 1]]},
        {"tokens": [1], "ranges": [[[-1, 0]]]},
        {"tokens": [1], "ranges": [[[2, 1]]]},
        {"byte_tokens": "eA==", "byte_ranges": [[5, 3], [0, 3]]},
        {"byte_tokens": "eA==", "byte_ranges": [[-1, 0]]},
    ],
)
@pytest.mark.parametrize("read", [decode_tokens, token_length])
def test_incomplete_or_malformed_stored_alignment_is_rejected(
    record: EncodedTokens, read: Callable[[EncodedTokens], object]
) -> None:
    with pytest.raises(ValueError):
        read(record)


@pytest.mark.parametrize(
    "record",
    [
        {"tokens": [True], "ranges": [[[0, 1]]]},
        {"tokens": [1], "ranges": [[[False, 1]]]},
        {"tokens": [1], "ranges": [["01"]]},
    ],
)
def test_pool_reads_reject_legacy_json_token_records(record: JSON) -> None:
    import json

    with closing(TokenCache()) as cache:
        cache["model"] = TokenList([1], [[(0, 1)]])
        cache.database.execute("UPDATE tokens SET data=?", (json.dumps(record).encode(),))
        with pytest.raises(ValueError, match="malformed protobuf payload"):
            cache["model"]


def test_measuring_large_model_rows_does_not_reconstruct_token_or_alignment_lists() -> None:
    count = 100_000
    row: EncodedTokens = {"tokens": list(range(count)), "ranges": [[[0, 1]]] * count}
    tracemalloc.start()
    try:
        assert token_length(row) == count
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 64 * 1024
