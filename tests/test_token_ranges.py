"""Token slices preserve byte alignment across retained source intervals."""

from pathlib import Path

import pytest
from blake3 import blake3
from hypothesis import given
from hypothesis import strategies as st

from premixdb.contracts import Interval
from premixdb.engine.curation import RetainedDocument
from premixdb.engine.datasets import (
    ByteRanges,
    ByteTokens,
    HuggingFaceTokenizer,
    _token_ranges,
    encoded_tokens,
)
from premixdb.engine.queries import Row
from premixdb.engine.snapshots import Document

SLICES = st.builds(
    slice,
    st.one_of(st.none(), st.integers(-100, 100)),
    st.one_of(st.none(), st.integers(-100, 100)),
    st.one_of(st.none(), st.sampled_from([-3, -2, -1, 1, 2, 3])),
)


@pytest.mark.parametrize("index", [slice(4, 1), slice(100, 1), slice(-1, -4)])
def test_empty_forward_slices_have_zero_length(index: slice) -> None:
    ranges = ByteRanges(((2, 5), (8, 10)))[index]
    assert isinstance(ranges, ByteRanges)
    assert len(ranges) == 0
    assert list(ranges) == []
    assert list(ranges.intervals()) == []


@given(
    segments=st.lists(st.tuples(st.integers(0, 8), st.integers(0, 8)), max_size=8),
    first=SLICES,
    second=SLICES,
)
def test_nested_slices_match_byte_and_alignment_sequences(
    segments: list[tuple[int, int]], first: slice, second: slice
) -> None:
    sources: list[Interval] = []
    cursor = 0
    for gap, length in segments:
        start = cursor + gap
        cursor = start + length
        sources.append((start, cursor))
    alignment = [[(position, position + 1)] for a, b in sources for position in range(a, b)]
    data = bytes(a for ranges in alignment for a, _ in ranges)
    tokens = ByteTokens(data, ByteRanges(sources))
    for actual, expected_data, expected_ranges in (
        (tokens[first], data[first], alignment[first]),
        (tokens[first][second], data[first][second], alignment[first][second]),
    ):
        assert list(actual) == list(expected_data)
        assert len(actual.ranges) == len(expected_data)
        assert list(actual.ranges) == expected_ranges
        if isinstance(actual.ranges, ByteRanges):
            assert [
                [(position, position + 1)]
                for a, b in actual.ranges.intervals()
                for position in range(a, b)
            ] == expected_ranges


def test_one_token_can_span_separate_retained_intervals() -> None:
    original = Document("01" * 16, "source", "hel秘密lo")
    row = Row(0, RetainedDocument(original, "hello", ((0, 3), (9, 11))))
    asset = (Path(__file__).parent / "fixtures" / "wordpiece.json").read_bytes()
    tokenizer = HuggingFaceTokenizer.from_bytes(asset, blake3(asset).hexdigest(), 1024)
    encoded = encoded_tokens(row, tokenizer)
    assert list(encoded) == [4]
    assert list(encoded.ranges) == [[(0, 3), (9, 11)]]
    byte_encoded = encoded_tokens(row)
    assert list(byte_encoded) == list(b"hello")
    assert list(byte_encoded.ranges) == [[(0, 1)], [(1, 2)], [(2, 3)], [(9, 10)], [(10, 11)]]


@given(
    segments=st.lists(st.tuples(st.integers(0, 8), st.integers(0, 8)), max_size=16),
    bounds=st.lists(st.tuples(st.integers(0, 128), st.integers(0, 128)), max_size=24),
)
def test_indexed_token_alignment_preserves_overlapping_and_repeated_offsets(
    segments: list[tuple[int, int]], bounds: list[tuple[int, int]]
) -> None:
    sources: list[Interval] = []
    positions: list[int] = []
    cursor = 0
    for gap, length in segments:
        start = cursor + gap
        cursor = start + length
        sources.append((start, cursor))
        positions.extend(range(start, cursor))
    offsets = []
    for a, b in bounds:
        a, b = a % (len(positions) + 1), b % (len(positions) + 1)
        offsets.append((min(a, b), max(a, b)))
    mapped = _token_ranges(sources, offsets)
    for (start, end), ranges in zip(offsets, mapped, strict=True):
        assert [position for a, b in ranges for position in range(a, b)] == positions[start:end]
        assert all(a < b for a, b in ranges)


def test_many_retained_unicode_segments_keep_model_token_alignment() -> None:
    parts = ["hello ", "é ", "🌍 ", "world "] * 128
    original = Document("01" * 16, "fragmented", "秘密".join(parts))
    sources = []
    positions = []
    cursor = 0
    for part in parts:
        end = cursor + len(part.encode())
        sources.append((cursor, end))
        positions.extend(range(cursor, end))
        cursor = end + len("秘密".encode())
    text = "".join(parts)
    row = Row(0, RetainedDocument(original, text, tuple(sources)))
    asset = (Path(__file__).parent / "fixtures" / "wordpiece.json").read_bytes()
    tokenizer = HuggingFaceTokenizer.from_bytes(asset, blake3(asset).hexdigest(), 8192)
    expected_tokens, offsets = tokenizer.encode_with_offsets(text)
    encoded = encoded_tokens(row, tokenizer)
    assert list(encoded) == expected_tokens
    assert len(encoded.ranges) == len(offsets)
    for (start, end), ranges in zip(offsets, encoded.ranges, strict=True):
        assert [position for a, b in ranges for position in range(a, b)] == positions[start:end]
