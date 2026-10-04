"""Durable encodings preserve selection identity and retained source alignment."""

from pathlib import Path
from unittest.mock import patch

import pytest
from blake3 import blake3

from premixdb.engine.curation import RetainedDocument
from premixdb.engine.datasets import HuggingFaceTokenizer, encoded_tokens
from premixdb.engine.queries import Row
from premixdb.engine.snapshots import Document
from premixdb.internal import derivation_pb2 as d
from premixdb.runtime.catalog import wire
from premixdb.runtime.encodings import Encodings
from premixdb.storage.objects import ObjectStore


@pytest.fixture
def tokenizer() -> HuggingFaceTokenizer:
    data = (Path(__file__).parent / "fixtures" / "wordpiece.json").read_bytes()
    return HuggingFaceTokenizer.from_bytes(data, blake3(data).hexdigest(), 1024)


def selection() -> Row:
    original = Document("01" * 16, "source", "xxhel秘密lozz")
    return Row(0, RetainedDocument(original, "hello", ((2, 5), (11, 13))))


def test_original_retained_and_empty_encodings_reopen_separately(
    tmp_path: Path, tokenizer: HuggingFaceTokenizer
) -> None:
    retained = selection()
    assert isinstance(retained.document, RetainedDocument)
    original = retained.document.original
    rows = [Row(0, original), retained, Row(0, RetainedDocument(original, "", ()))]
    expected = [encoded_tokens(row, tokenizer) for row in rows]
    assert list(expected[1]) == [4]
    assert list(expected[1].ranges) == [[(2, 5), (11, 13)]]
    assert list(expected[2]) == []
    with ObjectStore(tmp_path) as store:
        encodings = Encodings(store)
        for row, tokens in zip(rows, expected, strict=True):
            result = encodings(row, tokenizer)
            assert list(result) == list(tokens)
            assert list(result.ranges) == list(tokens.ranges)
        manifests = store.list("tokenizer", d.TokenEncoding, suffix=".encoding")
        assert len({manifest.id for manifest in manifests}) == 3
    with (
        ObjectStore(tmp_path) as store,
        patch.object(
            HuggingFaceTokenizer,
            "encode_with_offsets",
            side_effect=AssertionError("reencoded cached selection"),
        ),
    ):
        encodings = Encodings(store)
        for row, tokens in zip(rows, expected, strict=True):
            result = encodings(row, tokenizer)
            assert list(result) == list(tokens)
            assert list(result.ranges) == list(tokens.ranges)


@pytest.mark.parametrize("start,end", [(0, 1), (5, 6), (4, 12), (14, 16)])
def test_cached_alignment_cannot_point_outside_retained_text(
    tmp_path: Path, tokenizer: HuggingFaceTokenizer, start: int, end: int
) -> None:
    row = selection()
    with ObjectStore(tmp_path) as store:
        encodings = Encodings(store)
        encodings(row, tokenizer)
        manifest = store.list("tokenizer", d.TokenEncoding, suffix=".encoding")[0]
        shard = d.TokenEncodingShard.FromString(store.read_object("tokenizer", manifest.shards[0]))
        shard.tokens[0].ranges[0].start = start
        shard.tokens[0].ranges[0].end = end
        manifest.shards[0].CopyFrom(store.put("tokenizer", wire(shard)))
        store.metadata.save("tokenizer", manifest.id, manifest, suffix=".encoding", mutable=True)
        with (
            patch.object(
                HuggingFaceTokenizer,
                "encode_with_offsets",
                side_effect=AssertionError("reencoded corrupt cache"),
            ),
            pytest.raises(ValueError, match="cached token alignment"),
        ):
            encodings(row, tokenizer)


def test_empty_selection_rejects_cached_source_ranges(
    tmp_path: Path, tokenizer: HuggingFaceTokenizer
) -> None:
    retained = selection()
    assert isinstance(retained.document, RetainedDocument)
    row = Row(0, RetainedDocument(retained.document.original, "", ()))
    with ObjectStore(tmp_path) as store:
        encodings = Encodings(store)
        encodings(row, tokenizer)
        manifest = store.list("tokenizer", d.TokenEncoding, suffix=".encoding")[0]
        shard = d.TokenEncodingShard(
            tokens=[d.EncodedToken(value=4, ranges=[d.ByteRange(start=2, end=3)])]
        )
        manifest.shards.append(store.put("tokenizer", wire(shard)))
        manifest.tokens = 1
        store.metadata.save("tokenizer", manifest.id, manifest, suffix=".encoding", mutable=True)
        with pytest.raises(ValueError, match="includes removed text"):
            encodings(row, tokenizer)
