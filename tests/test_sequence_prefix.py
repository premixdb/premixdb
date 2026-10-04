"""Bounded sequence decoding still verifies complete stored token and mask spans."""

import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from premixdb import RangeReader
from premixdb.storage.objects import ObjectStore
from premixdb.training.sequences import Sequence
from premixdb.v1 import data_mixture_pb2 as d
from premixdb.v1.storage_pb2 import SpanRef


def stored(store: ObjectStore, data: bytes) -> SpanRef:
    object = store.put("dataset", data)
    return SpanRef(object=object, end=len(data), blake3_digest=object.blake3_digest)


def record(store: ObjectStore) -> d.Sequence:
    return d.Sequence(
        ordinal=0,
        tokens=stored(store, struct.pack("<512I", *range(512))),
        loss_mask=stored(store, bytes([1] * 300 + [0] * 212)),
        attention_mask=stored(store, bytes([1] * 400 + [0] * 112)),
    )


@pytest.mark.parametrize("shared", [False, True])
def test_prefixes_and_full_values_share_verified_bytes_and_remain_detached(
    tmp_path: Path, shared: bool
) -> None:
    with ObjectStore(tmp_path) as store, RangeReader(local_root=tmp_path) as reader:
        value = record(store)
        if shared:
            value.attention_mask.CopyFrom(value.loss_mask)
        sequence = Sequence(value, reader)
        expected_attention = [True] * (300 if shared else 400) + [False] * (212 if shared else 112)
        with patch.object(reader, "read", wraps=reader.read) as read:
            for limit in (0, 1, 256, 512, 1000):
                tokens, mask, attention = sequence._preview(limit)
                assert tokens == list(range(512))[:limit]
                assert mask == ([True] * 300 + [False] * 212)[:limit]
                assert attention == expected_attention[:limit]
            tokens[0], mask[0], attention[0] = 999, False, False
            assert sequence.tokens == list(range(512))
            assert sequence.mask == [True] * 300 + [False] * 212
            assert sequence.attention_mask == expected_attention
            assert read.call_count == (2 if shared else 3)
            with patch.object(
                Sequence, "_token_values", side_effect=AssertionError("decoded full tokens again")
            ):
                assert sequence.tokens == list(range(512))


@pytest.mark.parametrize("kind", ["span", "object"])
def test_shared_mask_locations_keep_distinct_digest_expectations(tmp_path: Path, kind: str) -> None:
    with ObjectStore(tmp_path) as store, RangeReader(local_root=tmp_path) as reader:
        value = record(store)
        value.attention_mask.CopyFrom(value.loss_mask)
        if kind == "span":
            value.attention_mask.blake3_digest = b"x" * 32
        else:
            value.attention_mask.object.blake3_digest = b"x" * 32
        sequence = Sequence(value, reader)
        assert sequence.mask == [True] * 300 + [False] * 212
        with pytest.raises(ValueError, match="integrity"):
            _ = sequence.attention_mask


@pytest.mark.parametrize("field", ["loss_mask", "attention_mask"])
@pytest.mark.parametrize("invalid", [2, 255])
def test_invalid_mask_suffix_is_rejected_before_returning_a_prefix(
    tmp_path: Path, field: str, invalid: int
) -> None:
    with ObjectStore(tmp_path) as store, RangeReader(local_root=tmp_path) as reader:
        value = record(store)
        getattr(value, field).CopyFrom(stored(store, bytes([1] * 511 + [invalid])))
        with pytest.raises(ValueError, match="invalid stored .*mask"):
            Sequence(value, reader)._preview(256)


def test_token_corruption_beyond_the_prefix_is_rejected(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store, RangeReader(local_root=tmp_path) as reader:
        value = record(store)
        path = tmp_path / "dataset/objects" / value.tokens.object.blake3_digest.hex()
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with pytest.raises(ValueError, match="integrity"):
            Sequence(value, reader)._token_values(256)
