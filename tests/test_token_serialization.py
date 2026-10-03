"""Token shards use little-endian uint32 bytes without changing sequence values."""

import struct
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from blake3 import blake3

from premixdb.engine.datasets import Sequence
from premixdb.engine.identity import CodeVersion
from premixdb.engine.queries import CorpusIndex
from premixdb.engine.snapshots import Snapshot


@pytest.mark.parametrize("order", ["little", "big"])
@pytest.mark.parametrize("tokens", [[], [0, 255, 2**31, 2**32 - 1]])
def test_serialization_preserves_values_in_both_host_byte_orders(
    order: str, tokens: list[int]
) -> None:
    sequence = Sequence(0, tokens, [])
    # Represent the array memory as it would appear on the simulated host.
    if order != sys.byteorder:
        sequence._tokens.byteswap()
    memory = sequence._tokens.tobytes()
    with patch("premixdb.engine.datasets.byteorder", order):
        assert sequence._token_bytes() == struct.pack(f"<{len(tokens)}I", *tokens)
    assert sequence._tokens.tobytes() == memory


def test_shards_preserve_uint32_values_masks_and_digests(tmp_path: Path) -> None:
    code = CodeVersion("local://serialization", "a" * 40, "09" * 32)
    snapshot = Snapshot("01" * 16, [("a", "\0é")], code)
    query = CorpusIndex([snapshot]).execute([], code)
    dataset = query.dataset(8, 2**31, 2**32 - 1)
    tokens = [0, 195, 169, 2**31, *[2**32 - 1] * 4]
    expected = struct.pack("<8I", *tokens)
    mask = bytes([1] * 4 + [0] * 4)
    shards = dataset.write_token_shards(tmp_path)
    first, token_path, mask_path, digests, sequences = next(shards)
    assert first == 0
    assert token_path.read_bytes() == expected
    assert mask_path.read_bytes() == mask
    assert digests == [(blake3(expected).hexdigest(), blake3(mask).hexdigest())]
    assert sequences[0].tokens == tokens
    assert sequences[0].mask == [bool(value) for value in mask]
    assert list(shards) == []


@pytest.mark.parametrize("limit", [0, 2, 3, 4, 5, 6, 8])
def test_bounded_prefix_preserves_content_separator_and_padding_masks(limit: int) -> None:
    sequence = Sequence(
        0,
        [97, 98, 99, 256, 257, 257],
        [
            dict(start=0, end=3, kind="content"),
            dict(start=3, end=4, kind="separator"),
            dict(start=4, end=6, kind="padding"),
        ],
    )
    tokens, mask = sequence._preview(limit)
    assert tokens == [97, 98, 99, 256, 257, 257][:limit]
    assert mask == [1, 1, 1, 1, 0, 0][:limit]
    assert sequence.mask == [True, True, True, True, False, False]
