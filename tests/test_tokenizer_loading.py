"""Admitted tokenizer assets load from bytes without filesystem staging."""

from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import coordinator
from blake3 import blake3

import premixdb as p
from premixdb.engine.datasets import HuggingFaceTokenizer
from premixdb.v1.storage_pb2 import ObjectRef


def test_coordinator_loads_inline_tokenizer_without_file_io(tmp_path: Path) -> None:
    path = Path(__file__).parent / "fixtures" / "wordpiece.json"
    data = path.read_bytes()
    digest = blake3(data)
    expected = HuggingFaceTokenizer(path, digest.hexdigest(), 1024)
    asset = ObjectRef(uri="inline://tokenizer", blake3_digest=digest.digest(), size_bytes=len(data))
    with p.PremixDB(storage=tmp_path) as db:
        service = coordinator(db)
        with (
            patch("tempfile.NamedTemporaryFile", side_effect=AssertionError("staged asset")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("reread asset")),
        ):
            loaded = service._tokenizer_asset(asset, 1024, inline=data)
        assert loaded.asset_bytes == expected.asset_bytes
        assert loaded.asset_digest == expected.asset_digest
        assert loaded.definition == expected.definition
        text = "hello world é秘密"
        tokens, offsets = expected.encode_with_offsets(text)
        assert loaded.encode(text) == tokens
        assert loaded.encode_with_offsets(text) == (tokens, offsets)
        assert loaded.decode(tokens) == expected.decode(tokens)
        with pytest.raises(NotImplementedError, match="whole-document limit"):
            loaded.encode("x" * 1025)


@pytest.mark.parametrize("invalid", ["digest", "json", "limit"])
def test_byte_constructor_validates_asset_and_limit(invalid: str) -> None:
    data = (Path(__file__).parent / "fixtures" / "wordpiece.json").read_bytes()
    if invalid == "digest":
        with pytest.raises(RuntimeError, match="digest mismatch"):
            HuggingFaceTokenizer.from_bytes(data, "00" * 32, 1024)
    elif invalid == "json":
        data = b"not JSON"
        with pytest.raises(ValueError, match="invalid tokenizer JSON"):
            HuggingFaceTokenizer.from_bytes(data, blake3(data).hexdigest(), 1024)
    else:
        with pytest.raises(ValueError, match="max_document_bytes must be positive"):
            HuggingFaceTokenizer.from_bytes(data, blake3(data).hexdigest(), 0)
