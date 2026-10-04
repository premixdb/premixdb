"""Pinned GPT-2 BPE assets for reproducible, offline demo tokenization."""

from functools import cache
from pathlib import Path

from blake3 import blake3

from .v1.data_mixture_pb2 import HuggingFaceTokenizer, Tokenizer
from .v1.storage_pb2 import ObjectRef

_GPT2_DIGEST = bytes.fromhex("1f9b61de3382db2e111c702730ef4ad5b12788d3c040db87936da6c7f988f861")
_GPT2_EOS = 50256


@cache
def _asset_bytes() -> bytes:
    data = (Path(__file__).parent / "data/gpt2-tokenizer.json").read_bytes()
    if blake3(data).digest() != _GPT2_DIGEST:
        raise ValueError("bundled GPT-2 tokenizer failed its integrity check")
    return data


def _gpt2_tokenizer() -> Tokenizer:
    data = _asset_bytes()
    return Tokenizer(
        hugging_face=HuggingFaceTokenizer(
            asset=ObjectRef(
                uri="inline://tokenizer", blake3_digest=_GPT2_DIGEST, size_bytes=len(data)
            ),
            json=data,
            max_document_bytes=8 * 1024 * 1024,
        )
    )
