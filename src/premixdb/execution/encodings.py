"""Verified, durable model-token encodings shared by profiles, pools and packing."""

from __future__ import annotations

from threading import RLock

from .. import _runtime
from .._protobuf import parse
from ..engine.datasets import ByteTokens, HuggingFaceTokenizer, TokenList, encoded_tokens
from ..engine.queries import Row
from ..internal import derivation_pb2 as d
from .catalog import identity, wire
from .storage import ObjectStore

SHARD_BYTES = 8 * 1024 * 1024


class Encodings:
    def __init__(self, storage: ObjectStore) -> None:
        self.storage = storage
        self.lock = RLock()

    def __call__(self, row: Row, tokenizer: HuggingFaceTokenizer) -> TokenList | ByteTokens:
        # A warmer cache must not bypass the caller's whole-document bound.
        if row.document.size > tokenizer.max_document_bytes:
            raise NotImplementedError("tokenizer input exceeds whole-document limit")
        selection = d.SelectedDocument(document_id=bytes.fromhex(row.id))
        source = getattr(row.document, "source_ranges", None)
        if source is not None:
            selection.transformed = True
            selection.ranges.extend(d.ByteRange(start=a, end=b) for a, b in source)
        key = identity(
            "premixdb-token-encoding/v1",
            bytes.fromhex(tokenizer.definition),
            _runtime.current_code().canonical_digest(),
            wire(selection),
        )
        with self.lock:
            try:
                manifest = self.storage.load("tokenizer", key, d.TokenEncoding, suffix=".encoding")
            except KeyError:
                tokens = encoded_tokens(row, tokenizer)
                manifest = d.TokenEncoding(id=key, tokens=len(tokens))
                shard, size = d.TokenEncodingShard(), 0

                def flush() -> None:
                    manifest.shards.append(self.storage.put("tokenizer", wire(shard)))
                    shard.Clear()

                for value, ranges in zip(tokens, tokens.ranges, strict=True):
                    token = d.EncodedToken(
                        value=value, ranges=[d.ByteRange(start=a, end=b) for a, b in ranges]
                    )
                    width = token.ByteSize() + 10
                    if shard.tokens and size + width > SHARD_BYTES:
                        flush()
                        size = 0
                    shard.tokens.append(token)
                    size += width
                if shard.tokens:
                    flush()
                self.storage.save("tokenizer", key, manifest, suffix=".encoding")
                return tokens
            if manifest.id != key:
                raise ValueError("token encoding belongs to different inputs")
            tokens, ranges = [], []
            bound = getattr(row.document, "original", row.document).size
            for ref in manifest.shards:
                try:
                    data = self.storage.read_object("tokenizer", ref)
                except KeyError as exc:
                    raise ValueError("cached token encoding shard is missing") from exc
                for token in parse(d.TokenEncodingShard, data).tokens:
                    if any(not 0 <= r.start <= r.end <= bound for r in token.ranges):
                        raise ValueError("cached token alignment is outside its source document")
                    tokens.append(token.value)
                    ranges.append([(r.start, r.end) for r in token.ranges])
            if len(tokens) != manifest.tokens:
                raise ValueError("cached token encoding has incomplete coverage")
            return TokenList(tokens, ranges)
