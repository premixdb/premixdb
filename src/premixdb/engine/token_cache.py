"""Ephemeral disk token pools keep model encodings out of corpus-sized RAM."""

from __future__ import annotations

import json
from contextlib import ExitStack
from threading import RLock
from typing import TYPE_CHECKING

from .._typing import checked_record, load_json
from .contracts import EncodedTokens
from .spill import _database
from .token_codec import decode_tokens, encode_tokens

if TYPE_CHECKING:
    from .datasets import ByteTokens, HuggingFaceTokenizer, TokenList
    from .queries import Query


def token_pool(query: Query, tokenizer: HuggingFaceTokenizer) -> TokenCache:
    from .datasets import encoded_tokens

    pool = TokenCache()
    encode = query._encoding_provider or encoded_tokens
    try:
        for row in query:
            if row.id not in pool:
                pool[row.id] = encode(row, tokenizer)
    except BaseException:
        pool.close()
        raise
    return pool


class TokenCache:
    def __init__(self) -> None:
        self.lock = RLock()
        with ExitStack() as startup:
            self.database = startup.enter_context(_database("token-pool", check_same_thread=False))
            self.database.execute(
                "CREATE TABLE tokens (id TEXT PRIMARY KEY, length INTEGER, data BLOB) WITHOUT ROWID"
            )
            self._lifetime = startup.pop_all()

    def __contains__(self, identity: str) -> bool:
        with self.lock:
            return (
                self.database.execute("SELECT 1 FROM tokens WHERE id=?", (identity,)).fetchone()
                is not None
            )

    def __setitem__(self, identity: str, tokens: ByteTokens | TokenList) -> None:
        data = json.dumps(encode_tokens(tokens), separators=(",", ":")).encode()
        with self.lock:
            self.database.execute(
                "INSERT INTO tokens VALUES (?,?,?)", (identity, len(tokens), data)
            )

    def length(self, identity: str) -> int:
        with self.lock:
            row = self.database.execute(
                "SELECT length FROM tokens WHERE id=?", (identity,)
            ).fetchone()
        if row is None:
            raise KeyError(identity)
        return int(row[0])

    def __getitem__(self, identity: str) -> ByteTokens | TokenList:
        with self.lock:
            row = self.database.execute(
                "SELECT data FROM tokens WHERE id=?", (identity,)
            ).fetchone()
        if row is None:
            raise KeyError(identity)
        return decode_tokens(checked_record(load_json(row[0]), EncodedTokens))

    def close(self) -> None:
        """Release temporary storage; repeated calls are safe."""
        with self.lock:
            self._lifetime.close()

    def __del__(self) -> None:
        if hasattr(self, "_lifetime"):
            self.close()
