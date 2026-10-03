"""Ephemeral disk token pools keep model encodings out of corpus-sized RAM."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TYPE_CHECKING

from .._typing import json_integers, json_list, json_object, load_json

if TYPE_CHECKING:
    from .datasets import ByteTokens, HuggingFaceTokenizer, TokenList
    from .queries import Query


def token_pool(query: Query, tokenizer: HuggingFaceTokenizer) -> TokenCache:
    from .datasets import encoded_tokens

    pool = TokenCache()
    encode = query._encoding_provider or encoded_tokens
    for row in query._rows:
        if row.id not in pool:
            pool[row.id] = encode(row, tokenizer)
    return pool


class TokenCache:
    def __init__(self) -> None:
        self.directory = TemporaryDirectory(prefix="premixdb-token-pool-")
        self.database = sqlite3.connect(
            Path(self.directory.name) / "tokens.sqlite3", check_same_thread=False
        )
        self.lock = RLock()
        self.database.execute("PRAGMA cache_size=-8192")
        self.database.execute(
            "CREATE TABLE tokens (id TEXT PRIMARY KEY, length INTEGER, data BLOB) WITHOUT ROWID"
        )

    def __contains__(self, identity: str) -> bool:
        with self.lock:
            return (
                self.database.execute("SELECT 1 FROM tokens WHERE id=?", (identity,)).fetchone()
                is not None
            )

    def __setitem__(self, identity: str, tokens: ByteTokens | TokenList) -> None:
        data = json.dumps(
            dict(tokens=list(tokens), ranges=tokens.ranges), separators=(",", ":")
        ).encode()
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

    def __getitem__(self, identity: str) -> TokenList:
        from .datasets import TokenList

        with self.lock:
            row = self.database.execute(
                "SELECT data FROM tokens WHERE id=?", (identity,)
            ).fetchone()
        if row is None:
            raise KeyError(identity)
        data = json_object(load_json(row[0]))
        ranges = []
        for token in json_list(data["ranges"]):
            intervals = []
            for interval in json_list(token):
                start, end = json_integers(interval)
                intervals.append((start, end))
            ranges.append(intervals)
        return TokenList(json_integers(data["tokens"]), ranges)

    def __del__(self) -> None:
        if hasattr(self, "database"):
            self.database.close()
        if hasattr(self, "directory"):
            self.directory.cleanup()
