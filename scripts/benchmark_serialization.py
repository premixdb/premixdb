"""Measure final private codecs against the previous JSON boundaries.

Run: uv run --locked python scripts/benchmark_serialization.py --output reports/serialization.json
Timing excludes tokenization/model inference. CPU detection is left unchanged.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import platform
import random
import sqlite3
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import google.protobuf
from blake3 import blake3
from google.protobuf.internal import api_implementation
from google.protobuf.message import Message

from premixdb.contracts import JSON, checked_record, field_value, json_list, json_object, load_json
from premixdb.engine.contracts import EncodedTokens
from premixdb.engine.datasets import HuggingFaceTokenizer, TokenList
from premixdb.engine.spill import _database
from premixdb.engine.token_cache import TokenCache
from premixdb.engine.token_codec import (
    decode_tokens,
    encode_tokens,
    load_tokens,
    packed_tokens,
    unpack_tokens,
)
from premixdb.engine.value_cache import ValueCache
from premixdb.enrichment.dupekit import DupekitIndex
from premixdb.enrichment.types import Document
from premixdb.internal import derivation_pb2 as e
from premixdb.internal import transport_pb2 as t
from premixdb.runtime.enrichment import DedupeRow
from premixdb.runtime.partition_types import request, result
from premixdb.schemas.binary import CODEC_VERSION, decode_value, encode_value, known, parse
from premixdb.schemas.protobuf import wire


class LegacyInput(TypedDict):
    id: str
    text: str
    ordinal: int
    ranges: list[list[int]]


class LegacyToken(EncodedTokens):
    id: str
    ordinal: int


class LegacyIndex(TypedDict):
    id: str
    exact_hash: str
    minhash: list[int] | None
    lsh_buckets: list[int] | None


def json_wire(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def measure(cases: dict[str, Callable[[], object]], repeats: int) -> dict[str, JSON]:
    samples: dict[str, list[float]] = {name: [] for name in cases}
    for case in cases.values():
        case()
    for iteration in range(repeats):
        names = list(cases)
        random.Random(iteration).shuffle(names)
        for name in names:
            gc.collect()
            started = time.perf_counter()
            outcome = cases[name]()
            samples[name].append((time.perf_counter() - started) * 1000)
            del outcome
    return {
        name: {"median_ms": statistics.median(values), "samples_ms": cast(list[JSON], values)}
        for name, values in samples.items()
    }


def boundary(
    json_convert: Callable[[], object],
    proto_convert: Callable[[], Message],
    json_decode: Callable[[bytes], object],
    proto_decode: Callable[[bytes], object],
    repeats: int,
) -> dict[str, JSON]:
    json_value, proto_value = json_convert(), proto_convert()
    json_data, proto_data = json_wire(json_value), wire(proto_value)
    assert json_decode(json_data) == proto_decode(proto_data)
    timing = measure(
        {
            "json_conversion": json_convert,
            "protobuf_conversion": proto_convert,
            "json_encoding": lambda: json_wire(json_value),
            "protobuf_encoding": lambda: wire(proto_value),
            "json_decoding": lambda: json_decode(json_data),
            "protobuf_decoding": lambda: proto_decode(proto_data),
            "json_round_trip": lambda: json_decode(json_wire(json_convert())),
            "protobuf_round_trip": lambda: proto_decode(wire(proto_convert())),
        },
        repeats,
    )
    timing["json_bytes"], timing["protobuf_bytes"] = len(json_data), len(proto_data)
    return timing


def database_bytes(database: sqlite3.Connection) -> int:
    return Path(database.execute("PRAGMA database_list").fetchone()[2]).stat().st_size


def cache_benchmark(
    tokens: list[TokenList], vectors: list[list[float]], repeats: int
) -> dict[str, JSON]:
    sizes: dict[str, int] = {}
    with (
        _database("benchmark-token-json") as old_tokens,
        _database("benchmark-vector-json") as old_vectors,
    ):
        old_tokens.execute(
            "CREATE TABLE tokens (id TEXT PRIMARY KEY,length INTEGER,data BLOB) WITHOUT ROWID"
        )
        old_vectors.execute(
            "CREATE TABLE values_index (id TEXT PRIMARY KEY,value TEXT) WITHOUT ROWID"
        )
        new_tokens = TokenCache()
        new_vectors = ValueCache()
        try:

            def token_build(codec: str) -> None:
                database = old_tokens if codec == "json" else new_tokens.database
                database.execute("DELETE FROM tokens")
                for i, value in enumerate(tokens):
                    if codec == "json":
                        database.execute(
                            "INSERT INTO tokens VALUES (?,?,?)",
                            (str(i), len(value), json_wire(encode_tokens(value))),
                        )
                    else:
                        new_tokens[str(i)] = value
                database.commit()
                sizes[codec + "_tokens"] = database_bytes(database)

            def vector_build(codec: str) -> None:
                database = old_vectors if codec == "json" else new_vectors._database
                database.execute("DELETE FROM values_index")
                for i, value in enumerate(vectors):
                    if codec == "json":
                        database.execute(
                            "INSERT INTO values_index VALUES (?,?)",
                            (
                                str(i),
                                json.dumps(
                                    value,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                ),
                            ),
                        )
                    else:
                        new_vectors[str(i)] = cast(list[str | int | float | bool | None], value)
                database.commit()
                sizes[codec + "_vectors"] = database_bytes(database)

            selected = random.Random(42).sample(range(len(tokens)), min(64, len(tokens)))

            def token_lookup(codec: str) -> object:
                if codec == "protobuf":
                    return [new_tokens[str(i)] for i in selected]
                return [
                    decode_tokens(
                        checked_record(
                            load_json(
                                old_tokens.execute(
                                    "SELECT data FROM tokens WHERE id=?", (str(i),)
                                ).fetchone()[0]
                            ),
                            EncodedTokens,
                        )
                    )
                    for i in selected
                ]

            def vector_lookup(codec: str) -> object:
                if codec == "protobuf":
                    return [new_vectors[str(i)] for i in range(64)]
                return [
                    field_value(
                        load_json(
                            old_vectors.execute(
                                "SELECT value FROM values_index WHERE id=?", (str(i),)
                            ).fetchone()[0]
                        )
                    )
                    for i in range(64)
                ]

            timings = measure(
                {
                    "json_token_insert_commit": lambda: token_build("json"),
                    "protobuf_token_insert_commit": lambda: token_build("protobuf"),
                    "json_vector_insert_commit": lambda: vector_build("json"),
                    "protobuf_vector_insert_commit": lambda: vector_build("protobuf"),
                },
                repeats,
            )
            for value, i in zip(tokens, range(len(tokens)), strict=True):
                assert list(new_tokens[str(i)]) == list(value)
                assert new_tokens[str(i)].ranges == value.ranges
            assert list(new_vectors.items()) == sorted(
                ((str(i), value) for i, value in enumerate(vectors))
            )
            timings.update(
                measure(
                    {
                        "json_64_token_lookups": lambda: token_lookup("json"),
                        "protobuf_64_token_lookups": lambda: token_lookup("protobuf"),
                        "json_64_vector_lookups": lambda: vector_lookup("json"),
                        "protobuf_64_vector_lookups": lambda: vector_lookup("protobuf"),
                    },
                    repeats,
                )
            )
            timings["committed_database_bytes"] = cast(dict[str, JSON], sizes)
            return timings
        finally:
            new_tokens.close()
            new_vectors.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    corpus = (root / "src/premixdb/data/tiny_shakespeare.txt").read_text()
    texts = [
        corpus[(i * 7919) % (len(corpus) - 4096) :][:4096]
        + ("\nCafé — 東京 🌍" if i % 8 == 0 else "")
        for i in range(128)
    ]
    asset = (root / "src/premixdb/data/gpt2-tokenizer.json").read_bytes()
    tokenizer = HuggingFaceTokenizer.from_bytes(asset, blake3(asset).hexdigest(), 8 * 1024 * 1024)
    tokens = [
        TokenList(values, [[pair] for pair in offsets])
        for values, offsets in map(tokenizer.encode_with_offsets, texts)
    ]
    documents = [Document(f"{i:064x}", text) for i, text in enumerate(texts)]
    evidence = [
        checked_record(row, DedupeRow) for row in DupekitIndex().compute(documents).to_pylist()
    ]

    def text_json() -> object:
        return dict(
            version=1,
            tokenizer="",
            rows=[
                dict(id=doc.id, text=doc.text, ordinal=i, ranges=[[0, len(doc.text.encode())]])
                for i, doc in enumerate(documents)
            ],
        )

    def text_proto() -> Message:
        value = t.WorkerRequest(version=CODEC_VERSION)
        value.tokenize.tokenizer.byte.SetInParent()
        value.tokenize.rows.extend(
            t.InputRow(
                id=doc.id,
                text=doc.text,
                ordinal=i,
                ranges=[e.ByteRange(end=len(doc.text.encode()))],
            )
            for i, doc in enumerate(documents)
        )
        return value

    def text_json_read(data: bytes) -> object:
        return [
            (row["id"], row["text"], row["ordinal"], row["ranges"])
            for raw in json_list(json_object(load_json(data))["rows"])
            for row in [checked_record(raw, LegacyInput)]
        ]

    def text_proto_read(data: bytes) -> object:
        return [
            (row.id, row.text, row.ordinal, [[r.start, r.end] for r in row.ranges])
            for row in request(data, "tokenize").tokenize.rows
        ]

    def tokens_json() -> object:
        return dict(
            version=1,
            rows=[
                dict(id=doc.id, ordinal=i, **encode_tokens(value))
                for i, (doc, value) in enumerate(zip(documents, tokens, strict=True))
            ],
        )

    def tokens_proto() -> Message:
        value = t.WorkerResult(version=CODEC_VERSION)
        value.tokenize.rows.extend(
            t.TokenRow(id=doc.id, ordinal=i, encoding=packed_tokens(token))
            for i, (doc, token) in enumerate(zip(documents, tokens, strict=True))
        )
        return value

    def tokens_json_read(data: bytes) -> object:
        return [
            (row["id"], row["ordinal"], list(decoded), list(decoded.ranges))
            for raw in json_list(json_object(load_json(data))["rows"])
            for row in [checked_record(raw, LegacyToken)]
            for decoded in [decode_tokens(row)]
        ]

    def tokens_proto_read(data: bytes) -> object:
        value = result(data, "tokenize")
        known(value.tokenize)
        rows = []
        for row in value.tokenize.rows:
            known(row)
            if not row.id:
                raise ValueError("missing token document identity")
            decoded = unpack_tokens(row.encoding)
            rows.append((row.id, row.ordinal, list(decoded), list(decoded.ranges)))
        return rows

    def evidence_json() -> object:
        return dict(
            version=1,
            rows=[
                dict(row, exact_hash=base64.b64encode(row["exact_hash"]).decode())
                for row in evidence
            ],
        )

    def evidence_proto() -> Message:
        value = t.WorkerResult(version=CODEC_VERSION)
        value.indexes.SetInParent()
        for row in evidence:
            target = value.indexes.rows.add(id=row["id"], exact_hash=row["exact_hash"])
            for name in ("minhash", "lsh_buckets"):
                values = row[name]
                if values is not None:
                    column = getattr(target, name)
                    column.SetInParent()
                    column.values.extend(values)
        return value

    def evidence_json_read(data: bytes) -> object:
        return [
            (
                row["id"],
                base64.b64decode(row["exact_hash"], validate=True),
                row["minhash"],
                row["lsh_buckets"],
            )
            for raw in json_list(json_object(load_json(data))["rows"])
            for row in [checked_record(raw, LegacyIndex)]
        ]

    def evidence_proto_read(data: bytes) -> object:
        value = result(data, "indexes")
        known(value.indexes)
        rows = []
        for row in value.indexes.rows:
            known(row)
            known(row.minhash)
            known(row.lsh_buckets)
            if not row.id or len(row.exact_hash) != 32:
                raise ValueError("invalid partition dedupe evidence")
            rows.append(
                (
                    row.id,
                    row.exact_hash,
                    list(row.minhash.values) if row.HasField("minhash") else None,
                    list(row.lsh_buckets.values) if row.HasField("lsh_buckets") else None,
                )
            )
        return rows

    rng = random.Random(41)
    vectors = [[rng.uniform(-2, 2) for _ in range(1024)] for _ in range(512)]
    token_message = packed_tokens(tokens[0])
    token_json = encode_tokens(tokens[0])
    vector_message = encode_value(cast(list[str | int | float | bool | None], vectors[0]))
    token_data, token_json_data = wire(token_message), json_wire(token_json)
    vector_data, vector_json_data = wire(vector_message), json_wire(vectors[0])
    report: dict[str, JSON] = {
        "environment": {
            "python": platform.python_version(),
            "protobuf": google.protobuf.__version__,
            "protobuf_backend": api_implementation.Type(),
            "detected_cpus": os.cpu_count(),
            "repeats": args.repeats,
        },
        "workload": {
            "documents": len(texts),
            "utf8_bytes": sum(len(text.encode()) for text in texts),
            "tokens": sum(map(len, tokens)),
            "vectors": 512,
            "vector_width": 1024,
        },
        "worker_text": boundary(
            text_json, text_proto, text_json_read, text_proto_read, args.repeats
        ),
        "worker_tokens": boundary(
            tokens_json, tokens_proto, tokens_json_read, tokens_proto_read, args.repeats
        ),
        "worker_dedupe": boundary(
            evidence_json, evidence_proto, evidence_json_read, evidence_proto_read, args.repeats
        ),
        "cache_boundary": measure(
            {
                "protobuf_token_conversion": lambda: packed_tokens(tokens[0]),
                "json_token_conversion": lambda: encode_tokens(tokens[0]),
                "protobuf_token_encoding": lambda: wire(token_message),
                "json_token_encoding": lambda: json_wire(token_json),
                "protobuf_token_decoding": lambda: load_tokens(token_data),
                "json_token_decoding": lambda: decode_tokens(
                    checked_record(load_json(token_json_data), EncodedTokens)
                ),
                "protobuf_vector_conversion": lambda: encode_value(
                    cast(list[str | int | float | bool | None], vectors[0])
                ),
                "protobuf_vector_encoding": lambda: wire(vector_message),
                "json_vector_encoding": lambda: json_wire(vectors[0]),
                "protobuf_vector_decoding": lambda: decode_value(
                    parse(vector_data, t.TransportValue())
                ),
                "json_vector_decoding": lambda: field_value(load_json(vector_json_data)),
            },
            args.repeats,
        ),
        "caches": cache_benchmark(tokens, vectors, args.repeats),
        "limits": "Boundary timings only; SQLite writes include commit, lookups are warm. Excludes process dispatch, model inference, cold disk and peak memory. No compression or quantization.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
