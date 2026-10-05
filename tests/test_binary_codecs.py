"""Private codecs retain exact values and reject invalid domain payloads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from premixdb.contracts import FieldValue
from premixdb.engine.datasets import ByteRanges, ByteTokens, TokenList
from premixdb.engine.identity import CodeVersion
from premixdb.engine.queries import CorpusIndex
from premixdb.engine.snapshots import Snapshot
from premixdb.engine.token_cache import TokenCache
from premixdb.engine.token_codec import dump_tokens, load_tokens, packed_length, packed_tokens
from premixdb.engine.value_cache import ValueCache
from premixdb.internal import derivation_pb2 as e
from premixdb.internal import transport_pb2 as t
from premixdb.runtime.partition_types import alignment, request, result, spans
from premixdb.runtime.partitions import Kernel, PartitionStore, PartitionTask
from premixdb.runtime.pipeline import PartitionPipeline, processor
from premixdb.schemas.binary import CODEC_VERSION, decode_value, encode_value, parse
from premixdb.schemas.protobuf import wire
from premixdb.v1 import data_mixture_pb2 as d


@pytest.mark.parametrize(
    "tokens",
    [
        TokenList([], []),
        TokenList(
            [0, 2**32 - 1, 17, 9], [[], [(0, 1), (3, 5)], [(2**64 - 2, 2**64 - 1)], [(9, 9)]]
        ),
        TokenList([1, 2], [[(0, 3), (1, 2)], [(0, 3)]]),
        ByteTokens(b"", ByteRanges([])),
        ByteTokens("pré 🌍".encode(), ByteRanges([(2**64 - 12, 2**64 - 3)])),
        ByteTokens(b"abcdef", ByteRanges([(1, 4), (8, 11)]))[1:5],
    ],
)
def test_tokens_round_trip_and_cache(tokens: TokenList | ByteTokens) -> None:
    decoded = load_tokens(dump_tokens(tokens))
    assert type(decoded) is type(tokens)
    assert list(decoded) == list(tokens)
    assert list(decoded.ranges) == list(tokens.ranges)
    assert packed_length(packed_tokens(tokens)) == len(tokens)
    cache = TokenCache()
    try:
        cache["a"] = tokens
        assert "a" in cache and "b" not in cache
        # Length remains independent of decoding the BLOB.
        with patch("premixdb.engine.token_cache.load_tokens", side_effect=AssertionError):
            assert cache.length("a") == len(tokens)
        assert list(cache["a"].ranges) == list(tokens.ranges)
        with pytest.raises(KeyError):
            cache["b"]
        with pytest.raises(KeyError):
            cache.length("b")
    finally:
        cache.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: setattr(m, "version", 0),
        lambda m: m.ClearField("numeric"),
        lambda m: m.numeric.interval_offsets.pop(),
        lambda m: m.numeric.interval_offsets.__setitem__(0, 1),
        lambda m: m.numeric.interval_offsets.__setitem__(-1, 0),
        lambda m: m.numeric.interval_offsets.__setitem__(1, 2),
        lambda m: m.numeric.ends.pop(),
        lambda m: m.numeric.ends.__setitem__(0, 0),
    ],
)
def test_invalid_packed_layout(mutate: Callable[[t.PackedTokens], object]) -> None:
    message = packed_tokens(TokenList([1, 2, 3], [[(1, 2)], [], []]))
    mutate(message)
    with pytest.raises(ValueError):
        load_tokens(wire(message))


def test_invalid_bytes_and_truncated_or_unknown_wire() -> None:
    message = packed_tokens(ByteTokens(b"a", ByteRanges([(0, 1)])))
    message.byte.ends[0] = 2
    with pytest.raises(ValueError, match="coverage"):
        load_tokens(wire(message))
    valid = dump_tokens(TokenList([1], [[(0, 1)]]))
    for data in (valid[:-1], b"\xff", valid + b"\xa0\x06\x01"):
        with pytest.raises(ValueError):
            load_tokens(data)
    for token in (-1, 2**32):
        with pytest.raises(ValueError):
            dump_tokens(TokenList([token], [[]]))
    for endpoint in (-1, 2**64):
        with pytest.raises(ValueError):
            dump_tokens(TokenList([1], [[(endpoint, endpoint)]]))
    with pytest.raises(ValueError, match="coverage"):
        dump_tokens(TokenList([1], []))


VALUES: list[FieldValue] = [
    None,
    False,
    True,
    0,
    -(2**63),
    2**63 - 1,
    2**64 - 1,
    -(2**100),
    2**100,
    1.0,
    -0.0,
    "pré 🌍",
    "",
    [],
    [1.0, -0.0, 1.23456789012345],
    [True, 1, 1.0, None, "a", 2**64 - 1],
]


@pytest.mark.parametrize("value", VALUES)
def test_exact_values_and_cache_mapping(value: FieldValue) -> None:
    decoded = decode_value(parse(wire(encode_value(value)), t.TransportValue()))
    assert decoded == value and type(decoded) is type(value)
    if isinstance(value, list):
        assert isinstance(decoded, list)
        assert list(map(type, decoded)) == list(map(type, value))
        for before, after in zip(value, decoded, strict=True):
            if type(before) is float and before == 0:
                assert isinstance(after, float)
                assert math.copysign(1, before) == math.copysign(1, after)
    cache = ValueCache([("a", value)])
    try:
        assert cache["a"] == value and type(cache["a"]) is type(value)
        assert list(cache) == ["a"] and len(cache) == 1
        assert list(cache.items()) == [("a", value)]
        assert cache.get("missing") is None
        cache["b"] = True
        cache["a"] = 2**64 - 1
        assert dict(cache.items()) == {"a": 2**64 - 1, "b": True}
        assert cache.pop("b") is True
        del cache["a"]
        assert not cache
        with pytest.raises(KeyError):
            cache["a"]
        with pytest.raises(KeyError):
            del cache["a"]
        cache.update({"a": value})
        if type(value) is float and value == 0:
            cached = cache["a"]
            assert isinstance(cached, float)
            assert math.copysign(1, cached) == math.copysign(1, value)
    finally:
        cache.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [1.0, float("nan")]])
def test_nonfinite_cache_values_fail(value: FieldValue) -> None:
    with pytest.raises(ValueError, match="finite"):
        ValueCache([("a", value)])


@pytest.mark.parametrize(
    "payload",
    [
        t.TransportValue(),
        t.TransportValue(field=e.FieldValue(null=False)),
        t.TransportValue(field=e.FieldValue(document_id=b"unexpected", integer=1)),
        t.TransportValue(field=e.FieldValue(number=float("inf"))),
        t.TransportValue(field=e.FieldValue(vector=e.NumericVector(values=[float("nan")]))),
        t.TransportValue(big_integer="1e2"),
        t.TransportValue(big_integer="01"),
        t.TransportValue(
            scalars=t.ScalarValues(values=[t.TransportValue(scalars=t.ScalarValues())])
        ),
    ],
)
def test_invalid_field_outcomes(payload: t.TransportValue) -> None:
    with pytest.raises(ValueError):
        decode_value(parse(wire(payload), t.TransportValue()))


@pytest.mark.parametrize("kind", [int, float])
def test_numeric_subclasses_keep_json_compatible_values(kind: type[int] | type[float]) -> None:
    # JSON normalized numeric subclasses to builtins; retain that mapping contract.
    if kind is int:
        from enum import IntEnum

        class Number(IntEnum):
            VALUE = 2**64 - 1

        value: int | float = Number.VALUE
    else:

        class Float(float):
            pass

        value = Float(1.25)
    decoded = decode_value(encode_value(value))
    assert decoded == value and type(decoded) is kind
    cache = ValueCache([("a", value)])
    try:
        assert cache["a"] == value and type(cache["a"]) is kind
        cache["a"] = value
        assert list(cache.items()) == [("a", value)]
    finally:
        cache.close()


def test_cache_corruption_fails_in_lookup_and_items() -> None:
    cache = ValueCache([("a", 1)])
    try:
        cache._database.execute("UPDATE values_index SET kind='unknown'")
        with pytest.raises(ValueError, match="invalid cached value"):
            cache["a"]
        with pytest.raises(ValueError, match="invalid cached value"):
            list(cache.items())
    finally:
        cache.close()


def test_request_presence_version_and_semantics() -> None:
    for message in (
        t.WorkerRequest(),
        t.WorkerRequest(version=99, pack=t.PackingRequest(length=1)),
    ):
        with pytest.raises(ValueError):
            request(wire(message), "pack")
    message = t.WorkerRequest(
        version=CODEC_VERSION, pack=t.PackingRequest(length=1, total_sequences=1)
    )
    assert not request(wire(message), "pack").pack.HasField("separator")
    message.pack.separator = 0
    message.pack.padding = 0
    assert request(wire(message), "pack").pack.HasField("separator")
    message.pack.length = 0
    with pytest.raises(ValueError, match="geometry"):
        request(wire(message), "pack")
    for message in (
        t.WorkerResult(),
        t.WorkerResult(version=CODEC_VERSION, features=t.FeatureRows()),
    ):
        with pytest.raises(ValueError):
            result(wire(message), "tokenize")


def test_index_null_and_empty_and_uint64_remain_distinct(tmp_path: Path) -> None:
    pipeline = PartitionPipeline(PartitionStore(tmp_path))
    policy = e.EnrichmentProducer(dupekit=e.DupekitProducer())
    message = t.WorkerResult(version=CODEC_VERSION)
    message.indexes.rows.add(id="null", exact_hash=b"x" * 32)
    empty = message.indexes.rows.add(id="empty", exact_hash=b"y" * 32)
    empty.minhash.SetInParent()
    empty.lsh_buckets.SetInParent()
    wide = message.indexes.rows.add(id="wide", exact_hash=b"z" * 32)
    wide.minhash.values.append(2**64 - 1)
    wide.lsh_buckets.values.append(2**64 - 1)
    try:
        artifact = pipeline._input(wire(message))
        with patch.object(pipeline, "_execute", return_value=(artifact,)):
            rows = list(pipeline.features(policy, b"{}", [[]]))[0]
        assert rows == [
            dict(id="null", exact_hash=b"x" * 32, minhash=None, lsh_buckets=None),
            dict(id="empty", exact_hash=b"y" * 32, minhash=[], lsh_buckets=[]),
            dict(id="wide", exact_hash=b"z" * 32, minhash=[2**64 - 1], lsh_buckets=[2**64 - 1]),
        ]
    finally:
        pipeline.close()


def test_feature_worker_preserves_nullable_outcomes_and_urls(tmp_path: Path) -> None:
    from test_enrichment_service import ControlledFields

    policy = e.EnrichmentProducer()
    policy.language.SetInParent()
    message = t.WorkerRequest(version=CODEC_VERSION)
    message.features.producer.CopyFrom(policy)
    message.features.definition = b'{"provider":"test-controlled-model","version":1}'
    message.features.rows.add(id="empty", text="", url="")
    message.features.rows.add(id="unicode", text="pré 🌍", url="https://example.org/b")
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_bytes(wire(message))
    task = PartitionTask(b"k" * 32, b"e" * 32, Kernel.FEATURES, (), output.as_uri())
    # Avoid the process-local producer cache influencing this controlled worker.
    with (
        patch("premixdb.runtime.pipeline._FEATURE_WORKERS", {}),
        patch("premixdb.runtime.enrichment.producer", ControlledFields),
    ):
        processor(task, (source,), output)
    decoded = result(output.read_bytes(), "features")
    rows = [
        {key: decode_value(value) for key, value in row.values.items()}
        for row in decoded.features.rows
    ]
    assert rows[0]["language.en"] is None
    assert rows[1]["language.en"] == 0.95


def test_packed_provenance_semantics() -> None:
    row = t.PackedRow(
        tokens=[1],
        spans=[t.PackedSpan(start=0, end=1, kind="content", occurrence=0, offset=0)],
        alignment=[t.TokenRange(token=0, occurrence=0, start=1, end=2)],
    )
    assert spans(row)[0]["offset"] == 0
    assert alignment(row)[0]["occurrence"] == 0
    row.alignment[0].occurrence = 1
    with pytest.raises(ValueError, match="provenance"):
        alignment(row)
    row.spans[0].start = 1
    with pytest.raises(ValueError, match="coverage"):
        spans(row)


@pytest.mark.integration
def test_actual_worker_tokenization_restart_and_packing(tmp_path: Path) -> None:
    code = CodeVersion("local://test", "a" * 40, "09" * 32)
    snapshot = Snapshot("01" * 16, [("a", "pré 🌍"), ("b", ""), ("c", "hello")], code)
    query = CorpusIndex([snapshot]).execute([], code)
    policy = d.Tokenizer()
    policy.byte.SetInParent()
    expected = [list(sequence.tokens) for sequence in query.dataset(4, 0, 0).iter_sequences()]
    for _ in range(2):
        pipeline = PartitionPipeline(PartitionStore(tmp_path / "partitions"), workers=2)
        try:
            lengths, encoded = pipeline.tokenize(query, policy)
            assert lengths == query.lengths()
            assert [(row.id, bytes(tokens)) for row, tokens in encoded] == [
                (row.id, row.text.encode()) for row in query
            ]
            dataset = pipeline.pack(query.dataset(4, 0, 0, stream=True))
            sequences = list(dataset.iter_sequences())
            assert [list(sequence.tokens) for sequence in sequences] == expected
        finally:
            pipeline.close()
