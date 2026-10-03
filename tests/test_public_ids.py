"""Public IDs round-trip through saved recipes, provenance, and reader state."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import premixdb as p
from premixdb._ids import _decode_id, _encode_id
from premixdb._requests import _id


@pytest.mark.parametrize("size", [16, 32])
@pytest.mark.parametrize("raw", [0, 255, 17])
def test_base64url_ids_round_trip_without_padding(size: int, raw: int) -> None:
    identity = bytes([raw]) * size
    encoded = _encode_id(identity)
    assert len(encoded) == (22 if size == 16 else 43)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", encoded)
    assert _id(encoded, size) == identity
    assert _id(identity.hex(), size) == identity


@pytest.mark.parametrize("value", ["!" * 43, "A", "A" * 42 + "B", "A" * 44, None])
def test_invalid_ids_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        _id(value, 32)


def test_public_ids_match_listings_provenance_profiles_and_old_checkpoints(tmp_path: Path) -> None:
    db = p.PremixDB(storage=tmp_path)
    try:
        snapshot = db.corpus("ids", [p.Source(str(i), "hello world") for i in range(8)])
        assert len(snapshot.id) == 43 and len(snapshot.corpus_id) == 22
        assert db._snapshot(snapshot.id).id == snapshot.id
        assert db._snapshot(_decode_id(snapshot.id).hex()).id == snapshot.id
        assert db.corpus.list()[0]["id"] == snapshot.corpus_id
        rows = snapshot.preview(limit=5, max_characters=0)
        assert len(rows) == 5
        query = snapshot.query(
            steps=[p.where(p.document_id.is_in([row["id"] for row in rows]))]
        ).wait()
        assert {row["id"] for row in query._list_document()} == {row["id"] for row in rows}
        assert set(query._provenance()) == {
            row["id"] for row in snapshot.preview(limit=8, max_characters=0)
        }
        assert all(
            value["corpus_id"] == snapshot.corpus_id for value in query._provenance().values()
        )
        assert all(value["snapshots"] == [snapshot.id] for value in query._provenance().values())
        mix = query.mix(tokens=8, sequence_length=4, bounds=p.Bounds(lower={snapshot.corpus_id: 1}))
        assert mix.weights == [{snapshot.corpus_id: 1.0}] * 3
        assert f"Lower bounds: {{'{snapshot.corpus_id}': 1.0}}" in repr(mix)
        dataset = mix[0].wait()
        assert set(dataset.profile().source_tokens) == {snapshot.corpus_id}
        assert set(dataset.profile().stratum_tokens) == {snapshot.corpus_id}
        assert set(dataset[0].document_ids()) <= set(query._provenance())
        reader = dataset.reader()
        next(reader)
        checkpoint = reader.checkpoint()
        assert checkpoint["dataset"] == dataset.id
        older = checkpoint.copy()
        older["dataset"] = _decode_id(dataset.id).hex()
        assert [item.ordinal for item in dataset.reader(checkpoint=older)] == [
            item.ordinal for item in dataset.reader(checkpoint=checkpoint)
        ]
        assert db._executions(snapshot.id) == db._executions(_decode_id(snapshot.id).hex())
        assert db._executions(snapshot.id)[0].resource_id == snapshot.id
    finally:
        db.close()
