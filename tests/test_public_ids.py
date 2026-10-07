"""Public IDs round-trip through saved recipes, provenance, and reader state."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

import premixdb as p
from premixdb.schemas.ids import _decode_id, _encode_id
from premixdb.schemas.requests import _id


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
        snapshot = db.Corpus("ids", [p.Source(str(i), "hello world") for i in range(8)])
        assert len(snapshot.id) == 43 and len(snapshot.corpus_id) == 22
        assert db._snapshot(snapshot.id).id == snapshot.id
        assert db._snapshot(_decode_id(snapshot.id).hex()).id == snapshot.id
        assert db.Corpus.list()[0]["id"] == snapshot.corpus_id
        rows = snapshot.preview(limit=5, max_characters=0)
        assert len(rows) == 5
        query = snapshot.query(
            steps=[p.where(p.document_id.is_in([row["id"] for row in rows]))]
        ).wait()
        assert {row["id"] for row in query.preview(limit=5, max_characters=0)} == {
            row["id"] for row in rows
        }
        assert set(query._provenance()) == {
            row["id"] for row in snapshot.preview(limit=8, max_characters=0)
        }
        assert all(
            value["corpus_id"] == snapshot.corpus_id for value in query._provenance().values()
        )
        assert all(value["snapshots"] == [snapshot.id] for value in query._provenance().values())
        mix = query.mix(
            tokenizer=p.ByteTokenizer(),
            tokens=8,
            sequence_length=4,
            bounds=p.Bounds(lower={snapshot.corpus_id: 1}),
        )
        assert mix.weights == [{snapshot.corpus_id: 1.0}]
        assert f"Lower bounds: {{'{snapshot.corpus_id}': 1.0}}" in repr(mix)
        dataset = mix[0].wait()
        assert set(dataset.profile().source_tokens) == {snapshot.corpus_id}
        assert set(dataset.profile().stratum_tokens) == {snapshot.corpus_id}
        assert set(dataset[0].document_ids()) <= set(query._provenance())
        reader = dataset._reader()
        next(reader)
        checkpoint = reader.checkpoint()
        assert checkpoint["dataset"] == dataset.id
        older = checkpoint.copy()
        older["dataset"] = _decode_id(dataset.id).hex()
        assert [item.ordinal for item in dataset._reader(checkpoint=older)] == [
            item.ordinal for item in dataset._reader(checkpoint=checkpoint)
        ]
        assert db._executions(snapshot.id) == db._executions(_decode_id(snapshot.id).hex())
        assert db._executions(snapshot.id)[0].resource_id == snapshot.id
    finally:
        db.close()


def test_line_dedupe_witnesses_use_public_document_ids(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.Corpus(
            "dedupe", [p.Source("a", "same\nfirst"), p.Source("b", "same\nsecond")]
        )
        ids = {row["source_key"]: row["id"] for row in snapshot.preview()}
        query = snapshot.query(
            steps=[p.dedupe(algorithm=p.DedupeAlgorithm.EXACT_LINE, order_by=[p.object.uri.asc()])]
        )
        selection = query._provenance()[ids["b"]]["selection"]
        kept = selection["kept"]
        assert isinstance(kept, dict)
        assert kept["document"] == ids["a"]
        assert selection["matched"]["document"] == ids["b"]
        assert (kept["start"], kept["end"]) == (0, 4)


@pytest.mark.parametrize("granularity", ["document", "span"])
def test_contamination_witnesses_use_public_reference_ids(
    tmp_path: Path, granularity: Literal["document", "span"]
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        target = db.Corpus("target", [p.Source("a", "pré\n秘密\nfin")])
        reference = db.Corpus("reference", [p.Source("ref", "秘密")])
        query = target.query(
            decontaminate=p.Decontaminate(reference, algorithm="line", granularity=granularity)
        )
        origin = next(iter(query._provenance().values()))
        witness = origin["contamination"][0]
        assert witness["reference"] == reference.preview()[0]["id"]
        assert (witness["start"], witness["end"]) == (5, 11)
        if granularity == "document":
            assert origin["selection"]["references"] == origin["contamination"]
        else:
            assert origin["retained_ranges"] == [(0, 5), (11, 15)]


@pytest.mark.integration
def test_read_only_lineage_needs_no_selection_or_query_execution_modules(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        target = db.Corpus("target", [p.Source("a" * 64, "pré\n秘密\nfin")])
        reference = db.Corpus("reference", [p.Source("ref", "秘密")])
        query = target.query(
            decontaminate=p.Decontaminate(reference, algorithm="line", granularity="span")
        )
        lineage = query._provenance()
        assert next(iter(lineage.values()))["source_key"] == "a" * 64
        expected = json.dumps(lineage, sort_keys=True)
        identity = query.id
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys

sys.modules["premixdb.storage.selections"] = None
sys.modules["premixdb.engine.queries"] = None
import premixdb as p

with p.PremixDB(storage=sys.argv[1], read_only=True) as db:
    print(json.dumps(db._query(sys.argv[2])._provenance(), sort_keys=True))
""",
            str(tmp_path),
            identity,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.strip() == expected
