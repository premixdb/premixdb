"""Packaged demo values support real queries without inference or downloads."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb.cli.main import _benchmark_sources, _demo_sources
from premixdb.enrichment import QuRating, WebOrganizer
from premixdb.enrichment.types import ComputedRow, Document
from premixdb.internal import derivation_pb2 as e
from premixdb.runtime import catalog, demo_enrichment, enrichment
from premixdb.runtime import environment as runtime


def test_demo_scores_match_sources_and_schemas() -> None:
    fixture = json.loads(files("premixdb").joinpath("data/demo-enrichment.json").read_bytes())
    sources = _demo_sources()
    hashes = {source.key: sha256(source.text.encode()).hexdigest() for source in sources}
    for name in ("quality.writing_style", "weborganizer.topic"):
        policy = catalog.recipe(name)
        worker = enrichment.producer(policy)
        assert isinstance(worker, (QuRating, WebOrganizer))
        saved = demo_enrichment.load(
            policy, worker.fields, [Document(source.key, source.text) for source in sources]
        )
        assert saved is not None
        assert len(saved.rows) == 8
        assert len(saved.rows[0][1]) == (4 if isinstance(worker, QuRating) else 1)
        for entry in fixture["producers"]:
            assert {row["source_key"]: row["text_sha256"] for row in entry["rows"]} == hashes
        if isinstance(worker, WebOrganizer):
            for _, values in saved.rows:
                logits = values["weborganizer.topic"]
                assert isinstance(logits, list) and len(logits) == 24


def test_readme_demo_runs_without_models_and_reopens(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    with (
        patch("premixdb.enrichment.models._sequence_model", side_effect=AssertionError("model")),
        patch.object(QuRating, "compute", side_effect=AssertionError("quality inference")),
        patch.object(WebOrganizer, "compute", side_effect=AssertionError("topic inference")),
    ):
        with p.PremixDB(storage=tmp_path, progress=False) as db:
            db.Corpus("demo", _demo_sources())
            db.Corpus("benchmark", _benchmark_sources())
            query = db.Corpus("demo").query(
                steps=[
                    p.where(p.text.characters >= 100),
                    p.dedupe(),
                    p.where(p.quality.writing_style >= 0.8),
                ],
                decontaminate=p.Decontaminate(db.Corpus("benchmark")),
            )
            mixture = query.mix(
                domains=p.Topic,
                weights=p.RegMix(),
                replacement=True,
                splits=p.Splits(train=0.9, validation=0.05, test=0.05),
            )
            batch = next(iter(DataLoader(mixture[0].train.torch(), batch_size=1)))
            assert tuple(batch["input_ids"].shape) == (1, 2048)
            expected = query.preview(limit=100, max_characters=4096)
            assert expected
            assert "speech/5385" not in {row["source_key"] for row in expected}
            identity = query.id
            service = coordinator(db)
            for item in service._storage.list("field", e.FieldBuild):
                _, manifest = enrichment.load_build(
                    service, "field", item.snapshot.id, item.snapshot.snapshot_ids
                )
                definition = json.loads(manifest.definition_json)
                assert definition["artifact"]["kind"] == "packaged_demo"
                assert len(definition["artifact"]["cohort_text_sha256"]) == 8
        # A different environment gets new derivation IDs and still uses the
        # frozen fixture's original generation environment and batch provenance.
        changed = replace(runtime.current_code(), environment="ab" * 32)
        with patch.object(runtime, "current_code", return_value=changed):
            with p.PremixDB(storage=tmp_path, progress=False) as db:
                assert db._query(identity).preview(limit=100, max_characters=4096) == expected
                assert (
                    db.Corpus("demo")
                    .query(steps=[p.where(p.quality.writing_style >= 3)])
                    .profile()
                    .output_documents
                    == 3
                )
                topic_query = db.Corpus("demo").query(steps=[p.where(p.topic.art_and_design >= 0)])
                assert topic_query.profile().output_documents == 8


@pytest.mark.parametrize("name", ["quality.writing_style", "weborganizer.topic"])
def test_fixture_reuses_subsets_and_requires_matching_inputs_and_policy(name: str) -> None:
    source = _demo_sources()[0]
    policy = catalog.recipe(name)
    worker = enrichment.producer(policy)
    assert isinstance(worker, (QuRating, WebOrganizer))
    docs = [Document("different-occurrence", source.text)]
    saved = demo_enrichment.load(policy, worker.fields, docs)
    assert saved is not None
    assert saved.rows[0][0] == "different-occurrence"
    assert demo_enrichment.load(policy, worker.fields, []) is None
    assert (
        demo_enrichment.load(policy, worker.fields, [*docs, Document("changed", source.text + "!")])
        is None
    )
    changed = e.EnrichmentProducer()
    changed.CopyFrom(policy)
    changed.model.revision = "ab" * 20
    assert demo_enrichment.load(changed, worker.fields, docs) is None
    changed.CopyFrom(policy)
    changed.model.batch_size += 1
    assert demo_enrichment.load(changed, worker.fields, docs) is None
    assert demo_enrichment.load(policy, [], docs) is None


def test_changed_demo_text_uses_normal_inference(tmp_path: Path) -> None:
    source = _demo_sources()[0]
    worker = enrichment.producer(catalog.recipe("quality.writing_style"))
    assert isinstance(worker, QuRating)
    fields = worker.fields

    def compute(self: QuRating, docs: Sequence[Document]) -> list[ComputedRow]:
        assert [doc.text for doc in docs] == [source.text + "!"]
        return [{"id": doc.id, **{spec.name: 0.25 for spec in fields}} for doc in docs]

    with (
        patch.object(QuRating, "compute", autospec=True, side_effect=compute) as inference,
        p.PremixDB(storage=tmp_path, progress=False) as db,
    ):
        snapshot = db.Corpus("demo", [p.Source(source.key, source.text + "!")])
        assert (
            snapshot.query(steps=[p.where(p.quality.writing_style >= 0.8)])
            .profile()
            .output_documents
            == 0
        )
        inference.assert_called_once()


def test_fixture_changes_invalidate_execution_identity(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    asset = tmp_path / "data/demo-enrichment.json"
    asset.write_text('{"version": 1}')
    before = runtime._source_digest(tmp_path)
    asset.write_text('{"version": 2}')
    assert runtime._source_digest(tmp_path) != before
