"""Real training and automatic RegMix selection with controlled classifiers."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable, Sequence
from dataclasses import is_dataclass, replace
from pathlib import Path
from types import FunctionType
from typing import cast

import pytest
import torch
from torch import Tensor, nn

import premixdb as p
from premixdb.enrichment.types import ComputedRow, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.internal import derivation_pb2 as d
from premixdb.runtime import enrichment

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPT = EXAMPLES / "11_c4_pretraining_ablation.py"


@pytest.fixture
def lesson(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return cast(dict[str, object], runpy.run_path(str(SCRIPT), run_name="ablation_test"))


def configure(lesson: dict[str, object], **categories: dict[str, object]) -> None:
    main = lesson["main"]
    assert isinstance(main, FunctionType)
    for category, changes in categories.items():
        config = main.__globals__[category]
        assert is_dataclass(config) and not isinstance(config, type)
        main.__globals__[category] = replace(config, **changes)


def test_transformer_cannot_see_future_training_targets(lesson: dict[str, object]) -> None:
    constructor = cast(Callable[[object], nn.Module], lesson["LanguageModel"])
    config = lesson["MODEL"]
    assert is_dataclass(config) and not isinstance(config, type)
    torch.manual_seed(17)
    model = constructor(replace(config, width=16)).eval()
    ids = torch.randint(0, 256, (2, 12))
    altered = ids.clone()
    altered[:, 6:] = (altered[:, 6:] + 1) % 256
    with torch.no_grad():
        original, changed = model(ids), model(altered)
    assert isinstance(original, Tensor) and isinstance(changed, Tensor)
    torch.testing.assert_close(original[:, :6], changed[:, :6], rtol=0, atol=1e-6)
    assert not torch.equal(original[:, 6:], changed[:, 6:])


class Classifier:
    definition = {"provider": "ablation-test", "version": 1}

    def __init__(self, policy: d.EnrichmentProducer) -> None:
        self.topic = policy.model.kind == d.ModelProducer.TOPIC
        labels = p.Topic if self.topic else p.ContentType
        self.fields = (
            field(
                "weborganizer.topic" if self.topic else "weborganizer.content_type",
                width=24,
                classes=tuple(label.value for label in labels),
            ),
        )

    def compute(self, documents: Sequence[FeatureDocument]) -> list[ComputedRow]:
        result: list[ComputedRow] = []
        for doc in documents:
            # Give the natural C4 sample two controlled labels as well as the
            # explicit labels in the generated fixture. No model download.
            local_label = (
                not doc.text.startswith(
                    ("Science document", "Tutorial document", "Travel document")
                )
                and len(doc.text) % 2 == 0
            )
            if self.topic:
                label = (
                    p.Topic.SCIENCE_AND_TECH
                    if doc.text.startswith("Science") or local_label
                    else p.Topic.TRAVEL
                )
                labels = p.Topic
            else:
                label = (
                    p.ContentType.TUTORIAL
                    if doc.text.startswith("Tutorial") or local_label
                    else p.ContentType.NEWS_ARTICLE
                )
                labels = p.ContentType
            result.append(
                {
                    "id": doc.id,
                    self.fields[0].name: [10.0 if item == label else 0.0 for item in labels],
                }
            )
        return result


@pytest.mark.integration
@pytest.mark.parametrize("domains", (p.Topic, p.ContentType))
@pytest.mark.parametrize("local_sample", (False, True))
def test_regmix_search_trains_every_candidate_and_selects_reusable_weights(
    lesson: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    domains: type[p.Topic] | type[p.ContentType],
    local_sample: bool,
) -> None:
    source = tmp_path / "web.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": f"{kind} document {index}. " * 20}) + "\n"
            for kind in ("Science", "Tutorial", "Travel")
            for index in range(40)
        ),
        encoding="utf-8",
    )
    if local_sample:
        source = EXAMPLES / "data/c4.jsonl"
    main = lesson["main"]
    assert isinstance(main, FunctionType)
    configure(
        lesson,
        DATA={"input": source, "storage": tmp_path / "store", "limit": 1000},
        TRAINING={"steps": 3, "batch_size": 2},
        MODEL={"sequence_length": 16, "width": 16},
        EVALUATION={"batches": 2},
        MIX={"domains": domains, "candidates": 3},
    )
    monkeypatch.setattr(enrichment, "producer", Classifier)
    old_threads = torch.get_num_threads()
    try:
        main()
    finally:
        torch.set_num_threads(old_threads)
    output = tmp_path / "store/results"
    report = json.loads((output / "report.json").read_text())
    runs = report["runs"]
    assert [row["name"] for row in runs] == ["regmix-0", "regmix-1", "regmix-2"]
    winner = min(runs, key=lambda row: row["validation_loss"])
    assert report["selected"] == winner["name"]
    assert report["selected_weights"] == winner["weights"]
    assert report["selected_dataset_id"] == winner["dataset_id"]
    assert report["domains"] == domains.__name__
    assert set(report["test_loss"]) == {"regmix-0", report["selected"]}
    assert report["reference_dataset_id"] == runs[0]["dataset_id"]
    assert report["training"]["seed"] == 0
    assert all(row["steps"] == 3 and row["content_tokens"] == 96 for row in runs)
    assert all(torch.isfinite(torch.tensor(row["validation_loss"])) for row in runs)
    assert runs[0]["delta_from_first"] == 0.0
    generated = runs
    assert {row["candidate_index"] for row in generated} == {0, 1, 2}
    assert len({row["mixture_id"] for row in generated}) == 1
    assert len({row["dataset_id"] for row in generated}) == 3
    assert len({tuple(sorted(row["weights"].items())) for row in generated}) == 3
    labels = (
        {p.Topic.SCIENCE_AND_TECH.value, p.Topic.TRAVEL.value}
        if domains is p.Topic
        else {p.ContentType.TUTORIAL.value, p.ContentType.NEWS_ARTICLE.value}
    )
    for row in runs:
        assert set(row["weights"]) == labels
        assert sum(row["weights"].values()) == pytest.approx(1.0)
        assert sum(row["domain_tokens"].values()) == 96
        for label, weight in row["weights"].items():
            if weight == 0.0:
                assert row["domain_tokens"].get(label, 0) == 0

    with p.PremixDB(storage=tmp_path / "store", progress=False) as db:
        reference = db._dataset(report["reference_dataset_id"])
        snapshot = db.Corpus("tutorial/c4-ablation")
        # Candidates are budgeted; use the full population to verify membership.
        population = snapshot.query(steps=[p.where(p.text.bytes > 0), p.dedupe()])
        full = population.mix(
            tokenizer=p.ByteTokenizer(),
            sequence_length=16,
            packing=p.Concat(),
        )[0]
        train_ids = {id for sequence in full.train for id in sequence.document_ids()}
        heldout_ids = {
            id
            for view in (reference.validation, reference.test)
            for sequence in view
            for id in sequence.document_ids()
        }
        assert train_ids and heldout_ids and train_ids.isdisjoint(heldout_ids)
        for row in runs:
            actual = {
                id
                for sequence in db._dataset(row["dataset_id"]).train
                for id in sequence.document_ids()
            }
            assert actual and actual <= train_ids
        replay = population.mix(
            domains=domains,
            weights=report["selected_weights"],
            tokens=96,
            replacement=True,
            tokenizer=p.ByteTokenizer(),
            sequence_length=16,
            packing=p.Concat(),
        )[0]
        assert replay.id == report["selected_dataset_id"]

    checkpoint = torch.load(output / "winner.pt", map_location="cpu", weights_only=True)
    assert checkpoint["selected"] == report["selected"]
    assert checkpoint["weights"] == report["selected_weights"]
    assert checkpoint["split"] == "train" and checkpoint["training"]["steps"] == 3
    assert checkpoint["optimizer_state_dict"]["state"]
    torch.manual_seed(0)
    constructor = cast(Callable[[object], nn.Module], lesson["LanguageModel"])
    fresh = constructor(main.__globals__["MODEL"])
    assert not torch.equal(
        fresh.state_dict()["tokens.weight"], checkpoint["model_state_dict"]["tokens.weight"]
    )
    assert all(torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values())

    # Default seeds reproduce proposals, draws, selection, and trained weights.
    try:
        main()
    finally:
        torch.set_num_threads(old_threads)
    assert json.loads((output / "report.json").read_text()) == report
    repeated = torch.load(output / "winner.pt", map_location="cpu", weights_only=True)
    for name, value in checkpoint["model_state_dict"].items():
        torch.testing.assert_close(value, repeated["model_state_dict"][name], rtol=0, atol=0)
