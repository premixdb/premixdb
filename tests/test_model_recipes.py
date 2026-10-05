"""Execute paper adaptations on custom inputs without downloading model assets."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from recipes import apply, falcon, gopher, gpt3, llama, mix, source_weights, t5

import premixdb as p
from premixdb.enrichment import language as language_service
from premixdb.enrichment.types import ComputedRow, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.v1 import field_pb2 as f

TEXT = (
    "The researchers mapped distant stars and measured changes in their brightness across "
    "several nights. A telescope gathered images while observers compared the signals with "
    "earlier records of neighboring galaxies. Careful analysis revealed differences between "
    "young clusters and older systems with cooler planets. These findings offer useful "
    "evidence about how matter moves through space, why certain patterns persist, and where "
    "future instruments might uncover additional details about our universe."
)


@pytest.fixture
def offline_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    from datatrove.pipeline.filters.c4_filters import C4BadWordsFilter

    monkeypatch.setattr(
        C4BadWordsFilter, "_get_badwords", lambda self, lang: re.compile(r"\bblocked\b")
    )

    class LanguageScores:
        definition = {"provider": "recipe-language-test", "version": 1}
        cache_scope = "document"
        fields = (field("language.en"), field("language.label", element_type=f.VALUE_STRING))

        def compute(self, documents: Sequence[FeatureDocument]) -> list[ComputedRow]:
            return [
                {
                    "id": doc.id,
                    "language.en": 0.1 if doc.text.startswith("French:") else 0.999,
                    "language.label": "fr" if doc.text.startswith("French:") else "en",
                }
                for doc in documents
            ]

    monkeypatch.setattr(
        language_service, "LanguageScores", lambda *args, **kwargs: LanguageScores()
    )


@pytest.mark.parametrize("model", ["falcon", "gopher"])
def test_web_heuristics_reject_short_repetitive_and_symbol_pages(model: str) -> None:
    recipe = {"falcon": falcon, "gopher": gopher}[model]
    sources = [
        p.Source("good", TEXT),
        p.Source("short", "A tiny page."),
        p.Source("repeat", "the telescope measured the sky. " * 80),
        p.Source("symbols", "# " * 100),
        p.Source("empty", ""),
    ]
    assert list(recipe.sources(sources)) == [sources[0]]


@pytest.mark.parametrize("model", ["falcon", "gopher", "t5", "llama"])
def test_queries_filter_language_and_dedupe_custom_crawl(
    model: str, tmp_path: Path, offline_assets: None
) -> None:
    recipe = {"falcon": falcon, "gopher": gopher, "t5": t5, "llama": llama}[model]
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.Corpus(
            "crawl",
            [p.Source("a", TEXT), p.Source("copy", TEXT), p.Source("fr", "French: " + TEXT)],
        )
        selected = recipe.query(snapshot)
        assert selected.profile().output_documents == 1
        assert selected.preview()[0]["text"] == TEXT
        assert recipe.query(snapshot).id == selected.id


def test_gopher_can_exclude_a_held_out_document(tmp_path: Path, offline_assets: None) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        independent = "A separate training article describes farming practices and soil conditions."
        training = db.Corpus("crawl", [p.Source("a", TEXT), p.Source("b", independent)])
        evaluation = db.Corpus("heldout", [p.Source("held", TEXT)])
        selected = gopher.query(training, evaluation=evaluation)
        assert selected.profile().output_documents == 1
        assert selected.preview()[0]["text"] == independent


def test_t5_cleans_lines_and_uses_paper_sentence_word_thresholds(offline_assets: None) -> None:
    # Exactly three sentences, each with five words; catches swapped DT defaults.
    clean = "The telescope reveals distant stars.\nThese observations explain changing brightness.\nOur research uncovers interesting patterns."
    sources = [
        p.Source("clean", "Navigation\n" + clean + "\nOur privacy policy uses cookies."),
        p.Source("few-sentences", "\n".join(clean.splitlines()[:2])),
        p.Source("few-words", "Only three words.\n" * 3),
        p.Source("blocked", clean + "\nThis page contains a blocked word."),
        p.Source("code", clean + "\nHere is some code { example."),
    ]
    prepared = list(t5.sources(sources))
    assert prepared == [p.Source("clean", clean)]


def test_llama_requires_and_applies_the_callers_reference_classifier() -> None:
    pages = [p.Source("reference", TEXT), p.Source("other", TEXT)]
    assert list(
        llama.sources(pages, is_wikipedia_reference=lambda source: source.key == "reference")
    ) == [pages[0]]


def test_gpt3_resampling_is_reproducible_and_preserves_the_published_probability() -> None:
    pages = [p.Source(f"page/{i}", TEXT) for i in range(2048)]
    first = list(gpt3.sources(pages, score=lambda source: 0.5, seed=42))
    reversed_order = list(gpt3.sources(reversed(pages), score=lambda source: 0.5, seed=42))
    assert first == list(reversed(reversed_order))
    # At score=.5, P(keep)=1/(1.5**9), about 2.6%, not a hard cutoff.
    assert 20 < len(first) < 95
    assert list(gpt3.sources(pages, score=lambda source: 1.0)) == pages
    assert first != list(gpt3.sources(pages, score=lambda source: 0.5, seed=43))


@pytest.mark.parametrize("score", [-0.1, 1.1, math.nan, math.inf])
def test_gpt3_rejects_invalid_classifier_scores(score: float) -> None:
    with pytest.raises(ValueError, match="finite in"):
        list(gpt3.sources([p.Source("page", TEXT)], score=lambda source: score))


@pytest.mark.parametrize("model", ["falcon", "gopher", "t5", "llama", "gpt3"])
def test_published_weights_allocate_a_custom_source_token_budget(
    model: str, tmp_path: Path
) -> None:
    recipe = {"falcon": falcon, "gopher": gopher, "t5": t5, "llama": llama, "gpt3": gpt3}[model]
    with p.PremixDB(storage=tmp_path) as db:
        snapshots = {
            name: db.Corpus(name, [p.Source(name, name + " " + TEXT)]) for name in recipe.WEIGHTS
        }
        weights = source_weights(recipe.WEIGHTS, snapshots)
        first, *rest = snapshots.values()
        dataset = (
            first.union(*rest)
            .query()
            .mix(
                weights=weights,
                splits=p.Splits(train=1, validation=0, test=0),
                tokens=100,
                tokenizer=p.ByteTokenizer(),
                sequence_length=10,
                replacement=False,
            )[0]
        )
        profile = dataset.profile()
        assert profile.content_tokens == 100
        for corpus_id, fraction in weights.items():
            assert abs(profile.source_tokens[corpus_id] - 100 * fraction) <= 1


def test_mixture_binding_rejects_missing_sources_and_shared_corpora(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        web = db.Corpus("web", [p.Source("a", TEXT)])
        with pytest.raises(ValueError, match="missing=.*c4"):
            source_weights(llama.WEIGHTS, {"common_crawl": web})
        with pytest.raises(ValueError, match="distinct corpus"):
            source_weights({"one": 0.5, "two": 0.5}, {"one": web, "two": web})


@pytest.mark.parametrize("model", ["falcon", "gopher", "t5"])
def test_custom_crawl_runner(
    model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    offline_assets: None,
) -> None:
    crawl = tmp_path / "crawl.jsonl"
    crawl.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in (TEXT, TEXT, "Short.")),
        encoding="utf-8",
    )
    monkeypatch.setattr(apply, "INPUT", crawl)
    monkeypatch.setattr(apply, "STORAGE", tmp_path / "store")
    monkeypatch.setattr(apply, "MODEL", model)
    apply.main()
    output = capsys.readouterr().out
    assert f"Recipe adaptation: {model}" in output
    assert "After preprocessing: 2" in output
    assert "After query: 1" in output


def test_published_mixture_runner_uses_custom_prepared_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = {}
    for category in gpt3.WEIGHTS:
        input_file = tmp_path / f"{category}.jsonl"
        input_file.write_text(json.dumps({"text": TEXT}) + "\n", encoding="utf-8")
        inputs[category] = input_file
    monkeypatch.setattr(mix, "MODEL", "gpt3")
    monkeypatch.setattr(mix, "INPUTS", inputs)
    monkeypatch.setattr(mix, "STORAGE", tmp_path / "store")
    monkeypatch.setattr(mix, "TOKENS", 100)
    monkeypatch.setattr(mix, "SEQUENCE_LENGTH", 10)
    monkeypatch.setattr(mix, "TOKENIZER", p.ByteTokenizer())
    mix.main()
    assert "Model recipe: gpt3" in capsys.readouterr().out
