"""Shared adapters for preprocessing text before an immutable capture."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from importlib.metadata import version
from typing import TYPE_CHECKING, Protocol

import premixdb as p

if TYPE_CHECKING:
    from datatrove.data import Document


class TextFilter(Protocol):
    def filter(self, doc: Document) -> bool | tuple[bool, str]: ...


def tokenizer_language() -> str:
    """Use spaCy on both supported DataTrove versions, without NLTK downloads."""
    if version("datatrove") == "0.3.0":
        from datatrove.utils.word_tokenizers import WORD_TOKENIZER_CACHE, SpaCyTokenizer

        name = "premixdb-recipes-en"
        WORD_TOKENIZER_CACHE.setdefault(name, SpaCyTokenizer("en"))
        return name
    return "en"


def filtered_sources(
    sources: Iterable[p.Source], filters: Iterable[TextFilter]
) -> Iterator[p.Source]:
    """Keep original keys and capture any line edits made by the upstream filters."""
    from datatrove.data import Document

    pipeline = tuple(filters)
    for source in sources:
        if not source.text.strip():
            continue
        document = Document(id=source.key, text=source.text)
        for stage in pipeline:
            result = stage.filter(document)
            keep = result[0] if isinstance(result, tuple) else result
            if not keep:
                break
        else:
            if document.text.strip():
                yield p.Source(source.key, document.text)


def gopher_filters() -> tuple[TextFilter, TextFilter]:
    """MassiveWeb quality rules and Table A1 repetition thresholds."""
    from datatrove.pipeline.filters import GopherQualityFilter, GopherRepetitionFilter

    language = tokenizer_language()
    return (
        GopherQualityFilter(
            min_doc_words=50,
            max_doc_words=100_000,
            min_avg_word_length=3,
            max_avg_word_length=10,
            max_symbol_word_ratio=0.1,
            max_bullet_lines_ratio=0.9,
            max_ellipsis_lines_ratio=0.3,
            max_non_alpha_words_ratio=0.8,
            min_stop_words=2,
            language=language,
        ),
        GopherRepetitionFilter(
            dup_line_frac=0.3,
            dup_para_frac=0.3,
            dup_line_char_frac=0.2,
            dup_para_char_frac=0.2,
            # DataTrove annotates these variadic tuples as tuples of length one.
            top_n_grams=((2, 0.2), (3, 0.18), (4, 0.16)),  # ty: ignore[invalid-argument-type]
            dup_n_grams=((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.1)),  # ty: ignore[invalid-argument-type]
            language=language,
        ),
    )


def source_weights(
    fractions: Mapping[str, float], snapshots: Mapping[str, p.Snapshot]
) -> dict[str, float]:
    """Bind every published source category to a distinct captured corpus."""
    if set(fractions) != set(snapshots):
        missing = sorted(set(fractions) - set(snapshots))
        extra = sorted(set(snapshots) - set(fractions))
        raise ValueError(
            f"source categories must match the recipe: missing={missing}, extra={extra}"
        )
    result = {snapshots[name].corpus_id: fraction for name, fraction in fractions.items()}
    if len(result) != len(snapshots):
        raise ValueError("each source category requires a distinct corpus")
    return result
