"""Use DataTrove readers and statistics without dropping source documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.metadata import version
from typing import TYPE_CHECKING, Iterator, Protocol, Sequence, cast

from premixdb.contracts import Metadata, Scalar
from premixdb.enrichment.types import ComputedRow

if TYPE_CHECKING:
    from datatrove.data import Document as DTDocument

from premixdb.enrichment.types import Document, check_documents, field, package_versions
from premixdb.v1 import field_pb2 as f

DOC_METRICS = (
    "length",
    "white_space_ratio",
    "non_alpha_digit_ratio",
    "digit_ratio",
    "uppercase_ratio",
    "elipsis_ratio",
    "punctuation_ratio",
)
WORD_METRICS = (
    "n_words",
    "avg_word_length",
    "avg_words_per_line",
    "short_word_ratio_3",
    "long_word_ratio_7",
    "type_token_ratio",
    "uppercase_word_ratio",
    "capitalized_word_ratio",
    "stop_word_ratio",
)


class TroveDocument(Protocol):
    @property
    def id(self) -> str | int: ...
    @property
    def text(self) -> str: ...
    @property
    def metadata(self) -> Mapping[str, object]: ...


class Stats(Protocol):
    def extract_stats(self, document: DTDocument) -> dict[str, Scalar]: ...


def from_datatrove(documents: Iterable[TroveDocument], *, namespace: str) -> Iterator[Document]:
    """Bridge any DataTrove reader/extractor; namespace disambiguates sources.

    Upstream readers remain responsible for stable, unique IDs. Text and URL
    metadata are kept without generating identity from mutable row order.
    """
    if not namespace:
        raise ValueError("a source namespace is required")
    for doc in documents:
        # Length-prefixing prevents collisions between namespace and document ID.
        url = doc.metadata.get("url")
        if url is not None and not isinstance(url, str):
            raise ValueError("document URL must be a string")
        yield Document(f"{len(namespace)}:{namespace}:{doc.id}", doc.text, url)


class DataTroveFields:
    """Independent document maps using DataTrove's actual statistic kernels.

    Empty-text ratios are null (undefined), with length/n_words equal to zero.
    Word tokenization follows DataTrove's selected language, not model tokens.
    """

    def __init__(self, *, language: str = "en") -> None:
        from datatrove.pipeline.stats import DocStats, WordStats

        # extract_stats does not write summaries or invoke domain extraction.
        self.doc_stats = cast(Stats, DocStats(output_folder="memory://premixdb-stats"))
        word_language = language
        if language == "en" and version("datatrove") == "0.3.0":
            from datatrove.utils.word_tokenizers import WORD_TOKENIZER_CACHE, SpaCyTokenizer

            # Keep current English word statistics on the NumPy 1 compatible
            # release, without NLTK downloads or changing upstream's "en" entry.
            word_language = "premixdb-spacy-en"
            WORD_TOKENIZER_CACHE.setdefault(word_language, SpaCyTokenizer("en"))
        self.word_stats = cast(
            Stats, WordStats(output_folder="memory://premixdb-stats", language=word_language)
        )
        self.language = language

    @property
    def definition(self) -> dict[str, Metadata]:
        """Describe the pinned assets and settings that determine output identity."""
        return {
            "provider": "datatrove",
            "version": 1,
            "language": self.language,
            "packages": package_versions("datatrove", "spacy"),
            "empty_ratios": "null",
            "english_word_tokenizer": "spacy",
        }

    @property
    def fields(self) -> tuple[f.Field, ...]:
        """Return the field definitions produced for each input document."""
        return tuple(
            field(
                f"datatrove.{key}",
                element_type=f.VALUE_INT64 if key in ("length", "n_words") else f.VALUE_FLOAT64,
            )
            for key in (*DOC_METRICS, *WORD_METRICS)
        )

    cache_scope = "document"

    def compute(self, documents: Sequence[Document]) -> list[ComputedRow]:
        """Compute one result per document, preserving input order and empty documents."""
        from datatrove.data import Document as DTDocument

        check_documents(documents)
        rows: list[ComputedRow] = []
        for doc in documents:
            if not doc.text.strip():
                values: dict[str, Scalar] = {key: None for key in (*DOC_METRICS, *WORD_METRICS)}
                if doc.text:
                    values.update(
                        self.doc_stats.extract_stats(DTDocument(id=doc.id, text=doc.text))
                    )
                values.update(length=len(doc.text), n_words=0)
            else:
                source = DTDocument(id=doc.id, text=doc.text)
                values = {
                    **self.doc_stats.extract_stats(source),
                    **self.word_stats.extract_stats(source),
                }
            rows.append({"id": doc.id, **{f"datatrove.{k}": v for k, v in values.items()}})
        return rows
