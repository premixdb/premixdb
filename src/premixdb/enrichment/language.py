"""DataTrove fastText language scores, without filtering source documents."""

from __future__ import annotations

import logging
import math
from functools import cached_property
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence, cast

from blake3 import blake3

from premixdb.contracts import Metadata
from premixdb.enrichment.types import (
    ComputedRow,
    Document,
    check_documents,
    field,
    package_versions,
)
from premixdb.fields import Language
from premixdb.v1.field_pb2 import Field

if TYPE_CHECKING:
    from datatrove.data import Document as DTDocument


class LabelModel(Protocol):
    def get_labels(self) -> list[str]: ...


class LanguageDetector(Protocol):
    MODEL_URL: str
    MODEL_SUBFOLDER: str
    model: LabelModel

    def predict(self, document: DTDocument) -> tuple[str, dict[str, float]]: ...


def _model_path(url: str, subfolder: str) -> Path:
    from datatrove.io import cached_assets_path, download_file, safely_create_file, strip_protocol

    path = Path(cached_assets_path(library_name="datatrove", namespace="lid", subfolder=subfolder))
    path = path / str(strip_protocol(url)).replace("/", "_")

    def download() -> None:
        try:
            download_file(url, str(path), progress=False)
        except Exception:
            logging.getLogger(__name__).exception("fastText model download failed: %s", url)
            raise

    safely_create_file(str(path), download)
    return path


def _fasttext_package() -> str:
    try:
        version("fasttext-numpy2-wheel")
    except PackageNotFoundError:
        return "fasttext-wheel"
    return "fasttext-numpy2-wheel"


class LanguageScores:
    def __init__(
        self, languages: Sequence[Language | str] = tuple(Language), *, model_digest: bytes = b""
    ) -> None:
        self.languages = tuple(sorted({Language(value).value for value in languages}))
        if not self.languages:
            raise ValueError("select at least one language")
        if not isinstance(model_digest, bytes) or (model_digest and len(model_digest) != 32):
            raise ValueError("language model digest must contain 32 bytes")
        self._model_digest = model_digest.hex()

    @cached_property
    def _lid(self) -> LanguageDetector:
        from datatrove.utils.lid import FT176LID

        lid = cast(LanguageDetector, FT176LID(list(self.languages), k=-1))
        path = _model_path(lid.MODEL_URL, lid.MODEL_SUBFOLDER)
        digest = blake3()
        with open(path, "rb") as model:
            for chunk in iter(lambda: model.read(1024 * 1024), b""):
                digest.update(chunk)
        if self._model_digest and digest.hexdigest() != self._model_digest:
            raise ValueError("language model asset does not match its pinned digest")
        self._model_digest = digest.hexdigest()
        labels = {label.removeprefix("__label__") for label in lid.model.get_labels()}
        if not set(self.languages) <= labels:
            raise ValueError("language vocabulary does not match fastText model")
        return lid

    @property
    def model_digest(self) -> str:
        """Return the verified language model BLAKE3 digest in hexadecimal."""
        if not self._model_digest:
            _ = self._lid
        return self._model_digest

    @property
    def fields(self) -> tuple[Field, ...]:
        """Return the field definitions produced for each input document."""
        from premixdb.v1.field_pb2 import VALUE_STRING

        return (
            *tuple(field(f"language.{code}") for code in self.languages),
            field("language.label", element_type=VALUE_STRING),
        )

    @property
    def definition(self) -> dict[str, Metadata]:
        """Describe the pinned assets and settings that determine output identity."""
        return {
            "provider": "datatrove-ft176",
            "version": 2,
            "languages": self.languages,
            "packages": package_versions("datatrove", _fasttext_package()),
            "model_blake3": self.model_digest,
            "k": -1,
            "output": "probabilities",
            "numerical_clipping": [0, 1],
            "empty_document": "null",
            "label": "highest-probability/code-order-ties",
        }

    cache_scope = "document"

    def compute(self, documents: Sequence[Document]) -> list[ComputedRow]:
        """Compute one result per document, preserving input order and empty documents."""
        from datatrove.data import Document as DTDocument

        check_documents(documents)
        rows: list[ComputedRow] = []
        for doc in documents:
            if not doc.text.strip():
                scores: dict[str, float | None] = {language: None for language in self.languages}
            else:
                _, predicted = self._lid.predict(DTDocument(id=doc.id, text=doc.text))
                if any(not math.isfinite(value) for value in predicted.values()):
                    raise ValueError("non-finite language score")
                # fastText adds a small epsilon and can return slightly >1.
                scores = {code: min(1.0, max(0.0, value)) for code, value in predicted.items()}
            label = (
                max(self.languages, key=lambda code: scores[code] or 0.0)
                if doc.text.strip()
                else None
            )
            rows.append(
                {
                    "id": doc.id,
                    "language.label": label,
                    **{f"language.{code}": scores[code] for code in self.languages},
                }
            )
        return rows
