"""Pinned, batched inference using the published model interfaces."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal, Mapping, Sequence, cast

from .._typing import Metadata, field_value
from ..v1.field_pb2 import Field
from .interfaces import ModelTokenizer, SequenceModel
from .types import ComputedRow, Document

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from torch import Tensor

from .types import ModelPin, check_documents, field, matrix, package_versions, positive

QUALITY_DIMENSIONS = (
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value",
)
EMBEDDING_MODELS = {
    "harrier": "microsoft/harrier-oss-v1-0.6b",
}


def _sequence_model(
    pin: ModelPin, device: str, *, web: bool = False
) -> tuple[ModelTokenizer, SequenceModel]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = cast(
        ModelTokenizer,
        AutoTokenizer.from_pretrained(pin.repository, revision=pin.revision, trust_remote_code=web),
    )
    if tokenizer is None:
        raise ValueError("model repository did not provide a tokenizer")
    if web:
        model = cast(
            SequenceModel,
            AutoModelForSequenceClassification.from_pretrained(
                pin.repository,
                revision=pin.revision,
                trust_remote_code=True,
                use_memory_efficient_attention=False,
                unpad_inputs=False,
            ),
        )
    else:
        model = cast(
            SequenceModel,
            AutoModelForSequenceClassification.from_pretrained(
                pin.repository,
                revision=pin.revision,
                trust_remote_code=False,
            ),
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("classifier tokenizer must have a padding token")
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model.to(device).eval()


def _logits(
    model: SequenceModel, inputs: Mapping[str, Tensor], rows: int, width: int
) -> list[list[float]]:
    import torch

    with torch.inference_mode():
        logits = model(**{key: value.to(model.device) for key, value in inputs.items()}).logits
    return matrix(logits.float().cpu().tolist(), rows, width)


class QuRating:
    """Four independent regression heads; scores are NOT class probabilities.

    All text is scored in <=512-token windows including special tokens. Document
    scores are means weighted by content-token counts, following the model card.
    Empty documents have null scores. No invented high/low quality threshold.
    """

    def __init__(self, pin: ModelPin, *, device: str = "cpu", batch_size: int = 8) -> None:
        if pin.repository != "princeton-nlp/QuRater-1.3B":
            raise ValueError("QuRating head order is defined for princeton-nlp/QuRater-1.3B")
        positive(batch_size, "batch_size")
        self.pin, self.batch_size = pin, batch_size
        self.device = device

    @cached_property
    def _model_pair(self) -> tuple[ModelTokenizer, SequenceModel]:
        tokenizer, model = _sequence_model(self.pin, self.device)
        if model.config.num_labels != 4:
            raise ValueError("QuRater must expose exactly four regression heads")
        return tokenizer, model

    @property
    def _tokenizer(self) -> ModelTokenizer:
        return self._model_pair[0]

    @property
    def _model(self) -> SequenceModel:
        return self._model_pair[1]

    @property
    def fields(self) -> tuple[Field, ...]:
        """Return the field definitions produced for each input document."""
        return tuple(field(f"quality.{name}") for name in QUALITY_DIMENSIONS)

    @property
    def definition(self) -> dict[str, Metadata]:
        """Describe the pinned assets and settings that determine output identity."""
        return {
            "provider": "qurating",
            "version": 1,
            "model": {"repository": self.pin.repository, "revision": self.pin.revision},
            "packages": package_versions("transformers", "torch"),
            "device": self.device,
            "batch_size": self.batch_size,
            "dimensions": QUALITY_DIMENSIONS,
            "output": "unnormalized_regression",
            "window_tokens": 512,
            "aggregation": "content_token_weighted_mean",
            "empty_document": "null",
        }

    # Padding/window batching can affect floating-point results. Reuse only the
    # exact ordered computation cohort; never assume numerical batch invariance.
    cache_scope = "batch"

    def compute(self, documents: Sequence[Document]) -> list[ComputedRow]:
        """Compute one result per document, preserving input order and empty documents."""
        check_documents(documents)
        totals = [[0.0] * 4 for _ in documents]
        counts = [0] * len(documents)
        windows: list[dict[str, list[int]]] = []
        owners: list[tuple[int, int]] = []
        capacity = None

        def flush() -> None:
            if not windows:
                return
            inputs = self._tokenizer.pad(windows, padding=True, return_tensors="pt")
            for (owner, weight), scores in zip(
                owners, _logits(self._model, inputs, len(windows), 4)
            ):
                counts[owner] += weight
                for head, score in enumerate(scores):
                    totals[owner][head] += weight * score
            windows.clear()
            owners.clear()

        for i, doc in enumerate(documents):
            if not doc.text.strip():
                continue
            if capacity is None:
                capacity = 512 - self._tokenizer.num_special_tokens_to_add(pair=False)
                positive(capacity, "window capacity")
            encoded = self._tokenizer(
                doc.text,
                add_special_tokens=False,
                truncation=False,
            ).encodings[0]
            if not encoded.ids:
                continue
            # Split content first, then add the model's special tokens to each
            # window. This preserves the tail without re-tokenizing text.
            encoded.truncate(capacity)
            for chunk in (encoded, *encoded.overflowing):
                prepared = self._tokenizer.backend_tokenizer.post_process(
                    chunk, add_special_tokens=True
                )
                windows.append(
                    {"input_ids": prepared.ids, "attention_mask": prepared.attention_mask}
                )
                owners.append((i, len(chunk.ids)))
                if len(windows) == self.batch_size:
                    flush()
        flush()
        return [
            {
                "id": doc.id,
                **{
                    f"quality.{name}": totals[i][head] / counts[i] if counts[i] else None
                    for head, name in enumerate(QUALITY_DIMENSIONS)
                },
            }
            for i, doc in enumerate(documents)
        ]


class WebOrganizer:
    """Topic/content-type class logits in numeric model.config.id2label order.

    Content type is the paper's 'format' axis. URL models require URLs; use the
    explicit -NoURL checkpoints when URLs are unavailable. Inputs are truncated
    at max_length tokens and that policy is recorded in the producer definition.
    """

    def __init__(
        self,
        pin: ModelPin,
        *,
        task: Literal["topic", "content_type"],
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 8192,
        classes: Sequence[str] | None = None,
    ) -> None:
        if task not in ("topic", "content_type"):
            raise ValueError("task must be topic or content_type")
        stem = "Topic" if task == "topic" else "Format"
        base = f"WebOrganizer/{stem}Classifier"
        if pin.repository not in (base, base + "-NoURL"):
            raise ValueError(f"{task} requires {base} or its -NoURL checkpoint")
        positive(batch_size, "batch_size")
        positive(max_length, "max_length")
        if max_length > 8192:
            raise ValueError("WebOrganizer supports at most 8192 tokens")
        self.pin, self.task = pin, task
        self.batch_size, self.max_length = batch_size, max_length
        self.with_url = not pin.repository.endswith("-NoURL")
        self.device, self._expected_classes = device, None if classes is None else tuple(classes)

    @cached_property
    def _model_pair(self) -> tuple[ModelTokenizer, SequenceModel]:
        tokenizer, model = _sequence_model(self.pin, self.device, web=True)
        labels = model.config.id2label
        ordered = {int(key): value for key, value in labels.items()}
        if set(ordered) != set(range(24)) or len(set(ordered.values())) != 24:
            raise ValueError("WebOrganizer requires 24 distinct, contiguous model labels")
        if (
            self._expected_classes is not None
            and tuple(ordered[i] for i in range(24)) != self._expected_classes
        ):
            raise ValueError("WebOrganizer model labels differ from the pinned vocabulary")
        return tokenizer, model

    @property
    def _tokenizer(self) -> ModelTokenizer:
        return self._model_pair[0]

    @property
    def _model(self) -> SequenceModel:
        return self._model_pair[1]

    @cached_property
    def _classes(self) -> tuple[str, ...]:
        if self._expected_classes is not None:
            return self._expected_classes
        labels = self._model.config.id2label
        ordered = {int(key): value for key, value in labels.items()}
        return tuple(ordered[i] for i in range(24))

    @property
    def fields(self) -> tuple[Field, ...]:
        """Return the field definitions produced for each input document."""
        return (
            field(f"weborganizer.{self.task}", width=len(self._classes), classes=self._classes),
        )

    @property
    def definition(self) -> dict[str, Metadata]:
        """Describe the pinned assets and settings that determine output identity."""
        return {
            "provider": "weborganizer",
            "version": 1,
            "model": {"repository": self.pin.repository, "revision": self.pin.revision},
            "packages": package_versions("transformers", "torch", "einops"),
            "device": self.device,
            "batch_size": self.batch_size,
            "task": self.task,
            "classes": self._classes,
            "output": "logits",
            "transform": "softmax",
            "input": "url\\n\\ntext" if self.with_url else "text",
            "max_length": self.max_length,
            "truncation": True,
        }

    # Padding/window batching can affect floating-point results. Reuse only the
    # exact ordered computation cohort; never assume numerical batch invariance.
    cache_scope = "batch"

    def compute(self, documents: Sequence[Document]) -> list[ComputedRow]:
        """Compute one result per document, preserving input order and empty documents."""
        check_documents(documents)
        if self.with_url and any(not doc.url or not doc.url.strip() for doc in documents):
            raise ValueError("URL-trained WebOrganizer requires a URL for every document")
        rows: list[ComputedRow] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            texts = [f"{doc.url}\n\n{doc.text}" if self.with_url else doc.text for doc in batch]
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            values = _logits(self._model, inputs, len(batch), len(self._classes))
            rows.extend(
                {"id": doc.id, f"weborganizer.{self.task}": field_value(logits)}
                for doc, logits in zip(batch, values)
            )
        return rows


class Embeddings:
    """Harrier embeddings with model-defined pooling and L2 normalization."""

    def __init__(
        self,
        pin: ModelPin,
        *,
        family: Literal["harrier"],
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 8192,
        width: int | None = None,
    ) -> None:
        if family not in EMBEDDING_MODELS or pin.repository != EMBEDDING_MODELS[family]:
            raise ValueError("select the documented harrier model")
        positive(batch_size, "batch_size")
        positive(max_length, "max_length")
        if max_length > 32768:
            raise ValueError("embedding context exceeds 32768 tokens")
        self.pin, self.family, self.batch_size = pin, family, batch_size
        self.device, self.max_length, self._expected_width = device, max_length, width
        if width is not None:
            positive(width, "embedding dimension")

    @cached_property
    def _model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            self.pin.repository, revision=self.pin.revision, device=self.device
        )
        model.max_seq_length = self.max_length
        width = model.get_embedding_dimension()
        if width is None:
            raise ValueError("model did not provide an embedding dimension")
        positive(width, "embedding dimension")
        if self._expected_width is not None and width != self._expected_width:
            raise ValueError("embedding dimension differs from its pinned schema")
        return model

    @cached_property
    def _width(self) -> int:
        width = (
            self._expected_width
            if self._expected_width is not None
            else self._model.get_embedding_dimension()
        )
        if width is None:
            raise ValueError("model did not provide an embedding dimension")
        return width

    @property
    def fields(self) -> tuple[Field, ...]:
        """Return the field definitions produced for each input document."""
        return (field(f"embedding.{self.family}", width=self._width),)

    @property
    def definition(self) -> dict[str, Metadata]:
        """Describe the pinned assets and settings that determine output identity."""
        return {
            "provider": "sentence-transformers",
            "version": 1,
            "model": {"repository": self.pin.repository, "revision": self.pin.revision},
            "packages": package_versions("sentence-transformers", "transformers", "torch"),
            "device": self.device,
            "batch_size": self.batch_size,
            "width": self._width,
            "normalize": True,
            "document_prompt": "",
            "max_length": self.max_length,
            "truncation": True,
        }

    def _encode(
        self, texts: list[str], *, prompt: str | None = None, prompt_name: str | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=True,
            prompt=prompt,
            prompt_name=prompt_name,
        )
        return matrix(vectors.tolist(), len(texts), self._width)

    # Padding/window batching can affect floating-point results. Reuse only the
    # exact ordered computation cohort; never assume numerical batch invariance.
    cache_scope = "batch"

    def compute(self, documents: Sequence[Document]) -> list[ComputedRow]:
        """Compute one result per document, preserving input order and empty documents."""
        check_documents(documents)
        vectors = self._encode([doc.text for doc in documents], prompt="")
        return [
            {"id": doc.id, f"embedding.{self.family}": field_value(vector)}
            for doc, vector in zip(documents, vectors)
        ]

    def encode_queries(self, queries: list[str]) -> list[list[float]]:
        """Encode search queries with the model family's query prompt."""
        return self._encode(queries, prompt_name="web_search_query")
