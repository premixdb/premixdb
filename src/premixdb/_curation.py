"""Typed constructors for pinned curation policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal

from ._policies import ByteTokenizer
from ._protobuf import parse
from .v1 import dataset_pb2 as datasets

if TYPE_CHECKING:
    from ._resources import Snapshot


from typing import Mapping, Protocol, runtime_checkable

from ._field_expr import FieldProjection, VectorField
from ._field_ids import field_id
from ._requests import _Field, _id, _uint
from .fields import ContentType, Language, Topic
from .v1 import query_pb2 as q
from .v1.storage_pb2 import ObjectRef


@runtime_checkable
class ClassifierProjection(Protocol):
    @property
    def label(self) -> FieldProjection: ...


type FieldSelector = (
    str
    | _Field[int]
    | _Field[str]
    | q.FieldComparison
    | FieldProjection
    | VectorField
    | ClassifierProjection
    | type[Topic]
    | type[ContentType]
    | type[Language]
)


def selector(field: FieldSelector) -> q.FieldComparison:
    if isinstance(field, q.FieldComparison):
        return parse(q.FieldComparison, field.SerializeToString())
    if isinstance(field, _Field):
        return q.FieldComparison(field=field_id(field.name), projection=q.FieldComparison.SCALAR)
    from .fields import ContentType, Language, Topic, content_type, language, topic

    if field is Topic:
        field = topic.label
    elif field is ContentType:
        field = content_type.label
    elif field is Language:
        field = language.label
    # A classifier namespace naturally selects its label; individual projections
    # remain available for explicit scalar probabilities and multi-field strata.
    if isinstance(field, ClassifierProjection):
        field = field.label
    if not isinstance(field, (str, FieldProjection, VectorField)):
        raise TypeError("expected a field projection or classifier namespace")
    name = field if isinstance(field, str) else field.name
    if not isinstance(name, str):
        raise TypeError("expected a field projection with a string name")
    result = q.FieldComparison(
        field=field_id(name),
        projection=field.projection
        if isinstance(field, FieldProjection)
        else q.FieldComparison.SCALAR,
        class_name=field.class_name if isinstance(field, FieldProjection) else "",
    )
    component = field.component_index if isinstance(field, FieldProjection) else None
    if component is not None:
        result.component = component
    return result


def decontaminate(
    *references: Snapshot | bytes | str,
    algorithm: Literal["document", "line", "ngram"] = "ngram",
    n: int = 13,
    granularity: Literal["document", "span"] = "document",
) -> q.Decontaminate:
    """Remove reference overlap using pinned snapshots and the chosen removal unit."""
    algorithms = {
        "document": q.Decontaminate.ALGORITHM_EXACT_DOCUMENT,
        "line": q.Decontaminate.ALGORITHM_EXACT_LINE,
        "ngram": q.Decontaminate.ALGORITHM_EXACT_NGRAM,
    }
    if algorithm not in algorithms or granularity not in ("document", "span"):
        raise ValueError("unknown decontamination policy")
    return q.Decontaminate(
        snapshot_ids=[
            _id(reference if isinstance(reference, (bytes, str)) else reference.id, 32)
            for reference in references
        ],
        algorithm=algorithms[algorithm],
        n=_uint(n, 32, "n", positive=True) if algorithm == "ngram" else 0,
        granularity=q.Decontaminate.DOCUMENT if granularity == "document" else q.Decontaminate.SPAN,
    )


def sample(
    *,
    seed: int = 0,
    documents: int | None = None,
    fraction: float | None = None,
    bytes: int | None = None,
    characters: int | None = None,
    tokens: int | None = None,
    domains: FieldSelector | list[FieldSelector] | tuple[FieldSelector, ...] = (),
    weights: Mapping[str, float] | None = None,
    replacement: bool = False,
    tokenizer_asset: ObjectRef | None = None,
    tokenizer: datasets.Tokenizer | ByteTokenizer | None = None,
    max_document_bytes: int = 8 * 1024 * 1024,
) -> q.QuerySampling:
    """Describe deterministic document sampling with explicit weights and ordering."""
    budgets = {
        k: v
        for k, v in dict(
            documents=documents,
            fraction=fraction,
            bytes=bytes,
            characters=characters,
            tokens=tokens,
        ).items()
        if v is not None
    }
    if not budgets:
        fraction = 1.0
        budgets["fraction"] = fraction
    if len(budgets) != 1 or type(replacement) is not bool:
        raise ValueError("sampling requires exactly one budget and a boolean replacement policy")
    name, value = next(iter(budgets.items()))
    if name != "fraction":
        _uint(value, 64, name)
    if not isinstance(domains, (tuple, list)):
        domains = (domains,)
    inline = b""
    if tokens is not None and tokenizer is None and tokenizer_asset is None:
        from ._default_tokenizer import _gpt2_tokenizer

        tokenizer = _gpt2_tokenizer()
    if tokenizer is not None:
        if tokenizer_asset is not None:
            raise ValueError("provide tokenizer or tokenizer_asset")
        if isinstance(tokenizer, ByteTokenizer):
            tokenizer = tokenizer._to_proto()
        if tokenizer.HasField("hugging_face"):
            policy = tokenizer.hugging_face
            tokenizer_asset, inline = policy.asset, policy.json
            max_document_bytes = policy.max_document_bytes
        elif not tokenizer.HasField("byte"):
            raise ValueError("token sampling requires a byte or Hugging Face tokenizer")
    return q.QuerySampling(
        seed=_uint(seed, 64, "seed"),
        replacement=replacement,
        domains=[selector(f) for f in domains],
        weights=weights or {},
        tokenizer_asset=tokenizer_asset,
        tokenizer_json=inline,
        max_document_bytes=max_document_bytes,
        documents=documents,
        fraction=fraction,
        bytes=bytes,
        characters=characters,
        tokens=tokens,
    )


def similarity_dedupe(
    *,
    threshold: float,
    n: int = 5,
    embedding: FieldSelector | None = None,
    order_by: Iterable[q.OrderBy] = (),
) -> q.Operation:
    """Describe greedy near-duplicate removal with the given similarity threshold."""
    return q.Operation(
        similarity_dedupe=q.SimilarityDedupe(
            algorithm=q.SimilarityDedupe.JACCARD
            if embedding is None
            else q.SimilarityDedupe.COSINE,
            threshold=threshold,
            n=_uint(n, 32, "n", positive=True) if embedding is None else 0,
            embedding=None if embedding is None else selector(embedding),
            order_by=list(order_by),
        )
    )
