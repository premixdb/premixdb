"""Built-in derivations known to planning, without loading inference runtimes.

Changing a recipe changes its pin. Existing artifacts are immutable and reused;
model revisions, language vocabulary and inference policies are defined by premixdb.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from premixdb.engine.identity import identity
from premixdb.fields import ContentType, DedupeIndex, Language, Topic
from premixdb.fields.ids import DERIVED_FIELD_NAMES as FIELD_NAMES
from premixdb.fields.ids import field_name, selector_field
from premixdb.internal import derivation_pb2 as d
from premixdb.runtime import environment as _runtime
from premixdb.schemas.protobuf import wire
from premixdb.v1 import query_pb2 as q

CATALOG_VERSION = 3
MODELS = {
    "quality": (
        d.ModelProducer.QUALITY,
        "princeton-nlp/QuRater-1.3B",
        "bd61c778c2f42c6e406b7bac8064290ffc183ae1",
    ),
    "weborganizer.topic": (
        d.ModelProducer.TOPIC,
        "WebOrganizer/TopicClassifier-NoURL",
        "8f60a8f97177f5664a81ad1c2789412fbb38351d",
    ),
    "weborganizer.content_type": (
        d.ModelProducer.CONTENT_TYPE,
        "WebOrganizer/FormatClassifier-NoURL",
        "74d5efb924e1843e84b28b59c26f6ceaa873dc16",
    ),
    "embedding.harrier": (
        d.ModelProducer.HARRIER,
        "microsoft/harrier-oss-v1-0.6b",
        "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
    ),
}


def recipe(name: str, *, index: bool = False) -> d.EnrichmentProducer:
    if index:
        if name not in (DedupeIndex.EXACT_DOCUMENT.value, "dupekit.lsh"):
            raise NotImplementedError(
                "LSH candidates require an explicit verification/clustering policy"
            )
        return d.EnrichmentProducer(
            dupekit=d.DupekitProducer(num_perms=128, num_bands=16, ngram_size=5, seed=42)
        )
    if name not in FIELD_NAMES:
        raise ValueError(f"unknown built-in field: {name}")
    family = name.split(".")[0]
    if family == "language":
        return d.EnrichmentProducer(
            language=d.LanguageProducer(
                languages=sorted(value.value for value in Language),
                model_digest=bytes.fromhex(
                    "220074c3411dae97da55ddba25718875e6391103f9d635731fa767afd9b060a7"
                ),
            )
        )
    if family == "datatrove":
        return d.EnrichmentProducer(datatrove=d.DataTroveProducer(language="en"))
    kind, repository, revision = MODELS[family if family == "quality" else name]
    return d.EnrichmentProducer(
        model=d.ModelProducer(
            kind=kind,
            repository=repository,
            revision=revision,
            device="cpu",
            batch_size=8,
            max_length=512 if family == "quality" else 8192,
        )
    )


def plan(
    snapshot_ids: Iterable[bytes], name: str, git_commit: bytes, *, index: bool = False
) -> d.DerivationPlan:
    result = d.DerivationPlan(
        snapshot_ids=sorted(set(snapshot_ids)),
        producer=recipe(name, index=index),
        git_commit=git_commit,
        catalog_version=CATALOG_VERSION,
        runtime_digest=bytes.fromhex(_runtime.current_code().environment),
    )
    result.id = identity("premixdb-derivation/v1", wire(result))
    return result


def build_id(plan: d.DerivationPlan, name: str) -> bytes:
    return identity("premixdb-derived-field-index/v1", plan.id, name.encode())


def resolve(query: q.CreateQueryRequest | q.Query) -> tuple[d.DerivationPlan, ...]:
    """Pin known recipes during planning; no data scans, downloads or inference."""
    plans = {}
    fields, indexes = set(), set()
    for selector in selectors(query):
        if selector.field in (1, 2, 3, 4):
            continue
        selector.field = selector_field(selector)
        selector.ClearField("field_name")
        name = field_name(selector.field)
        selected = plan(query.snapshot_ids, name, query.git_commit)
        pin = build_id(selected, name)
        if selector.field_snapshot_id and selector.field_snapshot_id != pin:
            raise ValueError("field derivation pin does not match the built-in recipe")
        selector.field_snapshot_id = pin
        fields.add(pin)
        plans[selected.id] = selected
    for operation in query.operations:
        if not operation.HasField("indexed_dedupe"):
            continue
        selector = operation.indexed_dedupe
        selected = plan(query.snapshot_ids, selector.index_name, query.git_commit, index=True)
        pin = build_id(selected, selector.index_name)
        if selector.index_snapshot_id and selector.index_snapshot_id != pin:
            raise ValueError("index derivation pin does not match the built-in recipe")
        selector.index_snapshot_id = pin
        indexes.add(pin)
        plans[selected.id] = selected
    for name, expected in (("field_snapshot_ids", fields), ("index_snapshot_ids", indexes)):
        supplied = set(getattr(query, name))
        if supplied and supplied != expected:
            raise ValueError(
                "query build pins must exactly match referenced field/index operations"
            )
        del getattr(query, name)[:]
        getattr(query, name).extend(sorted(expected))
    return tuple(plans[id] for id in sorted(plans))


def selectors(query: q.Query | q.CreateQueryRequest) -> Iterator[q.FieldComparison]:
    yield from query.fields
    for operation in query.operations:
        kind = operation.WhichOneof("kind")
        if kind == "field_where":
            yield operation.field_where
        if kind == "dedupe":
            orders = operation.dedupe.order_by
        elif kind == "indexed_dedupe":
            orders = operation.indexed_dedupe.order_by
        elif kind == "similarity_dedupe":
            orders = operation.similarity_dedupe.order_by
        else:
            continue
        for order in orders:
            if order.HasField("selector"):
                yield order.selector
        if kind == "similarity_dedupe" and operation.similarity_dedupe.HasField("embedding"):
            yield operation.similarity_dedupe.embedding
    if query.HasField("sampling"):
        yield from query.sampling.domains


def validate_logical_selector(selector: q.FieldComparison) -> None:
    """Catch catalog/type errors before scheduling a model download."""
    from premixdb.enrichment.types import field
    from premixdb.runtime.enrichment import validate_selector
    from premixdb.v1 import field_pb2 as f
    from premixdb.v1 import query_pb2 as q

    name = field_name(selector_field(selector))
    classes = (
        tuple(value.value for value in Topic)
        if name == "weborganizer.topic"
        else tuple(value.value for value in ContentType)
        if name == "weborganizer.content_type"
        else ()
    )
    if name.startswith("embedding."):
        width = 1024
    else:
        width = len(classes)
    value_type = (
        f.VALUE_STRING
        if name == "language.label"
        else f.VALUE_INT64
        if name in ("datatrove.length", "datatrove.n_words")
        else f.VALUE_FLOAT32
    )
    if name.startswith("language.") and selector.projection not in (
        q.FieldComparison.SCALAR,
        q.FieldComparison.IS_NULL,
    ):
        raise ValueError("language scores are scalar probabilities")
    validate_selector(field(name, width=width, classes=classes, element_type=value_type), selector)
