"""Human-readable resource recipes, using only metadata already on the handle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from . import _resources as sdk

from ._default_tokenizer import _GPT2_DIGEST
from ._field_ids import FIELD_NAMES
from ._ids import _decode_id, _encode_id
from .v1 import data_mixture_pb2 as d
from .v1 import profile_pb2 as p
from .v1 import query_pb2 as q
from .v1 import snapshot_pb2 as s
from .v1 import status_pb2 as status
from .v1.storage_pb2 import Source

_OPERATORS = {1: "==", 2: "!=", 3: "<", 4: "<=", 5: ">", 6: ">="}


def _id(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return _encode_id(value) if len(value) in (16, 32) else value.hex()[:12]
    try:
        return _encode_id(_decode_id(value))
    except ValueError:
        return value[:12]


def _items(values: list[str]) -> str:
    """Keep inventories bounded while preserving order."""
    head = list(values[:8])
    if len(values) > 8:
        head.append(f"... ({len(values) - 8} more)")
    return ", ".join(head) or "none"


def _mapping(values: Mapping[str, float], *, corpus_ids: bool = False) -> str:
    return (
        "{"
        + ", ".join(
            f"{_id(key) if corpus_ids else key!r}: {values[key]!r}" for key in sorted(values)
        )
        + "}"
    )


def _field(value: q.IntrinsicField) -> str:
    return FIELD_NAMES.get(value, f"field({value})")


def _selector(value: q.FieldComparison) -> str:
    name = _field(value.field) if value.field else value.field_name
    if not name:
        name = f"field_snapshot({_id(value.field_snapshot_id)})"
    if value.projection == q.FieldComparison.CLASS_PROBABILITY:
        return f"{name}.probability({value.class_name!r})"
    if value.projection == q.FieldComparison.TOP_CLASS:
        return f"{name}.label"
    if value.projection == q.FieldComparison.IS_NULL:
        return f"{name}.is_null()"
    if value.projection == q.FieldComparison.VECTOR_COMPONENT:
        return f"{name}.component({value.component})"
    return name


def _comparison(value: q.Comparison | q.FieldComparison, field: str) -> str:
    kind = value.WhichOneof("value")
    literal = repr(getattr(value, kind)) if kind else "?"
    return f"{field} {_OPERATORS.get(value.operator, '?')} {literal}"


def _orders(values: Iterable[q.OrderBy]) -> str:
    return (
        ", ".join(
            f"{_selector(v.selector) if v.HasField('selector') else _field(v.field)} "
            f"{'desc' if v.direction == q.OrderBy.DIRECTION_DESC else 'asc'}"
            for v in values
        )
        or "document ID asc"
    )


def _operation(value: q.Operation) -> str:
    kind = value.WhichOneof("kind")
    if kind == "where":
        return "where " + _comparison(value.where, _field(value.where.field))
    if kind == "field_where":
        return "where " + _comparison(value.field_where, _selector(value.field_where))
    if kind == "document_ids":
        return "select document IDs: " + _items([_id(v) for v in value.document_ids.ids])
    if kind == "dedupe":
        policy = value.dedupe
        algorithm = q.Dedupe.Algorithm.Name(policy.algorithm).removeprefix("ALGORITHM_").lower()
        removal = (
            f"source group (separator={policy.source_group_separator!r})"
            if policy.HasField("source_group_separator")
            else "document"
        )
        return f"dedupe {algorithm}; remove {removal}; order by {_orders(policy.order_by)}"
    if kind == "indexed_dedupe":
        policy = value.indexed_dedupe
        index = policy.index_name or f"index_snapshot({_id(policy.index_snapshot_id)})"
        threshold = f"; threshold={policy.threshold:g}" if policy.HasField("threshold") else ""
        return f"indexed dedupe {index}{threshold}; order by {_orders(policy.order_by)}"
    if kind == "similarity_dedupe":
        policy = value.similarity_dedupe
        algorithm = q.SimilarityDedupe.Algorithm.Name(policy.algorithm).lower()
        detail = (
            f"n={policy.n}"
            if policy.algorithm == q.SimilarityDedupe.JACCARD
            else f"embedding={_selector(policy.embedding)}"
        )
        return (
            f"similarity dedupe {algorithm}; threshold={policy.threshold:g}; {detail}; "
            f"order by {_orders(policy.order_by)}"
        )
    return "unspecified operation"


def _source(value: Source) -> str:
    kind = value.WhichOneof("location")
    if kind == "memory":
        result = f"memory ({len(value.memory.documents)} documents)"
    elif kind == "files":
        result = "files: " + _items([repr(v.path) for v in value.files.documents])
    elif kind == "manifest":
        result = "manifest: " + _items([repr(v) for v in sorted(value.manifest.objects)])
    elif kind == "hugging_face":
        source = value.hugging_face
        options = ", ".join(
            f"{name}={getattr(source, name)!r}"
            for name in ("configuration", "split", "revision", "text_column", "key_column")
            if getattr(source, name)
        )
        result = f"Hugging Face {source.repository!r}" + (f" ({options})" if options else "")
    else:
        result = "unspecified"
    if value.HasField("limit"):
        result += f"; limit={value.limit}"
    return result


def _tokenizer(value: d.Tokenizer) -> str:
    kind = value.WhichOneof("kind")
    if kind == "byte":
        return "ByteTokenizer()"
    if kind == "hugging_face":
        policy = value.hugging_face
        if policy.asset.blake3_digest == _GPT2_DIGEST:
            return "GPT2Tokenizer()"
        return (
            f"HuggingFaceTokenizer(asset={policy.asset.uri!r}, "
            f"digest={_id(policy.asset.blake3_digest)}, "
            f"max_document_bytes={policy.max_document_bytes})"
        )
    return f"tokenizer definition {_id(value.definition_digest)}"


def _packing(value: d.Packing) -> str:
    if value.WhichOneof("policy") != "concat":
        return "unspecified"
    policy = value.concat
    separator = policy.separator_token_id if policy.HasField("separator_token_id") else None
    pad = policy.pad_token_id if policy.HasField("pad_token_id") else None
    drop = policy.drop_remainder if policy.HasField("drop_remainder") else pad is None
    return f"Concat(separator={separator!r}, pad={pad!r}, drop_remainder={drop!r})"


def _domains(value: d.Domains) -> str:
    kind = value.WhichOneof("kind")
    if kind == "field":
        return _field(value.field)
    if kind == "fields":
        return ", ".join(_selector(v) for v in value.fields.selectors)
    if kind == "assignments":
        return f"explicit assignments ({len(value.assignments.documents)} documents)"
    return "all documents"


def _sampling(value: d.Sampling) -> list[str]:
    lines = [
        f"Sampling: {value.tokens:,} content tokens; seed={value.seed}; "
        f"replacement={value.replacement}",
        f"Domains: {_domains(value.domains)}",
        f"Weights: {_mapping(value.weights, corpus_ids=value.domains.field == q.FIELD_SOURCE_CORPUS_ID)}",
    ]
    if value.HasField("max_epochs"):
        lines.append(f"Max epochs: {value.max_epochs}")
    return lines


def _query(value: q.Query) -> list[str]:
    lines = ["Snapshots: " + _items([_id(v) for v in value.snapshot_ids])]
    lines.extend(f"{i}. {_operation(v)}" for i, v in enumerate(value.operations, 1))
    if not value.operations:
        lines.append("Select all documents")
    if value.HasField("decontaminate"):
        policy = value.decontaminate
        algorithm = q.Decontaminate.Algorithm.Name(policy.algorithm).removeprefix("ALGORITHM_")
        refs = _items([_id(v) for v in policy.snapshot_ids])
        granularity = q.Decontaminate.Granularity.Name(policy.granularity).lower()
        n = f"; n={policy.n}" if policy.algorithm == q.Decontaminate.ALGORITHM_EXACT_NGRAM else ""
        lines.append(f"Decontaminate: {algorithm.lower()}; {granularity}{n}; references: {refs}")
    if value.HasField("sampling"):
        policy = value.sampling
        budget = policy.WhichOneof("budget")
        amount = getattr(policy, budget) if budget else "?"
        lines.append(
            f"Sampling: {amount} {budget or 'unspecified budget'}; "
            f"seed={policy.seed}; replacement={policy.replacement}"
        )
        if policy.domains:
            lines.append("Sample domains: " + ", ".join(_selector(v) for v in policy.domains))
        if policy.weights:
            lines.append("Sample weights: " + _mapping(policy.weights))
        if budget == "tokens":
            lines.append(
                "Sample tokenizer: "
                + (
                    repr(policy.tokenizer_asset.uri)
                    if policy.HasField("tokenizer_asset")
                    else "byte"
                )
                + f"; max_document_bytes={policy.max_document_bytes}"
            )
    else:
        count = (
            f" ({value.profile.output_documents:,} documents)" if value.HasField("profile") else ""
        )
        lines.append("Sampling: all selected documents once; replacement=False" + count)
    if value.fields:
        lines.append("Fields: " + ", ".join(_selector(v) for v in value.fields))
    for label, ids in (
        ("Field snapshots", value.field_snapshot_ids),
        ("Index snapshots", value.index_snapshot_ids),
    ):
        if ids:
            lines.append(label + ": " + _items([_id(v) for v in ids]))
    return lines


def _documents(value: p.DocumentEstimate) -> str:
    count = f"{value.lower:,}" if value.lower == value.upper else f"{value.lower:,}–{value.upper:,}"
    return f"{count} documents"


def _profile(value: s.Snapshot | q.Query | d.Dataset) -> list[str]:
    """Format attached statistics without fetching metadata or executing a recipe."""
    if isinstance(value, q.Query):
        if value.HasField("profile"):
            query_profile = value.profile
            return [
                f"Input: {query_profile.input_documents:,} documents",
                f"Output: {query_profile.output_documents:,} document occurrences",
                f"Text: {query_profile.output_content_bytes:,} bytes; "
                f"{query_profile.output_characters:,} characters",
            ]
        if value.HasField("estimate"):
            lines = ["Input: " + _documents(value.estimate.input)]
            # The histogram estimate covers operations, before these policies.
            if value.HasField("decontaminate") or value.HasField("sampling"):
                lines.append("Estimated selection: " + _documents(value.estimate.output))
                lines.append("Output: unknown until decontamination or sampling is evaluated")
            else:
                lines.append("Estimated output: " + _documents(value.estimate.output))
            if value.estimate.unavailable_fields:
                lines.append(
                    "Unprofiled fields: "
                    + _items([_field(field) for field in value.estimate.unavailable_fields])
                )
            return lines
        return ["Input: unknown; output: unknown (profile not available)"]
    if not value.HasField("profile"):
        if isinstance(value, d.Dataset):
            return ["Content tokens: unknown; sequences: unknown (profile not computed)"]
        return ["Documents: unknown; text: unknown (profile not available)"]
    if isinstance(value, s.Snapshot):
        snapshot_profile = value.profile
        return [
            f"Documents: {snapshot_profile.documents:,}",
            f"Text: {snapshot_profile.content_bytes:,} bytes; {snapshot_profile.characters:,} characters",
            f"Changes: {snapshot_profile.added:,} added; {snapshot_profile.changed:,} changed; "
            f"{snapshot_profile.removed:,} removed",
        ]
    profile = value.profile
    label = "Output" if value.status == status.STATUS_COMPLETED else "Planned output"
    return [
        f"{label}: {profile.content_tokens:,} content tokens; {profile.sequences:,} sequences",
        f"Source: {profile.source_documents:,} unique documents; "
        f"{profile.document_occurrences:,} document occurrences",
        f"Packing totals: {profile.separator_tokens:,} separator tokens; "
        f"{profile.padding_tokens:,} padding tokens; {profile.dropped_tokens:,} dropped tokens",
    ]


def _resource_repr(
    handle: sdk.Corpus | sdk.Snapshot | sdk.Query | sdk.DataMixture | sdk.Dataset,
) -> str:
    from . import _resources as sdk
    from ._resources import Corpus, DataMixture

    value = handle._resource
    if isinstance(handle, Corpus):
        return f"Corpus(name={handle.name!r}, id={_id(handle._resource.id)!r})"
    header = f"{type(handle).__name__}(id={_id(value.id)!r}"
    if isinstance(handle, (sdk.Snapshot, sdk.Query, sdk.Dataset)):
        header += f", status={handle.status.value!r}"
    header += ")"
    if isinstance(value, s.Snapshot):
        source = (
            _source(value.source)
            if value.source.WhichOneof("location")
            else f"captured inventory ({value.profile.documents:,} documents)"
            if value.HasField("profile")
            else "unavailable"
        )
        lines = [f"Corpus: {_id(value.corpus_id)}", f"Source: {source}"]
        if value.parent_snapshot_id:
            lines.append(f"Base snapshot: {_id(value.parent_snapshot_id)}")
    elif isinstance(value, q.Query):
        lines = _query(value)
    elif isinstance(value, (d.Dataset, d.Mix)):
        lines = [
            f"Query: {_id(value.query_id)}",
            f"Tokenizer: {_tokenizer(value.tokenizer)}",
            f"Sequence length: {value.sequence_length:,}",
            f"Packing: {_packing(value.packing)}",
        ]
        if isinstance(value, d.Dataset):
            if value.HasField("sampling"):
                lines.extend(_sampling(value.sampling))
        else:
            assert isinstance(handle, DataMixture)
            lines.extend(
                [
                    f"Candidates: {len(handle)} of {value.n_candidates or len(value.dataset_ids)}",
                    "Dataset IDs: "
                    + (
                        _items([_id(value.dataset_ids[i]) for i in handle._indices])
                        if value.dataset_ids
                        else "unresolved"
                    ),
                    f"Budget: {value.tokens:,} content tokens"
                    if value.tokens
                    else "Budget: complete query population",
                    f"Domains: {_domains(value.domains)}",
                    f"Draw: seed={value.seed}; replacement={value.replacement}",
                ]
            )
            if value.algorithm.WhichOneof("kind") == "regmix":
                policy = value.algorithm.regmix
                lines.append(
                    f"Weights: RegMix(seed={policy.seed}, prior_power={policy.prior_power:g}, "
                    f"concentration_range=({policy.min_concentration:g}, {policy.max_concentration:g}), "
                    f"concentration_steps={policy.concentration_steps}, "
                    f"minimum_weight={policy.minimum_weight:g}, oversample={policy.oversample})"
                )
            bounds = value.bounds
            corpus_ids = value.domains.field == q.FIELD_SOURCE_CORPUS_ID
            if bounds.lower:
                lines.append("Lower bounds: " + _mapping(bounds.lower, corpus_ids=corpus_ids))
            if bounds.upper:
                lines.append("Upper bounds: " + _mapping(bounds.upper, corpus_ids=corpus_ids))
            for name in ("max_epochs", "reference_tokens"):
                if bounds.HasField(name):
                    lines.append(f"{name}: {getattr(bounds, name):,}")
    else:
        return f"{type(handle).__name__}(name={value.name!r}, id={_id(value.id)!r})"
    if isinstance(value, (s.Snapshot, q.Query, d.Dataset)):
        lines = _profile(value) + lines
    if isinstance(value, (s.Snapshot, q.Query, d.Dataset, d.Mix)) and value.git_commit:
        lines.append(f"Revision: {value.git_commit.hex()[:12]}")
    if hasattr(value, "error") and value.error:
        lines.append(f"Error: {value.error!r}")
    return header + "\n" + "\n".join("  " + line for line in lines)
