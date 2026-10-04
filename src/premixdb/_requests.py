"""Build typed requests without IO, execution handles, or execution-time resource IDs."""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from os import PathLike
from typing import TYPE_CHECKING, Iterable, Literal, Mapping, cast
from urllib.parse import urlsplit

from ._default_tokenizer import _GPT2_DIGEST, _GPT2_EOS, _gpt2_tokenizer
from ._enums import DedupeAlgorithm, IntrinsicField, RemovalUnit
from ._field_expr import FieldPredicate, ScalarField
from ._field_ids import field_id
from ._ids import _decode_id
from ._protobuf import copy_message
from .fields import ContentType, DedupeIndex, Language, Topic
from .v1 import corpus_pb2 as corpora
from .v1 import dataset_pb2 as datasets
from .v1 import query_pb2 as queries
from .v1 import snapshot_pb2 as snapshots
from .v1 import storage_pb2 as source_types
from .v1.storage_pb2 import ObjectRef

if TYPE_CHECKING:
    from ._mixing import Bounds, RegMixSampler, Tokens
    from ._policies import ByteTokenizer, Concat
    from ._profiles import ProfileSelector
    from ._resources import DomainInput


def _id(value: bytes | str, size: int) -> bytes:
    if isinstance(value, str):
        value = _decode_id(value)
    if not isinstance(value, bytes) or len(value) != size:
        raise ValueError(f"ID must contain {size} bytes from a resource")
    return value


def _uint(value: builtins.object, bits: int, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or not int(positive) <= value < 2**bits:
        raise ValueError(
            f"{name} must be {'a positive' if positive else 'an unsigned'} {bits}-bit integer"
        )
    return value


def _preview_options(
    limit: int,
    offset: int,
    max_characters: int,
    *,
    unit: Literal["documents", "sequences"],
) -> tuple[int, int, int]:
    limit = _uint(limit, 32, "limit")
    offset = _uint(offset, 64, "offset")
    max_characters = _uint(max_characters, 32, "max_characters")
    if limit > 1000 or max_characters > 1_000_000:
        raise ValueError(f"preview supports at most 1000 {unit} and 1,000,000 characters")
    return limit, offset, max_characters


@dataclass(frozen=True)
class _TextFields:
    bytes: ScalarField[int] = field(default_factory=lambda: ScalarField(IntrinsicField.BYTES, int))
    characters: ScalarField[int] = field(
        default_factory=lambda: ScalarField(IntrinsicField.CHARACTERS, int)
    )


@dataclass(frozen=True)
class _SourceFields:
    corpus_id: ScalarField[str] = field(
        default_factory=lambda: ScalarField(IntrinsicField.CORPUS_ID, str)
    )


@dataclass(frozen=True)
class _ObjectFields:
    uri: ScalarField[str] = field(
        default_factory=lambda: ScalarField(IntrinsicField.OBJECT_URI, str)
    )


text = _TextFields()
source = _SourceFields()
object = _ObjectFields()


@dataclass(frozen=True)
class _DocumentPredicate:
    ids: tuple[bytes, ...]

    def __bool__(self) -> bool:
        raise TypeError("use where(document_id.is_in(ids)) instead of and/or")


class _DocumentIdField:
    def is_in(self, ids: Iterable[bytes | str]) -> _DocumentPredicate:
        """Select document IDs supplied as base64url, hexadecimal, or raw bytes."""
        if isinstance(ids, (str, bytes)):
            raise TypeError("is_in() expects an iterable of document IDs")
        return _DocumentPredicate(tuple(sorted({_id(value, 32) for value in ids})))


document_id = _DocumentIdField()


def where(predicate: FieldPredicate | _DocumentPredicate) -> queries.Operation:
    """Build an ordered filter operation; never evaluate corpus data."""
    if isinstance(predicate, FieldPredicate):
        return predicate._operation()
    if isinstance(predicate, _DocumentPredicate):
        return queries.Operation(document_ids=queries.DocumentSelection(ids=predicate.ids))
    raise TypeError("where() expects a field comparison")


@dataclass(frozen=True)
class SourceGroup:
    """Remove members sharing a corpus and source-key prefix before separator."""

    separator: str = "/"

    def __post_init__(self) -> None:
        if not isinstance(self.separator, str):
            raise TypeError("separator must be a string")
        if (
            len(self.separator) != 1
            or self.separator == "\0"
            or 0xD800 <= ord(self.separator) <= 0xDFFF
        ):
            raise ValueError("separator must be one non-NUL Unicode scalar")


def dedupe(
    *,
    algorithm: DedupeAlgorithm = DedupeAlgorithm.EXACT_DOCUMENT,
    removal: RemovalUnit | SourceGroup = RemovalUnit.DOCUMENT,
    order_by: Iterable[queries.OrderBy] = (),
) -> queries.Operation:
    """The algorithm fixes the comparison unit; policies choose winners/removals."""
    if not isinstance(algorithm, DedupeAlgorithm):
        raise TypeError("algorithm must be a DedupeAlgorithm")
    algorithms = {
        DedupeAlgorithm.EXACT_DOCUMENT: queries.Dedupe.ALGORITHM_EXACT_DOCUMENT,
        DedupeAlgorithm.EXACT_LINE: queries.Dedupe.ALGORITHM_EXACT_LINE,
    }
    policy = queries.Dedupe(algorithm=algorithms[algorithm], order_by=list(order_by))
    if isinstance(removal, SourceGroup):
        policy.source_group_separator = removal.separator
    elif removal != RemovalUnit.DOCUMENT:
        raise TypeError("removal must be RemovalUnit.DOCUMENT or SourceGroup")
    return queries.Operation(dedupe=policy)


def gpt2_tokenizer() -> datasets.Tokenizer:
    """Use the bundled GPT-2 BPE vocabulary without downloading model weights."""
    return _gpt2_tokenizer()


def byte_tokenizer() -> datasets.Tokenizer:
    """Describe byte tokenization; the executor resolves its definition identity."""
    return datasets.Tokenizer(byte=datasets.ByteTokenizer())


def _asset(asset: ObjectRef) -> None:
    location = urlsplit(asset.uri)
    if location.scheme != "file" or location.netloc not in ("", "localhost"):
        raise ValueError("assets require a local file URI")
    if len(asset.blake3_digest) != 32:
        raise ValueError("assets require a 32-byte BLAKE3 pin")


def hugging_face_tokenizer(
    asset: ObjectRef | str | PathLike[str],
    *,
    digest: bytes | str | None = None,
    max_document_bytes: int = 8 * 1024 * 1024,
) -> datasets.Tokenizer:
    """Describe a pinned local tokenizer asset, capturing JSON from a file when supplied."""
    data = b""
    if not isinstance(asset, ObjectRef):
        from pathlib import Path

        from blake3 import blake3

        data = Path(asset).read_bytes()
        actual = blake3(data).digest()
        if digest is None or _id(digest, 32) != actual:
            raise ValueError("local tokenizer requires its expected BLAKE3 digest")
        asset = ObjectRef(uri="inline://tokenizer", blake3_digest=actual, size_bytes=len(data))
    else:
        _asset(asset)
    return datasets.Tokenizer(
        hugging_face=datasets.HuggingFaceTokenizer(
            asset=asset,
            json=data,
            max_document_bytes=_uint(max_document_bytes, 64, "max_document_bytes", positive=True),
        )
    )


def concat(
    *,
    separator: int | None = None,
    drop_remainder: bool = True,
    pad_token: int | None = None,
) -> datasets.Packing:
    """Describe explicit separators and drop/pad behavior, including presence bits."""
    from ._policies import Concat

    return Concat(separator, drop_remainder, pad_token)._to_proto()


def _commit(value: bytes | str | None = None) -> bytes:
    if value is None or value == b"":
        return b""
    if isinstance(value, str):
        value = bytes.fromhex(value)
    if not isinstance(value, bytes) or len(value) not in (20, 32):
        raise ValueError("git_commit must be a full 20- or 32-byte Git object ID")
    return value


def corpus(name: str, *, request_id: str = "") -> corpora.CreateCorpusRequest:
    """Build a request to create or reopen a named corpus."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("corpus name must be a nonempty string")
    return corpora.CreateCorpusRequest(name=name, request_id=request_id)


def snapshot(
    corpus: corpora.Corpus | bytes | str,
    *,
    source: source_types.Source,
    base: snapshots.Snapshot | bytes | str | None = None,
    git_commit: bytes | str | None = None,
    request_id: str = "",
) -> snapshots.CreateSnapshotRequest:
    """Capture a full source inventory, optionally relative to a previous snapshot."""
    if not isinstance(source, source_types.Source) or source.WhichOneof("location") is None:
        raise ValueError("source must be a protobuf Source with a location")
    kind = source.WhichOneof("location")
    if kind == "files" and any(not item.path.strip() for item in source.files.documents):
        raise ValueError("source file paths must be nonempty")
    if kind == "memory" and any(not item.uri for item in source.memory.documents):
        raise ValueError("source URIs must be nonempty")
    if kind == "hugging_face":
        hf = source.hugging_face
        if not hf.repository.strip() or not hf.split.strip():
            raise ValueError("Hugging Face sources require repository and split")
        if len(hf.revision) != 40 or any(c not in "0123456789abcdef" for c in hf.revision):
            raise ValueError("Hugging Face sources require a full lowercase commit revision")
    if kind == "manifest":
        for key, asset in source.manifest.objects.items():
            if not key:
                raise ValueError("source keys must be nonempty")
            _asset(asset)
    corpus_id = _id(corpus.id if isinstance(corpus, corpora.Corpus) else corpus, 16)
    if isinstance(base, snapshots.Snapshot):
        if base.corpus_id != corpus_id:
            raise ValueError("base snapshot must belong to the same corpus")
        base = base.id
    return snapshots.CreateSnapshotRequest(
        request_id=request_id,
        corpus_id=corpus_id,
        source=source,
        parent_snapshot_id=_id(base, 32) if base is not None else b"",
        git_commit=_commit(git_commit),
    )


def query(
    *inputs: snapshots.Snapshot | bytes | str,
    steps: Iterable[queries.Operation] = (),
    fields: Iterable[ProfileSelector] = (),
    git_commit: bytes | str | None = None,
    field_snapshot_ids: Iterable[bytes | str] = (),
    index_snapshot_ids: Iterable[bytes | str] = (),
    decontaminate: queries.Decontaminate | None = None,
    sampling: queries.QuerySampling | None = None,
    request_id: str = "",
) -> queries.CreateQueryRequest:
    """Build a query request over unique snapshots, applying ordered steps and projections."""
    ids = sorted(
        {_id(item.id if isinstance(item, snapshots.Snapshot) else item, 32) for item in inputs}
    )
    if not ids:
        raise ValueError("query requires at least one snapshot")
    operations = list(steps)
    for index, op in enumerate(operations):
        if isinstance(op, queries.Decontaminate):
            raise ValueError(
                f"steps[{index}] contains p.decontaminate(...); pass it as "
                "query(decontaminate=p.decontaminate(...)) instead of inside steps"
            )
        if isinstance(op, queries.QuerySampling):
            raise ValueError(
                f"steps[{index}] contains p.sample(...); pass it as "
                "query(sampling=p.sample(...)) instead of inside steps"
            )
        if not isinstance(op, queries.Operation):
            raise ValueError(
                f"steps[{index}] is {type(op).__name__}; expected a query step "
                "created by p.where(...), p.dedupe(...), or another step constructor"
            )
        if op.WhichOneof("kind") is None:
            raise ValueError(
                f"steps[{index}] is an empty query step; "
                "use p.where(...), p.dedupe(...), or another step constructor"
            )
    from ._curation import selector

    projections = [selector(value) for value in fields]
    if any(value.operator or value.WhichOneof("value") is not None for value in projections):
        raise ValueError("query fields must be projections without comparison values")
    return queries.CreateQueryRequest(
        request_id=request_id,
        snapshot_ids=ids,
        operations=operations,
        fields=projections,
        git_commit=_commit(git_commit),
        decontaminate=decontaminate,
        sampling=sampling,
        field_snapshot_ids=sorted({_id(id, 32) for id in field_snapshot_ids}),
        index_snapshot_ids=sorted({_id(id, 32) for id in index_snapshot_ids}),
    )


def dataset(
    query: queries.Query | bytes | str,
    *,
    tokenizer: datasets.Tokenizer | ByteTokenizer | None = None,
    sequence_length: int = 2048,
    packing: datasets.Packing | Concat | None = None,
    sampling: datasets.Sampling | None = None,
    git_commit: bytes | str | None = None,
    request_id: str = "",
) -> datasets.CreateDatasetRequest:
    """Default to offline GPT-2 BPE, 2,048-token sequences, and a padded final sequence."""
    query_id = _id(query.id if isinstance(query, queries.Query) else query, 32)
    tokenizer = gpt2_tokenizer() if tokenizer is None else tokenizer
    from ._policies import ByteTokenizer, Concat

    if isinstance(tokenizer, ByteTokenizer):
        tokenizer = tokenizer._to_proto()
    if not isinstance(tokenizer, datasets.Tokenizer) or tokenizer.WhichOneof("kind") is None:
        raise ValueError("tokenizer requires a typed policy")
    if tokenizer.HasField("hugging_face"):
        if tokenizer.hugging_face.json:
            from blake3 import blake3

            if (
                blake3(tokenizer.hugging_face.json).digest()
                != tokenizer.hugging_face.asset.blake3_digest
            ):
                raise ValueError("tokenizer asset digest mismatch")
        else:
            _asset(tokenizer.hugging_face.asset)
        _uint(tokenizer.hugging_face.max_document_bytes, 64, "max_document_bytes", positive=True)
    if packing is None:
        packing = (
            concat(separator=256, drop_remainder=False, pad_token=257)
            if tokenizer.HasField("byte")
            else concat(separator=_GPT2_EOS, drop_remainder=False, pad_token=_GPT2_EOS)
            if tokenizer.hugging_face.asset.blake3_digest == _GPT2_DIGEST
            else concat()
        )
    if isinstance(packing, Concat):
        packing = packing._to_proto()
    if not isinstance(packing, datasets.Packing) or packing.WhichOneof("policy") != "concat":
        raise ValueError("packing requires a concat policy")
    packing = copy_message(packing)
    policy = packing.concat
    if not policy.HasField("drop_remainder"):
        policy.drop_remainder = not policy.HasField("pad_token_id")
    if policy.drop_remainder == policy.HasField("pad_token_id"):
        raise ValueError("pad_token_id is required exactly when drop_remainder=False")
    if sampling is not None and sampling.domains.field == queries.FIELD_SOURCE_CORPUS_ID:
        sampling = copy_message(sampling)
        weights = {_id(key, 16).hex(): value for key, value in sampling.weights.items()}
        sampling.weights.clear()
        sampling.weights.update(weights)
    return datasets.CreateDatasetRequest(
        request_id=request_id,
        query_id=query_id,
        tokenizer=tokenizer,
        packing=packing,
        sequence_length=_uint(sequence_length, 32, "sequence_length", positive=True),
        sampling=sampling,
        git_commit=_commit(git_commit),
    )


def mix(
    query: queries.Query | bytes | str,
    *,
    domains: DomainInput | None = None,
    sampler: RegMixSampler | None = None,
    size: Tokens | None = None,
    tokens: int | None = None,
    tokenizer: datasets.Tokenizer | ByteTokenizer | None = None,
    sequence_length: int = 2048,
    packing: datasets.Packing | Concat | None = None,
    bounds: Bounds | None = None,
    n_candidates: int = 3,
    replacement: bool = True,
    seed: int = 0,
    git_commit: bytes | str | None = None,
    request_id: str = "",
) -> datasets.CreateMixRequest:
    """Three seeded RegMix candidates over corpus domains by default.

    An omitted token budget resolves to one population's worth of content tokens
    during execution. Pass n_candidates and a budget to explore larger searches.
    """
    from ._mixing import Bounds, RegMixSampler, Tokens

    if size is not None:
        if not isinstance(size, Tokens):
            raise TypeError("size must be Tokens(count, tokenizer=...)")
        if tokens is not None or tokenizer is not None:
            raise ValueError("use size or tokens/tokenizer, not both")
        tokens, tokenizer = size.count, size.tokenizer
    sampler = RegMixSampler() if sampler is None else sampler
    if not isinstance(sampler, RegMixSampler):
        raise TypeError("sampler must be RegMixSampler")
    if bounds is not None and not isinstance(bounds, Bounds):
        raise TypeError("bounds must be Bounds")
    if type(replacement) is not bool:
        raise TypeError("replacement must be a bool")
    partition = datasets.Domains()
    strata = source.corpus_id if domains is None else domains
    if (
        isinstance(strata, ScalarField)
        and field_id(strata.name) <= queries.FIELD_SOURCE_CORPUS_ID
        and strata.projection == queries.FieldComparison.SCALAR
    ):
        partition.field = field_id(strata.name)
    elif isinstance(strata, Mapping):
        labels = {_id(id, 32).hex(): label for id, label in cast(Mapping[str, str], strata).items()}
        if any(not isinstance(label, str) or not label for label in labels.values()):
            raise ValueError("assignment labels must be nonempty strings")
        partition.assignments.CopyFrom(datasets.MixAssignments(documents=labels))
    elif isinstance(strata, datasets.Domains):
        partition.CopyFrom(strata)
    else:
        from ._curation import ClassifierProjection, FieldSelector, selector
        from ._field_expr import FieldProjection, VectorField

        values: Iterable[FieldSelector]
        if isinstance(strata, type):
            if strata is not Topic and strata is not ContentType and strata is not Language:
                raise TypeError("expected Topic, ContentType, or Language")
            values = (strata,)
        elif isinstance(strata, (str, FieldProjection, ClassifierProjection, VectorField)):
            values = (strata,)
        else:
            values = strata
        partition.fields.selectors.extend(selector(value) for value in values)
    template = dataset(
        query,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        packing=packing,
        git_commit=git_commit,
    )
    bounds_proto = (bounds or Bounds())._to_proto()
    if partition.field == queries.FIELD_SOURCE_CORPUS_ID:
        for bound_values in (bounds_proto.lower, bounds_proto.upper):
            normalized = {_id(key, 16).hex(): value for key, value in bound_values.items()}
            bound_values.clear()
            bound_values.update(normalized)
    return datasets.CreateMixRequest(
        request_id=request_id,
        query_id=template.query_id,
        tokenizer=template.tokenizer,
        sequence_length=template.sequence_length,
        packing=template.packing,
        domains=partition,
        algorithm=sampler._to_proto(),
        bounds=bounds_proto,
        tokens=0 if tokens is None else _uint(tokens, 64, "tokens", positive=True),
        n_candidates=_uint(n_candidates, 32, "n_candidates", positive=True),
        seed=_uint(seed, 64, "data seed"),
        replacement=replacement,
        git_commit=template.git_commit,
    )


def indexed_dedupe(
    index: DedupeIndex = DedupeIndex.EXACT_DOCUMENT,
    *,
    order_by: Iterable[queries.OrderBy] = (),
    threshold: float | None = None,
) -> queries.Operation:
    """Request a built-in index; its evidence is derived lazily by the planner."""
    if not isinstance(index, DedupeIndex):
        raise TypeError("indexed_dedupe expects DedupeIndex")
    return queries.Operation(
        indexed_dedupe=queries.IndexedDedupe(
            index_name=index.value, order_by=list(order_by), threshold=threshold
        )
    )
