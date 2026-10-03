"""Python inspection helpers for profiles and deterministic example drilldowns."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, cast, overload

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

from .._ids import _decode_id
from .._typing import JSON, FieldValue, Scalar, json_object, load_json, scalar
from ..engine.contracts import Provenance
from ..v1 import corpus_pb2 as c
from ..v1 import dataset_pb2 as d
from ..v1 import query_pb2 as q
from ..v1 import snapshot_pb2 as s
from ..v1 import status_pb2
from .enrichment import numeric_vector

if TYPE_CHECKING:
    from .coordinator import Coordinator

type Parameters = Mapping[str, Sequence[str]]
type Kind = Literal["snapshot", "query", "dataset", "corpus"]


def _json_record(value: object) -> dict[str, JSON]:
    return json_object(load_json(json.dumps(value)))


def _parameter(parameters: Parameters, name: str) -> str | None:
    values = parameters.get(name)
    return values[0] if values else None


def _between(value: FieldValue, lower: Scalar, upper: Scalar) -> bool:
    if isinstance(value, str) and isinstance(lower, str) and isinstance(upper, str):
        return lower <= value <= upper
    if (
        isinstance(value, (int, float))
        and isinstance(lower, (int, float))
        and isinstance(upper, (int, float))
    ):
        return lower <= value <= upper
    return False


class Example(TypedDict):
    id: str
    source: str
    text: str


class Cell(TypedDict):
    topic: str
    quality: str
    documents: int
    bytes: int
    examples: list[Example]


class DocumentExample(TypedDict):
    id: str
    ordinal: int
    source: str
    text: str
    provenance: Provenance | None


class DocumentPage(TypedDict):
    total: int
    offset: int
    rows: list[DocumentExample]


class BuildEntry(TypedDict):
    id: str
    name: str
    git_commit: str
    documents: str
    shards: int
    pinned: bool
    profile: NotRequired[dict[str, JSON]]
    approximate: NotRequired[bool]


class Coverage(TypedDict):
    fields: list[BuildEntry]
    indexes: list[BuildEntry]


class IndexExample(Example):
    group: str


class IndexStatistics(TypedDict):
    documents: str
    duplicate_groups: str
    duplicate_documents: str
    group_sizes: dict[str, str]
    candidate_links: str
    candidate_documents: str
    approximate: bool
    examples: list[IndexExample]


class SequenceExample(TypedDict):
    ordinal: int
    tokens: list[int]
    text: str
    regions: list[dict[str, JSON]]
    truncated: bool


class SequencePage(TypedDict):
    total: int
    offset: int
    rows: list[SequenceExample]


@overload
def resource(service: Coordinator, kind: Literal["snapshot"], identity: str) -> s.Snapshot: ...
@overload
def resource(service: Coordinator, kind: Literal["query"], identity: str) -> q.Query: ...
@overload
def resource(service: Coordinator, kind: Literal["dataset"], identity: str) -> d.Dataset: ...
@overload
def resource(service: Coordinator, kind: Literal["corpus"], identity: str) -> c.Corpus: ...
@overload
def resource(
    service: Coordinator, kind: Kind, identity: str
) -> s.Snapshot | q.Query | d.Dataset | c.Corpus: ...
def resource(
    service: Coordinator, kind: Kind, identity: str
) -> s.Snapshot | q.Query | d.Dataset | c.Corpus:
    decoded = _decode_id(identity)
    if kind == "snapshot":
        return service.GetSnapshot(s.GetSnapshotRequest(id=decoded)).snapshot
    if kind == "query":
        return service.GetQuery(q.GetQueryRequest(id=decoded)).query
    if kind == "dataset":
        return service.GetDataset(d.GetDatasetRequest(id=decoded)).dataset
    return service.GetCorpus(c.GetCorpusRequest(id=decoded)).corpus


def view(message: Message) -> dict[str, JSON]:
    result = _json_record(
        MessageToDict(
            message, preserving_proto_field_name=True, always_print_fields_with_no_presence=True
        )
    )
    id = getattr(message, "id", None)
    if isinstance(id, bytes):
        result["id"] = id.hex()
    return result


def catalog(service: Coordinator) -> dict[str, list[dict[str, JSON]]]:
    return dict(
        corpus=[view(r) for r in service._storage.list("corpus", c.Corpus)],
        snapshot=[view(r) for r in service._storage.list("snapshot", s.Snapshot)],
        query=[view(r) for r in service._resources("query", q.Query, service._queries, ".pending")],
        dataset=[
            view(r) for r in service._resources("dataset", d.Dataset, service._datasets, ".recipe")
        ],
    )


def history(service: Coordinator, identity: str) -> list[dict[str, JSON]]:
    identity = _decode_id(identity).hex()
    events, page = [], b""
    while True:
        response = service.ListExecutions(status_pb2.ListExecutionRequest(page_token=page))
        events.extend(response.events)
        page = response.next_page_token
        if not page:
            break
    times = {event.resource_id: event.started_ns for event in reversed(events)}
    return [
        _json_record(dict(**view(snapshot), captured_ns=str(times.get(snapshot.id, 0))))
        for snapshot in sorted(
            (
                s
                for s in service._storage.list("snapshot", s.Snapshot)
                if s.corpus_id.hex() == identity
            ),
            key=lambda snapshot: (times.get(snapshot.id, 0), snapshot.id),
        )
    ]


def rows(service: Coordinator, kind: Kind, identity: str, parameters: Parameters) -> DocumentPage:
    from ..engine.queries import Row
    from ..enrichment.classification import probabilities, top_class
    from .enrichment import decode_value, load_build, read_rows

    offset = int(parameters.get("offset", ["0"])[0])
    if offset < 0:
        raise ValueError("invalid page offset")
    recipe = resource(service, kind, identity)
    if not isinstance(recipe, (q.Query, s.Snapshot)):
        raise ValueError("document drilldown requires a query or snapshot")
    selected: Iterable[Row]
    population = parameters.get("population", ["output"])[0]
    if population not in ("input", "output"):
        raise ValueError("invalid document population")
    if isinstance(recipe, q.Query) and population == "output":
        selected = service._query(_decode_id(identity)).rows()
    elif kind == "snapshot" or kind == "query":
        snapshots = recipe.snapshot_ids if isinstance(recipe, q.Query) else [_decode_id(identity)]
        documents = service._query_index(snapshots).documents
        selected = [Row(i, doc) for i, doc in enumerate(documents.values())]
    else:
        raise ValueError("document drilldown requires a query or snapshot")
    lineage: dict[str, Provenance] = {}
    if isinstance(recipe, q.Query):
        lineage = service._query(_decode_id(identity)).provenance()
        if "stage" in parameters:
            stage = int(parameters["stage"][0])
            if not 0 <= stage < len(recipe.operations):
                raise ValueError("invalid ordered query stage")
            outcome = parameters.get("outcome", ["retained"])[0]
            if outcome not in ("removed", "retained"):
                raise ValueError("invalid stage outcome")
            selected = [
                row
                for row in selected
                if (
                    lineage[row.id]["selection"].get("step") == stage
                    if outcome == "removed"
                    else lineage[row.id]["selection"].get("step", len(recipe.operations)) > stage
                )
            ]
        if "contamination" in parameters:
            matched = parameters["contamination"][0]
            if matched not in ("matched", "removed"):
                raise ValueError("invalid contamination predicate")
            selected = [
                row
                for row in selected
                if lineage[row.id].get("contamination")
                and (matched != "removed" or lineage[row.id]["selection"]["kind"] == "contaminated")
            ]
        if "stratum" in parameters:
            from .._field_ids import field_id
            from ..engine.curation import selector_key
            from .enrichment import project

            projections: dict[str, dict[str, FieldValue]] = {}
            for selection in recipe.sampling.domains:
                if selection.field <= q.FIELD_SOURCE_CORPUS_ID:
                    projections[selector_key(selection)] = {
                        row.id: row.document.size
                        if selection.field == q.FIELD_TEXT_BYTES
                        else row.document.characters
                        if selection.field == q.FIELD_TEXT_CHARACTERS
                        else row.source_key
                        if selection.field == q.FIELD_OBJECT_URI
                        else row.corpus_id
                        for row in selected
                    }
                else:
                    build, manifest = load_build(
                        service, "field", selection.field_snapshot_id, recipe.snapshot_ids
                    )
                    if field_id(build.field.name) != selection.field:
                        raise ValueError("sampling projection does not match its field")
                    projections[selector_key(selection)] = {
                        value.document_id.hex(): project(
                            build.field, selection, decode_value(build.field, value)
                        )
                        for value in read_rows(service, "field", manifest)
                    }

            def stratum(row: Row) -> str:
                values = [projections[selector_key(s)][row.id] for s in recipe.sampling.domains]
                return (
                    values[0]
                    if len(values) == 1 and isinstance(values[0], str)
                    else json.dumps(values, ensure_ascii=False, separators=(",", ":"))
                )

            selected = [row for row in selected if stratum(row) == parameters["stratum"][0]]
    if "field" in parameters:
        name = parameters["field"][0]
        field = q.IntrinsicField.Value(name)
        values: dict[str, FieldValue] = {}
        if field <= q.FIELD_SOURCE_CORPUS_ID:
            for row in selected:
                values[row.id] = (
                    row.document.size
                    if field == q.FIELD_TEXT_BYTES
                    else row.document.characters
                    if field == q.FIELD_TEXT_CHARACTERS
                    else row.source_key
                    if field == q.FIELD_OBJECT_URI
                    else row.corpus_id
                )
        else:
            population_ids = recipe.snapshot_ids if isinstance(recipe, q.Query) else [recipe.id]
            build_ids = (
                [bytes.fromhex(parameters["build"][0])]
                if "build" in parameters
                else recipe.field_snapshot_ids
                if isinstance(recipe, q.Query)
                else []
            )
            if not build_ids:
                raise ValueError("derived drilldown requires an available field build")
            for build_id in build_ids:
                build, manifest = load_build(service, "field", build_id, population_ids)
                from .._field_ids import field_id

                if field_id(build.field.name) != field:
                    continue
                projection = parameters.get("projection", ["SCALAR"])[0]
                for value_row in read_rows(service, "field", manifest):
                    value = decode_value(build.field, value_row)
                    if value is not None:
                        if projection == "TOP_CLASS":
                            value = top_class(build.field, numeric_vector(value))[0]
                        elif projection == "CLASS_PROBABILITY":
                            value = probabilities(build.field, numeric_vector(value))[
                                parameters["class_name"][0]
                            ]
                    values[value_row.document_id.hex()] = value

        def endpoint(name: str) -> Scalar:
            value = json_object(load_json(parameters[name][0]))
            key, raw = next(iter(value.items()))
            value = scalar(raw)
            if key in ("count", "integer"):
                if not isinstance(value, (str, int, float)):
                    raise ValueError("expected integer bound")
                return int(value)
            return value

        lower, upper = endpoint("lower"), endpoint("upper")
        selected = [r for r in selected if _between(values.get(r.id), lower, upper)]
    selected = list(selected)
    return DocumentPage(
        total=len(selected),
        offset=offset,
        rows=[
            dict(
                id=r.id,
                ordinal=r.ordinal,
                source=r.source_key,
                text=r.text[:4096],
                provenance=lineage.get(r.id),
            )
            for r in selected[offset : offset + 100]
        ],
    )


def coverage(service: Coordinator, kind: Kind, identity: str) -> Coverage:
    """List only completed builds covering this exact immutable population."""
    from ..internal import derivation_pb2 as e
    from .enrichment import load_build

    recipe = resource(service, kind, identity)
    if not isinstance(recipe, (q.Query, s.Snapshot)):
        raise ValueError("coverage requires a snapshot or query")
    population = list(recipe.snapshot_ids) if isinstance(recipe, q.Query) else [recipe.id]
    result: Coverage = Coverage(fields=[], indexes=[])
    for prefix, items in (
        ("field", service._storage.list("field", e.FieldBuild)),
        ("index", service._storage.list("index", e.IndexBuild)),
    ):
        for item in items:
            if list(item.snapshot.snapshot_ids) != population:
                continue
            if isinstance(item, e.FieldBuild):
                build, manifest = load_build(service, "field", item.snapshot.id, population)
                definition = build.field
            else:
                build, manifest = load_build(service, "index", item.snapshot.id, population)
                definition = build.index
            entry: BuildEntry = BuildEntry(
                id=build.snapshot.id.hex(),
                name=definition.name,
                git_commit=build.snapshot.git_commit.hex(),
                documents=str(manifest.documents),
                shards=len(manifest.shards),
                pinned=isinstance(recipe, q.Query)
                and build.snapshot.id
                in (recipe.field_snapshot_ids if prefix == "field" else recipe.index_snapshot_ids),
            )
            if isinstance(build, e.FieldBuild):
                entry["profile"] = view(build.snapshot.profile)
                result["fields"].append(entry)
            else:
                entry.update(approximate=build.index.approximate)
                result["indexes"].append(entry)
    return result


def index_statistics(
    service: Coordinator, identity: str, parameters: Parameters
) -> IndexStatistics:
    """Exact group and candidate counts; approximate links stay labelled candidates."""
    import sqlite3
    from tempfile import TemporaryDirectory

    from ..internal import derivation_pb2 as e
    from .enrichment import load_build, read_rows

    build = service._storage.load("index", _decode_id(identity), e.IndexBuild)
    build, manifest = load_build(service, "index", build.snapshot.id, build.snapshot.snapshot_ids)
    with TemporaryDirectory(prefix="premixdb-index-profile-") as directory:
        database = sqlite3.connect(Path(directory) / "profile.sqlite3")
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                "CREATE TABLE exact (hash BLOB,id TEXT,PRIMARY KEY(hash,id)) WITHOUT ROWID"
            )
            database.execute(
                "CREATE TABLE bands (band INTEGER,bucket BLOB,id TEXT,PRIMARY KEY(band,bucket,id)) WITHOUT ROWID"
            )
            for row in read_rows(service, "index", manifest):
                if len(row.exact_hash) != 32:
                    raise ValueError("invalid exact hash evidence")
                identity = row.document_id.hex()
                database.execute("INSERT INTO exact VALUES (?,?)", (row.exact_hash, identity))
                database.executemany(
                    "INSERT INTO bands VALUES (?,?,?)",
                    (
                        (band, bucket.to_bytes(8, "big"), identity)
                        for band, bucket in enumerate(row.lsh_buckets)
                    ),
                )
            database.execute(
                "CREATE TABLE groups AS SELECT hash,COUNT(*) AS size FROM exact GROUP BY hash HAVING COUNT(*)>1"
            )
            histogram = dict(
                cast(
                    Iterable[tuple[int, int]],
                    database.execute(
                        "SELECT size,COUNT(*) FROM groups GROUP BY size ORDER BY size"
                    ),
                )
            )
            database.execute(
                "CREATE TABLE candidates AS SELECT DISTINCT a.id AS a,b.id AS b FROM bands a JOIN bands b ON a.band=b.band AND a.bucket=b.bucket WHERE a.id<b.id"
            )
            candidate_links = database.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            participants = database.execute(
                "SELECT COUNT(*) FROM (SELECT a FROM candidates UNION SELECT b FROM candidates)"
            ).fetchone()[0]
            size = _parameter(parameters, "size")
            if size is not None and int(size) < 2:
                raise ValueError("duplicate group size must be at least two")
            hashes = database.execute(
                "SELECT hash FROM groups WHERE size=? ORDER BY hash LIMIT 10"
                if size
                else "SELECT hash FROM groups ORDER BY hash LIMIT 10",
                (int(size),) if size else (),
            ).fetchall()
            index = service._query_index(build.snapshot.snapshot_ids)
            examples: list[IndexExample] = []
            for (hash,) in hashes:
                for (id,) in database.execute(
                    "SELECT id FROM exact WHERE hash=? ORDER BY id LIMIT 10", (hash,)
                ):
                    document = index.documents[id]
                    examples.append(
                        dict(
                            id=id,
                            source=document.source_key,
                            text=document.text[:4096],
                            group=hash.hex(),
                        )
                    )
            return IndexStatistics(
                documents=str(manifest.documents),
                duplicate_groups=str(sum(histogram.values())),
                duplicate_documents=str(
                    sum((size - 1) * count for size, count in histogram.items())
                ),
                group_sizes={str(size): str(count) for size, count in histogram.items()},
                candidate_links=str(candidate_links),
                candidate_documents=str(participants),
                approximate=build.index.approximate,
                examples=examples,
            )
        finally:
            database.close()


def sequences(service: Coordinator, identity: str, parameters: Parameters) -> SequencePage:
    """Scan verified sequence metadata, reading token bytes only for page examples."""
    from blake3 import blake3

    from .._resources import Sequence
    from .._storage import RangeReader
    from .tokens import decode_preview

    offset = int(parameters.get("offset", ["0"])[0])
    if offset < 0:
        raise ValueError("invalid page offset")
    recipe = resource(service, "dataset", identity)
    tokenizer = service._tokenizer(recipe)
    source = _parameter(parameters, "source")
    if source is not None:
        source = _decode_id(source).hex()
    stratum = _parameter(parameters, "stratum")
    if stratum is not None and recipe.sampling.domains.field == q.FIELD_SOURCE_CORPUS_ID:
        stratum = _decode_id(stratum).hex()
    documents = _parameter(parameters, "documents")
    crossing = _parameter(parameters, "crossing")
    padding = _parameter(parameters, "padding")
    if crossing not in (None, "true") or padding not in (None, "true"):
        raise ValueError("invalid sequence predicate")
    lineage = service._query(recipe.query_id).provenance() if source else {}
    labels = service._sampling(recipe)[0].labels if stratum else {}
    store = service._storage
    reader = RangeReader(local_root=store.root)
    rows: list[SequenceExample] = []
    total, ordinal = 0, 0
    for ref in recipe.sequences:
        data = store.read_object("dataset", ref.object)[ref.start : ref.end]
        if blake3(data).digest() != ref.blake3_digest:
            raise ValueError("sequence metadata integrity check failed")
        batch = d.SequenceBatch()
        batch.ParseFromString(data)
        for sequence in batch.sequences:
            if sequence.ordinal != ordinal:
                raise ValueError("sequence metadata coverage is not contiguous")
            ordinal += 1
            members = {r.document_id.hex() for r in sequence.regions if r.document_id}
            content = {
                r.document_id.hex()
                for r in sequence.regions
                if r.kind == d.TokenRegion.KIND_CONTENT
            }
            if (
                (source and not any(lineage[id]["corpus_id"] == source for id in content))
                or (stratum and not any(labels[id] == stratum for id in content))
                or (documents is not None and len(members) != int(documents))
                or (crossing and len(members) <= 1)
                or (
                    padding
                    and not any(r.kind == d.TokenRegion.KIND_PADDING for r in sequence.regions)
                )
            ):
                continue
            if offset <= total < offset + 20:
                tokens = Sequence(sequence, reader).tokens[:256]
                rows.append(
                    dict(
                        ordinal=sequence.ordinal,
                        tokens=tokens,
                        text=decode_preview(tokens, sequence.regions, tokenizer),
                        regions=[view(r) for r in sequence.regions],
                        truncated=recipe.sequence_length > len(tokens),
                    )
                )
            total += 1
    if ordinal != recipe.profile.sequences:
        raise ValueError("incomplete sequence metadata")
    return SequencePage(total=total, offset=offset, rows=rows)


def matrix(service: Coordinator, identity: str) -> list[Cell]:
    from .. import quality, topic
    from .._curation import selector
    from .._field_ids import field_id
    from ..engine.curation import selector_key
    from .enrichment import decode_value, load_build, project, read_rows

    query = service._query(_decode_id(identity))
    recipe = resource(service, "query", identity)
    selectors = [selector(topic.label), selector(quality.educational_value)]
    values: dict[str, dict[str, FieldValue]] = {}
    for build_id in recipe.field_snapshot_ids:
        build, manifest = load_build(service, "field", build_id, recipe.snapshot_ids)
        for selection in selectors:
            if field_id(build.field.name) == selection.field:
                values[selector_key(selection)] = {
                    row.document_id.hex(): project(
                        build.field, selection, decode_value(build.field, row)
                    )
                    for row in read_rows(service, "field", manifest)
                }
    missing = [selection for selection in selectors if selector_key(selection) not in values]
    if missing:
        raise ValueError(
            "topic and quality must be requested by the query: "
            "query(fields=[topic.label, quality.educational_value])"
        )
    cells: dict[tuple[str, str], Cell] = {}
    for row in query.rows():
        label, score = [values[selector_key(s)][row.id] for s in selectors]
        if label is not None and not isinstance(label, str):
            raise ValueError("topic label must be a string")
        if score is not None and not isinstance(score, (float, int)):
            raise ValueError("quality score must be numeric")
        band = (
            "unknown"
            if score is None
            else "<1"
            if score < 1
            else "1–2"
            if score < 2
            else "2–3"
            if score < 3
            else "≥3"
        )
        key = (label or "unknown", band)
        cell = cells.setdefault(
            key, dict(topic=key[0], quality=key[1], documents=0, bytes=0, examples=[])
        )
        cell["documents"] += 1
        cell["bytes"] += row.document.size
        if len(cell["examples"]) < 10:
            cell["examples"].append(dict(id=row.id, source=row.source_key, text=row.text[:4096]))
    return [cells[key] for key in sorted(cells)]
