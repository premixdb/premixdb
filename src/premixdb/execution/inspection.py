"""Python inspection helpers for profiles and deterministic example drilldowns."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, cast, overload

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

from .._ids import _decode_id
from .._typing import JSON, FieldValue, Scalar, is_json, json_object, load_json, scalar
from ..engine.contracts import Provenance
from ..v1 import corpus_pb2 as c
from ..v1 import dataset_pb2 as d
from ..v1 import query_pb2 as q
from ..v1 import snapshot_pb2 as s
from ..v1 import status_pb2

if TYPE_CHECKING:
    from .coordinator import Coordinator

type Parameters = Mapping[str, Sequence[str]]
type Kind = Literal["snapshot", "query", "dataset", "corpus"]


def _json_record(value: object) -> dict[str, JSON]:
    if not is_json(value):
        raise ValueError("expected a JSON value")
    return json_object(value)


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
    from ..engine.curation import intrinsic_value
    from ..engine.queries import Row
    from .enrichment import decode_value, load_build, project, read_rows
    from .previewing import bounded_text

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
    handle = service._query(_decode_id(identity)) if isinstance(recipe, q.Query) else None
    if handle is not None and population == "output":
        selected = handle
    elif kind == "snapshot" or kind == "query":
        snapshots = recipe.snapshot_ids if isinstance(recipe, q.Query) else [_decode_id(identity)]
        documents = service._query_index(snapshots).documents
        selected = [Row(i, doc) for i, doc in enumerate(documents.values())]
    else:
        raise ValueError("document drilldown requires a query or snapshot")
    lineage: dict[str, Provenance] = {}
    if isinstance(recipe, q.Query):
        assert handle is not None
        lineage = handle.provenance()
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
            from .._mixing import _domain_key
            from ..engine.curation import selector_key
            from .enrichment import project

            projections: dict[str, dict[str, FieldValue]] = {}
            for selection in recipe.sampling.domains:
                if selection.field <= q.FIELD_SOURCE_CORPUS_ID:
                    projections[selector_key(selection)] = {
                        row.id: intrinsic_value(row.document, selection.field) for row in selected
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
                return _domain_key(values)

            selected = [row for row in selected if stratum(row) == parameters["stratum"][0]]
    if "field" in parameters:
        name = parameters["field"][0]
        field = q.IntrinsicField.Value(name)
        values: dict[str, FieldValue] = {}
        if field <= q.FIELD_SOURCE_CORPUS_ID:
            for row in selected:
                values[row.id] = intrinsic_value(row.document, field)
        else:
            selection = q.FieldComparison(
                field=field,
                projection=q.FieldComparison.Projection.Value(
                    parameters.get("projection", ["SCALAR"])[0]
                ),
                class_name=parameters.get("class_name", [""])[0],
            )
            component = _parameter(parameters, "component")
            if component is not None:
                selection.component = int(component)
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
                for value_row in read_rows(service, "field", manifest):
                    values[value_row.document_id.hex()] = project(
                        build.field, selection, decode_value(build.field, value_row)
                    )

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
    examples: list[DocumentExample] = []
    total = 0
    for row in selected:
        if offset <= total < offset + 100:
            examples.append(
                dict(
                    id=row.id,
                    ordinal=row.ordinal,
                    source=row.source_key,
                    text=bounded_text(service._storage, row, 4096)[0],
                    provenance=lineage.get(row.id),
                )
            )
        total += 1
    return DocumentPage(total=total, offset=offset, rows=examples)


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
    from ..engine.queries import Row
    from ..engine.spill import _database
    from ..internal import derivation_pb2 as e
    from .enrichment import load_build, read_rows
    from .previewing import bounded_text

    build = service._storage.load("index", _decode_id(identity), e.IndexBuild)
    build, manifest = load_build(service, "index", build.snapshot.id, build.snapshot.snapshot_ids)
    with _database("index-profile") as database:
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
                database.execute("SELECT size,COUNT(*) FROM groups GROUP BY size ORDER BY size"),
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
                        text=bounded_text(service._storage, Row(0, document), 4096)[0],
                        group=hash.hex(),
                    )
                )
        return IndexStatistics(
            documents=str(manifest.documents),
            duplicate_groups=str(sum(histogram.values())),
            duplicate_documents=str(sum((size - 1) * count for size, count in histogram.items())),
            group_sizes={str(size): str(count) for size, count in histogram.items()},
            candidate_links=str(candidate_links),
            candidate_documents=str(participants),
            approximate=build.index.approximate,
            examples=examples,
        )


def sequences(service: Coordinator, identity: str, parameters: Parameters) -> SequencePage:
    """Scan verified sequence metadata, reading token bytes only for page examples."""
    from .._sequences import Sequence, decode_preview, preview_decoder, sequence_page
    from .._storage import RangeReader

    offset = int(parameters.get("offset", ["0"])[0])
    if offset < 0:
        raise ValueError("invalid page offset")
    source = _parameter(parameters, "source")
    if source is not None:
        source = _decode_id(source).hex()
    count = _parameter(parameters, "documents")
    documents = int(count) if count is not None else None
    if documents is not None and documents < 0:
        raise ValueError("invalid sequence document count")
    crossing = _parameter(parameters, "crossing")
    padding = _parameter(parameters, "padding")
    if crossing not in (None, "true") or padding not in (None, "true"):
        raise ValueError("invalid sequence predicate")
    recipe = resource(service, "dataset", identity)
    stratum = _parameter(parameters, "stratum")
    if stratum is not None and recipe.sampling.domains.field == q.FIELD_SOURCE_CORPUS_ID:
        stratum = _decode_id(stratum).hex()
    lineage = service._query(recipe.query_id).provenance() if source else {}
    labels = service._sampling(recipe)[0].labels if stratum else {}
    rows: list[SequenceExample] = []
    total, ordinal = 0, 0
    decoder: Callable[[list[int]], str] | None = None
    with RangeReader(local_root=service._storage.root) as reader:
        for page, ref in enumerate(recipe.sequences):
            for sequence in sequence_page(recipe, reader.read(ref), page):
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
                    or (documents is not None and len(members) != documents)
                    or (crossing and len(members) <= 1)
                    or (
                        padding
                        and not any(r.kind == d.TokenRegion.KIND_PADDING for r in sequence.regions)
                    )
                ):
                    continue
                if offset <= total < offset + 20:
                    tokens = Sequence(sequence, reader)._token_values(256)
                    if decoder is None and recipe.tokenizer.HasField("hugging_face"):
                        decoder = preview_decoder(recipe.tokenizer, reader)
                    rows.append(
                        dict(
                            ordinal=sequence.ordinal,
                            tokens=tokens,
                            text=decode_preview(tokens, sequence.regions, decoder),
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
    from .previewing import bounded_text

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
        raise ValueError("topic and quality must be requested by the query")
    cells: dict[tuple[str, str], Cell] = {}
    for row in query:
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
            cell["examples"].append(
                dict(
                    id=row.id,
                    source=row.source_key,
                    text=bounded_text(service._storage, row, 4096)[0],
                )
            )
    return [cells[key] for key in sorted(cells)]
