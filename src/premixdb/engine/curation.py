"""Reference decontamination, whole-record sampling and verified similarity selection.

UTF-8 ranges always refer to the captured source. N-grams are case-sensitive
Unicode whitespace-delimited tokens, without implicit normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import (
    TYPE_CHECKING,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
)

from .._typing import Edge, EvidenceRange, FieldValue, FieldValues, Interval, Orders, Scalar
from ..v1 import query_pb2 as q
from .contracts import Contamination, Provenance, SamplingStatistics

if TYPE_CHECKING:
    from .datasets import HuggingFaceTokenizer

from blake3 import blake3

from .._mixing import _domain_key
from .identity import identity_domain
from .snapshots import Document

type SelectedDocument = Document | RetainedDocument
type Unit = bytes | tuple[str, ...]


class EvidenceLookup(Protocol):
    def get(self, value: Unit) -> EvidenceRange | None: ...


type ReferenceProvider = Callable[
    [Sequence[Document], q.Decontaminate], ContextManager[EvidenceLookup]
]


def word_ranges(text: str, n: int) -> Iterator[tuple[tuple[str, ...], int, int]]:
    from collections import deque

    words: deque[tuple[str, int, int]] = deque(maxlen=n)
    previous_character = previous_byte = 0
    for word in re.finditer(r"\S+", text):
        start = previous_byte + len(text[previous_character : word.start()].encode())
        end = start + len(word.group().encode())
        words.append((word.group(), start, end))
        if len(words) == n:
            yield tuple(w[0] for w in words), words[0][1], words[-1][2]
        previous_character, previous_byte = word.end(), end


def units(
    document: SelectedDocument, algorithm: int, n: int = 0
) -> Iterator[tuple[Unit, int, int]]:
    if algorithm == 1:
        data = document.text.encode()
        yield data, 0, len(data)
    elif algorithm == 2:
        data = document.text.encode()
        start = 0
        for line in data.split(b"\n"):
            if line:
                yield line, start, start + len(line)
            start += len(line) + 1
    else:
        yield from word_ranges(document.text, n)


def coalesce(ranges: Iterable[Interval]) -> list[Interval]:
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


@dataclass(frozen=True)
class RetainedDocument:
    original: Document
    text: str
    source_ranges: tuple[Interval, ...]

    @property
    def id(self) -> str:
        return self.original.id

    @property
    def corpus_id(self) -> str:
        return self.original.corpus_id

    @property
    def source_key(self) -> str:
        return self.original.source_key

    @property
    def content(self) -> bytes:
        return self.original.content

    @property
    def size(self) -> int:
        return len(self.text.encode())

    @property
    def characters(self) -> int:
        return len(self.text)


def decontaminate(
    documents: list[SelectedDocument],
    references: Sequence[Document],
    policy: q.Decontaminate,
    provenance: dict[str, Provenance],
    *,
    provider: ReferenceProvider | None = None,
    evidence: EvidenceLookup | None = None,
) -> list[SelectedDocument]:
    if evidence is None:
        from .spill import references as reference_index

        with (provider or reference_index)(references, policy) as lookup:
            return decontaminate(documents, references, policy, provenance, evidence=lookup)
    retained = []
    for document in documents:
        matches: list[Contamination] = [
            dict(
                start=start,
                end=end,
                reference=witness[0],
                reference_start=witness[1],
                reference_end=witness[2],
            )
            for value, start, end in units(document, policy.algorithm, policy.n)
            if (witness := evidence.get(value)) is not None
        ]
        if not matches:
            retained.append(document)
            continue
        record = provenance[document.id]
        record["contamination"] = matches
        if policy.granularity == 1:
            record["selection"] = dict(kind="contaminated", references=matches)
            continue
        removed = coalesce((m["start"], m["end"]) for m in matches)
        data, cursor, ranges = document.text.encode(), 0, []
        for start, end in removed:
            if start > cursor:
                ranges.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < len(data):
            ranges.append((cursor, len(data)))
        text = b"".join(data[a:b] for a, b in ranges).decode()
        record["retained_ranges"] = ranges
        # Empty retained records are valid and still receive explicit separators.
        original = document.original if isinstance(document, RetainedDocument) else document
        retained.append(RetainedDocument(original, text, tuple(ranges)))
    return retained


def ordering(
    documents: Iterable[SelectedDocument], orders: Orders, values: FieldValues
) -> list[SelectedDocument]:
    def value(document: SelectedDocument, field: str) -> Scalar:
        if field.startswith("external:"):
            from .._typing import scalar

            return scalar(values[field].get(document.id))
        return (
            document.size
            if field == "bytes"
            else document.characters
            if field == "characters"
            else document.source_key
        )

    def compare(a: SelectedDocument, b: SelectedDocument) -> int:
        for field, descending in orders:
            av, bv = value(a, field), value(b, field)
            if av is None or bv is None:
                result = (av is None) - (bv is None)
                if result:
                    return result
                continue
            if isinstance(av, str) and isinstance(bv, str):
                result = (av > bv) - (av < bv)
            elif isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                result = (av > bv) - (av < bv)
            else:
                raise TypeError("ordering requires values of comparable scalar types")
            if result:
                return -result if descending else result
        return (a.id > b.id) - (a.id < b.id)

    return sorted(documents, key=cmp_to_key(compare))


def greedy(
    documents: list[SelectedDocument],
    edges: Iterable[Edge],
    orders: Orders,
    values: FieldValues,
    provenance: dict[str, Provenance],
    ordinal: int,
) -> list[SelectedDocument]:
    from .spill import edge_database

    result = []
    with edge_database(edges) as database:
        for document in ordering(documents, orders, values):
            winner = database.execute(
                "SELECT k.id FROM kept k JOIN (SELECT b AS id FROM edges WHERE a=? UNION SELECT a AS id FROM edges WHERE b=?) n ON k.id=n.id ORDER BY k.rank LIMIT 1",
                (document.id, document.id),
            ).fetchone()
            if winner is not None:
                provenance[document.id]["selection"] = dict(
                    kind="duplicate", step=ordinal, kept=winner[0]
                )
            else:
                database.execute("INSERT INTO kept VALUES (?,?)", (document.id, len(result)))
                result.append(document)
    return result


def jaccard_edges(
    documents: list[SelectedDocument],
    n: int,
    threshold: float,
    candidates: Iterable[Edge] | None = None,
) -> Iterator[Edge]:
    from .spill import jaccard_edges as verified_edges

    yield from verified_edges(documents, n, threshold, candidates)


def cosine_edges(
    vectors: Mapping[str, Sequence[float] | None],
    threshold: float,
    *,
    selected: set[str] | None = None,
) -> Iterator[Edge]:
    from .spill import cosine_edges as verified_edges

    yield from verified_edges(vectors, threshold, selected)


def label(
    document: SelectedDocument, selector: q.FieldComparison, values: FieldValues
) -> FieldValue:
    if selector.field == 1:
        return document.size
    if selector.field == 2:
        return document.characters
    # Actual intrinsic enum values are resolved by the planner, via values.
    return values[selector_key(selector)].get(document.id)


def selector_key(selector: q.FieldComparison) -> str:
    from .identity import Canonical

    digest = (
        Canonical("field-projection/v1")
        .u64(selector.field)
        .fixed(selector.field_snapshot_id or bytes(32))
        .u64(selector.projection)
        .string(selector.class_name)
        .fixed(bytes([selector.HasField("component")]))
    )
    if selector.HasField("component"):
        digest.u64(selector.component)
    return "external:" + digest.finish().hex()


def allocations(weights: Mapping[str, int | float], total: int) -> dict[str, int]:
    quotas = {
        k: Fraction(v) * total / sum(Fraction(w) for w in weights.values())
        for k, v in weights.items()
    }
    result = {k: int(v) for k, v in quotas.items()}
    order = sorted(quotas, key=lambda k: (-(quotas[k] - result[k]), k))
    for key in order[: total - sum(result.values())]:
        result[key] += 1
    return result


def sample(
    documents: list[SelectedDocument],
    policy: q.QuerySampling,
    values: FieldValues,
    tokenizer: HuggingFaceTokenizer | None,
    provenance: dict[str, Provenance],
    statistics: SamplingStatistics | None = None,
) -> list[SelectedDocument]:
    strata: dict[str, list[SelectedDocument]] = {}
    for document in documents:
        labels = [label(document, s, values) for s in policy.domains]
        key = _domain_key(labels)
        strata.setdefault(key, []).append(document)
    budget = policy.WhichOneof("budget")
    if budget is None:
        raise ValueError("sampling requires a budget")
    total = (
        int(len(documents) * policy.fraction)
        if budget == "fraction"
        else policy.documents
        if budget == "documents"
        else policy.bytes
        if budget == "bytes"
        else policy.characters
        if budget == "characters"
        else policy.tokens
    )
    if statistics is not None:
        statistics.update(
            unit=budget,
            requested=total,
            realized=0,
            overshoot=0,
            unique_documents=0,
            document_occurrences=0,
        )
    if not strata:
        if total:
            raise ValueError("sampling allocation exceeds empty population")
        return []
    if total == 0:
        for document in documents:
            provenance[document.id]["selection"] = dict(kind="not_sampled")
        return []

    def size(d: SelectedDocument) -> int:
        if budget in ("documents", "fraction"):
            return 1
        if budget == "bytes":
            return d.size
        if budget == "characters":
            return d.characters
        return len(tokenizer.encode(d.text)) if tokenizer else d.size

    weights = dict(policy.weights) or {k: sum(size(d) for d in v) for k, v in strata.items()}
    if not sum(weights.values()):
        raise ValueError("sampling allocation has zero capacity")
    if set(weights) != set(strata):
        raise ValueError("sampling weights must exactly cover available strata")
    targets = allocations(weights, total)
    result = []
    realized_domains = {}
    seed = policy.seed.to_bytes(8, "big")
    for key, target in sorted(targets.items()):
        population = sorted(
            strata[key],
            key=lambda d: (
                blake3(identity_domain("sample") + seed + bytes.fromhex(d.id)).digest(),
                d.id,
            ),
        )

        if target and not any(size(d) for d in population):
            raise ValueError("sampling allocation has zero capacity")
        if not policy.replacement and sum(size(d) for d in population) < target:
            raise ValueError("sampling allocation exceeds capacity")
        amount, draw = 0, 0
        while amount < target:
            if policy.replacement:
                rank = blake3(
                    identity_domain("sample-replacement")
                    + seed
                    + key.encode()
                    + draw.to_bytes(8, "big")
                ).digest()
                document = population[int.from_bytes(rank, "big") % len(population)]
            else:
                document = population[draw]
            result.append(document)
            amount += size(document)
            draw += 1
        realized_domains[key] = amount
    selected = {d.id for d in result}
    if statistics is not None:
        realized = sum(realized_domains.values())
        statistics.update(
            unit=budget,
            requested=total,
            realized=realized,
            overshoot=max(0, realized - total),
            unique_documents=len(selected),
            document_occurrences=len(result),
            requested_domains=targets,
            realized_domains=realized_domains,
        )
    for document in documents:
        if document.id not in selected:
            provenance[document.id]["selection"] = dict(kind="not_sampled")
    return result
