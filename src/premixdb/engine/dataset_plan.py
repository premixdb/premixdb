"""Resolved dataset identity and exact packing arithmetic, without execution IO."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .contracts import Counts, PackingGeometry, PackingProfile, PackingSummary
from .identity import Canonical, CodeVersion, digest, unsigned

BYTE_DEFINITION = (
    Canonical("tokenizer").string("text.utf8_bytes/u32/no-special-tokens/v1").finish().hex()
)


@dataclass(frozen=True)
class PackingTotals:
    content: int
    separators: int
    dropped_content: int
    dropped_separators: int
    padding: int
    sequences: int
    output: int
    occurrences: int

    def summary(self, counts: Counts) -> PackingSummary:
        return PackingSummary(
            input=counts.copy(),
            content_tokens=self.content,
            separator_tokens=self.separators,
            dropped_content_tokens=self.dropped_content,
            dropped_separator_tokens=self.dropped_separators,
            padding_tokens=self.padding,
            sequences=self.sequences,
            output_tokens=self.output,
        )

    def profile(self, counts: Counts) -> PackingProfile:
        return PackingProfile(
            source_documents=counts["documents"],
            source_content_bytes=counts["bytes"],
            source_characters=counts["characters"],
            document_occurrences=self.occurrences,
            planned_content_tokens=self.content,
            content_tokens=self.content - self.dropped_content,
            separator_tokens=self.separators - self.dropped_separators,
            dropped_tokens=self.dropped_content + self.dropped_separators,
            padding_tokens=self.padding,
            sequences=self.sequences,
            output_tokens=self.output,
            stratum_tokens={},
        )


@dataclass(frozen=True)
class PackingPlan:
    length: int
    separator: int | None = None
    padding: int | None = None

    def __post_init__(self) -> None:
        if not unsigned(self.length):
            raise ValueError("sequence length must be positive")
        for token in (self.separator, self.padding):
            if token is not None:
                unsigned(token, 32)

    def measure(self, lengths: Iterable[int]) -> PackingTotals:
        lengths = tuple(unsigned(n) for n in lengths)
        content = unsigned(sum(lengths))
        separators = len(lengths) if self.separator is not None else 0
        total = unsigned(content + separators)
        sequences = (
            total // self.length
            if self.padding is None
            else (total + self.length - 1) // self.length
        )
        output = unsigned(sequences * self.length)
        remaining = max(0, total - output)
        dropped_content = dropped_separators = 0
        for length in reversed(lengths):
            if self.separator is not None and remaining:
                remaining -= 1
                dropped_separators += 1
            drop = min(remaining, length)
            dropped_content += drop
            remaining -= drop
            if not remaining:
                break
        return PackingTotals(
            content,
            separators,
            dropped_content,
            dropped_separators,
            max(0, output - total),
            sequences,
            output,
            len(lengths),
        )

    def profile(self, counts: Counts, lengths: Iterable[int]) -> PackingProfile:
        return self.measure(lengths).profile(counts)

    def geometry(self, occurrences: Iterable[tuple[str, str, int]]) -> PackingGeometry:
        """Exact boundary/source counts without constructing tokens or sequences."""
        occurrences = list(occurrences)
        totals = self.measure([length for _, _, length in occurrences])
        from .curation import coalesce

        source_tokens: dict[str, int] = {}
        intervals: dict[str, list[tuple[int, int]]] = {}
        cursor = 0
        for document, source, length in occurrences:
            emitted = max(0, min(length, totals.output - cursor))
            source_tokens[source] = source_tokens.get(source, 0) + emitted
            end = min(totals.output, cursor + length + (self.separator is not None))
            if end > cursor:
                intervals.setdefault(document, []).append(
                    (cursor // self.length, (end - 1) // self.length + 1)
                )
            cursor += length + (self.separator is not None)
        events = {0: 0, totals.sequences: 0}
        for ranges in intervals.values():
            for start, end in coalesce(ranges):
                events[start] = events.get(start, 0) + 1
                events[end] = events.get(end, 0) - 1
        histogram, previous, active = {}, 0, 0
        for position, delta in sorted(events.items()):
            if position > previous:
                histogram[active] = histogram.get(active, 0) + position - previous
            active += delta
            previous = position
        return PackingGeometry(
            source_tokens=source_tokens,
            documents_per_sequence=histogram,
            boundary_crossing_sequences=sum(
                count for documents, count in histogram.items() if documents > 1
            ),
        )


@dataclass(frozen=True)
class DatasetPlan:
    input_id: str
    definition: str
    code: CodeVersion
    packing: PackingPlan

    @property
    def id(self) -> str:
        p = self.packing
        h = (
            Canonical("dataset")
            .fixed(digest(self.input_id))
            .fixed(digest(self.definition))
            .fixed(self.code.canonical_digest())
            .string("concat/v1")
            .u64(p.length)
            .fixed(bytes([p.separator is not None]))
        )
        if p.separator is not None:
            h.u64(p.separator)
        h.string("drop" if p.padding is None else "pad")
        if p.padding is not None:
            h.u64(p.padding)
        return h.finish().hex()
