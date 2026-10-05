"""Concatenate independently packed datasets without crossing their boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from premixdb.engine.contracts import Occurrence
from premixdb.engine.dataset_plan import DatasetPlan, PackingTotals
from premixdb.engine.datasets import Dataset, Sequence


class ConcatenatedDataset(Dataset):
    def __init__(self, plan: DatasetPlan, parts: tuple[Dataset, ...]) -> None:
        self.plan = plan
        self.parts = parts
        self._id = plan.id
        self.query_id = parts[0].query_id
        self._query = parts[0]._query
        self._token_pool = None
        self.tokenizer_definition = plan.definition
        self._sequences = None
        self._consumed = False
        self._lengths = tuple(n for part in parts for n in part._lengths)
        self._input_counts = {
            "documents": sum(p._input_counts["documents"] for p in parts),
            "bytes": sum(p._input_counts["bytes"] for p in parts),
            "characters": sum(p._input_counts["characters"] for p in parts),
        }
        self._totals = PackingTotals(
            content=sum(p._totals.content for p in parts),
            separators=sum(p._totals.separators for p in parts),
            dropped_content=sum(p._totals.dropped_content for p in parts),
            dropped_separators=sum(p._totals.dropped_separators for p in parts),
            padding=sum(p._totals.padding for p in parts),
            sequences=sum(len(p) for p in parts),
            output=sum(p._totals.output for p in parts),
            occurrences=sum(p._totals.occurrences for p in parts),
        )
        self.elapsed_seconds = sum(p.elapsed_seconds for p in parts)

    def close(self) -> None:
        for part in self.parts:
            part.close()

    def packed(self, pack: Callable[[Dataset], Dataset]) -> ConcatenatedDataset:
        """Apply distributed packing separately to every part."""
        return ConcatenatedDataset(self.plan, tuple(pack(part) for part in self.parts))

    def occurrence_document(self, ordinal: int) -> str:
        for part in self.parts:
            if ordinal < len(part._lengths):
                return part.occurrence_document(ordinal)
            ordinal -= len(part._lengths)
        raise IndexError("occurrence index out of range")

    @property
    def occurrence_count(self) -> int:
        return sum(part.occurrence_count for part in self.parts)

    def occurrences(self) -> list[Occurrence]:
        result = []
        offset = 0
        for part in self.parts:
            for occurrence in part.occurrences():
                occurrence["ordinal"] += offset
                result.append(occurrence)
            offset += len(part._lengths)
        return result

    def _pack_sequences(self) -> Iterator[Sequence]:
        if self._consumed:
            raise RuntimeError("packing stream has already been consumed")
        self._consumed = True
        ordinal = occurrence = 0
        for part in self.parts:
            for sequence in part.iter_sequences():
                spans = sequence.spans
                for span in spans:
                    if "occurrence" in span:
                        span["occurrence"] += occurrence
                ranges = [record.copy() for record in sequence.source_ranges]
                for record in ranges:
                    record["occurrence"] += occurrence
                yield Sequence(ordinal, sequence.tokens, spans, ranges)
                ordinal += 1
            occurrence += len(part._lengths)
        if ordinal != len(self):
            raise ValueError("concatenated sequence coverage differs from its plan")
