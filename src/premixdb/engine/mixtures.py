"""Mixture policy and profiles over a reusable Python token pool."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterable, Mapping

from .contracts import Counts, PackingGeometry, PackingProfile
from .datasets import HuggingFaceTokenizer

if TYPE_CHECKING:
    from .queries import Query

from .dataset_plan import BYTE_DEFINITION, PackingPlan
from .datasets import Dataset, encoded_tokens
from .identity import Canonical, digest, unsigned
from .snapshots import counts


class MixturePool:
    def __init__(
        self,
        query: Query,
        field: str,
        assignments: Mapping[str, str],
        tokenizer: HuggingFaceTokenizer | None = None,
    ) -> None:
        if field not in ("", "source.corpus_id", "object.uri"):
            raise NotImplementedError("unsupported mixture stratum field")
        self.query = query
        self.rows = query.rows()
        self.tokenizer = tokenizer
        self._encoded = None
        if tokenizer:
            from .token_cache import token_pool

            self._encoded = token_pool(query, tokenizer)
        self._sizes = [
            self._encoded.length(r.id) if self._encoded is not None else r.document.size
            for r in self.rows
        ]
        if (not field and set(assignments) != {r.id for r in self.rows}) or (field and assignments):
            raise ValueError("assignments must cover the query exactly")
        self.labels = {
            r.id: r.corpus_id
            if field == "source.corpus_id"
            else r.source_key
            if field == "object.uri"
            else assignments[r.id]
            for r in self.rows
        }
        if any(not isinstance(label, str) or not label for label in self.labels.values()):
            raise ValueError("mixture labels must be nonempty strings")
        self.domains: dict[str, list[int]] = {}
        self._inventory: dict[str, int] = {}
        for i, row in enumerate(self.rows):
            label, length = self.labels[row.id], self._sizes[i]
            self._inventory[label] = self._inventory.get(label, 0) + length
            if length:
                self.domains.setdefault(label, []).append(i)
        tokenizer_digest = digest(self.tokenizer.definition if self.tokenizer else BYTE_DEFINITION)
        hash_ = (
            Canonical("mixture-pool")
            .fixed(digest(query.id))
            .fixed(tokenizer_digest)
            .string(field)
            .u64(len(self.rows))
        )
        for row in self.rows:
            hash_.fixed(digest(row.id)).string(self.labels[row.id])
        self.id = hash_.finish()

    def inventory(self) -> dict[str, int]:
        return dict(self._inventory)

    def identity(
        self,
        allocations: Mapping[str, int],
        definition: str,
        seed: int,
        replacement: bool,
        max_epochs: int | None,
    ) -> str:
        if set(allocations) != set(self._inventory):
            raise ValueError("invalid mixture allocation domains")
        if type(replacement) is not bool:
            raise TypeError("replacement must be a bool")
        if max_epochs is not None and not unsigned(max_epochs, 32):
            raise ValueError("max_epochs must be positive")
        hash_ = (
            Canonical("mixture-draws")
            .fixed(self.id)
            .fixed(digest(definition))
            .u64(seed)
            .u64(int(replacement))
            .u64(max_epochs or 0)
        )
        cap = max_epochs if replacement else 1
        for label, target in sorted(allocations.items()):
            unsigned(target)
            available = self._inventory[label]
            if target and (not available or (cap is not None and target > available * cap)):
                raise ValueError(f"token allocation exceeds capacity for {label}")
            hash_.string(label).u64(target)
        return hash_.finish().hex()

    def _prepare(
        self,
        allocations: Mapping[str, int],
        definition: str,
        seed: int,
        replacement: bool,
        max_epochs: int | None,
    ) -> tuple[str, list[tuple[int, int]]]:
        identity = self.identity(allocations, definition, seed, replacement, max_epochs)
        return identity, self._draw(allocations, seed)

    def profile(
        self,
        allocations: Mapping[str, int],
        definition: str,
        seed: int,
        replacement: bool,
        max_epochs: int | None,
        sequence_length: int,
        separator: int | None,
        padding: int | None,
    ) -> PackingProfile:
        _, draws = self._prepare(allocations, definition, seed, replacement, max_epochs)
        profile = PackingPlan(sequence_length, separator, padding).profile(
            self._input_counts(draws), [length for _, length in draws]
        )
        remaining = profile["output_tokens"]
        exposure = dict.fromkeys(self._inventory, 0)
        for index, length in draws:
            exposure[self.labels[self.rows[index].id]] += min(remaining, length)
            remaining = max(0, remaining - length - (separator is not None))
        profile["stratum_tokens"] = exposure
        return profile

    def dataset(
        self,
        allocations: Mapping[str, int],
        definition: str,
        seed: int,
        replacement: bool,
        max_epochs: int | None,
        sequence_length: int,
        separator: int | None,
        padding: int | None,
        *,
        stream: bool = False,
    ) -> Dataset:
        start = time.monotonic()
        identity, draws = self._prepare(allocations, definition, seed, replacement, max_epochs)
        encoded = (
            (
                self.rows[i],
                self._encoded[self.rows[i].id][:size]
                if self._encoded is not None
                else encoded_tokens(self.rows[i])[:size],
            )
            for i, size in draws
        )
        return Dataset(
            self.query,
            identity,
            self._input_counts(draws),
            encoded,
            self.tokenizer.definition if self.tokenizer else BYTE_DEFINITION,
            sequence_length,
            separator,
            padding,
            start,
            [size for _, size in draws],
            stream=stream,
        )

    def geometry(
        self,
        allocations: Mapping[str, int],
        definition: str,
        seed: int,
        replacement: bool,
        max_epochs: int | None,
        sequence_length: int,
        separator: int | None,
        padding: int | None,
    ) -> PackingGeometry:
        _, draws = self._prepare(allocations, definition, seed, replacement, max_epochs)
        return PackingPlan(sequence_length, separator, padding).geometry(
            (self.rows[i].id, self.rows[i].corpus_id, length) for i, length in draws
        )

    def _input_counts(self, draws: Iterable[tuple[int, int]]) -> Counts:
        return counts({self.rows[i].id: self.rows[i].document for i, _ in draws}.values())

    def _draw(self, allocations: Mapping[str, int], seed: int) -> list[tuple[int, int]]:
        pieces = []
        for label, target in sorted(allocations.items()):
            remaining, epoch = target, 0
            while remaining:
                order = sorted(
                    (_draw_key("mixture-epoch", seed, label, epoch, self.rows[i].id), i)
                    for i in self.domains[label]
                )
                for _, i in order:
                    length = min(remaining, self._sizes[i])
                    pieces.append(
                        (
                            _draw_key("mixture-order", seed, label, epoch, self.rows[i].id),
                            i,
                            length,
                        )
                    )
                    remaining -= length
                    if not remaining:
                        break
                epoch += 1
        return [(i, length) for _, i, length in sorted(pieces)]

    def exposure(self, dataset: Dataset) -> dict[str, int]:
        if dataset.query_id != self.query.id or dataset.tokenizer_definition != BYTE_DEFINITION:
            raise ValueError("dataset does not belong to this token pool")
        totals = dict.fromkeys(self._inventory, 0)
        for sequence in dataset.iter_sequences():
            for span in sequence._spans:
                if span["kind"] == "content":
                    occurrence = span["occurrence"]
                    label = self.labels[dataset.occurrence_document(occurrence)]
                    totals[label] += span["end"] - span["start"]
        return totals


def _draw_key(kind: str, seed: int, label: str, epoch: int, document: str) -> bytes:
    return Canonical(kind).u64(seed).string(label).u64(epoch).fixed(digest(document)).finish()
