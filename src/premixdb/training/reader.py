"""Deterministic ordinal partitions and JSON checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

from premixdb.contracts import Checkpoint
from premixdb.schemas.ids import _decode_id


@dataclass(frozen=True)
class Topology:
    rank: int = 0
    world_size: int = 1
    worker: int = 0
    workers_per_rank: int = 1

    def _checkpoint_values(self) -> list[int]:
        return [self.rank, self.world_size, self.worker, self.workers_per_rank]

    def _partition(self) -> tuple[int, int]:
        if (
            any(
                type(value) is not int or not 0 <= value < 2**32
                for value in self._checkpoint_values()
            )
            or not 0 <= self.rank < self.world_size
            or not 0 <= self.worker < self.workers_per_rank
        ):
            raise ValueError("invalid reader rank/worker topology")
        return (
            self.rank * self.workers_per_rank + self.worker,
            self.world_size * self.workers_per_rank,
        )


class _OrdinalSequence(Protocol):
    @property
    def ordinal(self) -> int: ...


class _ReadableDataset[S: _OrdinalSequence](Protocol):
    @property
    def id(self) -> str: ...
    def __len__(self) -> int: ...
    def _page(self, ordinal: int) -> list[S]: ...


class Reader[S: _OrdinalSequence]:
    """Version-1 ordinal strides with optional deterministic shuffling."""

    def __init__(
        self,
        dataset: _ReadableDataset[S],
        topology: Topology,
        checkpoint: Checkpoint | None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(topology, Topology):
            raise TypeError("topology must be a Topology")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("shuffle seed must be uint64")
        self._seed = seed
        self._dataset, self._topology = dataset, topology
        first, self._stride = topology._partition()
        self._cache: dict[int, S] = {}
        ordinal: int | None = None
        if checkpoint is not None:
            if not isinstance(checkpoint, dict):
                raise ValueError("checkpoint must be a dictionary")
            try:
                checkpoint_id = _decode_id(checkpoint.get("dataset", ""))
            except ValueError as exc:
                raise ValueError("checkpoint has an invalid dataset ID") from exc
            values = checkpoint.get("topology")
            if (
                type(checkpoint.get("version")) is not int
                or checkpoint.get("version") != 1
                or checkpoint_id != _decode_id(dataset.id)
                or not isinstance(values, list)
                or any(type(value) is not int for value in values)
                or values != topology._checkpoint_values()
            ):
                raise ValueError("checkpoint is incompatible with dataset or topology")
            checkpoint_seed = checkpoint.get("shuffle_seed")
            if type(checkpoint_seed) is not type(seed) or checkpoint_seed != seed:
                raise ValueError("checkpoint shuffle policy differs")
            if "next_ordinal" not in checkpoint:
                raise ValueError("checkpoint is missing next_ordinal")
            ordinal = checkpoint["next_ordinal"]
            if ordinal is not None and (
                type(ordinal) is not int or ordinal < first or (ordinal - first) % self._stride
            ):
                raise ValueError("checkpoint ordinal is not in this partition")
        self._next = first if checkpoint is None else ordinal
        # An exhausted checkpoint needs no dataset reads or materialization.
        self._count = len(dataset) if self._next is not None else 0
        if self._next is not None and self._next >= self._count:
            if checkpoint is not None:
                raise ValueError("checkpoint ordinal is not in this partition")
            self._next = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> S:
        if self._next is None:
            raise StopIteration
        ordinal = (
            self._next if self._seed is None else permutation(self._next, self._count, self._seed)
        )
        if ordinal not in self._cache:
            self._cache = {seq.ordinal: seq for seq in self._dataset._page(ordinal)}
        sequence = self._cache.pop(ordinal)
        position = self._next + self._stride
        self._next = position if position < self._count else None
        return sequence

    def checkpoint(self) -> Checkpoint:
        """Return JSON-compatible state for resuming this dataset partition."""
        result: Checkpoint = Checkpoint(
            version=1,
            dataset=self._dataset.id,
            topology=self._topology._checkpoint_values(),
            next_ordinal=self._next,
        )

        if self._seed is not None:
            result["shuffle_seed"] = self._seed
        return result


def permutation(index: int, count: int, seed: int) -> int:
    """Six-round Feistel permutation with cycle walking; no index array required."""
    from blake3 import blake3

    if count <= 1:
        return index
    half = ((count - 1).bit_length() + 1) // 2
    mask = (1 << half) - 1
    value = index
    while True:
        left, right = value >> half, value & mask
        for round in range(6):
            key = seed.to_bytes(8, "big") + round.to_bytes(1, "big") + right.to_bytes(8, "big")
            left, right = right, left ^ (int.from_bytes(blake3(key).digest()[:8], "big") & mask)
        value = left << half | right
        if value < count:
            return value
