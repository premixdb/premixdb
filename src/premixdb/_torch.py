"""Batched verified reads and deterministic distributed streaming for PyTorch."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Iterator, TypedDict

import torch
import torch.distributed
from blake3 import blake3
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ._reader import permutation
from ._resources import _sequence_page
from ._storage import RangeReader
from .v1 import dataset_pb2 as d


class _TorchState(TypedDict):
    resource: d.Dataset
    reader: RangeReader
    _pages: OrderedDict[int, tuple[d.Sequence, ...]]


class TorchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, resource: d.Dataset, reader: RangeReader) -> None:
        self.resource = resource
        self.reader = reader
        self._pages: OrderedDict[int, tuple[d.Sequence, ...]] = OrderedDict()

    def __len__(self) -> int:
        return self.resource.profile.sequences

    def _index(self, index: int) -> int:
        if type(index) is not int:
            raise TypeError("sequence index must be an integer")
        index = index + len(self) if index < 0 else index
        if not 0 <= index < len(self):
            raise IndexError("sequence index out of range")
        return index

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def _load_pages(self, pages: Iterable[int]) -> dict[int, tuple[d.Sequence, ...]]:
        pages = list(dict.fromkeys(pages))
        missing = [page for page in pages if page not in self._pages]
        for page, data in zip(
            missing,
            self.reader.read_many([self.resource.sequences[p] for p in missing]),
            strict=True,
        ):
            self._pages[page] = _sequence_page(self.resource, data, page)
        result = {page: self._pages[page] for page in pages}
        # Keep only four index pages between batches, regardless of batch size.
        for page in pages:
            self._pages.move_to_end(page)
        while len(self._pages) > 4:
            self._pages.popitem(last=False)
        return result

    def __getitems__(self, indices: Iterable[int]) -> list[dict[str, torch.Tensor]]:
        indices = [self._index(index) for index in indices]
        pages = self._load_pages(index // 128 for index in indices)
        return self._items([pages[index // 128][index % 128] for index in indices])

    def _items(self, sequences: list[d.Sequence]) -> list[dict[str, torch.Tensor]]:
        if not sequences:
            return []
        if any(s.tokens.start % 4 for s in sequences):
            raise ValueError("token range is not uint32 aligned")
        spans = [span for s in sequences for span in (s.tokens, s.attention_mask, s.loss_mask)]
        values = self.reader.read_many(spans)
        length = self.resource.sequence_length
        for i in range(0, len(values), 3):
            if (
                len(values[i]) != length * 4
                or len(values[i + 1]) != length
                or len(values[i + 2]) != length
            ):
                raise ValueError("stored sequence has an invalid length")
            if any(v > 1 for data in values[i + 1 : i + 3] for v in data):
                raise ValueError("invalid stored token mask")
        tokens = bytearray(b"".join(values[0::3]))
        attention = bytearray(b"".join(values[1::3]))
        loss = bytearray(b"".join(values[2::3]))
        # Explicit little-endian decoding works on hosts with either byte order.
        import sys

        if sys.byteorder != "little":
            from array import array

            swap = array("I", tokens)
            swap.byteswap()
            tokens = bytearray(swap.tobytes())
        ids = (
            torch.frombuffer(tokens, dtype=torch.int32)
            .to(torch.long)
            .bitwise_and_(0xFFFFFFFF)
            .reshape(-1, length)
        )
        attention_tensor = (
            torch.frombuffer(attention, dtype=torch.uint8).to(torch.long).reshape(-1, length)
        )
        mask = torch.frombuffer(loss, dtype=torch.uint8).reshape(-1, length)
        labels = ids.clone()
        labels[mask == 0] = -100
        return [
            dict(input_ids=ids[i], attention_mask=attention_tensor[i], labels=labels[i])
            for i in range(len(sequences))
        ]

    def __getstate__(self) -> _TorchState:
        return _TorchState(resource=self.resource, reader=self.reader, _pages=OrderedDict())


def streaming_topology(
    *, seed: int, epoch: int, rank: int | None, world_size: int | None
) -> tuple[int, int]:
    """Validate options before materialization or distributed rank discovery."""
    if any(type(v) is not int or not 0 <= v < 2**64 for v in (seed, epoch)):
        raise ValueError("seed and epoch must be uint64")
    if (rank is None) != (world_size is None):
        raise ValueError("provide both rank and world_size, or neither")
    if rank is None:
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1
    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank or world_size")
    return rank, world_size


class StreamingDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Shuffle index pages and sequences; assign each page to one rank and worker."""

    def __init__(
        self,
        resource: d.Dataset,
        reader: RangeReader,
        *,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        rank, world_size = streaming_topology(
            seed=seed, epoch=epoch, rank=rank, world_size=world_size
        )
        self.data = TorchDataset(resource, reader)
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, epoch
        self._seed = int.from_bytes(
            blake3(seed.to_bytes(8, "big") + epoch.to_bytes(8, "big")).digest()[:8], "big"
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        number, workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        count = (len(self.data) + 127) // 128
        first = self.rank + self.world_size * number
        for position in range(first, count, self.world_size * workers):
            page = permutation(position, count, self._seed)
            sequences = self.data._load_pages([page])[page]
            order_seed = int.from_bytes(
                blake3(self._seed.to_bytes(8, "big") + page.to_bytes(8, "big")).digest()[:8], "big"
            )
            ordered = [
                sequences[permutation(i, len(sequences), order_seed)] for i in range(len(sequences))
            ]
            yield from self.data._items(ordered)
