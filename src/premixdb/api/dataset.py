"""Packed dataset handles and training adapters."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)

from premixdb.api.progress import report_progress
from premixdb.contracts import (
    Checkpoint,
    ExecutionError,
    PreviewSequence,
)
from premixdb.schemas.ids import _public_dataset_profile
from premixdb.schemas.protobuf import copy_message
from premixdb.training.reader import Reader, Topology
from premixdb.training.sequences import Sequence, read_page
from premixdb.v1 import data_mixture_pb2 as mix_pb
from premixdb.v1 import query_pb2 as queries

if TYPE_CHECKING:
    from premixdb.training.torch import StreamingDataset, TorchDataset


from premixdb.api.base import _Execution


class Dataset(_Execution[mix_pb.Dataset, mix_pb.CreateDatasetRequest]):
    @report_progress("Previewing dataset {id}")
    def preview(
        self, *, limit: int = 3, offset: int = 0, max_characters: int = 1024
    ) -> list[PreviewSequence]:
        """Pack only enough output for the requested examples when not materialized.

        Each example includes decoded text, the first 256 token IDs and masks,
        source document IDs, and a truncation flag. Offset is a sequence ordinal.
        """
        from premixdb.api.dataset_preview import preview

        return preview(self, limit=limit, offset=offset, max_characters=max_characters)

    @property
    def _tokenizer_definition(self) -> str:
        """Return the versioned identity of the tokenizer used for packing."""
        return self._resource.tokenizer.definition_digest.hex()

    @property
    def _recipe(self) -> mix_pb.CreateDatasetRequest:
        """Return a detached request that reproduces this dataset."""
        from premixdb.schemas.messages import copy_fields

        return copy_fields(self._resource, mix_pb.CreateDatasetRequest())

    @report_progress("Profiling dataset {id}")
    def profile(self) -> mix_pb.DatasetProfile:
        """Compute the planned profile if needed, without packing candidate tokens."""
        if not self._resource.HasField("profile"):
            self._db._require_open()
            if self._db._read_only:
                raise ExecutionError(
                    "dataset profile is not computed; use a writable session first"
                )
            from premixdb.runtime import Coordinator

            assert isinstance(self._db._executor, Coordinator)
            self._resource.profile.CopyFrom(
                self._db._executor._planned_dataset_profile(self._resource)
            )
        return _public_dataset_profile(
            copy_message(self._resource.profile),
            corpus_strata=self._resource.sampling.domains.field == queries.FIELD_SOURCE_CORPUS_ID,
        )

    @overload
    def torch(
        self,
        *,
        streaming: Literal[False] = False,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> TorchDataset: ...
    @overload
    def torch(
        self,
        *,
        streaming: Literal[True],
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> StreamingDataset: ...
    @overload
    def torch(
        self,
        *,
        streaming: bool,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> TorchDataset | StreamingDataset: ...
    def torch(
        self,
        *,
        streaming: bool = False,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> TorchDataset | StreamingDataset:
        """Return batched or streaming PyTorch data with input IDs, masks and labels.

        Materializes this candidate once. DataLoader workers then read verified
        ranges directly; no catalog or execution handle crosses processes.
        """
        if type(streaming) is not bool:
            raise TypeError("streaming must be boolean")
        if not streaming and (
            seed != 0 or epoch != 0 or rank is not None or world_size is not None
        ):
            raise ValueError(
                "seed, epoch, rank and world_size require streaming=True; use a sampler for map-style data"
            )
        from premixdb.training import torch as _torch

        if streaming:
            rank, world_size = _torch.streaming_topology(
                seed=seed, epoch=epoch, rank=rank, world_size=world_size
            )
        self.wait()
        if streaming:
            return _torch.StreamingDataset(
                self._proto,
                self._db._object_reader,
                seed=seed,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
            )
        return _torch.TorchDataset(self._proto, self._db._object_reader)

    def __len__(self) -> int:
        return self.wait()._resource.profile.sequences

    def __iter__(self) -> Reader[Sequence]:
        return self._reader()

    def _page(self, ordinal: int, size: int | None = None) -> list[Sequence]:
        self.wait()
        return read_page(self._resource, self._db._object_reader, ordinal, size)

    def __getitem__(self, index: int) -> Sequence:
        if type(index) is not int:
            raise TypeError("sequence index must be an integer")
        count = len(self)
        index = index + count if index < 0 else index
        if not 0 <= index < count:
            raise IndexError("sequence index out of range")
        return self._page(index, 1)[0]

    def _reader(
        self,
        *,
        topology: Topology | None = None,
        checkpoint: Checkpoint | None = None,
        seed: int | None = None,
    ) -> Reader[Sequence]:
        """Iterate packed sequences with optional shuffling, partitions and resume.

        Save reader.checkpoint() after processing a sequence, then pass it back
        with the same seed and topology to resume from the next sequence.
        """
        return Reader(self, Topology() if topology is None else topology, checkpoint, seed)
