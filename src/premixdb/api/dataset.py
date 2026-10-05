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
from premixdb.schemas.enums import ExecutionStatus
from premixdb.schemas.ids import _public_dataset_profile
from premixdb.schemas.protobuf import copy_message
from premixdb.training.reader import Reader, Topology
from premixdb.training.sequences import Sequence, read_page
from premixdb.v1 import data_mixture_pb2 as mix_pb
from premixdb.v1 import query_pb2 as queries

if TYPE_CHECKING:
    from premixdb.training.torch import StreamingDataset, TorchDataset


from premixdb.api.base import _Execution


class _Dataset(_Execution[mix_pb.Dataset, mix_pb.CreateDatasetRequest]):
    """Reading and training operations shared by full datasets and split views."""

    _split_name: str | None = None

    @property
    def _window(self) -> tuple[int, int] | None:
        return None

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
                window=self._window,
                seed=seed,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
            )
        return _torch.TorchDataset(self._proto, self._db._object_reader, window=self._window)

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


class Dataset(_Dataset):
    """A packed dataset with optional training, validation, and test splits."""

    @property
    def train(self) -> DatasetSplit:
        """Return the training split without materializing its parent."""
        return DatasetSplit(self, "train")

    @property
    def validation(self) -> DatasetSplit:
        """Return the fixed validation split without materializing its parent."""
        return DatasetSplit(self, "validation")

    @property
    def test(self) -> DatasetSplit:
        """Return the fixed test split without materializing its parent."""
        return DatasetSplit(self, "test")


class DatasetSplit(_Dataset):
    """A named immutable sequence view; storage and materialization belong to its parent."""

    def __init__(self, parent: Dataset, name: str) -> None:
        if name not in ("train", "validation", "test"):
            raise ValueError("split name must be train, validation, or test")
        if isinstance(parent, DatasetSplit):
            raise ValueError("a split view cannot be split again")
        if name != "train" and not parent._resource.HasField("splits"):
            raise ValueError(f"dataset has no {name} split; create it with mix(splits=Splits(...))")
        super().__init__(parent._db, parent._resource, parent._creation_request)
        self._parent = parent
        self._split_name = name

    def __repr__(self) -> str:
        return f"DatasetSplit(name={self._split_name!r}, parent={self._parent.id!r})"

    @property
    def status(self) -> ExecutionStatus:
        """Return the shared parent materialization status."""
        return self._parent.status

    @property
    def id(self) -> str:
        """Return a stable identity specific to this parent and split."""
        from blake3 import blake3

        from premixdb.schemas.ids import _encode_id

        assert self._split_name is not None
        return _encode_id(
            blake3(
                b"premixdb-dataset-view/v1\0" + self._resource.id + self._split_name.encode()
            ).digest()
        )

    def wait(self, *, timeout: float | None = None) -> DatasetSplit:
        """Materialize the parent once and refresh this split's saved ranges."""
        self._parent.wait(timeout=timeout)
        self._resource = self._parent._resource
        return self

    @property
    def _window(self) -> tuple[int, int]:
        if not self._resource.HasField("splits"):
            return 0, self._resource.profile.sequences
        assert self._split_name is not None
        window = getattr(self._resource.split_ranges, self._split_name)
        assert isinstance(window, mix_pb.SequenceRange)
        return window.start, window.stop

    def profile(self) -> mix_pb.DatasetProfile:
        """Return exact packing totals for this split without packing tokens."""
        self._parent.profile()
        self._parent._resource = self._db._get("Dataset", self._resource.id)
        self._resource = self._parent._resource
        if not self._resource.HasField("splits"):
            return self._parent.profile()
        assert self._split_name is not None
        return _public_dataset_profile(
            copy_message(getattr(self._resource.split_profiles, self._split_name)),
            corpus_strata=self._resource.sampling.domains.field == queries.FIELD_SOURCE_CORPUS_ID,
        )

    def __len__(self) -> int:
        self.wait()
        start, stop = self._window
        return stop - start

    def _page(self, ordinal: int, size: int | None = None) -> list[Sequence]:
        self.wait()
        start, stop = self._window
        if not 0 <= ordinal < stop - start:
            raise IndexError("sequence index out of range")
        values = read_page(self._resource, self._db._object_reader, start + ordinal, size)
        result = []
        for sequence in values:
            if start <= sequence.ordinal < stop:
                value = copy_message(sequence._value)
                value.ordinal -= start
                result.append(Sequence(value, sequence._reader))
        return result
