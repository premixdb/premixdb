"""Lazy mixture collections and candidate composition profiles."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    MutableMapping,
    overload,
)

from premixdb.api.base import _Resource
from premixdb.api.progress import report_progress
from premixdb.contracts import (
    ExecutionError,
)
from premixdb.schemas import requests as _requests
from premixdb.schemas.ids import _decode_id, _encode_id
from premixdb.schemas.protobuf import copy_message
from premixdb.v1 import data_mixture_pb2 as mix_pb
from premixdb.v1 import query_pb2 as queries

if TYPE_CHECKING:
    from premixdb.api.database import PremixDB
    from premixdb.api.dataset import Dataset


class DataMixture(_Resource[mix_pb.Mix, mix_pb.CreateMixRequest]):
    """A lazy mixture recipe resolving to an ordered collection of training datasets."""

    def __init__(
        self,
        client: PremixDB,
        resource: mix_pb.Mix,
        request: mix_pb.CreateMixRequest | None = None,
        *,
        indices: Iterable[int] | None = None,
        cache: dict[int, Dataset] | None = None,
    ) -> None:
        super().__init__(client, resource, request)
        self._indices = (
            tuple(range(resource.n_candidates or len(resource.dataset_ids)))
            if indices is None
            else tuple(indices)
        )
        self._cache = {} if cache is None else cache

    def _resolve(self) -> None:
        self._db._require_open()
        if len(self._resource.dataset_ids) == self._resource.n_candidates:
            return
        if self._db._read_only:
            raise ExecutionError("mixture is unresolved; resolve it in a writable session first")
        from premixdb.runtime import Coordinator

        assert isinstance(self._db._executor, Coordinator)
        self._resource = self._db._executor._resolve_mix(self._resource.id)

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        """The selected concrete recipes, resolved without packing output."""
        return tuple(self)

    @property
    def weights(self) -> list[dict[str, float]]:
        """Detached resolved weight vectors for the selected candidates."""
        return [dict(candidate.weights) for candidate in self.profile().candidates]

    @property
    def _configs(self) -> list[mix_pb.CreateDatasetRequest]:
        return [dataset._recipe for dataset in self]

    @report_progress("Profiling mixture {id}")
    def profile(self) -> mix_pb.MixProfile:
        """Inventory and composition of this mixture; never profile packed datasets."""
        self._resolve()
        if not self._resource.HasField("profile"):
            if self._db._read_only:
                raise ExecutionError("mixture profile is unavailable; use a writable session first")
            from premixdb.runtime import Coordinator

            assert isinstance(self._db._executor, Coordinator)
            self._resource = self._db._executor._profile_mix(self._resource.id)
        result = copy_message(self._resource.profile)
        candidates = [copy_message(result.candidates[i]) for i in self._indices]
        del result.candidates[:]
        result.candidates.extend(candidates)
        return _public_mix_profile(result)

    @report_progress("Previewing mixture {id}")
    def preview(self, *, limit: int = 3, offset: int = 0) -> mix_pb.MixPreview:
        """Show candidate compositions and allocations, without packing sequences."""
        limit = _requests._uint(limit, 32, "limit")
        offset = _requests._uint(offset, 64, "offset")
        if limit > 1000:
            raise ValueError("preview supports at most 1000 compositions")
        if not limit or offset >= len(self):
            return mix_pb.MixPreview()
        return mix_pb.MixPreview(candidates=self[offset : offset + limit].profile().candidates)

    def __len__(self) -> int:
        return len(self._indices)

    def _index(self, index: int) -> int:
        if type(index) is not int:
            raise TypeError("candidate index must be an integer")
        return self._indices[index]

    @overload
    def __getitem__(self, index: int) -> Dataset: ...
    @overload
    def __getitem__(self, index: slice) -> DataMixture: ...
    def __getitem__(self, index: int | slice) -> Dataset | DataMixture:
        if isinstance(index, slice):
            return DataMixture(
                self._db,
                self._resource,
                self._creation_request,
                indices=self._indices[index],
                cache=self._cache,
            )
        original = self._index(index)
        self._resolve()
        if original not in self._cache:
            dataset = self._db._dataset(self._resource.dataset_ids[original])
            dataset._creation_request = dataset._recipe
            self._cache[original] = dataset
        return self._cache[original]

    def __iter__(self) -> Iterator[Dataset]:
        for index in range(len(self)):
            yield self[index]


def _public_mix_profile(profile: mix_pb.MixProfile) -> mix_pb.MixProfile:
    if profile.domains.field == queries.FIELD_SOURCE_CORPUS_ID:
        _public_domain_keys(profile.domain_tokens)
        _public_domain_keys(profile.bounds.lower)
        _public_domain_keys(profile.bounds.upper)
        for candidate in profile.candidates:
            _public_domain_keys(candidate.weights)
            _public_domain_keys(candidate.tokens)
    return profile


def _public_domain_keys[T: int | float](values: MutableMapping[str, T]) -> None:
    encoded = {_encode_id(_decode_id(key)): value for key, value in values.items()}
    values.clear()
    values.update(encoded)
