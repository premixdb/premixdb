"""Query handles, profiles, provenance, and mixture recipes."""

from __future__ import annotations

from typing import (
    Iterable,
    Mapping,
)

from premixdb.api.base import _Execution
from premixdb.api.mixture import DataMixture
from premixdb.api.profiles import ProfileSelector
from premixdb.api.progress import report_progress
from premixdb.contracts import (
    PreviewDocument,
)
from premixdb.engine.contracts import Provenance
from premixdb.engine.mixing import Bounds, RegMix, Tokens
from premixdb.engine.policies import ByteTokenizer as BytePolicy
from premixdb.engine.policies import Concat as ConcatPolicy
from premixdb.fields.selectors import DomainInput
from premixdb.schemas import requests as _requests
from premixdb.schemas.protobuf import copy_message
from premixdb.storage.lineage import decode_lineage
from premixdb.v1 import data_mixture_pb2 as mix_pb
from premixdb.v1 import profile_pb2 as profiles
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import storage_pb2 as source_types


class Query(_Execution[queries.Query, queries.CreateQueryRequest]):
    def profile(self) -> queries.QueryProfile:
        """Execute the full query if needed, then return cached selection statistics.

        Even a query with no steps materializes its selection on the first call.
        Use snapshot.profile() for existing capture statistics without a query.
        """
        return copy_message(self.wait()._resource.profile)

    def preview(
        self, *, limit: int = 3, offset: int = 0, max_characters: int = 1024
    ) -> list[PreviewDocument]:
        """Wait for results and browse up to three selected documents, starting at offset."""
        return self._preview(limit=limit, offset=offset, max_characters=max_characters)

    def _with_fields(self, fields: Iterable[ProfileSelector]) -> Query:
        self._db._require_open()
        request = _requests.query(
            *self._resource.snapshot_ids,
            steps=self._resource.operations,
            fields=fields,
            decontaminate=self._resource.decontaminate
            if self._resource.HasField("decontaminate")
            else None,
            sampling=self._resource.sampling if self._resource.HasField("sampling") else None,
            git_commit=self._resource.git_commit,
        )
        if self._db._read_only:
            from premixdb.runtime.planner import compile_query

            return self._db._query(compile_query(request).id)
        from premixdb.runtime import Coordinator

        assert isinstance(self._db._executor, Coordinator)
        return Query(self._db, self._db._executor._plan_query(request), request)

    def _provenance(self) -> dict[str, Provenance]:
        """Trace each selected document to its source and query decisions."""
        resource = self.wait()._resource
        ref = resource.lineage
        return decode_lineage(
            self._db._object_reader.read(
                source_types.SpanRef(
                    object=ref, end=ref.size_bytes, blake3_digest=ref.blake3_digest
                )
            ),
            public=True,
        )

    @property
    def _estimate(self) -> profiles.QueryEstimate:
        """Refresh cardinality bounds and field distributions without waiting."""
        return copy_message(self._db._get("Query", self._resource.id).estimate)

    @report_progress("Creating mixture from query {id}")
    def mix(
        self,
        *,
        domains: DomainInput | None = None,
        weights: Mapping[str, float] | RegMix | None = None,
        size: Tokens | None = None,
        tokens: int | None = None,
        tokenizer: mix_pb.Tokenizer | BytePolicy | None = None,
        sequence_length: int = 2048,
        packing: mix_pb.Packing | ConcatPolicy | None = None,
        bounds: Bounds | None = None,
        n_candidates: int = 1,
        replacement: bool = False,
        seed: int = 0,
    ) -> DataMixture:
        """Register a lazy mixture recipe; defaults pack the query unchanged."""
        self._db._require_writable("plan mixtures")
        request = _requests.mix(
            self._resource.id,
            domains=domains,
            weights=weights,
            size=size,
            tokens=tokens,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            packing=packing,
            bounds=bounds,
            n_candidates=n_candidates,
            replacement=replacement,
            seed=seed,
        )
        id = self._db._submit(request).id
        response = self._db._executor.GetMix(
            mix_pb.GetMixRequest(id=id),
        )
        return DataMixture(self._db, response.mix, request)
