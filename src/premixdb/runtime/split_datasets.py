"""Plan and pack fixed content holdouts around a candidate's training mixture."""

from __future__ import annotations

import time
from contextlib import ExitStack, closing
from copy import copy
from typing import TYPE_CHECKING

from premixdb.engine.concatenated import ConcatenatedDataset
from premixdb.engine.dataset_plan import DatasetPlan, PackingPlan
from premixdb.engine.datasets import Dataset
from premixdb.engine.queries import Query, Row
from premixdb.runtime.mixing import canonical_digest
from premixdb.schemas.splits import SPLIT_NAMES, content_split
from premixdb.v1 import data_mixture_pb2 as d

if TYPE_CHECKING:
    from premixdb.runtime.coordinator import Coordinator


def partition(query: Query, policy: d.Splits, name: str) -> Query:
    result = copy(query)
    result._rows = [
        Row(i, row.document)
        for i, row in enumerate(
            r for r in query if content_split(r.document.content, policy) == name
        )
    ]
    from blake3 import blake3

    result._id = blake3(
        b"premixdb-split-query/v1\0"
        + bytes.fromhex(query.id)
        + canonical_digest("splits", policy)
        + name.encode()
    ).hexdigest()
    return result


def build(coordinator: Coordinator, spec: d.CreateDatasetRequest) -> ConcatenatedDataset:
    query = coordinator._packing_query(spec)
    tokenizer = coordinator._tokenizer(spec)
    parts = []
    with ExitStack() as cleanup:
        for name in SPLIT_NAMES:
            if name == "train" and spec.HasField("sampling"):
                pool, args = coordinator._sampling(spec)
                part = pool.dataset(*args, *coordinator._packing(spec), stream=True)
            else:
                selected = partition(query, spec.splits, name)
                plan = DatasetPlan(
                    selected.id,
                    spec.tokenizer.definition_digest.hex(),
                    query.code,
                    PackingPlan(*coordinator._packing(spec)),
                )
                if not selected.row_count:
                    part = Dataset(
                        selected,
                        plan,
                        selected.source_counts(),
                        (),
                        time.monotonic(),
                        (),
                        stream=True,
                    )
                elif coordinator.pipeline is None:
                    part = selected.dataset(
                        *coordinator._packing(spec), tokenizer=tokenizer, stream=True
                    )
                else:
                    lengths, encoded = coordinator.pipeline.tokenize(selected, spec.tokenizer)
                    part = Dataset(
                        selected,
                        plan,
                        selected.source_counts(),
                        encoded,
                        time.monotonic(),
                        lengths,
                        stream=True,
                    )
            cleanup.callback(part.close)
            parts.append(part)
        result = ConcatenatedDataset(coordinator._dataset_plan(spec), tuple(parts))
        cleanup.pop_all()
    return result


def profiles(
    coordinator: Coordinator, spec: d.CreateDatasetRequest
) -> tuple[d.DatasetProfile, d.DatasetSplitMetadata]:
    query = coordinator._packing_query(spec)
    packing = PackingPlan(*coordinator._packing(spec))
    tokenizer = coordinator._tokenizer(spec)
    metadata = d.DatasetSplitMetadata()
    total = d.DatasetProfile()
    cursor = 0
    for name in SPLIT_NAMES:
        planned = {}
        if name == "train" and spec.HasField("sampling"):
            pool, args = coordinator._sampling(spec)
            prepared = pool._prepare(*args)
            values = prepared.profile(packing)
            geometry = prepared.geometry(packing)
            planned = args[0]
        else:
            selected = partition(query, spec.splits, name)
            if not selected.row_count:
                lengths = []
            elif tokenizer is None:
                lengths = selected.lengths()
            elif coordinator.pipeline is not None:
                lengths, _ = coordinator.pipeline.tokenize(selected, spec.tokenizer)
            else:
                from premixdb.engine.token_cache import token_pool

                with closing(token_pool(selected, tokenizer)) as tokens:
                    lengths = [tokens.length(row.id) for row in selected]
            values = packing.profile(selected.source_counts(), lengths)
            geometry = packing.geometry(
                (r.id, r.corpus_id, n) for r, n in zip(selected, lengths, strict=True)
            )
        profile = d.DatasetProfile(**values, **geometry, planned_stratum_tokens=planned)
        getattr(metadata.profiles, name).CopyFrom(profile)
        getattr(metadata.ranges, name).CopyFrom(
            d.SequenceRange(start=cursor, stop=cursor + profile.sequences)
        )
        cursor += profile.sequences
        for field, value in profile.ListFields():
            if field.is_repeated:
                dest = getattr(total, field.name)
                for key, count in value.items():
                    dest[key] = dest.get(key, 0) + count
            else:
                setattr(total, field.name, getattr(total, field.name) + value)
    return total, metadata


def attach(coordinator: Coordinator, resource: d.Dataset) -> None:
    if resource.HasField("splits"):
        metadata = coordinator._storage.load(
            "dataset", resource.id, d.DatasetSplitMetadata, suffix=".split-metadata"
        )
        resource.split_ranges.CopyFrom(metadata.ranges)
        resource.split_profiles.CopyFrom(metadata.profiles)
