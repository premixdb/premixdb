"""Resolve, pack, and publish training dataset recipes."""

from __future__ import annotations

from contextlib import closing
from typing import (
    TYPE_CHECKING,
)

from blake3 import blake3

from premixdb.engine import execution
from premixdb.engine.dataset_plan import BYTE_DEFINITION, DatasetPlan, PackingPlan
from premixdb.runtime import environment as _runtime
from premixdb.runtime import mixing
from premixdb.runtime.planner import (
    copy_fields,
)
from premixdb.schemas import requests as _requests
from premixdb.schemas.protobuf import copy_message
from premixdb.storage import tokens
from premixdb.storage.catalog import Catalog
from premixdb.v1 import data_mixture_pb2 as datasets
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import status_pb2 as status
from premixdb.v1 import storage_pb2 as storage

if TYPE_CHECKING:
    from premixdb.runtime.coordinator import Coordinator


def resolve_recipe(
    coordinator: Coordinator, request: datasets.CreateDatasetRequest, *, _lazy: bool = False
) -> datasets.CreateDatasetRequest:
    parent = (
        Catalog.GetQuery(coordinator, queries.GetQueryRequest(id=request.query_id))
        if _lazy
        else coordinator.GetQuery(queries.GetQueryRequest(id=request.query_id))
    ).query
    code = _runtime.resolve_code(request.git_commit or parent.git_commit)
    if parent.git_commit != bytes.fromhex(code.commit):
        raise ValueError("dataset revision must match its query")
    spec = _requests.dataset(
        request.query_id,
        tokenizer=request.tokenizer if request.HasField("tokenizer") else None,
        sequence_length=request.sequence_length or 2048,
        packing=request.packing if request.HasField("packing") else None,
        sampling=request.sampling if request.HasField("sampling") else None,
        splits=request.splits if request.HasField("splits") else None,
        git_commit=code.commit,
    )
    if _lazy and spec.tokenizer.HasField("hugging_face"):
        from premixdb.engine.datasets import _tokenizer_definition
        from premixdb.runtime.assets import read

        policy = spec.tokenizer.hugging_face
        relative = "tokenizer/objects/" + policy.asset.blake3_digest.hex()
        inline = policy.json or None
        if inline is None and policy.asset.uri == coordinator._storage.object_uri(relative):
            inline = coordinator._storage._get(relative)
        data = read(policy.asset, local_root=coordinator._source_root, inline=inline)
        policy.asset.CopyFrom(coordinator._storage.put("tokenizer", data))
        policy.ClearField("json")
        definition = bytes.fromhex(_tokenizer_definition(policy.asset.blake3_digest.hex()))
    else:
        tokenizer = coordinator._tokenizer(spec)
        definition = bytes.fromhex(tokenizer.definition if tokenizer else BYTE_DEFINITION)
        if tokenizer is not None:
            policy = spec.tokenizer.hugging_face
            policy.asset.CopyFrom(coordinator._storage.put("tokenizer", tokenizer.asset_bytes))
            policy.ClearField("json")
    if spec.tokenizer.definition_digest and spec.tokenizer.definition_digest != definition:
        raise ValueError("tokenizer definition does not match its policy")
    spec.tokenizer.definition_digest = definition
    if spec.HasField("sampling"):
        sampling = spec.sampling
        if not sampling.HasField("seed"):
            sampling.seed = 0
        if not sampling.HasField("replacement"):
            sampling.replacement = True
        if sampling.domains.WhichOneof("kind") is None:
            sampling.domains.field = queries.FIELD_SOURCE_CORPUS_ID
        mixing.validate_sampling(sampling)
    return spec


def load_tokenizer_asset(
    coordinator: Coordinator, asset: storage.ObjectRef, limit: int, inline: bytes | None = None
) -> execution.HuggingFaceTokenizer:
    from premixdb.runtime.assets import read

    relative = "tokenizer/objects/" + asset.blake3_digest.hex()
    owned_uri = coordinator._storage.object_uri(relative)
    if inline is None and asset.uri == owned_uri:
        inline = coordinator._storage._get(relative)
    data = read(asset, local_root=coordinator._source_root, inline=inline)
    return execution.HuggingFaceTokenizer.from_bytes(data, asset.blake3_digest.hex(), limit)


def tokenizer(
    coordinator: Coordinator,
    spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset,
) -> execution.HuggingFaceTokenizer | None:
    if spec.tokenizer.HasField("byte"):
        return None
    policy = spec.tokenizer.hugging_face
    return coordinator._tokenizer_asset(
        policy.asset, policy.max_document_bytes, policy.json or None
    )


def mixture_pool(
    coordinator: Coordinator,
    spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset,
    strata: datasets.Domains,
) -> execution.MixturePool:
    if strata.HasField("fields"):
        from premixdb.fields.ids import field_name, selector_field
        from premixdb.runtime.catalog import build_id, plan
        from premixdb.runtime.planner import validate_projection

        query = coordinator.GetQuery(queries.GetQueryRequest(id=spec.query_id)).query
        for selector in strata.fields.selectors:
            selector.field = selector_field(selector)
            selector.ClearField("field_name")
            if selector.operator or selector.WhichOneof("value") is not None:
                raise ValueError("mixture domains must be field projections, not predicates")
            validate_projection(selector)
            if selector.field in (1, 2, 3, 4):
                if selector.field_snapshot_id:
                    raise ValueError("intrinsic domains cannot contain a derivation pin")
                continue
            definition = plan(query.snapshot_ids, field_name(selector.field), query.git_commit)
            pin = build_id(definition, field_name(selector.field))
            if selector.field_snapshot_id and selector.field_snapshot_id != pin:
                raise ValueError("domain pin differs from its built-in recipe")
            selector.field_snapshot_id = pin
    strata_digest = mixing.canonical_digest("mixture-strata", strata)
    if spec.HasField("splits"):
        strata_digest = blake3(
            strata_digest + mixing.canonical_digest("splits", spec.splits)
        ).digest()
    key = (
        spec.query_id,
        spec.tokenizer.definition_digest,
        strata_digest,
        _runtime.current_code().canonical_digest(),
    )
    with coordinator._mix_pool_lock:
        pool = coordinator._mix_pools.get(key)
        if pool is None:
            from premixdb.engine.mixtures import MixturePool

            query = coordinator._packing_query(spec)
            assignments = dict(strata.assignments.documents)
            if strata.HasField("fields"):
                from premixdb.engine.curation import selector_key
                from premixdb.engine.mixing import _domain_key
                from premixdb.runtime.enrichment import projections

                values = projections(coordinator, query, strata.fields.selectors)
                assignments = {
                    r.id: _domain_key(
                        [values[selector_key(s)][r.id] for s in strata.fields.selectors]
                    )
                    for r in query
                }
            field = mixing.domains_name(strata)
            if spec.HasField("splits"):
                from premixdb.runtime.split_datasets import partition

                original = query
                if not field and set(assignments) != {r.id for r in original}:
                    raise ValueError("assignments must cover the query exactly")
                labels = {
                    r.id: r.corpus_id
                    if field == "source.corpus_id"
                    else r.source_key
                    if field == "object.uri"
                    else assignments[r.id]
                    for r in original
                }
                query = partition(query, spec.splits, "train")
                assignments = {r.id: labels[r.id] for r in query} if not field else {}
            pool = MixturePool(query, field, assignments, coordinator._tokenizer(spec))
            if spec.HasField("splits"):
                for label in labels.values():
                    pool._inventory.setdefault(label, 0)
            coordinator._owned_mix_pools.add(pool)
            coordinator._mix_pools[key] = pool
        return pool


def sampling(
    coordinator: Coordinator, spec: datasets.CreateDatasetRequest | datasets.Dataset
) -> tuple[execution.MixturePool, tuple[dict[str, int], str, int, bool, int | None]]:
    sampling = spec.sampling
    pool = coordinator._mix_pool(spec, sampling.domains)
    counts = mixing.allocations(sampling.weights, sampling.tokens)
    cap = sampling.max_epochs if sampling.HasField("max_epochs") else None
    mixing.capacity_check(counts, pool.inventory(), sampling.replacement, cap)
    return pool, (
        counts,
        mixing.canonical_digest("mixture-sampling", sampling).hex(),
        sampling.seed,
        sampling.replacement,
        cap,
    )


def packing(spec: datasets.CreateDatasetRequest) -> tuple[int, int | None, int | None]:
    policy = spec.packing.concat
    return (
        spec.sequence_length,
        policy.separator_token_id if policy.HasField("separator_token_id") else None,
        policy.pad_token_id if policy.HasField("pad_token_id") else None,
    )


def plan(coordinator: Coordinator, spec: datasets.CreateDatasetRequest) -> DatasetPlan:
    identity = spec.query_id.hex()
    if spec.HasField("sampling"):
        pool, args = coordinator._sampling(spec)
        identity = pool.identity(*args)
    if spec.HasField("splits"):
        identity = blake3(
            b"premixdb-split-dataset/v1\0"
            + bytes.fromhex(identity)
            + mixing.canonical_digest("splits", spec.splits)
        ).hexdigest()
    return DatasetPlan(
        identity,
        spec.tokenizer.definition_digest.hex(),
        _runtime.resolve_code(spec.git_commit),
        PackingPlan(*coordinator._packing(spec)),
    )


def identity(coordinator: Coordinator, spec: datasets.CreateDatasetRequest) -> bytes:
    return bytes.fromhex(coordinator._dataset_plan(spec).id)


def packing_query(
    coordinator: Coordinator,
    spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset,
) -> execution.Query:
    from copy import copy

    query = coordinator._query(spec.query_id)
    code = _runtime.resolve_code(spec.git_commit)
    if query.code != code:
        # Selection is a frozen input. New packing pins the executing runtime
        # even when that input was produced by an older environment.
        query = copy(query)
        query.code = code
    return query


def build(coordinator: Coordinator, spec: datasets.CreateDatasetRequest) -> execution.Dataset:
    if spec.HasField("splits"):
        from premixdb.runtime import split_datasets

        return split_datasets.build(coordinator, spec)
    if spec.HasField("sampling"):
        pool, args = coordinator._sampling(spec)
        return pool.dataset(*args, *coordinator._packing(spec), stream=True)
    query = coordinator._packing_query(spec)
    if coordinator.pipeline is None:
        return query.dataset(
            *coordinator._packing(spec), tokenizer=coordinator._tokenizer(spec), stream=True
        )
    import time

    from premixdb.engine.datasets import Dataset

    plan = coordinator._dataset_plan(spec)
    lengths, encoded = coordinator.pipeline.tokenize(query, spec.tokenizer)
    return Dataset(
        query,
        plan,
        query.source_counts(),
        encoded,
        start=time.monotonic(),
        lengths=lengths,
        stream=True,
    )


def profile(
    coordinator: Coordinator, spec: datasets.CreateDatasetRequest
) -> datasets.DatasetProfile:
    spec = coordinator._resolve_dataset(spec)
    key = blake3(
        b"premixdb-dataset-profile-runtime/v1\0"
        + mixing.canonical_digest("dataset-profile", spec)
        + _runtime.current_code().canonical_digest()
    ).digest()
    with coordinator._profile_lock:
        if key in coordinator._dataset_profiles:
            return copy_message(coordinator._dataset_profiles[key])
        try:
            profile = coordinator._storage.load(
                "dataset", key, datasets.DatasetProfile, suffix=".profile"
            )
        except KeyError:
            pass
        else:
            coordinator._dataset_profiles[key] = profile
            return copy_message(profile)
        if spec.HasField("splits"):
            from premixdb.runtime import split_datasets

            profile, metadata = split_datasets.profiles(coordinator, spec)
            coordinator._storage.save(
                "dataset", coordinator._dataset_id(spec), metadata, suffix=".split-metadata"
            )
            coordinator._storage.save("dataset", key, profile, suffix=".profile")
            coordinator._dataset_profiles[key] = profile
            return copy_message(profile)
        planned = {}
        if spec.HasField("sampling"):
            pool, args = coordinator._sampling(spec)
            planned = args[0]
            packing = PackingPlan(*coordinator._packing(spec))
            prepared = pool._prepare(*args)
            values = prepared.profile(packing)
            geometry = prepared.geometry(packing)
        else:
            handle = coordinator._query(spec.query_id)
            tokenizer = coordinator._tokenizer(spec)
            if tokenizer:
                if coordinator.pipeline is not None:
                    lengths, _ = coordinator.pipeline.tokenize(handle, spec.tokenizer)
                else:
                    from premixdb.engine.token_cache import token_pool

                    with closing(token_pool(handle, tokenizer)) as tokens:
                        lengths = [tokens.length(row.id) for row in handle]
                values = PackingPlan(*coordinator._packing(spec)).profile(
                    handle.source_counts(),
                    lengths,
                )
            else:
                values = handle.profile(*coordinator._packing(spec))
                lengths = handle.lengths()
            geometry = PackingPlan(*coordinator._packing(spec)).geometry(
                (r.id, r.corpus_id, length) for r, length in zip(handle, lengths, strict=True)
            )
        profile = datasets.DatasetProfile(**values, **geometry, planned_stratum_tokens=planned)
        coordinator._storage.save("dataset", key, profile, suffix=".profile")
        coordinator._dataset_profiles[key] = profile
        return copy_message(profile)


def create(
    coordinator: Coordinator,
    request: datasets.CreateDatasetRequest,
    *,
    _lazy: bool = False,
    timeout: float | None = None,
) -> datasets.CreateDatasetResponse:
    spec = coordinator._resolve_dataset(request, _lazy=_lazy)
    if _lazy:
        identity = coordinator._dataset_id(spec)
        try:
            Catalog.GetDataset(coordinator, datasets.GetDatasetRequest(id=identity))
        except KeyError:
            resource = copy_fields(
                spec, datasets.Dataset(id=identity, status=status.STATUS_PENDING)
            )
            coordinator._storage.save("dataset", identity, resource, suffix=".recipe")
        return datasets.CreateDatasetResponse(id=identity)

    def run() -> datasets.Dataset:
        id = coordinator._dataset_id(spec)
        try:
            existing = coordinator._storage.load("dataset", id, datasets.Dataset)
            if existing.status == status.STATUS_COMPLETED:
                return existing
        except KeyError:
            pass
        with closing(coordinator._dataset_handle(spec)) as native:
            handle = (
                coordinator.pipeline.pack(native) if coordinator.pipeline is not None else native
            )
            if bytes.fromhex(handle.id) != id:
                raise RuntimeError("kernel dataset identity does not match its recipe")
            result = copy_fields(
                spec,
                datasets.Dataset(
                    id=id,
                    status=status.STATUS_COMPLETED,
                    profile=coordinator._profile_dataset(spec),
                ),
            )
            from premixdb.runtime.split_datasets import attach

            attach(coordinator, result)
            spans, batches, preview = tokens.publish(
                coordinator._storage,
                handle,
                profile=result.profile,
                lineage=coordinator._query(spec.query_id).provenance(),
                tokenizer=coordinator._tokenizer(spec),
            )
            result.tokens.extend(spans)
            result.sequences.extend(batches)
            result.preview.CopyFrom(preview)
            return result

    # Mix candidates and direct builds share the same registered recipe.
    id = coordinator._dataset_id(spec)
    try:
        coordinator._storage.load("dataset", id, datasets.Dataset)
    except KeyError:
        pass
    else:
        return datasets.CreateDatasetResponse(id=id)
    resource = copy_fields(
        spec,
        datasets.Dataset(
            id=id, status=status.STATUS_PENDING, profile=coordinator._profile_dataset(spec)
        ),
    )
    try:
        coordinator._storage.load("dataset", id, datasets.Dataset, suffix=".recipe")
    except KeyError:
        coordinator._storage.save("dataset", id, resource, suffix=".recipe")
    return coordinator._once(
        spec,
        lambda: datasets.CreateDatasetResponse(
            id=coordinator._materialize("dataset", resource, run).result().id
        ),
    )


def planned_profile(
    coordinator: Coordinator, resource: datasets.Dataset
) -> datasets.DatasetProfile:
    profile = coordinator._profile_dataset(copy_fields(resource, datasets.CreateDatasetRequest()))
    coordinator._storage.save("dataset", resource.id, profile, suffix=".planned-profile")
    return profile


def plan_resource(
    coordinator: Coordinator, request: datasets.CreateDatasetRequest
) -> datasets.Dataset:
    """Register a recipe without executing its query, profiling, or packing."""
    identity = coordinator.CreateDataset(request, _lazy=True).id
    result = Catalog.GetDataset(coordinator, datasets.GetDatasetRequest(id=identity)).dataset
    if result.status == status.STATUS_ERROR:
        result.status = status.STATUS_PENDING
        result.ClearField("error")
    return result


def get(
    coordinator: Coordinator, request: datasets.GetDatasetRequest
) -> datasets.GetDatasetResponse:
    resource = coordinator._materialized_resource(
        "dataset", request.id, datasets.Dataset, coordinator._datasets, ".recipe"
    )
    return datasets.GetDatasetResponse(dataset=coordinator._dataset_profile(resource))
