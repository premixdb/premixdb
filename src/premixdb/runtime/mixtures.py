"""Register and resolve lazy mixture candidates."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Mapping,
)

from blake3 import blake3

from premixdb.contracts import ExecutionError
from premixdb.runtime import environment as _runtime
from premixdb.runtime import mixing
from premixdb.runtime.planner import (
    copy_fields,
    reject_unknown,
)
from premixdb.schemas.protobuf import copy_message
from premixdb.storage.catalog import Catalog
from premixdb.v1 import data_mixture_pb2 as datasets
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import status_pb2 as status

if TYPE_CHECKING:
    from premixdb.runtime.coordinator import Coordinator


def create(
    coordinator: Coordinator, request: datasets.CreateMixRequest
) -> datasets.CreateMixResponse:
    """Pin and register the recipe without running its query or token inventory."""
    reject_unknown(request)

    def run() -> datasets.CreateMixResponse:
        template = coordinator._resolve_dataset(
            copy_fields(request, datasets.CreateDatasetRequest()), _lazy=True
        )
        spec = copy_fields(request, datasets.CreateMixRequest())
        from premixdb.schemas.splits import split_policy

        spec.splits.CopyFrom(split_policy(spec.splits if spec.HasField("splits") else None))
        for name in ("tokenizer", "packing"):
            getattr(spec, name).CopyFrom(getattr(template, name))
        spec.git_commit = template.git_commit
        spec.sequence_length = template.sequence_length
        if spec.domains.WhichOneof("kind") is None:
            spec.domains.field = queries.FIELD_SOURCE_CORPUS_ID
        if not spec.HasField("seed"):
            spec.seed = 0
        if not spec.HasField("replacement"):
            spec.replacement = False
        spec.n_candidates = spec.n_candidates or 1
        if spec.algorithm.WhichOneof("kind") is None:
            spec.ClearField("algorithm")
        else:
            from premixdb.engine.mixing import RegMix

            defaults = RegMix()._to_proto().regmix
            policy = spec.algorithm.regmix
            for name in (
                "prior_power",
                "min_concentration",
                "max_concentration",
                "concentration_steps",
                "oversample",
            ):
                if not getattr(policy, name):
                    setattr(policy, name, getattr(defaults, name))
            if not policy.HasField("seed"):
                policy.seed = 0
        if not spec.bounds.ListFields():
            spec.ClearField("bounds")
        mixing.validate_mix(spec)
        result = copy_fields(spec, datasets.Mix())
        result.pass_through = (
            not spec.tokens and not spec.weights and not spec.HasField("algorithm")
        )
        result.execution_fingerprint = _runtime.current_code().canonical_digest()
        result.id = blake3(
            mixing.canonical_digest("mix-plan", spec) + result.execution_fingerprint
        ).digest()
        if result.ByteSize() > 3 * 1024 * 1024:
            raise ValueError("mixture metadata exceeds 3 MiB")
        with coordinator._lock:
            try:
                coordinator._storage.load("mixture", result.id, datasets.Mix)
            except KeyError:
                coordinator._storage.save("mixture", result.id, result, suffix=".recipe")
            coordinator._mixes.pop(result.id, None)
        return datasets.CreateMixResponse(id=result.id)

    return coordinator._once(request, run)


def resolve(coordinator: Coordinator, identity: bytes) -> datasets.Mix:
    """Freeze concrete candidate recipes; profiles and packed output stay lazy."""

    def run() -> datasets.Mix:
        result = Catalog.GetMix(coordinator, datasets.GetMixRequest(id=identity)).mix
        if len(result.dataset_ids) == result.n_candidates:
            return result
        if (
            result.execution_fingerprint
            and result.execution_fingerprint != _runtime.current_code().canonical_digest()
        ):
            raise ExecutionError("mixture execution environment changed; create a new mix recipe")
        spec = copy_fields(result, datasets.CreateMixRequest())
        template = copy_fields(result, datasets.CreateDatasetRequest())
        recipes = []
        if result.pass_through:
            resource = copy_fields(
                template,
                datasets.Dataset(
                    id=coordinator._dataset_id(template), status=status.STATUS_PENDING
                ),
            )
            recipes.append(resource)
            if result.bounds.ListFields():
                inventory = coordinator._mix_pool(template, result.domains).inventory()
                result.profile.CopyFrom(coordinator._mix_profile(result, inventory, recipes))
        else:
            inventory = coordinator._mix_pool(template, spec.domains).inventory()
            spec.tokens = spec.tokens or sum(inventory.values())
            candidates = mixing.generate(spec, inventory)
            single_domain = sum(value > 0 for value in inventory.values()) == 1
            for index, weights in enumerate(candidates):
                recipe = copy_message(template)
                recipe.sampling.CopyFrom(
                    datasets.Sampling(
                        domains=spec.domains,
                        weights=weights,
                        tokens=spec.tokens,
                        seed=(spec.seed + index) % (2**64) if single_domain else spec.seed,
                        replacement=spec.replacement,
                    )
                )
                if spec.bounds.HasField("max_epochs"):
                    recipe.sampling.max_epochs = spec.bounds.max_epochs
                resource = copy_fields(
                    recipe,
                    datasets.Dataset(
                        id=coordinator._dataset_id(recipe), status=status.STATUS_PENDING
                    ),
                )
                recipes.append(resource)
            result.tokens = spec.tokens
            result.domains.CopyFrom(spec.domains)
            result.profile.CopyFrom(coordinator._mix_profile(result, inventory, recipes))
        result.dataset_ids.extend(resource.id for resource in recipes)
        if result.ByteSize() + sum(recipe.ByteSize() for recipe in recipes) > 3 * 1024 * 1024:
            raise ValueError("mixture metadata exceeds 3 MiB; use smaller candidate batches")
        for resource in recipes:
            try:
                Catalog.GetDataset(coordinator, datasets.GetDatasetRequest(id=resource.id))
            except KeyError:
                coordinator._storage.save("dataset", resource.id, resource, suffix=".recipe")
        coordinator._storage.save("mixture", result.id, result)
        with coordinator._lock:
            coordinator._mixes[result.id] = copy_message(result)
        return result

    return copy_message(coordinator._submissions.run(("resolve-mixture", identity), run))


def profile(
    coordinator: Coordinator,
    resource: datasets.Mix,
    inventory: Mapping[str, int],
    recipes: list[datasets.Dataset],
) -> datasets.MixProfile:
    total = sum(inventory.values())
    profile = datasets.MixProfile(
        domain_tokens=inventory,
        population_tokens=total,
        tokens=resource.tokens or total,
        domains=resource.domains,
        algorithm=resource.algorithm if resource.HasField("algorithm") else None,
        bounds=resource.bounds,
        pass_through=resource.pass_through,
        replacement=resource.replacement,
        seed=resource.seed,
    )
    natural = {key: count / total if total else 0 for key, count in inventory.items()}
    for index, recipe in enumerate(recipes):
        weights = dict(recipe.sampling.weights) if recipe.HasField("sampling") else natural
        counts = (
            mixing.allocations(weights, recipe.sampling.tokens)
            if recipe.HasField("sampling")
            else dict(inventory)
        )
        profile.candidates.add(index=index, dataset_id=recipe.id, weights=weights, tokens=counts)
    if resource.pass_through and total:
        spec = copy_fields(resource, datasets.CreateMixRequest())
        spec.tokens = total
        mixing.generate(spec, inventory)  # Validate natural composition against requested bounds.
    return profile


def profile_resource(coordinator: Coordinator, identity: bytes) -> datasets.Mix:
    def run() -> datasets.Mix:
        resource = coordinator._resolve_mix(identity)
        if resource.HasField("profile"):
            return resource
        template = copy_fields(resource, datasets.CreateDatasetRequest())
        inventory = coordinator._mix_pool(template, resource.domains).inventory()
        recipes = [
            Catalog.GetDataset(coordinator, datasets.GetDatasetRequest(id=id)).dataset
            for id in resource.dataset_ids
        ]
        resource.profile.CopyFrom(coordinator._mix_profile(resource, inventory, recipes))
        coordinator._storage.save("mixture", resource.id, resource.profile, suffix=".profile")
        with coordinator._lock:
            coordinator._mixes[resource.id] = copy_message(resource)
        return resource

    return copy_message(coordinator._submissions.run(("profile-mixture", identity), run))
