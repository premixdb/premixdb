"""Metadata-only mixture planning over the Python token pool."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from fractions import Fraction
from typing import Mapping

from blake3 import blake3
from google.protobuf.message import Message

from .. import _requests
from .._mixing import RegMixSampler, _weights_message, allocations
from .._protobuf import descriptor_name
from .._typing import JSON
from .._typing import scalar as json_scalar
from ..engine.identity import identity_domain
from ..v1 import dataset_pb2 as datasets
from ..v1 import query_pb2 as queries
from .planner import reject_unknown

MAX_CANDIDATES = 10_000
MAX_PROPOSALS = 1_000_000


def canonical_digest(kind: str, message: Message) -> bytes:
    """Versioned logical encoding, excluding request IDs and transport serialization.

    Sorted map keys; exact float.hex values; explicit optional presence. Protobuf
    field names/types are part of this versioned schema contract.
    """

    def encode(message: Message) -> dict[str, JSON]:
        result: dict[str, JSON] = {}
        for descriptor, value in message.ListFields():
            if descriptor.name == "request_id":
                continue
            if (
                descriptor_name(message) == "premixdb.v1.ObjectRef"
                and descriptor.name != "blake3_digest"
            ):
                continue
            if (
                descriptor_name(message) == "premixdb.v1.HuggingFaceTokenizer"
                and descriptor.name == "json"
            ):
                continue

            def scalar(item: object) -> JSON:
                if isinstance(item, bytes):
                    return {"bytes": item.hex()}
                if isinstance(item, float):
                    return {"float": item.hex()}
                if isinstance(item, Message):
                    return encode(item)
                return json_scalar(item)

            if descriptor.is_repeated:
                if descriptor.message_type and descriptor.message_type.GetOptions().map_entry:
                    if not isinstance(value, Mapping):
                        raise TypeError("protobuf map did not provide a mapping")
                    result[descriptor.name] = {str(k): scalar(v) for k, v in value.items()}
                else:
                    if not isinstance(value, Iterable):
                        raise TypeError("protobuf repeated field did not provide an iterable")
                    result[descriptor.name] = [scalar(v) for v in value]
            else:
                result[descriptor.name] = scalar(value)
        return result

    payload = json.dumps(encode(message), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return blake3(identity_domain(kind) + payload.encode()).digest()


def domains_name(strata: datasets.Domains) -> str:
    return {
        queries.FIELD_SOURCE_CORPUS_ID: "source.corpus_id",
        queries.FIELD_OBJECT_URI: "object.uri",
    }.get(strata.field, "")


def validate_strata(strata: datasets.Domains) -> None:
    kind = strata.WhichOneof("kind")
    if kind == "field":
        if not domains_name(strata):
            raise NotImplementedError(
                "mix supports source.corpus_id, object.uri, or explicit assignments"
            )
    elif kind == "fields":
        if not strata.fields.selectors:
            raise ValueError("stratum fields cannot be empty")
    elif kind == "assignments":
        for id, label in strata.assignments.documents.items():
            if _requests._id(id, 32).hex() != id or not label:
                raise ValueError("assignments require lowercase document IDs and nonempty labels")
    else:
        raise ValueError("mixture strata are required")


def validate_sampling(spec: datasets.Sampling) -> None:
    reject_unknown(spec)
    validate_strata(spec.domains)
    _weights_message(spec.weights)
    if not spec.tokens or not spec.HasField("seed") or not spec.HasField("replacement"):
        raise ValueError("sampling requires tokens, seed, and replacement")
    if spec.HasField("max_epochs") and spec.max_epochs == 0:
        raise ValueError("max_epochs must be positive")


def validate_mix(spec: datasets.CreateMixRequest) -> None:
    reject_unknown(spec)
    validate_strata(spec.domains)
    if not spec.tokens or not spec.HasField("seed") or not spec.HasField("replacement"):
        raise ValueError("mix requires tokens, seed, and replacement")
    if not 1 <= spec.n_candidates <= MAX_CANDIDATES:
        raise ValueError(f"n_candidates must be between 1 and {MAX_CANDIDATES}")
    bounds = spec.bounds
    for mapping in (bounds.lower, bounds.upper):
        if any(not key or not math.isfinite(v) or not 0 <= v <= 1 for key, v in mapping.items()):
            raise ValueError("weight bounds must be finite and in [0, 1]")
    for name in ("max_epochs", "reference_tokens"):
        if bounds.HasField(name) and getattr(bounds, name) == 0:
            raise ValueError(f"{name} must be positive")
    if spec.algorithm.WhichOneof("kind") != "regmix":
        raise ValueError("mix requires a RegMix algorithm")
    policy = spec.algorithm.regmix
    RegMixSampler(
        seed=policy.seed,
        prior_power=policy.prior_power,
        concentration_range=(policy.min_concentration, policy.max_concentration),
        concentration_steps=policy.concentration_steps,
        minimum_weight=policy.minimum_weight,
        oversample=policy.oversample,
    )._to_proto()
    if policy.concentration_steps > 1024 or policy.oversample * spec.n_candidates > MAX_PROPOSALS:
        raise ValueError("RegMix proposal work exceeds the service limit")


def capacity_check(
    counts: Mapping[str, int],
    inventory: Mapping[str, int],
    replacement: bool,
    max_epochs: int | None,
) -> None:
    if set(counts) != set(inventory):
        raise ValueError("weights must name exactly the available strata")
    epochs = max_epochs if replacement else 1
    for key, count in counts.items():
        if count and (
            inventory[key] == 0 or (epochs is not None and count > inventory[key] * epochs)
        ):
            raise ValueError(f"token allocation exceeds capacity for {key}")


def resolved_bounds(
    spec: datasets.CreateMixRequest, inventory: Mapping[str, int]
) -> tuple[dict[str, float], dict[str, float]]:
    bounds = spec.bounds
    if (set(bounds.lower) | set(bounds.upper)) - inventory.keys():
        raise ValueError("bounds contain unknown strata")
    epochs = bounds.max_epochs if bounds.HasField("max_epochs") else None
    if not spec.replacement:
        epochs = 1
    budget = max(spec.tokens, bounds.reference_tokens or spec.tokens)
    lower = {k: bounds.lower.get(k, 0.0) for k in inventory}
    upper = {
        k: min(
            bounds.upper.get(k, 1.0),
            float(Fraction(n * epochs, budget)) if epochs is not None else 1.0,
        )
        if n
        else 0.0
        for k, n in inventory.items()
    }
    if (
        any(lower[k] > upper[k] for k in inventory)
        or math.fsum(lower.values()) > 1 + 1e-12
        or math.fsum(upper.values()) < 1 - 1e-12
    ):
        raise ValueError("mixture bounds and token capacities are infeasible")
    return lower, upper


def generate(
    spec: datasets.CreateMixRequest, inventory: Mapping[str, int]
) -> list[dict[str, float]]:
    if not inventory or not any(inventory.values()):
        raise ValueError("mix requires a nonempty token population")
    lower, upper = resolved_bounds(spec, inventory)
    keys = sorted(inventory)
    cap = spec.bounds.max_epochs if spec.bounds.HasField("max_epochs") else None

    def validate(values: Mapping[str, float]) -> None:
        _weights_message(values)
        if set(values) != set(keys):
            raise ValueError("weights must name exactly the available strata")
        if any(not lower[k] <= values[k] <= upper[k] for k in keys):
            raise ValueError("candidate violates mixture bounds")
        for budget in {spec.tokens, spec.bounds.reference_tokens or spec.tokens}:
            capacity_check(allocations(values, budget), inventory, spec.replacement, cap)

    active = [key for key in keys if inventory[key]]
    if len(active) == 1:
        values = {key: float(key == active[0]) for key in keys}
        validate(values)
        return [dict(values) for _ in range(spec.n_candidates)]

    p = spec.algorithm.regmix
    rng = random.Random(p.seed)
    # Work in log space so even large prior powers cannot overflow inventories.
    logs = {k: math.log(v) for k, v in inventory.items() if v}
    largest = max(logs.values())
    prior = {k: math.exp((logs[k] - largest) * p.prior_power) if k in logs else 0 for k in keys}
    total = math.fsum(prior.values())
    prior = {k: v / total for k, v in prior.items()}
    strengths = [
        math.exp(
            math.log(p.min_concentration)
            + i
            * (math.log(p.max_concentration) - math.log(p.min_concentration))
            / max(p.concentration_steps - 1, 1)
        )
        for i in range(p.concentration_steps)
    ]
    unique = set()
    for _ in range(spec.n_candidates * p.oversample):
        strength = rng.choice(strengths)
        draw = [
            rng.gammavariate(prior[k] * strength, 1.0) if prior[k] * strength > 0 else 0
            for k in keys
        ]
        total = math.fsum(draw)
        if total == 0:
            continue
        draw = [v / total for v in draw]
        if p.minimum_weight:

            def quantize(value: float) -> float:
                if value < p.minimum_weight:
                    return 0
                units = value / p.minimum_weight
                # A subnormal quantum can overflow the quotient. At that scale
                # quantization cannot change this representable float.
                return round(units) * p.minimum_weight if math.isfinite(units) else value

            draw = [quantize(v) for v in draw]
        total = math.fsum(draw)
        if not total:
            continue
        values = dict(zip(keys, (v / total for v in draw)))
        try:
            validate(values)
        except ValueError:
            continue
        unique.add(tuple(values[k] for k in keys))
    if len(unique) < spec.n_candidates:
        raise ValueError(
            f"only {len(unique)} unique feasible mixtures found; increase oversample or loosen bounds"
        )
    chosen = rng.sample(sorted(unique), spec.n_candidates)
    return [dict(zip(keys, values)) for values in chosen]
