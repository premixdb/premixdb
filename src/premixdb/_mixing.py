"""Serializable mixture policies. These objects never read or process corpus data."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence

from ._protobuf import parse
from ._typing import FieldValue
from .v1 import dataset_pb2 as datasets


def _domain_key(labels: Sequence[FieldValue]) -> str:
    """Shared human-readable keys for string domains and typed tuple domains."""
    return (
        labels[0]
        if len(labels) == 1 and isinstance(labels[0], str)
        else json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    )


def _weights_message(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("mixture weights cannot be empty")
    values = {}
    for key, value in sorted(weights.items()):
        if not isinstance(key, str) or not key:
            raise ValueError("stratum keys must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("weights must be numbers")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("weights must be finite values between zero and one")
        values[key] = float(value)
    if not math.isclose(math.fsum(values.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("mixture weights must sum to one")
    return values


@dataclass(frozen=True)
class Tokens:
    """Exact content-token budget; separators and padding are accounted separately."""

    count: int
    tokenizer: datasets.Tokenizer | None = None


class Bounds:
    """Per-domain weight limits and optional whole-pass exposure limits.

    reference_tokens checks capacity for a later larger run as well as this one.
    Maps may omit domains; lower defaults to zero and upper to one.
    """

    def __init__(
        self,
        *,
        lower: Mapping[str, float] | None = None,
        upper: Mapping[str, float] | None = None,
        max_epochs: int | None = None,
        reference_tokens: int | None = None,
    ) -> None:
        from ._requests import _uint

        value = datasets.MixBounds()
        for name, mapping in (("lower", lower or {}), ("upper", upper or {})):
            for key, weight in mapping.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                    or not 0 <= weight <= 1
                ):
                    raise ValueError("bounds require nonempty labels and finite weights in [0, 1]")
                getattr(value, name)[key] = weight
        if max_epochs is not None:
            value.max_epochs = _uint(max_epochs, 32, "max_epochs", positive=True)
        if reference_tokens is not None:
            value.reference_tokens = _uint(reference_tokens, 64, "reference_tokens", positive=True)
        self._bytes = value.SerializeToString(deterministic=True)

    def _to_proto(self) -> datasets.MixBounds:
        return parse(datasets.MixBounds, self._bytes)


@dataclass(frozen=True)
class RegMixSampler:
    """Versioned RegMix-style proposals only; no predictor or training loop.

    Smooth the token prior, explore log-spaced Dirichlet concentrations, zero
    tiny weights, quantize, renormalize, validate bounds, deduplicate, and choose
    from a bounded oversampled pool. This is not bit-identical to upstream NumPy.
    """

    seed: int = 0
    prior_power: float = 0.5
    concentration_range: tuple[float, float] = (0.1, 5.0)
    concentration_steps: int = 15
    minimum_weight: float = 2e-4
    oversample: int = 100

    def _to_proto(self) -> datasets.MixAlgorithm:
        from ._requests import _uint

        low, high = self.concentration_range
        if (
            any(
                isinstance(v, bool) or not math.isfinite(v) or v <= 0
                for v in (self.prior_power, low, high)
            )
            or low > high
        ):
            raise ValueError("prior power and concentration range must be positive and finite")
        if high > sys.float_info.max / 4:
            raise ValueError("concentration is too large for finite gamma sampling arithmetic")
        if not math.isfinite(self.minimum_weight) or not 0 <= self.minimum_weight < 1:
            raise ValueError("minimum_weight must be finite and in [0, 1)")
        return datasets.MixAlgorithm(
            regmix=datasets.RegMix(
                seed=_uint(self.seed, 64, "sampler seed"),
                prior_power=self.prior_power,
                min_concentration=low,
                max_concentration=high,
                concentration_steps=_uint(
                    self.concentration_steps, 32, "concentration_steps", positive=True
                ),
                minimum_weight=self.minimum_weight,
                oversample=_uint(self.oversample, 32, "oversample", positive=True),
            )
        )
