"""Typed split policies and versioned captured-content membership."""

from __future__ import annotations

import math

from blake3 import blake3

from premixdb.schemas.messages import reject_unknown
from premixdb.schemas.protobuf import copy_message
from premixdb.v1.data_mixture_pb2 import Splits

SPLIT_NAMES = ("train", "validation", "test")


def split_policy(value: Splits | None = None) -> Splits:
    """Resolve defaults without changing a caller's mutable protobuf message."""
    if value is None:
        return Splits(train=0.8, validation=0.1, test=0.1, seed=0)
    if not isinstance(value, Splits):
        raise TypeError("splits must be a Splits protobuf message")
    reject_unknown(value)
    if not all(value.HasField(name) for name in SPLIT_NAMES):
        raise ValueError("splits requires train, validation, and test proportions")
    proportions = [getattr(value, name) for name in SPLIT_NAMES]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in proportions):
        raise ValueError("split proportions must be finite and between zero and one")
    if not math.isclose(math.fsum(proportions), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("split proportions must sum to one")
    return copy_message(value)


def content_split(content: bytes, policy: Splits) -> str:
    """Identical captured content stays together across sources and candidates."""
    value = int.from_bytes(
        blake3(b"premixdb-content-split/v1\0" + policy.seed.to_bytes(8, "big") + content).digest()[
            :8
        ],
        "big",
    )
    if value < int(policy.train * 2**64):
        return "train"
    if value < int((policy.train + policy.validation) * 2**64):
        return "validation"
    return "test"
