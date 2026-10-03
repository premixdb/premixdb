"""Interpret stored classifier logits using the field's actual class schema."""

from __future__ import annotations

import math
from collections.abc import Sequence

from premixdb.v1 import field_pb2 as fields


def probabilities(field: fields.Field, logits: Sequence[float]) -> dict[str, float]:
    """Convert one class vector; never apply this to regression/embedding fields."""
    if not field.HasField("classification"):
        raise ValueError("field has no classification metadata")
    schema = field.classification
    classes = tuple(schema.classes)
    if (
        not classes
        or len(set(classes)) != len(classes)
        or any(not c for c in classes)
        or len(classes) != field.length
        or len(logits) != field.length
        or field.element_type not in (fields.VALUE_FLOAT32, fields.VALUE_FLOAT64)
    ):
        raise ValueError("classification requires distinct classes aligned with numeric logits")
    temperature = schema.temperature if schema.HasField("temperature") else 1.0
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("classification logits must be finite")
    if schema.transform == fields.PROBABILITY_TRANSFORM_SOFTMAX:
        maximum = max(values)
        exps = [math.exp((value - maximum) / temperature) for value in values]
        total = sum(exps)
        result = [value / total for value in exps]
    elif schema.transform == fields.PROBABILITY_TRANSFORM_SIGMOID:
        result = []
        for value in values:
            value /= temperature
            e = math.exp(-abs(value))
            result.append(1 / (1 + e) if value >= 0 else e / (1 + e))
    else:
        raise ValueError("classification requires an explicit probability transform")
    return dict(zip(classes, result))


def top_class(field: fields.Field, logits: Sequence[float]) -> tuple[str, float]:
    """Return (label, probability) for a mutually exclusive classifier."""
    if field.classification.transform != fields.PROBABILITY_TRANSFORM_SOFTMAX:
        raise ValueError("top_class requires a softmax classifier")
    scores = probabilities(field, logits)
    label = max(scores, key=scores.__getitem__)
    return label, scores[label]
