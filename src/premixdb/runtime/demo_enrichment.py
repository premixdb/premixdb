"""Frozen model outputs for the packaged demo, with their original provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from typing import Iterable, TypedDict

from premixdb.contracts import JSON, FieldValue, checked_record, load_json
from premixdb.enrichment.types import Document
from premixdb.internal import derivation_pb2 as e
from premixdb.schemas.protobuf import wire
from premixdb.v1 import field_pb2 as f


class SavedRow(TypedDict):
    source_key: str
    text_sha256: str
    values: dict[str, FieldValue]


class SavedProducer(TypedDict):
    producer: str
    schema_ids: list[str]
    definition: dict[str, JSON]
    rows: list[SavedRow]


class DemoFixture(TypedDict):
    version: int
    producers: list[SavedProducer]


@dataclass(frozen=True)
class SavedEnrichment:
    definition_json: bytes
    rows: list[tuple[str, dict[str, FieldValue]]]


def load(
    policy: e.EnrichmentProducer,
    schemas: Iterable[f.Field],
    documents: Iterable[Document],
) -> SavedEnrichment | None:
    """Reuse a frozen demo population or subset; unmatched populations run normally."""
    if not policy.HasField("model") or policy.model.kind not in (
        e.ModelProducer.QUALITY,
        e.ModelProducer.TOPIC,
    ):
        return None
    data = files("premixdb").joinpath("data/demo-enrichment.json").read_bytes()
    fixture = checked_record(load_json(data), DemoFixture)
    if fixture["version"] != 1:
        raise ValueError("unsupported demo enrichment fixture version")
    selected = [item for item in fixture["producers"] if item["producer"] == wire(policy).hex()]
    if not selected:
        return None
    if len(selected) != 1:
        raise ValueError("duplicate demo enrichment producer")
    saved = selected[0]
    schemas = list(schemas)
    if saved["schema_ids"] != [spec.id.hex() for spec in schemas]:
        return None
    lookup = {row["text_sha256"]: row["values"] for row in saved["rows"]}
    if len(lookup) != len(saved["rows"]):
        raise ValueError("duplicate demo enrichment content hash")
    names = {spec.name for spec in schemas}
    if any(set(values) != names for values in lookup.values()):
        raise ValueError("demo enrichment field coverage mismatch")
    rows = []
    for doc in documents:
        values = lookup.get(sha256(doc.text.encode()).hexdigest())
        if values is None:
            return None
        rows.append((doc.id, values))
    if not rows:
        return None
    definition = dict(saved["definition"])
    definition["artifact"] = {
        "kind": "packaged_demo",
        "sha256": sha256(data).hexdigest(),
        "cohort_text_sha256": [row["text_sha256"] for row in saved["rows"]],
    }
    return SavedEnrichment(
        json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
        rows,
    )
