"""Capture pinned object manifests, local files and Hub dataset rows."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from ..v1.storage_pb2 import Source
from .assets import read


def capture(
    source: Source, local_root: str | Path | None = None
) -> Generator[tuple[str, str], None, None]:
    if source.HasField("manifest"):
        for key, reference in sorted(source.manifest.objects.items()):
            yield key, read(reference, local_root=local_root).decode("utf-8")
    elif source.HasField("hugging_face"):
        from datasets import load_dataset

        policy = source.hugging_face
        dataset = load_dataset(
            policy.repository,
            policy.configuration or None,
            split=policy.split,
            revision=policy.revision,
            streaming=True,
        )
        seen = set()
        for ordinal, row in enumerate(dataset):
            text = row.get(policy.text_column or "text")
            if not isinstance(text, str):
                raise ValueError("dataset text column must contain strings")
            key = str(row[policy.key_column]) if policy.key_column else str(ordinal)
            if key in seen:
                raise ValueError("duplicate dataset row key")
            seen.add(key)
            yield f"hf://{policy.repository}/{policy.configuration}/{policy.split}/{key}", text
    else:
        raise ValueError("unsupported source")
