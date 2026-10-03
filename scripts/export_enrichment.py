"""Low-level worker export: DataTrove, Dupekit and optional model outputs.

For normal querying, see examples/07_c4_quality_scores.py. This exporter inspects
raw outputs and producer policies.

Input: JSONL or gzip JSONL with text and optional unique string id and url.
C4 rows without id use stable shard/row occurrence keys. Run --help
for checkpoint pin options. A completed output has a manifest.json; failed runs
leave partial files without a manifest. Use a fresh output directory per run.
"""

from __future__ import annotations

import argparse
import gzip
import json
from itertools import islice
from pathlib import Path
from typing import Iterator

from blake3 import blake3
from google.protobuf.json_format import MessageToDict

from premixdb.enrichment import (
    DataTroveFields,
    Document,
    DupekitIndex,
    ModelPin,
    QuRating,
    WebOrganizer,
)
from premixdb.enrichment.types import ComputedRow


def documents(path: Path, limit: int | None = None) -> Iterator[Document]:
    seen = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for ordinal, line in enumerate(islice(source, limit)):
            row = json.loads(line)
            id = row.get("id", f"{path.name.removesuffix('.gz')}/{ordinal:08d}")
            if not isinstance(id, str) or not id or id in seen:
                raise ValueError(f"row {ordinal + 1}: IDs must be unique nonempty strings")
            seen.add(id)
            yield Document(id, row["text"], row.get("url"))


def digest(path: Path) -> str:
    result = blake3()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=1000, help="documents to read; 0 reads all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quality-revision", help="full commit SHA of princeton-nlp/QuRater-1.3B")
    parser.add_argument("--topic-revision", help="full commit SHA of WebOrganizer/TopicClassifier")
    parser.add_argument(
        "--content-type-revision", help="full commit SHA of WebOrganizer/FormatClassifier"
    )
    parser.add_argument("--no-url", action="store_true", help="use/pin -NoURL WebOrganizer models")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    providers = [DataTroveFields()]
    if args.quality_revision:
        providers.append(
            QuRating(
                ModelPin("princeton-nlp/QuRater-1.3B", args.quality_revision), device=args.device
            )
        )
    for task, stem, revision in (
        ("topic", "Topic", args.topic_revision),
        ("content_type", "Format", args.content_type_revision),
    ):
        if revision:
            repository = f"WebOrganizer/{stem}Classifier" + ("-NoURL" if args.no_url else "")
            providers.append(
                WebOrganizer(ModelPin(repository, revision), task=task, device=args.device)
            )
    dedupe = DupekitIndex()
    args.output.mkdir(parents=True, exist_ok=False)
    field_path, index_path = args.output / "fields.jsonl", args.output / "dedupe.parquet"
    source = iter(documents(args.input, args.limit or None))
    count = 0
    # This example expects unique source IDs. For distributed readers, namespace
    # occurrence IDs before dispatch and validate uniqueness at ingestion.
    with (
        field_path.open("w") as out,
        pq.ParquetWriter(index_path, dedupe.compute([]).schema) as index_out,
    ):
        while batch := list(islice(source, args.batch_size)):
            rows: list[ComputedRow] = [
                {"id": doc.id, "url": doc.url, "text_blake3": blake3(doc.text.encode()).hexdigest()}
                for doc in batch
            ]
            for provider in providers:
                for row, values in zip(rows, provider.compute(batch), strict=True):
                    if values["id"] != row["id"]:
                        raise ValueError("producer changed document alignment")
                    row.update(values)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            index_out.write_batch(dedupe.compute(batch))
            count += len(batch)
    manifest = {
        "version": 1,
        "documents": count,
        "fields": [MessageToDict(f) for p in providers for f in p.fields],
        "indexes": [MessageToDict(i) for i in dedupe.indexes],
        "producers": [p.definition for p in providers],
        "dedupe": dedupe.definition,
        "files": {
            p.name: {"blake3": digest(p), "bytes": p.stat().st_size}
            for p in (field_path, index_path)
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {count} documents to {args.output}")


if __name__ == "__main__":
    main()
