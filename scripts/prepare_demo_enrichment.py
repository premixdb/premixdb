"""Regenerate the packaged demo scores using the catalog's pinned models.

Run with `uv run python scripts/prepare_demo_enrichment.py`. Model assets may
download on first use. Generation runs on CPU; normal demo queries need no models.
"""

from __future__ import annotations

import gc
import json
from hashlib import sha256
from pathlib import Path

from premixdb.cli.main import _demo_sources
from premixdb.enrichment import QuRating, WebOrganizer
from premixdb.enrichment.types import Document
from premixdb.runtime.catalog import recipe
from premixdb.runtime.enrichment import producer
from premixdb.schemas.protobuf import wire

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/premixdb/data/demo-enrichment.json"


def main() -> None:
    import torch

    torch.set_num_threads(4)
    sources = _demo_sources()
    documents = [Document(source.key, source.text) for source in sources]
    producers = []
    for name in ("quality.writing_style", "weborganizer.topic"):
        policy = recipe(name)
        worker = producer(policy)
        if not isinstance(worker, (QuRating, WebOrganizer)):
            raise ValueError("demo enrichment requires a classifier")
        print(f"Computing {name} for {len(documents)} demo documents", flush=True)
        values = worker.compute(documents)
        producers.append(
            {
                "producer": wire(policy).hex(),
                "schema_ids": [schema.id.hex() for schema in worker.fields],
                "definition": worker.definition,
                "rows": [
                    {
                        "source_key": doc.id,
                        "text_sha256": sha256(doc.text.encode()).hexdigest(),
                        "values": {key: value for key, value in row.items() if key != "id"},
                    }
                    for doc, row in zip(documents, values, strict=True)
                ],
            }
        )
        if [row["id"] for row in values] != [doc.id for doc in documents]:
            raise ValueError("producer changed document coverage")
        del worker
        gc.collect()
    data = json.dumps({"version": 1, "producers": producers}, indent=2, allow_nan=False) + "\n"
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(data)
    temporary.replace(OUTPUT)
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
