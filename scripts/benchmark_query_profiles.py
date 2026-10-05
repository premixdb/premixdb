"""Measure fresh public query.profile calls, including durable result publication.

Run with: uv run python scripts/benchmark_query_profiles.py --documents 100000
Synthetic enrichment replaces inference only; capture, preparation, execution,
profiling, bounded preview and publication use the production implementations.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

import premixdb as p
from premixdb.contracts import FieldValue
from premixdb.enrichment.types import ComputedRow, Document, field
from premixdb.fields.ids import field_id
from premixdb.internal import derivation_pb2 as d
from premixdb.v1 import field_pb2 as f

QUALITY = ["educational_value", "writing_style", "facts_and_trivia", "required_expertise"]
PROJECTIONS = [*[getattr(p.quality, name) for name in QUALITY], p.language.en, p.language.fr]


class SyntheticFields:
    cache_scope = "batch"
    definition = {"provider": "benchmark", "version": 1}

    def __init__(
        self, policy: d.EnrichmentProducer, values: dict[str, NDArray[np.float64]]
    ) -> None:
        names = (
            ["language.en", "language.fr"]
            if policy.HasField("language")
            else [f"quality.{name}" for name in QUALITY]
        )
        self.fields = tuple(field(name, element_type=f.VALUE_FLOAT64) for name in names)
        self.values = values

    def compute(self, documents: Sequence[Document]) -> list[ComputedRow]:
        rows: list[ComputedRow] = []
        for document in documents:
            index = int(document.text.split(":", 1)[0], 16)
            row: dict[str, FieldValue] = {"id": document.id}
            row.update(
                {
                    spec.name: None if index % 97 == 0 else float(self.values[spec.name][index])
                    for spec in self.fields
                }
            )
            rows.append(row)
        return rows


def measure(
    documents: int, trials: int, root: Path, cache_bytes: int, reference: bool
) -> dict[str, object]:
    rng = np.random.default_rng(8127)
    values = {
        name: rng.random(documents)
        for name in [*[f"quality.{name}" for name in QUALITY], "language.en", "language.fr"]
    }
    sources = [
        p.Source(f"https://benchmark.test/{i:08x}", f"{i:08x}:" + "é" * (i % 64))
        for i in range(documents)
    ]
    report: dict[str, object] = {
        "documents": documents,
        "numeric_fields": 6,
        "intrinsic_fields": 4,
        "trials": trials,
        "cache_bytes": cache_bytes,
        "inference": "synthetic",
    }
    with p.PremixDB(storage=root, cache_bytes=cache_bytes) as client:
        started = time.perf_counter()
        snapshot = client.Corpus("profile-benchmark", sources)
        report["capture_and_intrinsic_preparation_seconds"] = time.perf_counter() - started
        del sources
        print(
            json.dumps(
                {"phase": "capture", "seconds": report["capture_and_intrinsic_preparation_seconds"]}
            ),
            flush=True,
        )
        started = time.perf_counter()
        with patch(
            "premixdb.runtime.enrichment.producer",
            side_effect=lambda policy: SyntheticFields(policy, values),
        ):
            snapshot.query()._with_fields(PROJECTIONS).wait()
        report["enrichment_and_field_preparation_seconds"] = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "phase": "enrichment",
                    "seconds": report["enrichment_and_field_preparation_seconds"],
                }
            ),
            flush=True,
        )
        snapshot_id = snapshot.id

        def run(a: float, b: float, *, warm_client: p.PremixDB = client) -> tuple[float, int]:
            started = time.perf_counter()
            selected = (
                warm_client._snapshot(snapshot_id)
                .query(steps=[p.where(p.quality.educational_value > a), p.where(p.language.en > b)])
                ._with_fields(PROJECTIONS)
            )
            profile = selected.profile()
            elapsed = (time.perf_counter() - started) * 1000
            indices = np.arange(documents)
            mask = (
                (indices % 97 != 0)
                & (values["quality.educational_value"] > a)
                & (values["language.en"] > b)
            )
            assert profile.output_documents == np.count_nonzero(mask)
            for name in QUALITY:
                spec = next(
                    field for field in profile.fields if field.field == field_id(f"quality.{name}")
                )
                numeric = spec.distributions[0].numeric
                assert np.isclose(
                    numeric.mean, np.mean(values[f"quality.{name}"][mask]), rtol=1e-12, atol=1e-12
                )
                assert np.isclose(
                    numeric.standard_deviation,
                    np.std(values[f"quality.{name}"][mask]),
                    rtol=1e-12,
                    atol=1e-12,
                )
            return elapsed, profile.output_documents

        results = {}
        for label, a, b in [
            ("narrow", 0.95, 0.95),
            ("medium", 0.7, 0.9),
            ("half", 0.5, 0.0),
            ("broad", 0.1, 0.1),
        ]:
            times, count = [], 0
            for trial in range(trials + 1):
                elapsed, count = run(a + (trial + 1) * 1e-6, b + (trial + 1) * 1e-6)
                if trial:
                    times.append(elapsed)
            results[label] = {
                "selected_documents": count,
                "p50_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "samples_ms": times,
            }
            print(json.dumps({"phase": label, **results[label]}), flush=True)
        report["fresh_queries"] = results
        if reference:
            with patch("premixdb.runtime.analytics.execute", return_value=None):
                elapsed, count = run(0.501234, 0.0001234)
            report["existing_executor_half_ms"] = elapsed
            report["existing_executor_selected_documents"] = count
            print(json.dumps({"phase": "existing-executor", "ms": elapsed}), flush=True)
    with p.PremixDB(storage=root, cache_bytes=0) as cold:
        elapsed, count = run(0.501235, 0.0001235, warm_client=cold)
        report["reopened_no_application_cache_ms"] = elapsed
        report["reopened_selected_documents"] = count
    report["storage_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--cache-mib", type=int, default=512)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/query-profiles.json"))
    args = parser.parse_args()
    with TemporaryDirectory(prefix="premixdb-query-benchmark-") as temporary:
        report = measure(
            args.documents,
            args.trials,
            Path(temporary),
            args.cache_mib * 1024 * 1024,
            args.reference,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
