"""GPT-3 mixture and scored CC resampling: https://arxiv.org/abs/2005.14165."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Iterator

import premixdb as p

# Table 2.2 reports rounded percentages summing to 101%. Normalize explicitly
# because premixdb requires token fractions summing to one.
REPORTED_WEIGHTS = {
    "common_crawl": 0.60,
    "webtext": 0.22,
    "books1": 0.08,
    "books2": 0.08,
    "wiki": 0.03,
}
WEIGHTS = {
    name: value / math.fsum(REPORTED_WEIGHTS.values()) for name, value in REPORTED_WEIGHTS.items()
}


def sources(
    crawl: Iterable[p.Source], *, score: Callable[[p.Source], float], seed: int = 0
) -> Iterator[p.Source]:
    """Appendix A's Pareto(alpha=9) selection, with caller-supplied classifier scores.

    Train a logistic classifier on curated positives and raw crawl negatives.
    The original classifier assets are unavailable. Deterministic per-key draws
    replace NumPy's order-dependent random draws for reproducible custom crawls.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for source in crawl:
        value = score(source)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"classifier score for {source.key!r} must be finite in [0, 1]")
        digest = hashlib.sha256(f"{seed}\0{source.key}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        # P(Pareto(9) > 1-score), for NumPy's zero-based Pareto distribution.
        if draw < (2.0 - value) ** -9:
            yield source


def query(snapshot: p.Snapshot, *, similarity: float = 0.8) -> p.Query:
    """Dedupe classifier-resampled CC with premixdb's adapted MinHash index.

    similarity=0.8 is an example choice; GPT-3 used Spark MinHashLSH with ten
    hashes. Cross-source WebText removal and benchmark span removal are upstream.
    """
    return snapshot.query(
        steps=[
            p.where(p.text.characters > 0),
            p.indexed_dedupe(p.DedupeIndex.MINHASH_LSH, threshold=similarity),
            p.dedupe(order_by=[p.object.uri.asc()]),
        ]
    )
