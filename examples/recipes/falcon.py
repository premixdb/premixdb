"""English RefinedWeb adaptation: https://arxiv.org/abs/2306.01116, §3.

Uses Gopher heuristics, fastText English >=0.65, and premixdb document dedupe.
Apply URL filtering, extraction, and line corrections upstream; exact repeated
substring removal is not supplied. See README.md before treating this as Falcon.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import premixdb as p

from ._common import filtered_sources, gopher_filters

# Falcon series report §5.1, Table 15: https://arxiv.org/abs/2311.16867.
# These are production-model fractions, not the web-only RefinedWeb experiment.
WEIGHTS = {
    "web_en": 0.76,
    "web_europe": 0.08,
    "books": 0.06,
    "conversations": 0.05,
    "code": 0.03,
    "technical": 0.02,
}


def sources(crawl: Iterable[p.Source]) -> Iterator[p.Source]:
    """Apply published MassiveWeb quality/repetition rules to extracted English text."""
    return filtered_sources(crawl, gopher_filters())


def query(snapshot: p.Snapshot, *, similarity: float = 0.8) -> p.Query:
    """Plan the English selection and adapted document-level deduplication stages.

    Run sources() before capture. similarity=0.8 is this example's choice;
    premixdb's MinHash index differs from Falcon's 9,000-hash configuration.
    """
    return snapshot.query(
        steps=[
            p.where(p.language.en >= 0.65),
            p.indexed_dedupe(p.DedupeIndex.MINHASH_LSH, threshold=similarity),
            p.dedupe(order_by=[p.object.uri.asc()]),
        ]
    )
