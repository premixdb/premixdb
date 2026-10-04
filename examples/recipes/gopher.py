"""MassiveWeb adaptation: https://arxiv.org/abs/2112.11446, Appendix A.1.1."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import premixdb as p

from ._common import filtered_sources, gopher_filters

# Gopher Table 2. Appendix A.3.1 selects a mixture using held-out losses.
WEIGHTS = {"web": 0.48, "books": 0.27, "c4": 0.10, "news": 0.10, "code": 0.03, "wiki": 0.02}


def sources(crawl: Iterable[p.Source]) -> Iterator[p.Source]:
    """Apply the MassiveWeb heuristics, including all Table A1 repetition cutoffs."""
    return filtered_sources(crawl, gopher_filters())


def query(snapshot: p.Snapshot, *, evaluation: p.Snapshot | None = None) -> p.Query:
    """Select English and dedupe; optionally remove evaluation-document overlap.

    Run sources() before capture. Language labels, the MinHash configuration,
    and exact-document decontamination are adaptations, not the original tools.
    The web heuristics should not be applied indiscriminately to books or code.
    """
    steps = [
        p.where(p.language.label == "en"),
        p.dedupe(order_by=[p.object.uri.asc()]),
        p.indexed_dedupe(p.DedupeIndex.MINHASH_LSH, threshold=0.8),
    ]
    if evaluation is None:
        return snapshot.query(steps=steps)
    return snapshot.query(
        steps=steps, decontaminate=p.decontaminate(evaluation, algorithm="document")
    )
