"""LLaMA 1 public-source recipe: https://arxiv.org/abs/2302.13971, §2.1/Table 1."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import premixdb as p

WEIGHTS = {
    "common_crawl": 0.67,
    "c4": 0.15,
    "code": 0.045,
    "wiki": 0.045,
    "books": 0.045,
    "arxiv": 0.025,
    "stack_exchange": 0.02,
}


def sources(
    ccnet_crawl: Iterable[p.Source], *, is_wikipedia_reference: Callable[[p.Source], bool]
) -> Iterator[p.Source]:
    """Keep pages classified as Wikipedia references after upstream CCNet processing.

    Supply your own trained reference-page classifier; its original weights are
    not provided by the paper. CCNet line dedupe and n-gram quality filtering
    must already have run. Use t5.sources() separately for the C4 component.
    """
    for source in ccnet_crawl:
        keep = is_wikipedia_reference(source)
        if type(keep) is not bool:
            raise TypeError("the reference-page classifier must return a bool")
        if keep:
            yield source


def query(snapshot: p.Snapshot) -> p.Query:
    """Apply adapted English/exact-document selection to the prepared CC component."""
    return snapshot.query(
        steps=[p.where(p.language.label == "en"), p.dedupe(order_by=[p.object.uri.asc()])]
    )
