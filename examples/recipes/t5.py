"""C4 adaptation: https://arxiv.org/abs/1910.10683, §2.2 and §3.4.1."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import premixdb as p

from ._common import filtered_sources, tokenizer_language

# Unlabeled corpus only; the final T5 recipe also mixed supervised tasks.
WEIGHTS = {"c4": 1.0}


def sources(crawl: Iterable[p.Source]) -> Iterator[p.Source]:
    """Clean lines and apply the C4 blocked-word list before capture.

    Explicitly use the paper's 3 sentences / 5 words, rather than DataTrove's
    defaults of 5 sentences / 3 words. The blocked-word asset downloads on use.
    """
    from datatrove.pipeline.filters import C4BadWordsFilter, C4QualityFilter

    return filtered_sources(
        crawl,
        (
            C4BadWordsFilter(default_language="en", keep_fraction=0.0, seed=0),
            C4QualityFilter(
                min_num_sentences=3,
                min_words_per_line=5,
                max_word_length=-1,
                language=tokenizer_language(),
            ),
        ),
    )


def query(snapshot: p.Snapshot) -> p.Query:
    """Select cleaned English pages; adapt language ID and dedupe to premixdb.

    Run sources() before capture. fastText >=0.99 replaces the paper's langdetect;
    exact-document dedupe does not implement the original three-sentence spans.
    """
    return snapshot.query(
        steps=[p.where(p.language.en >= 0.99), p.dedupe(order_by=[p.object.uri.asc()])]
    )
