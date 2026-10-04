"""Split a token budget across web pages, papers, and literature."""

from __future__ import annotations

from _tutorial import C4, DEFAULT_STORAGE, TINY, TRAIN_PAPERS, check_inputs, mixture_sources

import premixdb as p

WEB = C4
PAPERS = TRAIN_PAPERS
LITERATURE = TINY
STORAGE = DEFAULT_STORAGE
LIMIT = 100


def main() -> None:
    check_inputs(WEB, PAPERS, LITERATURE, limit=LIMIT)
    with p.PremixDB(storage=STORAGE) as db:
        web, science, literature = mixture_sources(
            db, web=WEB, papers=PAPERS, literature=LITERATURE, limit=LIMIT
        )
        names = {
            web.corpus_id: "web/C4",
            science.corpus_id: "science/peS2o-train",
            literature.corpus_id: "literature/Tiny-Shakespeare",
        }
        population = web.union(science, literature).query(steps=[p.where(p.text.characters > 0)])
        mixture = population.mix(
            weights=p.RegMix(),
            tokens=1024,
            replacement=True,
            # Give every source at least 10% for this example.
            bounds=p.Bounds(lower={domain: 0.1 for domain in names}),
            sequence_length=64,
            packing=p.Concat(separator=256, drop_remainder=False, pad_token=257),
        )
        print(
            "Proposed token fractions:",
            dict(sorted((names[k], v) for k, v in mixture.weights[0].items())),
        )
        profile = mixture[0].profile()
        print("Planned content tokens:", profile.planned_content_tokens)
        print(
            "Tokens per source:",
            dict(sorted((names[k], v) for k, v in profile.source_tokens.items())),
        )
        print(
            "Sequences / separators / padding:",
            profile.sequences,
            profile.separator_tokens,
            profile.padding_tokens,
        )


if __name__ == "__main__":
    main()
