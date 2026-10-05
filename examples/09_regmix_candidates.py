"""Compare mixtures with fixed budgets and seeds."""

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
        candidates = population.mix(
            # Try more proposals to find four that meet the source bounds.
            weights=p.RegMix(seed=123, oversample=1000),
            seed=42,  # Document draws.
            n_candidates=4,
            tokens=1024,
            replacement=True,
            bounds=p.Bounds(lower={domain: 0.1 for domain in names}),
            sequence_length=64,
            packing=p.Concat(separator=256, drop_remainder=False, pad_token=257),
        )
        for index, weights in enumerate(candidates.weights):
            print(f"Candidate {index}:", dict(sorted((names[k], v) for k, v in weights.items())))
            assert candidates[index].train.profile().planned_content_tokens == 1024
        selected = candidates[::2]
        first = selected[0].train[0]  # Pack one sequence from the first choice.
        print("First chosen sequence:", first.ordinal, first.tokens[:16])
        print("Record candidate IDs:", [dataset.id for dataset in candidates])
        print(
            "No winning mixture yet. Train the candidates and compare them on a separate evaluation set."
        )


if __name__ == "__main__":
    main()
