"""Compare C4 page lengths, then apply Gopher's published word-count limits."""

from __future__ import annotations

from _tutorial import C4, DEFAULT_STORAGE, check_inputs

import premixdb as p

INPUT = C4
STORAGE = DEFAULT_STORAGE
LIMIT = 100
MIN_CHARACTERS = 200
MIN_WORDS = 50
MAX_WORDS = 100_000


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    if MIN_CHARACTERS <= 0:
        raise ValueError("MIN_CHARACTERS must be positive")
    if not 0 < MIN_WORDS <= MAX_WORDS:
        raise ValueError("word limits must satisfy 0 < MIN_WORDS <= MAX_WORDS")
    with p.PremixDB(storage=STORAGE) as db:
        snapshot = db.Corpus("tutorial/c4", p.Source.read_jsonl(INPUT, limit=LIMIT))
        # C4 already has T5's line cleanup (§2.2): https://arxiv.org/abs/1910.10683.
        # MIN_CHARACTERS is a tutorial cutoff, not a T5 rule. Gopher's MassiveWeb
        # limits are 50–100,000 words (App. A.1.1): https://arxiv.org/abs/2112.11446.
        # DataTrove counts differ from the original tokenizer. For full heuristic
        # preprocessing plus dedupe, see recipes/gopher.py and recipes/falcon.py.
        query = snapshot.query(
            steps=[
                p.where(p.text.characters >= MIN_CHARACTERS),
                p.where(p.datatrove.n_words >= MIN_WORDS),
                p.where(p.datatrove.n_words <= MAX_WORDS),
                p.where(p.quality.writing_style >= 0.8),
            ]
        )
        print("Captured documents:", snapshot.profile().documents)
        print("Selected documents:", query.profile().output_documents)
        print(query.profile())


if __name__ == "__main__":
    main()
