"""Remove held-out text from training."""

from __future__ import annotations

from _tutorial import DEFAULT_STORAGE, TINY, check_inputs, tiny_sources

import premixdb as p

INPUT = TINY
STORAGE = DEFAULT_STORAGE
LIMIT = 100


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    sources = tiny_sources(INPUT, LIMIT)
    if len(sources) < 2:
        raise ValueError("need at least two dialogue blocks to create a held-out example")
    with p.PremixDB(storage=STORAGE) as db:
        # Intentionally leave the final held-out block in the training input.
        train = db.Corpus("tutorial/tiny-leaky-train", sources)
        held_out = db.Corpus("tutorial/tiny-held-out", [sources[-1]])
        clean = train.query(decontaminate=p.decontaminate(held_out, algorithm="document"))
        print("Leaky training documents:", train.profile().documents)
        print("After excluding held-out text:", clean.profile().output_documents)
        assert clean.profile().output_documents < train.profile().documents
        print("Held-out snapshot stays separate:", held_out.id)


if __name__ == "__main__":
    main()
