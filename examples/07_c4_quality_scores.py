"""Cache educational-value scores and try a different cutoff."""

from __future__ import annotations

import math

from _tutorial import C4, DEFAULT_STORAGE, check_inputs

import premixdb as p

INPUT = C4
STORAGE = DEFAULT_STORAGE
LIMIT = 8
MINIMUM = 1.0


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    if not math.isfinite(MINIMUM):
        raise ValueError("MINIMUM must be finite")
    with p.PremixDB(storage=STORAGE) as db:
        snapshot = db.Corpus("tutorial/c4-quality", p.Source.read_jsonl(INPUT, limit=LIMIT))
        selected = snapshot.query(steps=[p.where(p.quality.educational_value >= MINIMUM)])
        print(
            "Captured → selected documents:",
            snapshot.profile().documents,
            "→",
            selected.profile().output_documents,
        )
        print(selected.profile())
        print(selected.preview())


if __name__ == "__main__":
    main()
