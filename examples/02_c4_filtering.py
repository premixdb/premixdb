"""Filter web pages by length and inspect what remains."""

from __future__ import annotations

from _tutorial import C4, DEFAULT_STORAGE, check_inputs

import premixdb as p

INPUT = C4
STORAGE = DEFAULT_STORAGE
LIMIT = 100
MIN_CHARACTERS = 200


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    if MIN_CHARACTERS <= 0:
        raise ValueError("MIN_CHARACTERS must be positive")
    with p.PremixDB(storage=STORAGE) as db:
        snapshot = db.corpus("tutorial/c4", p.Source.read_jsonl(INPUT, limit=LIMIT))
        query = snapshot.query(steps=[p.where(p.text.characters >= MIN_CHARACTERS)])
        print("Captured documents:", snapshot.profile().documents)
        print("Selected documents:", query.profile().output_documents)
        print(query.profile())


if __name__ == "__main__":
    main()
