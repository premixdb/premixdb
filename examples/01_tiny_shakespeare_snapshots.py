"""Capture the same text twice and reopen its snapshot."""

from __future__ import annotations

from _tutorial import DEFAULT_STORAGE, TINY, check_inputs, tiny_sources

import premixdb as p

INPUT = TINY
STORAGE = DEFAULT_STORAGE
LIMIT = 100


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    with p.PremixDB(storage=STORAGE) as db:
        first = db.Corpus("tutorial/tiny-shakespeare", tiny_sources(INPUT, LIMIT))
        again = db.Corpus("tutorial/tiny-shakespeare", tiny_sources(INPUT, LIMIT))
        assert first.id == again.id
        assert db.Corpus("tutorial/tiny-shakespeare").id == first.id
        print("Snapshot unchanged:", first.id == again.id)
        print("Save this snapshot ID with your experiment:", first.id)
        print("Dialogue blocks:", first.profile().documents)
        print("One captured speech:", first.preview(limit=1)[0]["text"])


if __name__ == "__main__":
    main()
