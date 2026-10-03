"""Remove a copied page and count the surviving documents."""

from __future__ import annotations

from _tutorial import C4, DEFAULT_STORAGE, check_inputs

import premixdb as p

INPUT = C4
STORAGE = DEFAULT_STORAGE
LIMIT = 100


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    sources = list(p.Source.read_jsonl(INPUT, limit=LIMIT))
    if not sources:
        raise ValueError("need at least one C4 row to demonstrate a copied page")
    # Add a copy so this small sample has a duplicate to find.
    sources.append(p.Source("tutorial-copy", sources[0].text))
    with p.PremixDB(storage=STORAGE) as db:
        snapshot = db.corpus("tutorial/c4-with-copy", sources)
        query = snapshot.query(steps=[p.dedupe(order_by=[p.object.uri.asc()])])
        before, after = snapshot.profile().documents, query.profile().output_documents
        assert after < before
        print("Captured → unique documents:", before, "→", after)
        print("Removed copies:", before - after)
        print("Retained sample:", query.preview())


if __name__ == "__main__":
    main()
