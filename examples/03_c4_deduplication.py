"""Remove a copied C4 page; distinguish document dedupe from T5's span dedupe."""

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
        snapshot = db.Corpus("tutorial/c4-with-copy", sources)
        # T5/C4 (§2.2) deduped three-sentence spans, not just whole documents:
        # https://arxiv.org/abs/1910.10683. This bounded lesson removes exact pages.
        # Falcon used MinHash followed by exact-substring removal (§3.3):
        # https://arxiv.org/abs/2306.01116; see recipes/falcon.py for an adaptation.
        query = snapshot.query(steps=[p.dedupe(order_by=[p.object.uri.asc()])])
        before, after = snapshot.profile().documents, query.profile().output_documents
        assert after < before
        print("Captured → unique documents:", before, "→", after)
        print("Removed copies:", before - after)
        print("Retained sample:", query.preview())


if __name__ == "__main__":
    main()
