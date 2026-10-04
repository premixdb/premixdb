"""Inspect word counts before choosing a cutoff."""

from __future__ import annotations

from _tutorial import DEFAULT_STORAGE, PAPERS, check_inputs

import premixdb as p
from premixdb.v1.query_pb2 import FIELD_DATATROVE_N_WORDS

INPUT = PAPERS
STORAGE = DEFAULT_STORAGE
LIMIT = 100


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    with p.PremixDB(storage=STORAGE) as db:
        snapshot = db.Corpus(
            "tutorial/pes2o-validation",
            p.Source.read_jsonl(INPUT, key_column="id", limit=LIMIT),
        )
        query = snapshot.query(steps=[p.where(p.datatrove.n_words >= 0)])
        profile = query.profile()
        words = next(field for field in profile.fields if field.field == FIELD_DATATROVE_N_WORDS)
        print("Papers inspected:", profile.output_documents)
        distribution = words.distributions[0]
        if distribution.HasField("numeric"):
            print("Mean words per paper:", distribution.numeric.mean)
            print(
                "Word-count range:",
                distribution.numeric.minimum.integer,
                "→",
                distribution.numeric.maximum.integer,
            )
        else:
            print("No word counts available.")


if __name__ == "__main__":
    main()
