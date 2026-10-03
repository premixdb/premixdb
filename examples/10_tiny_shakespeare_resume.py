"""Save a reader position and resume at the next sequence."""

from __future__ import annotations

import json

from _tutorial import DEFAULT_STORAGE, TINY, check_inputs, tiny_sources

import premixdb as p

INPUT = TINY
STORAGE = DEFAULT_STORAGE
LIMIT = 100


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    with p.PremixDB(storage=STORAGE) as db:
        snapshot = db.corpus("tutorial/tiny-shakespeare", tiny_sources(INPUT, LIMIT))
        dataset = snapshot.query().dataset(sequence_length=64)
        reader = dataset.reader()
        delivered = next(reader, None)
        if delivered is None:
            print("No sequences: provide nonempty text.")
            return
        print("Processed sequence:", delivered.ordinal)
        # Save this position after the trainer has processed the sequence.
        saved = {"reader": reader.checkpoint()}
        path = STORAGE / "tutorial-10-checkpoint.json"
        path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        expected = next(reader, None)

    with p.PremixDB(storage=STORAGE) as db:
        saved = json.loads(path.read_text(encoding="utf-8"))
        reopened = db.corpus("tutorial/tiny-shakespeare").query().dataset(sequence_length=64)
        resumed = next(reopened.reader(checkpoint=saved["reader"]), None)
        if expected is None:
            assert resumed is None
            print("Resumed reader: no remaining sequences")
        else:
            assert resumed is not None
            assert resumed.ordinal == expected.ordinal == delivered.ordinal + 1
            assert resumed.tokens == expected.tokens
            print("Resumed reader: next sequence", resumed.ordinal, "with identical tokens")
        print("Saved reader state:", path)


if __name__ == "__main__":
    main()
