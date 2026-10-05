"""Save the next consumed sequence index and resume map-style PyTorch reads."""

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
        snapshot = db.Corpus("tutorial/tiny-shakespeare", tiny_sources(INPUT, LIMIT))
        dataset = snapshot.query().mix(sequence_length=64)[0]
        data = dataset.train.torch()
        if not len(data):
            print("No sequences: provide nonempty text.")
            return
        delivered = data[0]
        print("Processed sequence:", 0, delivered["input_ids"][:8].tolist())
        # Save only after the trainer has consumed this sequence. Save model,
        # optimizer and RNG state alongside it in an actual training checkpoint.
        saved = {"dataset": dataset.id, "split": "train", "next_index": 1}
        path = STORAGE / "tutorial-10-checkpoint.json"
        path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        expected = data[1] if len(data) > 1 else None

    with p.PremixDB(storage=STORAGE) as db:
        saved = json.loads(path.read_text(encoding="utf-8"))
        reopened = db.Corpus("tutorial/tiny-shakespeare").query().mix(sequence_length=64)[0]
        assert reopened.id == saved["dataset"]
        assert saved["split"] == "train"
        data = reopened.train.torch()
        ordinal = saved["next_index"]
        resumed = data[ordinal] if ordinal < len(data) else None
        if expected is None:
            assert resumed is None
            print("Resumed training: no remaining sequences")
        else:
            assert resumed is not None
            assert resumed["input_ids"].tolist() == expected["input_ids"].tolist()
            print("Resumed training: next sequence", ordinal, "with identical tokens")
        print("Saved training position:", path)


if __name__ == "__main__":
    main()
