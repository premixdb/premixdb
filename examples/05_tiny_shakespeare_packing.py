"""Pack text into sequences and read a PyTorch batch."""

from __future__ import annotations

from _tutorial import DEFAULT_STORAGE, TINY, check_inputs, tiny_sources
from torch.utils.data import DataLoader

import premixdb as p

INPUT = TINY
STORAGE = DEFAULT_STORAGE
LIMIT = 100


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    with p.PremixDB(storage=STORAGE) as db:
        snapshot = db.Corpus("tutorial/tiny-shakespeare", tiny_sources(INPUT, LIMIT))
        dataset = snapshot.query().mix(
            tokenizer=p.ByteTokenizer(),
            sequence_length=64,
            # Bytes use 0..255. Use 256 for boundaries and 257 for padding.
            packing=p.Concat(separator=256, drop_remainder=False, pad_token=257),
        )[0]
        profile = dataset.profile()
        print(
            "Content / separators / padding:",
            profile.content_tokens,
            profile.separator_tokens,
            profile.padding_tokens,
        )
        if not profile.sequences:
            print("No sequences: provide nonempty text.")
            return
        batch = next(iter(DataLoader(dataset.torch(), batch_size=2)))
        print("Batch shapes:", {name: tuple(tensor.shape) for name, tensor in batch.items()})
        tail = dataset.torch()[-1]
        assert (tail["labels"][tail["attention_mask"] == 0] == -100).all()
        print("Padding labels are -100. Configure the model for this 258-token vocabulary.")


if __name__ == "__main__":
    main()
