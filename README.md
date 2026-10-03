# premixdb

premixdb is a declarative library for building reproducible data mixtures to improve llm pretraining.

## Quick start

Run a premixdb shell without installing the library:

```bash
uvx --python 3.12 premixdb shell
```

The shell has already run these commands so you have `p` and `db` available:

```bash
import premixdb as p
db = PremixDB()
```

You can also use the tiny_shakespeare demo corpus of his plays:

```bash
from torch.utils.data import DataLoader

dataset = db.corpus('demo').query().mix()[0].torch()
batch = next(iter(DataLoader(dataset, batch_size=1)))
```

```
{'input_ids': tensor([[   35,    52,  7336,  ...,   628, 50256,  5097]]),
 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]]),
 'labels': tensor([[   35,    52,  7336,  ...,   628, 50256,  5097]])}
```

## Full example

Capture a corpus snapshot. Filter the snapshot with a query. Mix query results into datasets. Train with Pytorch.

```python
from torch.utils.data import DataLoader

c4 = db.corpus(
    "c4",
    p.HuggingFaceSource("datablations/c4-filter-small"),
    limit=8,
)
oscar = db.corpus(
    "oscar",
    p.HuggingFaceSource("datablations/oscar-filter-small"),
    limit=8,
)
query = c4.union(oscar).query(
    steps=[
        p.where(p.text.characters >= 200),
        p.dedupe(),
        p.where(p.language.en >= 0.75),
        p.where(p.quality.educational_value >= 1.0),
    ]
)

mixtures = query.mix(tokens=256, sequence_length=64)

dataset = mixtures[0]
print(dataset.preview())
batch = next(iter(DataLoader(dataset.torch(), batch_size=1)))
```

## Installation

To add as a dependency to your project use:

```bash
uv add premixdb
```

- [Runnable examples](examples/README.md)
- [Curation](docs/curation.md) · [Fields](docs/enrichment.md) · [Training](docs/training.md)
- [Storage](docs/persistence.md) · [Internals](docs/architecture.md) · [Development](docs/development.md)
