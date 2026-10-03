# premixdb

premixdb is a declarative language for building reproducible data mixtures to improve llm pretraining.

```bash
uv add premixdb
```

## Capture, filter, mix, train

```python
import premixdb as p
from torch.utils.data import DataLoader

with p.PremixDB(storage=".premixdb") as db:
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
        ]
    )
    print(query.preview())

    mixtures = query.mix(tokens=256, sequence_length=64)
    print(mixtures.weights)
    print(mixtures.profile())

    dataset = mixtures[0]
    print(dataset.preview())
    batch = next(iter(DataLoader(dataset.torch(), batch_size=1)))
```

A mix creates three candidate datasets by default. Repeat a recipe to reuse its
results. Keep `.premixdb` to reopen your corpora and datasets.

## Use the fields you need

```python
with p.PremixDB(storage=".premixdb") as db:
    query = db.corpus("c4").query(
        steps=[
            p.where(p.language.en >= 0.75),
            p.where(p.quality.educational_value >= 1.0),
        ]
    )
    print(query.profile())
    print(query.preview())
```

Fields are computed on demand and cached. Model fields download their assets on
first use. The default dataset tokenizer is bundled GPT-2; use your model's
tokenizer for training.

## Explore

```bash
uvx --python 3.12 premixdb shell
```

The shell opens with `db`, `p`, and an offline Shakespeare `demo` corpus.
From a checkout, use `uvx --from . premixdb shell`.

- [Runnable examples](examples/README.md)
- [Curation](docs/curation.md) · [Fields](docs/enrichment.md) · [Training](docs/training.md)
- [Storage](docs/persistence.md) · [Internals](docs/architecture.md) · [Development](docs/development.md)
