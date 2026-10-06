# premixdb

premixdb is a declarative library for building reproducible data mixtures to improve llm pretraining.

## Quick start

Run a premixdb shell without installing the library:

```bash
uvx --python 3.12 premixdb shell
```

The shell has already run these commands so you have `p` and `db` available:

```python
import premixdb as p
db = p.PremixDB()
```

You can also use the tiny_shakespeare demo corpus of his plays:

```python
from torch.utils.data import DataLoader

dataset = db.Corpus('demo').query().mix()[0].train.torch()
next(iter(DataLoader(dataset, batch_size=1)))
```

```
{'input_ids': tensor([[45472, 10426,  1677,  ...,   475,   286,   477]]),
 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]]),
 'labels': tensor([[45472, 10426,  1677,  ...,   475,   286,   477]])}
```

## Full example

Run from the repository root using the small checked-in C4 and S2ORC-derived
samples. Capture snapshots, filter them, and mix the results for PyTorch training.
Language and quality models download on first use.

```python
from pathlib import Path
from torch.utils.data import DataLoader

c4 = db.Corpus(
    "c4",
    p.Source.read_jsonl(Path("examples/data/c4.jsonl"), limit=8),
)
papers = db.Corpus(
    "papers",
    p.Source.read_jsonl(Path("examples/data/s2orc-train.jsonl"), key_column="id"),
)
query = c4.union(papers).query(
    steps=[
        p.where(p.text.characters >= 200),
        p.dedupe(),
        p.where(p.language.en >= 0.75),
        p.where(p.quality.educational_value >= 1.0),
    ]
)

mixtures = query.mix(tokens=256, replacement=True, sequence_length=64)

dataset = mixtures[0]
print(dataset.preview())
batch = next(iter(DataLoader(dataset.train.torch(), batch_size=1)))
```

## Installation

To add as a dependency to your project use:

```bash
uv add premixdb
```

- [Runnable examples](examples/README.md)
- [RegMix training search](examples/11_c4_pretraining_ablation.py): train C4 topic/content-type mixtures and select weights by validation loss
- [Model data recipes](examples/recipes/README.md): T5/C4, Falcon, Gopher, LLaMA 1, GPT-3 with local sample inputs
- [Curation](docs/curation.md) · [Fields](docs/enrichment.md) · [Training](docs/training.md)
- [Storage](docs/persistence.md) · [Internals](docs/architecture.md) · [Development](docs/development.md)
