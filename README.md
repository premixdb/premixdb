# premixdb

premixdb is a declarative library for building reproducible data mixtures to improve llm pretraining.

## Quick start

Run a premixdb shell without installing the library:

```bash
uvx --python 3.12 premixdb shell
```

The shell imports premixdb as `p` and opens `db`. These bindings are already available:

```python
import premixdb as p
db = p.PremixDB()
```

## Example

The shell creates two small corpus snapshots: `demo` contains eight Tiny
Shakespeare speeches, and `benchmark` contains one example from the
[LEAF Shakespeare benchmark](https://leaf.cmu.edu/). One demo speech overlaps
the reference at the default 13-word threshold. Both inputs ship with the package.

The following example demonstrates usage of premixdb to:

1. Filter, dedupe and decontaminate a corpus
2. Create a [RegMix](https://arxiv.org/abs/2407.01492) style data mixture
3. Tokenize and pack the result into a PyTorch dataset

```python
from torch.utils.data import DataLoader

mixtures = db.Corpus('demo').query(
    steps=[
        p.where(p.text.characters >= 100),
        p.dedupe(),
        p.where(p.quality.writing_style >= 0.8)
    ],
    decontaminate=p.Decontaminate(db.Corpus('benchmark'))
).mix(
    domains=p.Topic,
    weights=p.RegMix(),
    tokens=3,
    splits=p.Splits(train=0.9, validation=0.05, test=0.05)
)

next(iter(DataLoader(mixtures[0].train.torch(), batch_size=1)))
```

```
{'input_ids': tensor([[45472, 10426,  1677,  ...,   475,   286,   477]]),
 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]]),
 'labels': tensor([[45472, 10426,  1677,  ...,   475,   286,   477]])}
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
