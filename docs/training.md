# Training

```python
import premixdb as p

from torch.utils.data import DataLoader

with p.PremixDB(storage=".premixdb") as db:
    dataset = db.Corpus("training").query().mix(sequence_length=2048)[0]
    dataset.train.preview()
    loader = DataLoader(dataset.train.torch(), batch_size=32)
    for batch in loader:
        # input_ids, attention_mask, labels
        ...
```

Use your model's tokenizer when packing. Padding labels are `-100`.
The PyTorch adapter remains usable after closing the database.
Closing a session releases its own read threads. Caller-supplied readers stay
open; use `with p.RangeReader(...) as reader` or call `reader.close()` when done.

## Define mixtures

`query.mix()` returns a `DataMixture`. By default it contains one dataset with
80% training, 10% validation, and 10% test document membership. The proportions
are approximate: a seeded hash assigns whole captured-content groups, keeping
identical content together across source IDs and corpora. Small populations can
have empty splits. Token and packed-sequence proportions can differ.

Configure a different policy with the generated `Splits` protobuf message:

```python
dataset = query.mix(
    splits=p.Splits(train=0.90, validation=0.05, test=0.05, seed=42),
)[0]
train = dataset.train.torch()
validation = dataset.validation.torch()
test = dataset.test.torch()
```

All three finite, nonnegative proportions must be supplied and sum to one.
`splits=` accepts `p.Splits`, rather than a dictionary or tuple. Omission saves an
explicit 80/10/10 policy with split seed zero. Use
`p.Splits(train=1, validation=0, test=0)` for inputs already split elsewhere.

Weights, token budgets, and replacement apply to the training population.
Validation and test retain the same documents and packing across candidates and
sampling seeds. The split seed controls membership independently of RegMix,
mixture sampling, and reader seeds. A budget without weights uses the training
population's token proportions. Without weights or a budget, every query
occurrence is retained in its original order within its assigned split.

Each split is packed independently; padding or dropped tails are handled at each
boundary. Split views share the parent's stored tokens through half-open sequence
ranges. They support indexing, iteration, profiles, previews, and PyTorch adapters.
`dataset.profile()` describes all splits; `dataset.train.profile()` describes
training alone. `dataset.torch()` reads the combined dataset, so use
`dataset.train.torch()` for training.

Split selectors belong to the full dataset. If `dd = dataset.validation`, use
`dd.torch()` for validation batches and `dataset.train.torch()` for training batches.

```python
fixed = query.mix(
    domains=p.object.uri,
    weights={"web": 0.8, "papers": 0.2},
    tokens=100_000_000,
    replacement=True,
)
candidates = query.mix(
    domains=p.source.corpus_id,
    weights=p.RegMix(seed=42),
    n_candidates=16,
    tokens=100_000_000,
    replacement=True,
    seed=123,
)
```

Fixed weight keys must cover the available domains and sum to one. Use captured
source URIs for `p.object.uri` and public corpus IDs for `p.source.corpus_id`.
Multiple candidates require `weights=p.RegMix(...)`. Its seed controls proposed
proportions; the mix seed controls content selection. Replacement defaults to
false, so allocations must fit the population. With replacement enabled, sampling
uses repeated shuffled passes; `Bounds(max_epochs=...)` limits exposure.

A mixture's datasets are alternative complete training recipes. Index or slice
it to choose candidates; `mixture.datasets` exposes an immutable tuple of handles.
`mixture.profile()` reports the domain inventory, proportions, allocations,
constraints, and budgets. `mixture.preview()` shows three candidate compositions.
These operations do not profile or pack the datasets. For packing statistics or
training examples, use `mixture[0].profile()` or `mixture[0].preview()`.

Mixture creation is lazy. Indexing resolves candidate recipes; generated weights
require domain inventory, which may execute the query and tokenize its population.
Pass-through indexing requires neither inventory nor query execution. Exact
profiles may scan the full population. Pending dataset previews pack only enough
sequences for the requested window and leave the dataset pending; `.torch()` and
`.wait()` publish complete packed output. `streaming=True` controls training reads
of that output, rather than packing on demand.

## Shuffle and distribute

```python
data = dataset.train.torch(streaming=True, seed=42, epoch=0)
loader = DataLoader(data, batch_size=32, num_workers=4)
```

Streaming shuffles pages and their sequences, partitioning them across ranks and
workers. It uses an initialized distributed group, or explicit `rank` and
`world_size`. Use a new adapter for each epoch. Don't add a `DistributedSampler`
or `shuffle=True` to streaming data.

Ranks can have unequal lengths; use DDP join support or a shared step budget.
In scripts that spawn workers, create the loader inside the main guard.

## Resume training

Save the parent dataset ID and split name, model, optimizer, random state, and the number of consumed
batches. Restore the same dataset and reading order. With a map-style adapter,
a PyTorch sampler can begin at the next sequence index. DataLoader prefetching
means the number of fetched batches can exceed the number actually processed.

Checkpointed framework-independent iteration remains an internal implementation
facility. The public training interface is `dataset.train.torch()`.
