# Training

```python
import premixdb as p

from torch.utils.data import DataLoader

with p.PremixDB(storage=".premixdb") as db:
    dataset = db.Corpus("training").query().mix(sequence_length=2048)[0]
    dataset.preview()
    loader = DataLoader(dataset.torch(), batch_size=32)
    for batch in loader:
        # input_ids, attention_mask, labels
        ...
```

Use your model's tokenizer when packing. Padding labels are `-100`.
The PyTorch adapter remains usable after closing the database.
Closing a session releases its own read threads. Caller-supplied readers stay
open; use `with p.RangeReader(...) as reader` or call `reader.close()` when done.

## Define mixtures

`query.mix()` returns a `DataMixture`. By default it contains one dataset that
packs every query occurrence in its existing order. A token budget without
weights samples according to the population's token proportions.

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
data = dataset.torch(streaming=True, seed=42, epoch=0)
loader = DataLoader(data, batch_size=32, num_workers=4)
```

Streaming shuffles pages and their sequences, partitioning them across ranks and
workers. It uses an initialized distributed group, or explicit `rank` and
`world_size`. Use a new adapter for each epoch. Don't add a `DistributedSampler`
or `shuffle=True` to streaming data.

Ranks can have unequal lengths; use DDP join support or a shared step budget.
In scripts that spawn workers, create the loader inside the main guard.

## Resume training

Save the dataset ID, model, optimizer, random state, and the number of consumed
batches. Restore the same dataset and reading order. With a map-style adapter,
a PyTorch sampler can begin at the next sequence index. DataLoader prefetching
means the number of fetched batches can exceed the number actually processed.

Checkpointed framework-independent iteration remains an internal implementation
facility. The public training interface is `dataset.torch()`.
