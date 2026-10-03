# Training

```python
import premixdb as p

from torch.utils.data import DataLoader

with p.PremixDB(storage=".premixdb") as db:
    dataset = db.corpus("training").query().dataset(sequence_length=2048)
    dataset.preview()
    loader = DataLoader(dataset.torch(), batch_size=32)
    for batch in loader:
        # input_ids, attention_mask, labels
        ...
```

Use your model's tokenizer when packing. Padding labels are `-100`.
The PyTorch adapter remains usable after closing the database.

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

## Resume a reader

```python
reader = dataset.reader(seed=42)
sequence = next(reader)
# Save after processing the sequence.
checkpoint = reader.checkpoint()
resumed = dataset.reader(seed=42, checkpoint=checkpoint)
```

Resume with the same dataset, seed, and topology. Save model, optimizer, and RNG
state alongside the checkpoint. DataLoader prefetching needs separate tracking
of consumed batches.
