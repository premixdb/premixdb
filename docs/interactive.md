# Interactive use

```bash
uvx --python 3.12 premixdb --storage .premixdb shell
```

`db` and `p` are ready. Tab completes public names; history is saved in the database
directory. Leaving the shell closes the database.

```python
snapshot = db.Corpus("demo")
snapshot.preview()
snapshot.preview(offset=3)
query = snapshot.query(steps=[p.where(p.text.characters > 200)])
query.profile()
query.preview()
mixture = query.mix(sequence_length=64)
mixture.profile()
mixture.preview()
dataset = mixture[0]
dataset.preview()
```

Queries and mixtures are lazy. Each resource's `profile()` describes that
resource: capture statistics for snapshots, selection statistics for queries,
composition statistics for mixtures, and packing statistics for datasets.
Exact profiles may scan the population, while snapshot profiles reuse capture
metadata.

`preview()` defaults to three results and stops consuming input once its window
is satisfied. It does not compute full profiles or publish completed query or
dataset output. Filters and decontamination can stop early; ranking, deduplication,
sampling, and generated mixture weights can require complete evidence first.
Fewer than three matches may require exhausting the input. Reopened completed
resources use saved preview indexes and bounded text reads.
Long operations print status; pass `progress=False` to `PremixDB` to turn it off.
For capture statistics alone, use `snapshot.profile()`.

Press Ctrl-C to cancel a Hub snapshot capture and return to the prompt. Capture
fails if the Hub reader produces no data for 60 seconds. A cancelled or failed
capture keeps the corpus's last successful snapshot; rerun the capture to retry.

```bash
premixdb --storage .premixdb corpora
premixdb --storage .premixdb profile query QUERY_ID
premixdb --storage .premixdb preview dataset DATASET_ID --limit 3
```

These commands inspect saved results. The same Python API works in notebooks.
