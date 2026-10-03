# Interactive use

```bash
uvx --python 3.12 premixdb --storage .premixdb shell
```

`db` and `p` are ready. Tab completes public names; history is saved in the database
directory. Leaving the shell closes the database.

```python
snapshot = db.corpus("demo")
snapshot.preview()
snapshot.preview(offset=3)
query = snapshot.query(steps=[p.where(p.text.characters > 200)])
query.profile()
query.preview()
dataset = query.dataset(sequence_length=64)
dataset.preview()
```

Queries are lazy. `profile()`, `preview()`, and `wait()` run the work they need.
Long operations print status; pass `progress=False` to `PremixDB` to turn it off.
For capture statistics alone, use `snapshot.profile()`.

```bash
premixdb --storage .premixdb corpora
premixdb --storage .premixdb profile query QUERY_ID
premixdb --storage .premixdb preview dataset DATASET_ID --limit 3
```

These commands inspect saved results. The same Python API works in notebooks.
