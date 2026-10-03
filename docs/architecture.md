# Internals

```text
sources → snapshot → query → candidate datasets → packed sequences
```

Recipes pin their inputs, policies, seeds, model assets, and execution environment.
Completed results are reusable and remain readable after runtime changes.

SQLite stores recipes and metadata. Immutable files store text, fields, selections,
lineage, and tokens. Training reads token ranges directly. Selection and snapshot
inventories must fit in worker memory.

| Code | Responsibility |
| --- | --- |
| [`_resources.py`](../src/premixdb/_resources.py) | Python API |
| [`execution/`](../src/premixdb/execution) | Planning, execution, caching |
| [`engine/`](../src/premixdb/engine) | Capture, curation, tokenization, packing |
| [`proto/`](../proto/README.md) | Stored schemas |

`premixdb.local` is the in-memory reference adapter used by engine tests.
