# Internals

```text
sources → snapshot → query → mixture recipe → candidate datasets → packed sequences
```

Recipes pin their inputs, policies, seeds, model assets, and execution environment.
Completed results are reusable and remain readable after runtime changes.

SQLite stores recipes and metadata. Immutable files store text, fields, selections,
lineage, and tokens. Training reads token ranges directly. Selection and snapshot
inventories must fit in worker memory.

Queries and datasets share one materialization pool. Reads prefer completed results;
an active retry overrides an earlier failure. Recipes and failures are durable, so
cache eviction does not discard their state.

Mixtures register immutable recipes before resolving candidates. Recipe identity
pins the current execution fingerprint. Resolving a pass-through recipe does not
run the query; sampled mixtures resolve domain inventory and concrete weights.
Mixture profiles describe that inventory and composition, not packed datasets.

Preview execution consumes bounded query and packing streams without publishing
completed results or exact profiles. Global selection dependencies are resolved
when required to preserve final output semantics. Snapshot and completed-output
previews read only the requested stored document/sequence windows.

Local and partitioned execution share token serialization and disk-backed exact
evidence grouping. Evidence groups stay readable while their stream is open;
exhausting or closing the stream releases its temporary database.

| Code | Responsibility |
| --- | --- |
| [`_resources.py`](../src/premixdb/_resources.py) | Python API |
| [`_inputs.py`](../src/premixdb/_inputs.py) | Source values and request conversion |
| [`_files.py`](../src/premixdb/_files.py) | Publish complete immutable files and synchronize writes |
| [`_lineage.py`](../src/premixdb/_lineage.py) | Validated provenance and public witness IDs |
| [`_sequences.py`](../src/premixdb/_sequences.py) | Verified sequence reads and preview decoding |
| [`execution/`](../src/premixdb/execution) | Planning, execution, caching |
| [`engine/`](../src/premixdb/engine) | Capture, curation, tokenization, packing |
| [`engine/records.py`](../src/premixdb/engine/records.py) | Shared captured-document codec and structural checks |
| [`proto/`](../proto/README.md) | Stored schemas |

`tests/_reference.py` is the in-memory harness used by engine tests and
independent comparisons with the SDK. It is not shipped in the package.
