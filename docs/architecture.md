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
| [`api/`](../src/premixdb/api) | Sessions, resource handles, collections, and display |
| [`contracts.py`](../src/premixdb/contracts.py) | Shared records, checkpoints, scalar types, and JSON validation |
| [`fields/`](../src/premixdb/fields) | Typed field catalog, expressions, selectors, and stable field IDs |
| [`schemas/`](../src/premixdb/schemas) | Request builders, protobuf adapters, and wire validation |
| [`runtime/`](../src/premixdb/runtime) | Planning, scheduling, enrichment, and partition execution |
| [`storage/`](../src/premixdb/storage) | Metadata, immutable publication, catalogs, profiles, and restoration |
| [`training/`](../src/premixdb/training) | Verified sequence reads, checkpoints, and PyTorch adapters |
| [`cli/`](../src/premixdb/cli) | Command-line entry point and interactive shell |
| [`engine/`](../src/premixdb/engine) | Capture, curation, tokenization, packing |
| [`enrichment/`](../src/premixdb/enrichment) | Model and provider integrations |
| [`proto/`](../proto/README.md) | Stored schemas |

Application imports remain available from `premixdb`, with typed field namespaces
also available from `premixdb.fields`. Generated `v1/` and `internal/` schema
packages retain their existing paths. Other modules are implementation details;
the package name and its responsibility identify their home without a leading
underscore on every filename.

The API delegates execution to the runtime, which composes engine algorithms and
storage services. Engine, schema, field, storage, training, and provider modules
do not import API or runtime modules. Shared request and field vocabulary lives
below the API so planning and restoration use the same validation. Resource
classes live in individual API modules; `api/base.py` owns their shared identity
and waiting behavior.

`runtime/coordinator.py` owns session lifecycle, scheduling, and service entry
points. Dataset and mixture operations live in `runtime/datasets.py` and
`runtime/mixtures.py`; explicit delegates preserve the coordinator's request
tracking and method dispatch. Read-only catalogs and training readers can load
without importing the runtime. Stored recipes, protobuf names, and data codecs
retain their existing formats; moving Python implementation modules changes the
execution source fingerprint used for newly created recipes.

`tests/test_architecture.py` enforces these import boundaries. Place new code in
the package that owns its behavior instead of adding a general utilities package.

`tests/_reference.py` is the in-memory harness used by engine tests and
independent comparisons with the SDK. It is not shipped in the package.
