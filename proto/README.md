# Protobuf schemas

The Python API builds these requests for local execution.
The schemas also describe persisted objects; another language can read them
without importing the Python engine. Most users can stay in the Python API.

| Schema | Contents |
| --- | --- |
| [corpus.proto](premixdb/v1/corpus.proto) | Names and latest snapshot references |
| [snapshot.proto](premixdb/v1/snapshot.proto) | Captures, sources, profiles and status |
| [query.proto](premixdb/v1/query.proto) | Ordered selection recipes, dependencies and previews |
| [intrinsic.proto](premixdb/v1/intrinsic.proto) | Built-in field IDs |
| [field.proto](premixdb/v1/field.proto), [index.proto](premixdb/v1/index.proto) | Definitions and immutable builds |
| [dataset.proto](premixdb/v1/dataset.proto) | Tokenizers, packing, sampling, mixtures and stored sequences |
| [profile.proto](premixdb/v1/profile.proto) | Histograms, numeric summaries and estimates |
| [storage.proto](premixdb/v1/storage.proto) | Source inputs, objects and byte ranges |
| [status.proto](premixdb/v1/status.proto) | Resource state and execution history |

## Recipes and pins

Resources reference parents by ID. Queries canonicalize snapshot unions and keep
operation order. The planner resolves built-in field/index dependencies; clients
don't register producers or choose field build IDs. `Query.fields` requests
projections for inspection or mixture domains without adding filters.

`git_commit` contains raw bytes of the full revision, not ASCII hex. Empty pins
resolve through the coordinator or parent resource. Explicit unavailable revisions
fail. Query identities also include the execution fingerprint described in
[architecture](../docs/architecture.md). Model and tokenizer assets have separate pins.

[`internal/derivation.proto`](premixdb/internal/derivation.proto) stores producer
recipes, artifact receipts, query selections and token encodings. These are
internal persistence messages, with no public build operation.

Mixtures register ordered candidate dataset IDs and planned profiles. A candidate
packs when consumed. Dataset sampling budgets count content tokens; separators,
padding and dropped tails have separate accounting. RegMix proposes weights only.

## Reads and statistics

`ObjectRef` carries a BLAKE3 digest and local file location.
`SpanRef` identifies a half-open byte range and its digest. Dataset readers fetch
these ranges directly.
Token files use little-endian uint32 values and masks use uint8 values.
Sequence metadata stores regions and source provenance in batches.

`Preview` pages snapshot/query documents. Resource metadata also includes bounded
deterministic prefix previews. Previews aren't representative random samples.
An absent preview is unavailable; a present empty preview means empty output.

Profiles retain exact counts and numeric moments plus compacted histograms.
Quantiles return containing bucket bounds. Missing optional statistics mean
unknown, not zero. Query estimates describe input populations and conservative
selection bounds; they don't start inference or claim exact output counts.
See [interactive inspection](../docs/interactive.md).

## Generate and verify

```bash
make sync
uv run --locked python scripts/generate_protos.py
make protos
make build
```

Edit schemas and commit them. Generated `_pb2.py` and `_pb2.pyi`
files are ignored build outputs; don't edit or commit them. Editable installation
creates local bindings. Schema changes trigger an editable rebuild; `make protos`
checks that generated outputs match.

Setuptools generates messages and type stubs with the pinned
`grpcio-tools` build dependency. Wheels contain the bindings and need no compiler.
Source distributions contain schemas and regenerate bindings when built.

Changing field numbers affects saved protobuf messages as well as the wire format.
Update readers and stored-format handling together when changing a schema.
