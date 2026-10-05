# Serialization approach

## Decision and scope

Keep protobuf as the typed binary format for worker exchange and numeric cache
payloads. It is already a runtime dependency, and its compiler is already part of
the build. Do not add MessagePack or another serialization dependency.

Use Python dataclasses for internal domain values where they simplify the code.
Keep protobuf conversion at persistence and transport boundaries. Retain
`TypedDict` for records that actually cross JSON or Arrow interfaces; avoid
converting every record just for consistency.

Start with bulk worker JSON and ephemeral SQLite caches. Preserve the public API,
existing durable formats, logical identity codecs, and raw training-token files.
The initial implementation is complete: versioned protobuf worker requests and
results, packed ephemeral token caches, and tagged SQLite scalar/protobuf vector
caches. Durable token encodings remain on their existing per-token schema.
Generated bindings are build outputs; no serialization dependency was added.

## Evidence

Measured October 4, 2026 on this machine: Python 3.12.2, protobuf 6.33.6 with the
native `upb` backend, PyArrow 23.0.1, MessagePack 1.2.3. Values below are medians of
seven interleaved runs, in milliseconds. Implementations were measured in
separate runs using the same inputs.

| Workload | JSON | Protobuf | MessagePack | Arrow IPC → Python |
| --- | ---: | ---: | ---: | ---: |
| Worker text round trip | 4.05 | 1.28 | 0.84 | 1.89 |
| Worker tokens + ranges round trip | 315 | 242 | 218 | 973 |
| Worker dedupe evidence round trip | 6.35 | 2.98 | 2.51 | 9.70 |
| Token cache: insert + commit | 136 | 59 | 83 | Unmeasured |
| Token cache: 64 warm lookups | 143 | 57 | 120 | Unmeasured |
| Vector cache: build + commit | 411 | 34 | 42 | Unmeasured |

Worker data comprised 128 Shakespeare documents, 524,640 UTF-8 bytes, 157,785
tokens from the bundled GPT-2 tokenizer, and real Dupekit evidence. Vector data
comprised 512 synthetic 1024-dimensional float64 vectors. All codecs round-tripped
the same values exactly, without compression or quantization. SQLite writes
include commits; lookups are warm. These are boundary measurements, not complete
process-pool, model-inference, cold-disk, or peak-memory benchmarks.

The protobuf worker figures use protobuf's type enforcement instead of repeating
`checked_record()` dictionary shape validation after decoding. Packed-layout
version, coverage, and interval checks remain. Retaining the dictionary checker
produced protobuf times of **2.01 / 380 / 4.28 ms**, respectively. Removing only
redundant type checks is part of the proposed integration, not a free codec swap.

Schema layout is decisive. The existing `EncodedToken` submessage-per-token
representation took **594 ms** for the worker round trip even without the
dictionary checker. The packed-array prototype took **242 ms**. Its token-cache
database was **49% smaller** than JSON. Existing `NumericVector` protobufs made
the vector database **58% smaller**, with performance comparable to custom
packed float64 blobs.

MessagePack performed well but would become another direct dependency. Arrow is
useful when consumers retain batches: serializing Dupekit's existing batch took
0.13 ms versus 9.94 ms for conversion into JSON rows. Reconstructing nested Python
token records erased that benefit. Keep Arrow for Dupekit and Parquet export;
these measurements do not justify expanding it into all worker/cache interfaces.

## Format choices

- **Worker batches:** explicit protobuf request/result schemas, versioned at the
  envelope. Use `bytes` for digests and binary evidence rather than base64 strings.
  Reference existing producer/tokenizer policy messages instead of embedding
  serialized protobuf inside JSON.
- **Token caches:** a shared packed-array protobuf codec, preserving both
  `TokenList` and compact `ByteTokens` representations.
- **Vector caches:** reuse `NumericVector` and its packed `double` values.
  Preserve current precision; float32 conversion is a separate decision.
- **Scalar caches:** native SQLite columns when the field type is known. Preserve
  bool/int distinctions and nulls. SQLite INTEGER is signed 64-bit: never route
  uint64 values through REAL and lose precision. A heterogeneous `ValueCache`
  needs type tags or a schema; existing `FieldValue` protobufs are a reasonable
  fallback for supported types.
- **Small metadata:** retain JSON for receipts, manifests, producer definitions,
  CLI output, and external JSON/JSONL formats.
- **Training tokens:** retain little-endian uint32 token files and uint8 masks.

## Packed token layout

Use a versioned envelope with a representation discriminator:

```text
numeric:
  tokens:          packed uint32[N]
  interval_offsets: packed uint32[N + 1]
  starts:          packed uint64[M]
  ends:            packed uint64[M]

byte:
  tokens: bytes
  starts: packed uint64[K]
  ends:   packed uint64[K]
```

For numeric token `i`, its intervals occupy
`[interval_offsets[i], interval_offsets[i + 1])`. This preserves tokens with no
intervals, multiple intervals, and overlapping byte coverage. Byte tokens retain
their compact source intervals rather than expanding one record per byte.

Validate version and representation; offsets length `N + 1`; first offset zero;
last offset `M`; monotonic offsets; equal start/end lengths; ordered endpoints;
and applicable integer bounds. For byte tokens, interval lengths must cover the
byte count. Preserve the existing original-document bounds and retained-source
checks when restoring durable token encodings.

Protobuf parsing supplies field types, not domain validity. Keep checks for IDs,
coverage, vector widths/finiteness, artifact digests, and policy constraints.
Explicitly preserve absent versus empty values and nullable dedupe evidence;
repeated fields alone cannot distinguish null from an empty list.

## Implementation sequence

1. **Add internal schemas and shared codecs.** Extend the internal schema package
   with packed tokens and worker envelopes. Keep codecs below the runtime so
   engine caches and workers can share them. Do not create duplicate domain
   models or depend on runtime modules from the engine. Regenerate bindings using
   the existing build workflow; generated `_pb2.py`/`.pyi` files remain ignored.
2. **Convert ephemeral caches first.** Update `engine/token_cache.py` and
   `engine/value_cache.py`. Preserve locking, bounded SQLite memory, cleanup,
   independent length queries, and mapping behavior. Update every read/write
   path, including `items()`; benchmark cache subclasses only override selected
   methods and are not complete implementations.
3. **Convert worker payloads.** Update `runtime/pipeline.py` and
   `runtime/partition_types.py` for text, token, feature, evidence, and packing
   batches. Reuse existing field/evidence messages where suitable. Preserve
   document ordering, occurrence ordinals, retained ranges, masks, and provenance.
   Remove redundant dictionary validation only where the protobuf decoder
   constructs typed records and equivalent semantic checks remain.
4. **Preserve bounded execution.** Keep partition byte limits, bounded queues,
   immutable publication, coverage reconciliation, and receipt verification.
   Estimate sizes without serializing every row twice. Bind task/cache reuse to
   the codec version and artifact digests.
5. **Consider durable token shards separately.** `runtime/encodings.py` already
   stores per-token protobuf messages. Moving those to packed arrays requires a
   new format discriminator/version and an old-format reader or explicit
   migration. Do not reinterpret existing field numbers or silently overwrite
   published artifacts.

## Compatibility constraints

Leave snapshot JSON manifests, selection `source_record` JSON, lineage, and
existing SQLite resource metadata formats unchanged in the initial work.
Ephemeral cache changes do not require a historical database migration; persisted
worker artifacts need version-aware reuse or deliberate invalidation.

Preserve identity encodings in `engine/identity.py`,
`runtime/mixing.py:canonical_digest()`, and existing request/derivation hashing.
Some JSON is intentionally part of canonical identities. Changing a physical
codec must not casually redefine logical identities. Normal source/environment
fingerprint changes may still legitimately affect newly created recipes.

## Final integration and validation

The shared packed codec lives in `engine/token_codec.py`; shared scalar/vector
conversion lives in `schemas/binary.py`. `internal/transport.proto` defines the
physical envelopes. Workers consume typed protobuf messages directly and convert
to existing domain records at the engine/enrichment boundaries. JSON/Arrow record
checks remain where data still enters through those interfaces. No additional
domain models were needed.

Map planning uses protobuf byte sizes without serializing each row twice. Packing
uses the packed encoding's byte size, including multi-interval alignment. Codec
version and input artifact digests participate in physical task reuse through the
engine fingerprint and task keys. Queue windows, receipts and immutable
publication retain their existing behavior.

Run the repeatable final benchmark with normal CPU detection:

```bash
uv run --locked python scripts/benchmark_serialization.py --output reports/serialization.json
```

Final measurements use the same 128-document workload above, seven interleaved
samples, native `upb`, and 12 detected CPUs. The JSON baseline includes the former
record validation; protobuf includes version, presence, interval and coverage
validation. Round trips now include conversion from domain values and back, so
they should be compared within this table rather than to the prototype totals.
Independent phase medians need not sum to the round-trip median. All values are
milliseconds.

| Workload | JSON conversion | Proto conversion | JSON encoding | Proto encoding | JSON decoding | Proto decoding | JSON round trip | Proto round trip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Worker text | 0.20 | 1.19 | 4.17 | 0.74 | 3.14 | 1.72 | 7.68 | 3.70 |
| Worker tokens + ranges | 179.24 | 146.42 | 124.30 | 7.89 | 653.88 | 287.57 | 968.06 | 454.63 |
| Worker dedupe evidence | 0.09 | 1.98 | 2.65 | 0.65 | 7.45 | 1.87 | 9.54 | 4.10 |

| Actual cache write/read path | JSON | Protobuf |
| --- | ---: | ---: |
| Token inserts + commit | 298.10 | 147.47 |
| 64 warm token lookups | 307.96 | 95.20 |
| Vector inserts + commit | 645.78 | 146.95 |
| 64 warm vector lookups | 74.00 | 15.02 |

Committed token databases: 3,063,808 JSON bytes versus
1,572,864 protobuf bytes (48.7% smaller).
Committed vector databases: 10,784,768 JSON bytes versus
4,493,312 protobuf bytes (58.3% smaller).
The raw report also records individual token/vector conversion, encoding,
decoding samples. Cache timings exercise production cache writes and lookups,
include commits, and retain full float64 precision. These remain boundary
measurements; they exclude process dispatch, inference, cold disk and peak memory.

Regression coverage includes the full cache mapping interface, signed/unsigned
integer limits, bool/int distinctions, signed zero, numeric subclasses, heterogeneous scalar lists,
Unicode and empty values, numeric/compact-byte alignments, absent/empty fuzzy
evidence, malformed/truncated payloads, actual spawned workers, and restart,
packing, durable restoration, receipts and training reads.

Validation: `make check` passed with 948 tests, 91.44% coverage, lint/type/schema
checks, and fresh wheel/source-distribution builds with strict package validation.

## Verification and benchmark artifacts

Test exact round trips for Unicode, empty encodings, zero/multiple intervals,
uint32 token extremes, large uint64 endpoints/evidence, nullable outcomes, and
malformed/truncated payloads. Test the full cache mapping interface, actual worker
processes, token coverage, receipts, restart/reuse, and unchanged training reads.
If durable formats change later, add old-format restoration coverage.

Start with `tests/test_ephemeral_caches.py`, `tests/test_partitions.py`,
`tests/test_partition_planning.py`, `tests/test_encodings.py`,
`tests/test_enrichment_service.py`, `tests/test_reuse.py`, and
`tests/test_training_reads.py`. Regenerate with
`uv run --locked python scripts/generate_protos.py`; run relevant tests and
`make check`. Repeat the benchmarks on the final integration with normal CPU
detection. Report conversion, encoding, decoding, committed database size, and
lookup costs separately; do not extrapolate these results to whole-pipeline speed.

Local evidence is outside the checkout:

- [JSON/MessagePack/Arrow benchmark and reproduction notes](/Users/ariel/.codex/visualizations/2026/10/04/01a10687-88e9-7892-80a1-acf7408bda0f/json-proof/summary.txt)
- [JSON/MessagePack/Arrow raw samples](/Users/ariel/.codex/visualizations/2026/10/04/01a10687-88e9-7892-80a1-acf7408bda0f/json-proof/results-final.json)
- [Protobuf benchmark and reproduction notes](/Users/ariel/.codex/visualizations/2026/10/04/01a10687-88e9-7892-80a1-acf7408bda0f/protobuf-proof/summary.txt)
- [Protobuf raw samples](/Users/ariel/.codex/visualizations/2026/10/04/01a10687-88e9-7892-80a1-acf7408bda0f/protobuf-proof/results-final.json)
- [Prototype transport schemas](/Users/ariel/.codex/visualizations/2026/10/04/01a10687-88e9-7892-80a1-acf7408bda0f/protobuf-proof/benchmark_transport.proto)

The prototypes demonstrate the layout and performance tradeoffs. Production
codecs need the complete validation, compatibility, and API behavior above.
