# Storage format options

## Recommendation

Prototype Vortex for enrichment and analytical columns before changing the
default format. Consider Lance if premixdb should delegate corpus tables and
search to an existing storage engine. Keep the specialized training-token format
initially. Neither option has been benchmarked in this project.

Premixdb already uses complementary formats: SQLite metadata, immutable objects
with verified digests, Arrow analytical blocks, Roaring selection indexes, and
specialized token files. Ordinary filters use prepared indexes rather than full
column scans, so a new columnar format would not automatically accelerate them.
See [architecture](architecture.md), [query performance](query-performance.md),
and [serialization](serialization.md).

## Vortex

Vortex is the more focused experiment for the current architecture. Its column
layouts, pruning statistics, and compressed encodings suit quality scores,
classifier probabilities, labels, lengths, and document descriptors. Some
operations can defer decompression; benefits depend on the data and execution
path. See the [file layout](https://docs.vortex.dev/concepts/file-format) and
[array encodings](https://docs.vortex.dev/concepts/arrays).

Integration would start at the Arrow read/write boundary in
[`storage/analytics.py`](../src/premixdb/storage/analytics.py):

1. Publish immutable Vortex blocks with the existing population order, numeric
   types, and null semantics.
2. Record the codec, schema, population identity, and digest in artifact references.
3. Keep Roaring indexes for predicates and completed-query membership.
4. Gather selected values for profiles, previews, and downstream processing.
5. Retain readers for existing Arrow artifacts.

New enrichment could publish additional column blocks against the same immutable
population. Premixdb would continue to own population alignment, manifests,
indexing, publication, and provenance.

The likely advantages are smaller column storage and reduced memory traffic,
with potentially faster selected-value reads and profiles. The disadvantages are
another native dependency, compression costs during publication, compatibility
management, and decoding or Arrow/NumPy conversion overhead. Replacing Arrow
files while immediately decoding every value could save space but add latency.
Use supported writer editions explicitly; Vortex documents frozen-format
[compatibility guarantees](https://docs.vortex.dev/specs/file-format).

## Lance

Distinguish Lance files, Lance tables, and LanceDB. The
[Lance file format](https://lance.org/format/file/) provides columnar storage
designed for random row/range reads and Arrow processing. Table semantics and
search indexes live above the file layer. Lance files could therefore sit beneath
premixdb's existing manifests as an alternative physical codec.

Adopting [Lance tables](https://lance.org/format/table/) would be a broader change:

1. Store corpus text, identifiers, and intrinsic fields in fragments.
2. Publish enrichment columns in additional fragment files without rewriting
   existing columns.
3. Pin exact table versions in premixdb recipes.
4. Map Lance rows to premixdb document identities and selection ordinals.
5. Optionally delegate scalar or vector search to Lance.

Lance becomes more compelling when embeddings, similarity retrieval, shuffled
document access, or shared object-store corpora are central. Its table versioning,
feature evolution, and indexes could reduce custom code if premixdb delegates
those responsibilities. Keeping both systems would instead add overlapping
manifests, indexes, and version lifecycles. Lance row lineage does not replace
provenance through filtering, mixing, tokenization, and packing.

Premixdb's object store currently requires a local filesystem. A cloud-capable
file format alone would not add remote storage support.

## Training tokens

Keep little-endian uint32 token files and uint8 masks initially. The sequence
path already supports optional Zstd compression. A columnar container might
compress tokens, but adds decoding and metadata work to a simple range-read
workload. Evaluate changes against actual training throughput separately.

## Evaluation before adoption

Compare current Arrow storage, compressed Arrow IPC, Vortex, and Lance files on
real corpus and enrichment data. Measure publication time, stored size, peak
memory, cold and warm selected-value reads, complete query/profile latency, and
document retrieval. Verify exact types, nulls, population alignment, digests,
restart behavior, and old-format readability.

Choose based on end-to-end results. Vortex is a candidate for improving column
storage within the existing design; Lance tables are a candidate for delegating
corpus storage and search responsibilities.
