# Repository cruft removal guide

Reviewed the working tree on October 3, 2026, including the cleanup already in
progress. This document identifies 23 follow-up items; it does not remove code.
Items marked completed describe associated test/documentation work already done.
Locations use file links and symbol names because line numbers will move during
cleanup.

The largest removal candidate is `execution/inspection.py`: 597 lines with only
test callers in this repository. The next largest is the second API in `_api.py`
and `local.py`, totaling 363 lines, but it currently supports reference tests.
Neither count includes tests or dependent helpers, and neither implies a measured
runtime improvement.

## Suggested order

“Confirmed” means the stated condition was verified in the current repository.
It does not establish that an external application never imports the module.
“Decision” means the behavior still has consumers or a compatibility purpose.

| Item | Priority | Classification | Follow-up |
| --- | --- | --- | --- |
| 1. Unused comparison enum | First | Confirmed unused | Delete the enum |
| 2. Ineffective Hugging Face extra | First | Confirmed redundant dependency declaration | Remove or retain an empty compatibility extra |
| 3. Ignored timeout arguments | First | Confirmed inert private plumbing | Remove arguments and forwarding; retain wait deadlines |
| 4. Test-only inspection surface | Next | Confirmed absence of in-repository runtime callers | Retire it or give it an explicit supported use |
| 5. Second resource API | Next | Decision: reference-test dependency | Move or shrink the adapter |
| 6. Repeated field catalogs | Next | Structural duplication | Generate the repeated declarations |
| 7. Object state grafting | Next | Structural maintenance debt | Introduce explicit restoration/construction paths |
| 8. Compatibility branches | Later | Decision: saved-data and recipe compatibility | Set support cutoffs before removal |
| 9. Duplicate Shakespeare excerpt | Later | Confirmed duplicate bytes, both copies used | Choose one source for examples and packaging |
| 10. Installation/documentation leftovers | First | Concurrently edited documentation | Verify shell snippets and storage continuity |
| 11. Mandatory optional-feature dependencies | Later | Decision: installation contract | Consider dependency groups after defining supported installs |
| 12. Default-policy conversion methods | First | Confirmed unused methods | Delete two no-op adapters; keep the markers |
| 13. Two field-expression implementations | Next | Structural duplication | Unify operators while preserving integer semantics |
| 14. Private summary compatibility bridge | Next | Only test callers in repository | Assert profiles directly, then remove conversions |
| 15. Hidden document/field convenience methods | Next | Only test callers in repository | Replace wrappers or move them into test support |
| 16. `Datasets` alias | Later | Compatibility naming residue | Use `Mix` consistently; decide on alias deprecation |
| 17. Linear protobuf indexing helper | Next | Confirmed avoidable repeated traversal | Iterate rows once, then remove `at` |
| 18. Whole-catalog scans behind paginated lists | Next | Confirmed redundant data loading | Move filtering/paging closer to SQLite |
| 19. Empty object namespace directories | Later | Confirmed eager initialization | Create blob directories only when publishing |
| 20. Reference-API display implementation | Later | Coupled to item 5 | Remove the second formatter with the reference adapter |
| 21. Cross-layer access to private snapshot codecs | Next | Structural coupling | Extract the shared record contract before simplifying callers |
| 22. Stale tokenizer engine label | Next | Confirmed version-label mismatch | Define the label's meaning and remove misleading version text |
| 23. Repeated packaging training smoke tests | Completed | Confirmed repeated heavyweight setup | Full GPT-2/PyTorch smoke retained on an installed wheel |

## 1. Delete `ComparisonUnit`

**Evidence.** [`_enums.py`](../src/premixdb/_enums.py) defines `ComparisonUnit`
with `DOCUMENT` and `LINE`. A repository-wide symbol search finds only its
definition. It is not exported from
[`__init__.py`](../src/premixdb/__init__.py). The current dedupe request API uses
`DedupeAlgorithm`; the direct adapter accepts literal comparison strings.

**Removal.** Delete this class. Keep the other policy enums: they have callers
and public exports. Do not replace current literals with this enum merely to give
it a purpose.

**Validation.** Lint and type checking, plus
[`test_requests.py`](../tests/test_requests.py) and
[`test_field_expressions.py`](../tests/test_field_expressions.py).

## 2. Remove the ineffective `huggingface` extra

**Evidence.** [`pyproject.toml`](../pyproject.toml) declares
`datasets>=5.0.1,<6` both as a mandatory dependency and as the complete
`huggingface` extra. Installing the extra adds no dependency capability.
[`test_packaging.py`](../tests/test_packaging.py) asserts that the extra exists
in wheel metadata.

**Removal.** Keep `datasets` mandatory and remove the duplicate extra if that is
the intended installation contract. If existing `premixdb[huggingface]` users
matter, retain an empty extra during a compatibility period. Alternatively,
actually make Hub dataset capture optional; that is the larger decision in item
11, not just a metadata deletion.

**Validation.** Update the packaging assertion deliberately. Run the integration
packaging tests and inspect fresh wheel dependency metadata. Update the lockfile
without unrelated dependency upgrades.

## 3. Remove ignored local-executor timeout plumbing

**Evidence.** [`PremixDB._submit`](../src/premixdb/_resources.py) accepts a
`timeout` keyword but neither uses it nor forwards it. Twelve catalog methods in
[`catalog_reader.py`](../src/premixdb/execution/catalog_reader.py) and eight
create/get methods in
[`coordinator.py`](../src/premixdb/execution/coordinator.py) accept the same
unused argument. An AST check of argument reads confirms this. `_get` forwards
timeouts to these methods, and [`_catalog._pages`](../src/premixdb/_catalog.py)
passes them through `PageMethod`.

This resembles a transport call contract, but execution is local. The arguments
do not interrupt a blocked local method.

**Removal.** Remove the unused private arguments, overload declarations,
`PageMethod` keyword, and forwarding call sites together. Preserve the public
`PremixDB(timeout=...)` configuration and `_Execution.wait(timeout=...)`:
`wait` validates durations, computes a monotonic deadline, checks remaining
time, and bounds polling sleeps. Those behaviors are active. Also preserve actual
future wait timeouts in the materialization and worker code.

**Validation.** Run
[`test_initialization.py`](../tests/test_initialization.py),
[`test_service.py`](../tests/test_service.py), and
[`test_catalog_listings.py`](../tests/test_catalog_listings.py). Retain tests
covering invalid durations before execution and pending/error/timeout states.
Audit mocks and patched method signatures when changing the keyword contract.

## 4. Retire the inspection module or give it a supported caller

**Evidence.** [`inspection.py`](../src/premixdb/execution/inspection.py) is 597
lines. Its entry points include `catalog`, `history`, `rows`, `coverage`,
`index_statistics`, `sequences`, and `matrix`, with a separate family of
`TypedDict` response shapes and string-list query parameters. There are no
imports or calls from application source outside the module, CLI, scripts,
examples, or documentation. Its consumers are these tests:

- [`test_document_inspection.py`](../tests/test_document_inspection.py)
- [`test_sequence_inspection.py`](../tests/test_sequence_inspection.py)
- [`test_intrinsic_fields.py`](../tests/test_intrinsic_fields.py)
- Inspection portions of
  [`test_extended_resources.py`](../tests/test_extended_resources.py) and
  [`test_extended_curation.py`](../tests/test_extended_curation.py)

The supported interactive workflows already use `.preview()`, `.profile()`, and
resource collections. The inspector maintains another navigation/serialization
surface, including hexadecimal IDs and hard-coded topic/quality matrix bands.

**Decision.** Is an application intended to consume this inspector? If yes,
document that caller and its contract before retaining it. Otherwise remove the
module and its inspection-only DTOs, parameter parsing, and matrix/statistics
formatting. Move genuinely useful diagnostics into a development script only if
someone needs them.

**Removal sequence.** Map assertions in the five test files above to supported
behaviors first. Port relevant field projection, bounded preview, provenance,
sequence alignment, and corruption assertions to the public preview/profile or
storage tests. Delete tests that only exercise the retired inspector's response
format. Do not delete whole mixed-purpose test files to make imports disappear.

**Validation.** Search for remaining imports; run document/dataset preview,
intrinsic-field, profile, lineage/public-ID, and token-range tests. Run the full
suite and coverage after deletion. No speedup should be claimed until measured.

## 5. Reduce the second resource API

**Evidence.** [`local.py`](../src/premixdb/local.py) exposes the 317-line
[`_api.py`](../src/premixdb/_api.py) adapter. It provides another `PremixDB`,
`Corpus`, `Snapshot`, `SnapshotUnion`, `Query`, and `Dataset`. It repeats fluent
union/query wrappers also implemented in
[`_unions.py`](../src/premixdb/_unions.py) and the main resource API. The local
adapter persists snapshots but keeps query/dataset handles in memory and exposes
hexadecimal identities; the main API uses its catalog and public encoded IDs.

[`architecture.md`](architecture.md) calls this a reference adapter. Actual
consumers include `test_premixdb.py`, `test_curation.py`, `test_jsonl_sources.py`,
`tests/_type_support.py`, and the reference-reader comparison in
`test_service.py`. This is not dead code.

**Decision.** If `premixdb.local` is only a testing interface, move its adapter
into test support or replace broad API wrappers with narrow engine fixtures.
Preserve independent comparisons between service execution and engine results.
Routing both sides of a differential test through the service would weaken it.
If the import is externally supported, deprecate it before moving it.

**Associated removal.** Reassess the engine-handle wrapper
`_policies.HuggingFaceTokenizer` after shrinking the adapter. Do not confuse it
with the main tokenizer asset API. Replace duplicated policy conversion and
union scaffolding where the engine fixtures can use validated plans directly;
keep engine curation, packing, identity, and reader implementations.

**Validation.** Run the five caller groups above, dataset planning, tokenizer,
reader/checkpoint, and property tests. Check import/export compatibility if
keeping a deprecation shim.

## 6. Generate repeated field declarations

**Evidence.** The 176 language codes are repeated in `fields.Language`, the
176 typed `LanguageFields` attributes in
[`fields.py`](../src/premixdb/fields.py), the `LANGUAGE_*` members of
[`_enums.IntrinsicField`](../src/premixdb/_enums.py), and the protobuf intrinsic
field catalog. [`_field_ids.py`](../src/premixdb/_field_ids.py) already derives
the runtime name/ID mapping from the protobuf enum. The handwritten declarations
are used, and `test_field_expressions.py` checks their agreement.

**Refactor, then remove.** Make the protobuf catalog the authoritative source
for field identities and generate the repeated Python declarations. Preserve
typed attribute completion, explicit enum exports, and keyword spellings such
as `language.as_`, `language.is_`, and `language.or_`. Prefer generated typed
source or stubs over an untyped dynamic attribute bag. A smaller alternative is
to retire the public `IntrinsicField` enum after checking its users, leaving
typed field expressions as the primary interface.

**Validation.** Keep catalog agreement tests and type checking. Preserve existing
numeric protobuf IDs and serialized names; never regenerate IDs from list order.
Check language provider defaults and fixture recipe identities.

## 7. Replace object state grafting with explicit construction

**Evidence.** [`selections.restore`](../src/premixdb/execution/selections.py)
uses `Query.__new__(Query)` and manually assigns state, bypassing the constructor
that executes a recipe. `PackedPartitions.__init__` in
[`pipeline.py`](../src/premixdb/execution/pipeline.py) copies a `Dataset` through
`self.__dict__.update(handle.__dict__)`. Both depend on implicit knowledge of
another class's private state. Adding constructor fields can silently leave
restored values incomplete or copy unrelated state into a subclass.

**Refactor, then remove.** Add a typed engine restoration factory or completed
query value, and an explicit dataset construction path for packed outputs. Then
remove manual attribute grafting. Keep restoration separate from recipe
execution: replacing `__new__` with the normal executing constructor would rerun
completed work. Preserve the source/lineage/range integrity checks in restoration.

**Validation.** Exercise saved selections after cache eviction, reopening with a
different runtime, read-only preview/training, and local versus partitioned
packing. Relevant coverage lives in `test_service.py`, `test_training_reads.py`,
`test_readme_workflows.py`, `test_partition_planning.py`, and token publication
tests.

## 8. Set compatibility cutoffs before removing fallback branches

These are live compatibility paths, not confirmed dead code. Treat each as a
separate decision, even if they share a removal milestone.

### Legacy `.ref` resource migration

[`ObjectStore._migrate_metadata`](../src/premixdb/execution/storage.py) scans
legacy `.ref` files on writable initialization, verifies their spans/digests,
imports metadata into SQLite, and records `legacy-refs-v1`.
[`MetadataStore`](../src/premixdb/execution/metadata.py) has migration marker
methods and a migrations table used by this path.

If all supported stores use SQLite metadata, move the importer to an explicit
migration utility, then remove startup scanning and importer-only methods.
Existing migration records do not need destructive database cleanup. Keep old
files intact while migrating. Validate resumability, corruption rejection, and
original-file retention using `test_metadata.py`.

### Inline snapshot inventories and frames without profiles

[`Store._records`](../src/premixdb/engine/snapshots.py) supports version-1 inline
`documents` and version-2 paged inventories. Frame decoding accepts missing
profiles, and `Store._restore` recovers those profiles by reading frame text.
`test_snapshot_engine.py::test_legacy_inline_manifest_without_profiles_loads`
explicitly protects this support.

If the support floor becomes version 2 with profiled frames, provide a migration
that preserves logical snapshot identities and captured text before removing the
old layout and profile fallbacks. Stored frame encodings/digests may change;
references must remain valid. Keep strict rejection of mixed/invalid layouts and
corrupt inventories. Run snapshot loading, inventory, engine, and persistence
tests with old and current fixtures.

### Deprecated field selector names

[`query.proto`](../proto/premixdb/v1/query.proto) retains the deprecated
`FieldComparison.field_name` field. `_field_ids.selector_field` accepts it,
rejects disagreement with the enum, and callers canonicalize it away.
[`_display.py`](../src/premixdb/_display.py) also supports this spelling.
`test_enrichment_service.py::test_enum_requests_and_legacy_names_resolve_identically`
checks canonical identity and conflict handling.

This fallback is small and has little deletion payoff. Remove it only after
declaring old request/recipe support ended. Reserve protobuf field number 9 and
the old name if removing the schema field; never reuse them. Test old persisted
recipes and current canonical IDs. Keep the checkpoint ID compatibility in
`_ids.py` unless its support contract is separately changed.

## 9. Keep one authoritative Shakespeare excerpt

**Evidence.**
[`examples/data/tiny_shakespeare_excerpt.txt`](../examples/data/tiny_shakespeare_excerpt.txt)
and
[`src/premixdb/data/tiny_shakespeare_excerpt.txt`](../src/premixdb/data/tiny_shakespeare_excerpt.txt)
have identical bytes. `examples/_tutorial.py` and `test_shell_api.py` use the
example copy; the installed CLI uses the packaged copy. Both are currently used.

**Removal.** Have examples read the installed package resource, then remove the
example copy and adjust source-distribution expectations. Alternatively generate
both from one source during packaging if standalone example data is intentional.
Keep the full `tiny_shakespeare.txt`: it serves a separate shell demo.

**Validation.** Run CLI, shell API, example integration, and wheel/sdist tests.
Run examples from outside the repository root to check resource portability.

## 10. Finish the README shell instructions

**Evidence.** During the audit, [`README.md`](../README.md) contained an empty
shell bootstrap block, then a `PremixDB()` call without its `p.` qualifier and
Python snippets marked as Bash. Its installation command said `uv add premix`.

**Follow-up.** The README is being edited concurrently. Check the current snippets
for Python code marked as Bash, the missing `p.` qualifier, and use of `dataset`
before assignment in the demo. The shell's default storage and an explicit
`.premixdb` directory can differ; subsequent field examples should use the same
database as capture. Verify the installation command and heading after editing.
The workflow test now supplies the shell's `p`/`db` bindings and an offline demo,
isolates storage, executes Python blocks in order, and closes handles.

## 11. Consider separating optional-feature dependencies

**Evidence.** `pyproject.toml` makes `torch`, `transformers`,
`sentence-transformers`, and `datasets` mandatory. Provider loading in
[`enrichment/models.py`](../src/premixdb/enrichment/models.py) is lazy, Hub source
loading is capability-specific, and `_torch.py` is needed for `.torch()`.
Installing the package therefore includes dependencies for features that a local
byte-tokenizer workflow need not execute.

**Decision.** Define whether the default installation promises all these
features. If a smaller core is desired, introduce feature extras, useful
missing-dependency errors, and separate core/full environment checks. The README
currently demonstrates Hub capture and PyTorch training together, so update
installation instructions at the same time. This is packaging design work, not
proof that the dependencies are unused. Keep platform-specific pins while the
affected platforms remain supported.

**Validation.** Build a wheel, install it into fresh minimal and full environments,
and exercise local capture/query/packing plus each advertised optional feature.
Check provider identity/version handling when packages are absent. Do not remove
provider transitive dependencies just because application imports do not mention
them directly.

## 12. Delete the unused default-policy `_to_proto` methods

**Evidence.** `DecontaminateDefault._to_proto` and `SamplerDefault._to_proto` in
[`_policies.py`](../src/premixdb/_policies.py) both return `None`. No caller
invokes either method. [`_unions.py`](../src/premixdb/_unions.py) recognizes the
marker classes with `isinstance` and converts them to `None` itself.

**Removal.** Delete the two methods. Keep the classes unless the public policy
signature is deliberately changed: they are exported and appear in query
defaults. Other `_to_proto` methods have real callers and are not part of this
deletion. Validate query defaults, request construction, and explicit policies.

## 13. Unify the two field-expression systems

**Evidence.** [`_requests.py`](../src/premixdb/_requests.py) implements `_Field`,
`_Predicate`, comparison operators, ordering, and intrinsic namespace objects.
[`_field_expr.py`](../src/premixdb/_field_expr.py) implements `ScalarField`,
`FieldPredicate`, the same operators, ordering, and projection metadata.
`where`, `_curation.selector`, and profile selector unions handle both families.

**Refactor, then remove.** Give intrinsic and derived fields one expression
interface and remove duplicate operators and conversion branches. This needs a
numeric design decision: intrinsic counts use unsigned 64-bit comparison values,
while derived integer selectors use signed 64-bit values. Preserve that range
distinction, enum handling, rejection of boolean-as-integer inputs, and protection
against Python chained comparisons. Validate field expressions, request/planner
tests, large-count boundaries, ordering, and null/vector/class projections.

## 14. Remove the private resource summary compatibility bridge

**Evidence.** `_Execution._summary`, `_counts`, and `_query_counts` in
[`_resources.py`](../src/premixdb/_resources.py) convert public protobuf profiles
back to older dictionary summaries. All `_summary()` callers outside its
definition are tests. [`_types.py`](../src/premixdb/_types.py) maintains separate
`SnapshotSummary` and `QueryPopulationSummary` declarations with identical
document/byte/character keys, plus wrapper-specific query/dataset summary shapes.
The engine has its own richer summary records.

**Removal.** Assert public profile values directly in service tests. Keep any
reference-engine comparison conversion in test support, then delete the SDK
bridge and types that lose their last caller. Do not delete the engine summary
contract or `SnapshotSummary` blindly: engine counts currently reuse it. Validate
service/profile tests and the differential checks in `test_design.py`.

## 15. Reassess hidden document and field convenience methods

**Evidence.** `Query._list_document` is called only by `test_public_ids.py`;
`Query._describe` is called by `test_profiles.py` and internally uses
`_with_fields` to create a second query if a projection is missing. The supported
public workflow uses `.preview()` and `.profile()`, with field summarization in
`_profiles._describe_field`. Neither convenience method is on the public shell
surface.

**Removal.** Test document IDs through `preview(max_characters=0)` and field
summaries through published profiles. Delete the wrappers when no intended
internal consumer remains. Keep `_with_fields` until its actual inspection,
mixture, and test consumers are handled; keep `_describe_field` because the CLI
uses it. `_estimate` is a separate metadata feature and is not proven unused.
Validate public IDs, profiles, and metadata-only behavior.

## 16. Retire the `Datasets` naming alias

**Evidence.** [`_resources.py`](../src/premixdb/_resources.py) defines
`Datasets = Mix`. The old name remains in `Query.mix`'s return annotation,
`PremixDB._datasets`, and public imports/exports in `__init__.py`. There is one
implementation, and the documented workflow calls the resource a mix.

**Cleanup.** Use `Mix` in internal annotations and rename the private reopening
helper consistently. Decide whether to retain a public compatibility alias for
external imports. This is a small naming cleanup, not a second implementation
to delete. Validate mix slicing, reopening, type checking, and exports. Do not
rename the unrelated protobuf `ListDatasets` operation solely because of this
Python alias.

## 17. Remove repeated linear protobuf indexing

**Evidence.** [`_protobuf.at`](../src/premixdb/_protobuf.py) enumerates a container
from its start for each requested index to work around protobuf indexing stubs.
[`enrichment.cached_rows`](../src/premixdb/execution/enrichment.py) calls it inside
row loops, once per field column or evidence row. Reading indices 0 through
`n-1` consequently traverses about `n(n+1)/2` entries per column. Cohorts are
bounded, so this is bounded inefficiency rather than unbounded memory growth.

**Refactor, then remove.** Iterate columns/rows together, or materialize a typed
tuple once per column and index that. Remove `at` after its final caller is gone.
Keep strict row/document alignment, deterministic order, and missing-cache
handling. Validate field/index cache reuse and local/partitioned equivalence;
benchmark a full cohort with multiple columns rather than a single-row fixture.

## 18. Stop reconstructing whole catalogs for each page

**Evidence.** [`MetadataStore.list`](../src/premixdb/execution/metadata.py) loads
every payload in a namespace. `Catalog._listing` sorts resources and recomputes a
membership digest on every page. SDK `_pages` collects all pages before public
collection methods slice to `limit`/`offset`. `_corpus_queries` scans all queries;
`list_dataset` repeats dataset namespace scans for each matching query.

**Refactor.** Consolidate filtering and paging in the metadata layer, with parent
indexes or a saved listing snapshot where needed. Remove repeated Python
filter/sort/full-payload passes after the storage query supplies the required
window. Preserve completed/active/failed/pending precedence, rejection of stale
or cross-list continuation tokens, and each public list's ordering; raw digest
order and encoded public-ID order are not interchangeable. Validate catalog
listing tests and measure a large catalog with many queries per corpus. The
current small fixture timings are not evidence that this scales cheaply.

## 19. Avoid eager empty object directories

**Evidence.** `ObjectStore.__init__` creates an `objects` directory for every
`PREFIXES` entry, including `execution` and `submission`. Those two resources are
saved as SQLite metadata by the current execution paths, not object blobs.
[`_files.publish`](../src/premixdb/_files.py) already creates parents when an
immutable file is actually written.

**Cleanup.** Create blob directories on publication instead of at every writable
initialization. Keep namespace validation and legacy-import support separately.
Update legacy fixtures to construct their old directory layout explicitly if
they currently rely on eager creation. Validate fresh/read-only initialization,
migration, publication, and concurrent writes. Expect a tidier storage root;
measure before claiming a meaningful initialization speedup.

## 20. Remove the reference adapter's second display implementation

**Evidence.** [`_display.py`](../src/premixdb/_display.py) has `_LOCAL_FIELDS`,
`_LOCAL_OPERATORS`, and `_local_repr` alongside the protobuf resource formatter.
The local adapter's snapshot/query/dataset repr methods call this path, and
`test_resource_display.py` covers it.

**Removal.** Include this formatter and its local plan vocabulary in item 5's
adapter retirement. If the reference adapter moves into test support, its debug
formatting can move with it or become a minimal fixture repr. Keep the production
formatter's metadata-only guarantee. This is coupled work, not another
independently dead module. Validate display tests and update user-facing examples
only if an externally supported local API is being deprecated.

## 21. Extract the shared captured-document record contract

**Evidence.** [`selections.py`](../src/premixdb/execution/selections.py) imports
private snapshot helpers `_decode_document`, `_schema`, `_totals`, and `_profile`
from the engine module. It embeds a JSON `source_record` inside a protobuf
`SelectedDocument`, then repeats frame/record checks while restoring that record.
This ties selection persistence to private snapshot layout details.

**Refactor.** Move the common frame/document codec and structural checks into an
explicit shared storage-record module. Remove cross-layer imports of private
snapshot helpers and duplicate structural decoding once callers use that
contract. A typed protobuf source record could also replace the embedded JSON,
but requires a stored-selection migration. Preserve selection-specific lineage,
retained-range, corpus, and occurrence checks; legacy snapshot acceptance is
broader than the selection contract. Validate corruption tests, transformed
documents, cold restoration, and old persisted records.

## 22. Resolve the stale tokenizer engine label

**Evidence.** `_tokenizer_definition` in
[`engine/datasets.py`](../src/premixdb/engine/datasets.py) hashes the literal
`huggingface/tokenizers/0.22.2/python/onig/encode-no-special-tokens/v1`.
`pyproject.toml` currently requires `tokenizers>=0.23.1,<0.24`.

**Decision.** Is `0.22.2` a historical logical-codec label or a claim about the
executing engine? Document it as historical if intentional; otherwise replace
the misleading version label with an explicitly versioned encoding contract and
use the actual execution fingerprint for package versions. Do not casually
update this string: it participates in tokenizer/dataset identities. Validate
identity fixtures, persisted tokenizer reuse, and current encoding/offset
semantics. This is stale configuration text, not an unused dependency.

## 23. Split repeated packaging imports from installed training validation — completed

**Evidence.** [`test_packaging.py`](../tests/test_packaging.py)'s
`assert_importable` performs the same GPT-2 capture/query/PyTorch training-read
workflow for a direct wheel, a wheel rebuilt from the sdist, and both editable
build modes. Each invocation starts a fresh Python process and imports PyTorch.
The build modes need binding generation, path isolation, asset inclusion, and
import validation; repeating the whole tensor workflow adds heavyweight setup.

**Completed.** Every artifact/mode still gets fresh-process import, message,
asset, real capture/query, and byte-token dataset checks. The installed wheel in
`test_wheel_replaces_stale_build_outputs` additionally runs the full default
GPT-2 and PyTorch DataLoader workflow. This removes three repetitions of that
heavyweight setup while retaining installed-training validation. Fresh-process
checks remain in place to catch stale module/path leakage.

## Keep these distinctions during cleanup

- **Caches:** `MemoryCache` bounds resident handles; `ValueCache` and `TokenCache`
  provide temporary disk-backed computation; `Encodings` stores reusable durable
  token shards. They have different lifetimes and integrity requirements.
- **Concurrency:** `SingleFlight` deduplicates submissions; the materialization
  pool executes background recipes; partition workers run kernels in processes.
  These are separate responsibilities.
- **Read-only imports:** The lazy exports in `execution/__init__.py` keep catalog
  reads from importing execution machinery. Preserve the subprocess tests that
  enforce this boundary.
- **Validation:** Hash, size, schema, source-range, and lineage checks verify
  different boundaries. Repeated-looking checks are not sufficient evidence of
  redundancy.
- **Generated bindings:** Local `_pb2.py` and `_pb2.pyi` files are ignored build
  outputs. Edit schemas and generation code, not those outputs. The schemas are
  message contracts; they do not declare RPC services.
- **Manual scripts:** A script without an application import is not necessarily
  unused. Preparation and enrichment exporters are executable tools and need a
  usage decision before removal.

## Verification for follow-up changes

The associated speed work keeps six bounded pytest workers, defers heavy imports,
uses small real tokenizers for generic preview checks, reduces repeated installed
training smoke tests, and starts expensive integration files early. Local query
waits wake on job completion instead of sleeping until the next poll; regression
tests cover completion, errors, deadlines, and a stale running handle.

On the reviewed machine, the complete non-performance suite took **38.14 seconds
before the scheduling change and 28.42 seconds after**, with **802 passed and one
platform-related skip** in both runs. The final coverage run passed the same suite
in **50.69 seconds**, with **91.59% branch-enabled coverage** against an 85%
requirement. The final default fast suite took **15.60 seconds**, with **769
passed and one skip**. These are local timings, not CI guarantees. Lint,
formatting, types, and generated-binding checks also passed.

Use targeted tests while changing one item, then run `make check` for a completed
removal batch. The default fast suite excludes integration/performance tests;
package, example, and process changes need the relevant integration tests too.
Use `uv run --locked pytest -m integration ...` to select them explicitly.

For a deletion, search for the old import/symbol in source, tests, scripts,
examples, schemas, and documentation. Review remaining public exports and
packaging metadata. Measure test duration again if test-only features are retired;
do not preserve obsolete behavior solely to retain a coverage percentage.

The audit combined source inspection, repository-wide caller/export searches,
AST checks for unread arguments and unreferenced definitions, test inspection,
dependency metadata review, and byte comparison of duplicate assets. It does not
establish external usage or a saved-data support policy. Those decisions are
called out above rather than inferred from missing in-repository callers.
