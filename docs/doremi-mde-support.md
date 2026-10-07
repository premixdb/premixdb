# Implementation plan: mixture domain datasets for DoReMi and MDE

Status: implementation plan, not an implemented API.

This document is the implementation brief for a future session. Implement the
`DataMixture.domains` property completely, including durable execution, ordinary
dataset behavior, tests, documentation, and a runnable example. Do not stop at a
Python mapping wrapper or a demonstration using private engine handles.

## 1. Objective and scope

Expose the complete input population of each mixture domain through ordinary
`Dataset` handles:

```python
from collections.abc import Mapping

class DataMixture:
    @property
    def domains(self) -> Mapping[str, Dataset]:
        ...
```

This enables external DoReMi proxy experiments, external MDE expert experiments,
and external online samplers to access consistent domain inputs efficiently.
RegMix continues to use its existing candidate API.

PremixDB owns capture, enrichment inference and its materialization, domain
membership, tokenization, packing, sampling, provenance, and data access. The
application owns all model training, evaluation, loss computation, gradients,
regression fitting, mixture optimization, and experiment orchestration. This
boundary includes reference models, proxy models, and data experts used to select
a mixture. Model computation in the example is application code, not a library
feature.

The research context is [the data-mixture survey](https://arxiv.org/html/2505.21598v1).
DoReMi needs identifiable domain inputs for reference/proxy computation and a
final static mixture. MDE needs single-domain expert inputs and consistent
held-out inputs across experts. Neither requires PremixDB to implement its
optimization algorithm.

### Required in this implementation

- A read-only, lazy `Mapping[str, Dataset]` returned by `mixture.domains`.
- Full-population domain selection, followed by the existing content split rule.
- Every existing domain selector form, including enriched fields and explicit
  assignments.
- Durable domain membership and dataset recipes, restart/read-only behavior,
  profiles, previews, sequence provenance, and PyTorch adapters.
- Reuse of parent selections, enrichment results, and document token encodings.
- RegMix regression coverage and an application-owned end-to-end example.

### Separate future work

Do not add `query.prepare()`, a new public `DomainDataset` type, `mixture.mix()`,
`doremi()`/`mde()` algorithm APIs, or model-training/evaluation services. Do not
make `metadata=True`, adaptive reader weights, custom enrichment registration,
or virtual packed candidate reads a prerequisite for this feature. Those were
earlier design possibilities, not agreed API additions for this implementation.

Retain current `.torch()`/`.wait()` materialization behavior. This task makes
domain access reusable and inexpensive to plan; it does not promise that a new
packed ordering requires no output publication. Broadly replacing candidate
storage with shared-token virtual reads belongs in a separate change.

## 2. Public API and usage

```python
mixture = population.mix(
    domains=p.source.corpus_id,
    tokenizer=tokenizer,
    sequence_length=2048,
    splits=p.Splits(train=0.9, validation=0.05, test=0.05, seed=42),
)

for name, domain in mixture.domains.items():
    train = domain.train.torch()
    validation = domain.validation.torch()
    test = domain.test.torch()
    print(name, domain.id, domain.train.profile())
```

The value is a normal `Dataset`: `.train`, `.validation`, `.test`, `.profile()`,
`.preview()`, `.wait()`, indexing, iteration, `.torch()`, and dataset identity
behave as they do for existing datasets. An already selected split uses
`.torch()` directly; it does not expose another level of split selectors.

For DoReMi, the application constructs one loader per domain and manages finite
loader exhaustion, paired reference/proxy batches, loss weighting, and weight
averaging. For MDE, the application trains one expert per domain and reuses each
held-out loader for every expert. Existing PyTorch samplers can operate on the
map-style adapters.

RegMix remains unchanged:

```python
candidates = population.mix(
    domains=p.source.corpus_id,
    weights=p.RegMix(seed=42),
    n_candidates=4,
    tokens=100_000,
    replacement=True,
    tokenizer=tokenizer,
    sequence_length=2048,
)

domain_inputs = candidates.domains
for candidate in candidates:
    train = candidate.train.torch()
    # Training, evaluation, and regression are application code.
```

The application passes selected weights back through `population.mix(...)`,
including the same tokenizer, splits, and packing settings. Selecting weights
does not change the existing query API or require another preparation handle.

## 3. Exact semantic contract

### Population and membership

1. Use the mixture's full parent query output, after its filters, deduplication,
   decontamination, retained-text transformations, and query-level sampling.
2. Preserve query occurrences. If the parent query already contains repeated
   occurrences, domain datasets contain those occurrences too. Domain views add
   no new sampling, repetitions, or truncation.
3. Assign each selected document/occurrence to exactly one domain using the same
   labeling logic as candidate generation.
4. Include every domain actually present in that query, including domains with
   no training members, zero tokenizer tokens, or only held-out members. Do not
   manufacture absent classifier labels as empty domain datasets.
5. A query with no rows produces an empty mapping.

Domain datasets inherit the query, tokenizer asset and definition, sequence
length, packing policy, split policy, and execution revision. They ignore
candidate weights, token budgets, mixture draw seeds, proposal seeds,
replacement, exposure bounds, candidate count, and candidate slicing.

`mixture[0]` means a complete candidate recipe. `mixture.domains[key]` means the
full input population for that domain. Neither replaces the other.

Do not implement domain access by creating a one-hot sampled candidate. Sampling
only training would leave all domains in validation/test, and candidate budgets
would prevent access to the complete domain population.

### Splits and packing

Use `schemas/splits.py:content_split` unchanged. Split membership depends on
captured content and the split policy, not the domain name or candidate weights.
Identical captured content must stay in the same split across domains/corpora.

Filtering to a domain and applying that membership predicate must be equivalent
to taking that domain's members from the global split. Do not derive a new split
seed from the domain or resample a split to ensure that it is nonempty.

Pack independently inside each domain and each split. No domain dataset sequence
may contain content from another domain or split. Preserve separator, padding,
dropped-tail, attention-mask, and loss-mask rules. Empty splits behave exactly
like existing empty split views.

Domain sequences are not slices of a globally packed candidate: filtering
documents can change boundaries and ordinal numbering. Domain profile totals,
especially separators and dropped tails, need not sum to a candidate's totals.
Underlying document membership and content-split membership are the invariants.

### Keys and mapping behavior

- Return public corpus IDs for `p.source.corpus_id`, consistent with
  `mixture.weights` and `mixture.profile()`.
- Use exact source URI strings for `p.object.uri`.
- Use the current canonical label encoding for scalar fields, classifier labels,
  multi-field tuples, and explicit assignments. Reuse `_domain_key` and the
  existing public-ID conversion; do not invent a competing codec.
- Order keys deterministically, preferably by their public string values.
- Iteration yields keys; `.items()` yields `(str, Dataset)` pairs. Values are
  created lazily and cached per mapping/session. Do not instantiate all Python
  dataset handles merely to list keys.
- Assignment/deletion fail as for a read-only `Mapping`; an unknown key raises
  `KeyError` without materializing a dataset.
- Checking membership in the mapping must not create dataset handles or pack
  data. Override generic `Mapping` behavior where its default would call
  `__getitem__` unnecessarily.
- Candidate slices, including an empty slice, expose the same domain population
  as the original mixture. Share the mapping state with sliced handles.
- Domain access must not resolve RegMix proposals or validate unrelated candidate
  capacity/bounds. It is independently usable even if a later candidate request
  is infeasible. Mix-construction validation still applies as before.

### Laziness and session lifecycle

Obtaining `mixture.domains` returns a lightweight mapping without executing the
query, enrichment, tokenization, profiles, or packing. A short mapping `repr`
must also avoid execution.

The first operation requiring keys may execute the parent query and the
enrichment needed to label its population. It must not tokenize that population,
generate candidate weights, compute dataset profiles, or pack output. Creating
pending dataset recipe metadata is allowed.

`mapping[key]` resolves a pending ordinary dataset handle. `.profile()` and
`.preview()` perform the usual required work; a preview must remain bounded and
must not publish complete output. `.torch()` and `.wait()` retain the current
complete materialization behavior. Accessing one domain must not materialize its
siblings or candidate datasets.

Operations needing a live database respect existing closed-session errors.
Detached PyTorch adapters obtained while open retain their current lifetime
behavior after the database closes. Returning a cached mapping object does not
grant permission to perform new work after session closure.

## 4. Current implementation map

Inspect the checkout again before editing: this plan was written on October 6,
2026, in a working tree with unrelated edits, including storage/read-path work.
Preserve those changes. Source pointers below identify responsibilities, not a
guarantee that signatures will be unchanged in the next session.

| Source | Role in this change |
| --- | --- |
| `src/premixdb/api/mixture.py` | Property, lazy mapping, sliced-handle sharing |
| `src/premixdb/api/dataset.py` | Ordinary dataset/split behavior and recipe restoration |
| `src/premixdb/api/database.py` | Existing resource loading by ID |
| `src/premixdb/runtime/mixtures.py` | Independent domain discovery/recipe registration |
| `src/premixdb/runtime/datasets.py` | Domain pinning, packing input, profile and identity paths |
| `src/premixdb/runtime/split_datasets.py` | Domain selection before split-specific packing |
| `src/premixdb/runtime/coordinator.py` | Thin dispatch delegates and lifecycle |
| `src/premixdb/runtime/mixing.py` | Canonical domains, validation and public/internal labels |
| `src/premixdb/runtime/catalog.py` | Built-in enrichment pins |
| `src/premixdb/runtime/enrichment.py` | Reusable projections and enrichment builds |
| `src/premixdb/runtime/encodings.py` | Durable document token encodings |
| `src/premixdb/runtime/pipeline.py` | Partitioned tokenization/packing |
| `src/premixdb/engine/mixing.py` | `_domain_key` and existing proposal policies |
| `src/premixdb/engine/mixtures.py` | Current token pool, labels, inventory and draws |
| `src/premixdb/engine/dataset_plan.py` | Dataset identity and packing arithmetic |
| `src/premixdb/engine/concatenated.py` | Independent split packing and occurrence offsets |
| `src/premixdb/schemas/requests.py` | Recipe normalization and validation |
| `src/premixdb/schemas/messages.py` | Resource/request field copying |
| `src/premixdb/schemas/ids.py` | Public IDs and profile keys |
| `src/premixdb/storage/catalog.py` | Metadata-only reopening without runtime imports |
| `src/premixdb/storage/selections.py`, `storage/analytics.py` | Existing selected-population storage and restoration |
| `src/premixdb/storage/tokens.py` | Packed publication and provenance |
| `proto/premixdb/v1/data_mixture.proto` | Durable dataset recipe vocabulary |

Currently `_mix_pool()` discovers labels and constructs a token inventory over
training members. Calling it to enumerate domains would tokenize prematurely.
Extract shared label/pin resolution from that path; retain the existing sampling
semantics. Do not build a second labeling implementation beside it.

Follow `docs/architecture.md` and `tests/test_architecture.py`: engine, schemas,
storage and readers must not import API/runtime modules; runtime must not import
API modules. Public imports and read-only catalog reads must remain lightweight.

## 5. Recommended implementation design

### A. Separate domain membership from token inventory

Introduce an internal domain-population operation keyed by the parent query and
the canonical, pinned domain definition. It resolves labels over the complete
selected population in one pass and partitions occurrence ordinals by label.

Store compact membership references to the existing parent selection. Use the
existing bitmap/selection facilities where possible. Preserve occurrence
multiplicity: a canonical-document bitmap alone is insufficient for a query
with repeats. Do not copy document text, source descriptors, or a complete JSON
lineage map into every domain.

The partition manifest is independent of tokenizer, packing, splits and candidate
parameters. Its identity includes the parent query identity, domain selectors or
assignments, pinned enrichment identities, and the relevant versioned execution
definition. Persist verified schema/version, parent population binding, canonical
labels, membership references and occurrence counts. Publish completion only
after every reference is durable.

Reusing an unchanged query/domain definition across mixtures must reuse this
partition. An implementation may choose a more compact equivalent storage layout,
but it must meet the same identity, verification and read-only contracts.

### B. Represent domain selection in ordinary durable dataset recipes

Recommended approach: add an optional `DomainSelection` message to
`CreateDatasetRequest` and `Dataset` in `data_mixture.proto`. Its necessary
information is a verified domain-population reference and canonical domain label.
Existing `query_id` continues to identify the actual stored parent query.

The selector is applied to the packing input before split partitioning. Domain
datasets have no `Sampling` field. Reject requests combining this new domain-view
selector with sampling for this implementation; mixtures still sample from their
normal population. Do not accept an arbitrary label detached from its verified
parent population.

Use unused protobuf field numbers. Extend request/resource copying, normalization,
restoration and identity logic so the selector cannot be dropped on reopening.
Regenerate protobuf Python files and stubs through the existing generator; do not
edit generated files by hand. Avoid adding a new public request-builder argument
unless necessary for ordinary durable recipe construction.

The planner and engine must agree on the deterministic filtered-input identity.
A temporary in-memory filtered query can be a packing input, but its synthetic ID
must not masquerade as a public query resource that the catalog cannot load.
Preserve the real parent query for provenance publication and reopening.

If a different recipe representation is materially simpler, document it before
implementation and prove the same persistence and identity properties. Merely
attaching Python attributes to `Dataset` is not an acceptable substitute.

### C. Register and restore domain datasets independently of candidates

Resolve domain keys without calling `DataMixture._resolve()` or generating
candidate datasets. A separate runtime operation registers pending domain dataset
recipes and publishes a compact domain registry for the mixture's input/config.
Registering small pending recipes for all discovered keys is acceptable; creating
Python handles, profiles or packed output for all domains is not.

The registry may be paged if needed. Do not append all memberships or a large
domain map to the candidate `Mix` payload, and do not evade existing metadata size
limits. Cache the lightweight mapping on `DataMixture` and share it on slicing.

The same domain dataset ID must result from the same parent population, canonical
domain selection, tokenizer, packing, splits and execution revision regardless
of weights, budgets, proposal algorithm/seed, draw seed, replacement, bounds or
candidate count. A meaningful change to membership, tokenizer, packing or splits
must change its identity. When the new selector is absent, preserve existing
canonical recipe/identity rules rather than introducing a gratuitous format
change. The normal execution fingerprint can change when implementation code
changes; do not claim IDs stay identical across different runtime versions.

Domain datasets must be retrievable through the existing dataset catalog, not
only through the originating live mapping. Existing mixture resources should
gain domain discovery lazily in a writable session without forced recapture.

### D. Preserve ordinary data-plane behavior

Apply membership once through a shared packing-input path so previews, exact
profiles, local packing and partitioned packing cannot disagree. Preserve retained
source byte ranges, source corpus IDs, split sequence ranges, sequence ordinals,
token masks and source-document IDs.

Domain profiles should populate `planned_stratum_tokens` and `stratum_tokens`
with the selected domain key, including explicit zero values where applicable.
Counts reflect the unsampled domain input and emitted domain content respectively;
padding and separators remain separate. Ensure public corpus-ID conversion also
handles domain-selected datasets, which do not have a `Sampling` field.

Do not route a domain preview through a full training token pool. Use the existing
bounded preview machinery after the required global membership dependencies are
resolved. `.torch()` should remain a normal adapter and retain its current output
keys; callers know the domain from the mapping key. Mixed-candidate token-domain
metadata is outside this task.

### E. Read-only behavior and failures

Persist enough registry and recipe metadata that a reopened read-only session can
enumerate previously resolved domains and obtain their handles without executing
queries, enrichment or runtime imports. Completed domain datasets remain readable
after cache eviction and code/runtime changes, as existing completed data does.

If domain membership has never been resolved, read-only access requiring keys
raises a clear `ExecutionError` directing the caller to resolve it in a writable
session (for example, `list(mixture.domains)`). A known pending domain handle may
be returned, but its materialization still fails under the existing read-only
rules.

First-time domain resolution must honor the mixture's saved execution fingerprint
and enrichment pins. Do not bypass the existing changed-environment guard just
because candidate generation was skipped. Already completed registries and
datasets are frozen artifacts and should remain readable without recomputation.

Validate parent identity, membership coverage, labels, ordinals and artifact
digests. Missing/corrupt completed artifacts raise errors; do not silently
reclassify, rerun the query or publish a new partition. A genuinely unpublished
or failed build follows the existing explicit retry/single-flight behavior.

Concurrent mapping resolution must publish one logical registry. Avoid holding a
global mixture/cache lock across expensive query/enrichment execution. Failed
construction must not leave owned threads, encodings or temporary files alive.

## 6. Efficiency requirements and honest limits

Acceptance should use operation counters and small reproducible measurements,
not machine-dependent latency assertions:

- Resolving keys performs no tokenization, packing, candidate generation or model
  training/evaluation.
- Domain labeling is shared across profiles, candidate generation and domain
  access for equivalent populations; required enrichment inference runs once per
  unchanged pinned recipe/input under existing caching rules.
- Accessing several domains does not rescan and copy the parent population once
  per domain. Partition in one pass and retain compact membership references.
- Document model-token encodings are reused across domain datasets, candidate
  profiles and candidate materialization. Reopening should load durable encodings
  instead of invoking the tokenizer again for unchanged inputs.
- Test the local path and `process_workers` path. Do not assume local encoding
  caching proves that partitioned execution reuses the same work.
- Listing keys and creating handles never publish token output. Materializing one
  domain does not materialize another domain or candidate.
- Distinct packed domain/candidate orderings can still require distinct packed
  output objects. Report that cost accurately; this feature is not zero-copy
  packed storage or a production-scale throughput guarantee.

If reuse in the existing partition pipeline is inadequate, fix the narrow
encoding reuse path needed here and test it. Do not redesign all storage or add
an unrelated distributed inference system to finish this feature.

## 7. End-to-end application example

Add `examples/12_doremi_mde_domain_access.py` (adjust numbering only if occupied)
and a corresponding integration test. The example must exercise the new public
API, execute real small model computations, select external weights, and read a
final mixture. Printing domain names or running only mocked losses is insufficient.

Use a bounded synthetic corpus with two or three named source corpora, unique
document contents, and enough members for nonempty held-out splits. Default to
CPU, a short sequence length, fixed seeds, and a tiny causal language model. Keep
training/evaluation helpers in the example or an example-only helper module. No
network downloads, GPU, classifier model download, or package installation should
be needed. Use `p.ByteTokenizer()` for the runnable demonstration and separately
test durable model-tokenizer reuse with the existing WordPiece fixture.

Provide a main guard for DataLoader/spawn safety and configurable small step and
batch budgets. The integration test runs a deliberately smaller configuration.

### DoReMi branch

1. Create the full query and a pass-through mixture with explicit splits/packing.
2. Obtain all input loaders from `mixture.domains`, never private engine pools.
3. Train a tiny reference model using an application-owned balanced domain loop.
4. Train a fresh proxy while updating external domain weights from paired
   reference/proxy batches. Keep the reference frozen. Use causal shifted labels,
   unreduced token losses and valid-token masks; handle padding correctly.
5. Implement the documented small group-DRO/DoReMi-style rule in example code:
   clipped token excess losses, domain aggregation, normalized exponentiated
   weight updates, a stated exploration floor, weighted proxy objective and
   averaging of the weight history. Explain any simplification relative to the
   research implementation rather than claiming a complete paper reproduction.
6. Submit averaged weights through the original `query.mix(...)` settings, read
   the resulting training adapter, and verify its allocation/profile consistency.

This is an executable illustration of the data contract, not a benchmark showing
that a tiny proxy discovers scientifically optimal weights.

### MDE branch

1. Train one tiny expert on each `domain.train.torch()` using application code.
2. Evaluate every expert against every `domain.validation.torch()`. Reuse the
   exact same held-out adapters/order; handle causal masks and token denominators
   consistently. Store this expert-by-held-out-domain matrix in the example.
3. Construct a few RegMix candidates from the same query and obtain their domain
   handles. Assert that corresponding domain IDs match the original mixture.
4. Run a few bounded candidate proxy experiments externally, construct MDE-based
   features externally, and fit a small regression externally (NumPy ridge is
   sufficient; no new dependency is required).
5. Score a small grid of supplied weight vectors externally, choose a vector and
   submit it with `query.mix(weights=...)`. Read the final training data.
6. Keep test inputs out of weight selection. Any final test evaluation occurs
   only after selection, in application code.

Record a compact JSON report containing the parent query/mixture IDs, public
domain labels and dataset IDs, split identities or sequence signatures, seeds,
actual token counts, external weight histories/selected vectors, finite expert
scores, candidate IDs and final allocations. Reopening must reproduce recorded
domain datasets. Avoid storing all batches or model tensors in the report.

Do not assert that selected weights are nonuniform or beat a baseline after a
handful of updates. Assert real parameter updates, finite values, valid simplex
weights, reproducible data, and correct final allocations.

## 8. Test matrix

Add focused tests, preferably `tests/test_mixture_domains.py`, with independent
membership/split oracles rather than assertions that repeat implementation code.

### API and isolation

- Read-only mapping semantics, deterministic key order, unknown keys and empty
  populations. Membership tests and `repr` do not instantiate/execute values.
- The value and its splits have the advertised existing types and methods.
- Obtaining the property does no work; resolving keys does no tokenization,
  profiles, packing or RegMix proposal generation.
- Lookup returns pending handles; previews stay pending and bounded; one domain's
  `.wait()` does not materialize siblings or candidates.
- Full, sampled, zero-weight and sliced/empty-sliced mixtures share domain IDs.
- Unrelated candidate budget/capacity failures do not obstruct domain access.
- Invalid configuration and closed-session behavior follow existing contracts.

### Membership and splits

- Default corpus domains and public corpus-ID keys; URI domains; scalar/label
  projections; classifier namespaces; multiple field projections; explicit
  document assignments. Exercise null/typed labels according to current codec
  semantics and preserve candidate/domain key agreement.
- Assignment maps require exact coverage of the parent query, before splitting.
- Query filters, query-level repeats, retained text ranges and decontamination
  remain effective. Every domain contains exactly its expected occurrences.
- Domains containing only held-out documents, only empty text, or zero model
  tokens remain discoverable and usable with correct empty profiles/adapters.
- Captured-content duplicates across corpus IDs remain in the same split.
- No packed sequence crosses a domain or split boundary. Token provenance and
  retained byte offsets match selected source documents.
- Padding and dropped tails have independently computed expected totals. Do not
  equate separately packed domain sequence totals with global packing totals.

### Identity, persistence and failures

- Candidate weights/budget/seeds/replacement/bounds/count do not change domain IDs.
- Domain definition, parent selection, tokenizer, splits or packing changes do.
- Source corpus public/internal ID normalization is consistent in keys/profiles.
- Absent new selectors preserve existing canonical dataset identity behavior;
  compare recipes within one runtime and separately test cross-version reads.
- Reopen after cache eviction; reload domain datasets by recorded ID; enumerate
  a resolved mapping read-only without importing runtime/inference libraries.
- Previously unresolved mapping/pending data errors read-only without writes.
- Missing, corrupt, misbound or incomplete membership/recipe/token objects fail
  explicitly. Concurrent resolution and retry do not duplicate logical outputs.

### Reuse and reader behavior

- Controlled enrichment producer counters show unchanged outputs are reused when
  accessing domains, changing candidate weights and reopening. Keep real large
  enrichment inference out of fast tests.
- Actual tokenizer encode counters show durable reuse, including restored handles
  and partitioned execution where supported. Distinguish encoding inference from
  loading encodings or repacking already encoded tokens.
- Batched map-style reads, streaming reads, rank/worker partitions and detached
  adapter lifetime match ordinary datasets. Respect current uneven-rank guidance.
- Domain profiles agree with emitted sequences and published geometry.
- Existing RegMix proposals, candidate IDs, budgets, splits and ablation behavior
  remain correct.
- The example integration test runs both external experiment branches, consumes
  all required domains, checks real model updates, reads the final mixtures and
  verifies reproducible reopening.

## 9. Implementation sequence and validation

1. Inspect the current checkout, applicable `AGENTS.md` instructions and uncommitted
   changes. Read this document and the current mixture/dataset/selection code.
2. Define the shared domain-population identity/label contract and add independent
   tests for full-query membership and split behavior.
3. Add versioned durable membership and domain-selection recipe support. Implement
   normalization, validation, identity, registration and metadata-only restoration.
4. Refactor existing mixture labeling to share that operation without changing
   proposal generation or training-only sampling semantics.
5. Integrate selection with preview, profile, local/partitioned packing and token
   publication. Confirm encoding reuse and provenance invariants.
6. Add the lazy API mapping and sliced-handle sharing. Verify read-only and cache
   eviction behavior before declaring the public API complete.
7. Add the runnable application example and its real integration test. Update
   `docs/training.md`, `docs/curation.md`, and `examples/README.md` with concise
   usage and the training/evaluation ownership boundary. Correct contradictory
   documentation encountered in touched sections rather than copying stale text.
8. Run focused tests, relevant existing checks, the example, and the repository's
   required full checks. Fix failures caused by the implementation; preserve
   unrelated working-tree edits.

Typical validation commands (adjust new filenames if necessary):

```sh
uv run --locked pytest tests/test_mixture_domains.py tests/test_mix.py tests/test_dataset_splits.py tests/test_mixture_pool.py tests/test_training_reads.py tests/test_token_publication.py tests/test_architecture.py
uv run --locked pytest tests/test_doremi_mde_domain_access.py -m integration -n 0
uv run --locked python examples/12_doremi_mde_domain_access.py
make check
```

Keep fixtures small and checks explicit. Do not introduce pre-commit/pre-push
hooks. If a failure belongs to unrelated concurrent edits, identify it with
evidence and still complete the unaffected checks. Report actual verification,
not projected benchmark results.

## 10. Completion criteria

The feature is complete when an external application can discover the complete
domain populations of an ordinary or RegMix mixture, obtain ordinary domain
datasets and splits, reuse their prepared inputs in both example workflows, pass
externally selected weights to `query.mix`, and reopen the resulting resources.

No model-training/evaluation/mixture-optimization implementation has entered the
library. Domain discovery is independent of candidate generation and tokenization.
Persistence, provenance, split correctness, reader behavior, encoding reuse and
RegMix compatibility have automated coverage. The real example and required checks
pass, and the final implementation report states any remaining packing/storage
costs or unverified scale assumptions.
