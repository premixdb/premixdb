# Optional synthesis and strategy mixture optimization

Implementation plan, prepared October 7, 2026 against checkout `2a4bd21`.
This document is a design and implementation handoff. The APIs and schemas below
are proposed; synthesis is not implemented in this checkout. Recheck the owning
code before editing because unrelated work may have landed since this plan.

## 1. Objective and agreed product behavior

Add an optional `Query.synthesize()` step that accepts a list of built-in strategy
presets, such as `strategies=[p.SynthesisStrategy.FINEPHRASE_FAQ, p.SynthesisStrategy.FINEPHRASE_TUTORIAL]`.
Each strategy generates a separate synthetic inventory from the same selected
input. Return a lazy, mixable `SynthesisPool` containing the original training
selection and one source per strategy. Use the existing `.mix()` machinery to
propose different token allocations across these sources.

The intended workflow is:

```text
capture -> query/select -> [optional synthesis pool] -> mix candidates -> train/evaluate
                                      |
                     original + faq + tutorial + table + math
```

The user decisions this plan implements are:

- The method is `.synthesize()`, rather than `.augment()`.
- Synthesis is an optional step, separate from arguments to `.mix()`.
- The method takes a list of strategies, rather than one implicit generator.
- Strategies are alternative generation branches whose mixture proportions can
  be explored and optimized.
- Synthetic inventory is generated once and reused across mixture candidates.

The proposed first-release recipe set is FinePhrase's document-grounded structured
rewriting. This is a scope recommendation for fixed reusable corpora, not a claim
that FinePhrase is the best synthesis method overall. Section 14 compares
FineInstructions, SynPro, and RePro and explains their different requirements.
Users pass enum members directly, without constructors or a public `Generator`;
one shared teacher configuration provides a small advanced override for this
first recipe family.

Do not turn the list into a sequence of transformations. `faq` and `tutorial`
both consume the selected original training input; `tutorial` does not consume
the output of `faq`. Preserve a clean ordinary path: `query.mix(...)` must keep
its existing behavior and identities.

This feature supplies ingredients and candidate recipes for optimization.
Current `RegMix` proposes weight vectors; it does not train proxy models, fit a
predictor, or automatically choose a winning mixture. Keep that distinction
explicit in the API and documentation.

## 2. Scope and defaults

Implement these capabilities in the first release:

1. Enum-selected FinePhrase recipes: `SynthesisStrategy.FINEPHRASE_FAQ`,
   `FINEPHRASE_TUTORIAL`, `FINEPHRASE_TABLE`, and `FINEPHRASE_MATH`.
2. A thin adapter over DataTrove inference on supported runtimes, using the
   framework already adopted by FinePhrase and PremixDB. A small Transformers
   adapter supports local CPU/Intel-mac execution and offline model tests.
   Load/download model assets only on execution.
3. Lazy pool planning, durable generated text, resumable task receipts, and
   immutable completed outcomes.
4. Strategy labels as a real mixture dimension, including fixed weights,
   RegMix proposals, zero weights, and existing `Bounds` constraints.
5. Training-only generation and fixed original validation/test selections.
6. Query-based curation of the generated training population, preserving pool
   identity, strategy labels, and evaluation context.
7. Offline deterministic tests, an offline example, public typing, and packaging.

Defer multi-document discovery, entity graphs, online adaptive generation,
recursive synthesis, remote paid providers, reinforcement-learning rephrasers,
automatic experiment ranking, generation-budget optimization, custom prompts,
per-branch model settings, textbook curricula, and user-defined output parsers.
Leave the artifact and worker contracts capable of recording multiple parents, but reject
unsupported public strategy kinds rather than accepting inert configuration.

Do not add arbitrary Python callables or pickle function objects to saved
recipes. Keep extension points internal: versioned recipe assets and a generation
protocol. Add another public preset only when its research basis and validation
justify it. Additional backends can later implement the same internal protocol.

Use no universal synthetic-data ratio. Preserve existing natural-token-prior
behavior when weights are absent, and require explicit weights or RegMix when
the user wants another composition.

### 2.1 Nonnegotiable invariants

Use these as review gates throughout implementation:

- Selected original identities and retained text survive synthesis unchanged.
- Generated text is always a new captured document with explicit derivation.
- Training-only generation and original evaluation membership are persisted rules.
- No query, snapshot, mix, or reopen path can silently detach those rules.
- A completed pool binds one verified outcome; training never regenerates it.
- Strategy inventory is independent of candidate weights and the training tokenizer.
- Pool planning is bounded metadata work; execution triggers are documented.
- Ordinary queries/mixes and previously stored resources retain their semantics.

## 3. Proposed public API

### 3.1 Enum choices and a bounded generation recipe

Use public string enums following the existing `Quality`, `EmbeddingModel`,
and `ExecutionStatus` pattern in this checkout. Export `SynthesisStrategy`,
`SynthesisModel`, and `SynthesisSource` from `premixdb`. Enum values are stable
identifiers; resolved versioned recipes contain the actual prompt/model assets.
Do not encode a complete mutable recipe into an enum value.

```python
from enum import Enum

class SynthesisStrategy(str, Enum):
    FINEPHRASE_FAQ = "finephrase_faq"
    FINEPHRASE_TUTORIAL = "finephrase_tutorial"
    FINEPHRASE_TABLE = "finephrase_table"
    FINEPHRASE_MATH = "finephrase_math"

class SynthesisModel(str, Enum):
    SMOLLM2_1_7B_INSTRUCT = "smollm2_1_7b_instruct"

class SynthesisSource(str, Enum):
    ORIGINAL = "original"
    FAQ = "faq"
    TUTORIAL = "tutorial"
    TABLE = "table"
    MATH = "math"
```

Do not add speculative members for unimplemented families. In particular,
`ORIGINAL` is a source member, not a strategy. Enum-selected defaults resolve to
bundled pinned recipes without user constructors or a `p.Generator` object.
Convert selections into detached protobuf definitions at the request boundary.
The following sketch is proposed, not runnable against the current checkout:

```python
import premixdb as p

pool = query.synthesize(
    strategies=[
        p.SynthesisStrategy.FINEPHRASE_FAQ,
        p.SynthesisStrategy.FINEPHRASE_TUTORIAL,
        p.SynthesisStrategy.FINEPHRASE_TABLE,
        p.SynthesisStrategy.FINEPHRASE_MATH,
    ],
    max_documents=10_000,
    splits=p.Splits(train=0.8, validation=0.1, test=0.1, seed=42),
    seed=7,
)

mixture = pool.mix(
    weights=p.RegMix(seed=42),
    n_candidates=16,
    tokens=100_000_000,
    replacement=True,
    bounds=p.Bounds(lower={p.SynthesisSource.ORIGINAL: 0.2}, max_epochs=4),
    seed=123,
)

mixture.preview()
dataset = mixture[0]
training = dataset.train.torch()
validation = dataset.validation.torch()
```

The minimal call is valid with no extra configuration:

```python
pool = query.synthesize(
    strategies=[p.SynthesisStrategy.FINEPHRASE_FAQ, p.SynthesisStrategy.FINEPHRASE_TUTORIAL],
)
```

Recommended signature:

```python
def synthesize(
    self,
    *,
    strategies: Sequence[SynthesisStrategy],
    max_documents: int = 10_000,
    max_tokens: int | None = None,
    model: SynthesisModel | Path = SynthesisModel.SMOLLM2_1_7B_INSTRUCT,
    splits: Splits | None = None,
    seed: int = 0,
    run_key: str = "",
) -> SynthesisPool:
    ...
```

`max_documents` and optional `max_tokens` are ceilings **per strategy**, not
training budgets or a shared race between strategies. The 10,000-document default
is a conservative PremixDB execution policy, not a published FinePhrase setting.
The finite input population, one output attempt per unique input view/preset,
per-call output limit, and finite retry count bound work when `max_tokens` is
omitted. Do not silently change a ceiling into a target to fill using unrelated
prompts. A document ceiling caps accepted output; rejected tasks still cost work.

`max_tokens` counts accepted completion token IDs under the pinned generator
tokenizer. It excludes prompts and is distinct from `.mix(tokens=...)`, which
counts training content under the training tokenizer. Record actual input and
completion work separately, including rejected outputs and retries.

The default `model=p.SynthesisModel.SMOLLM2_1_7B_INSTRUCT` resolves a bundled,
versioned recipe using `HuggingFaceTB/SmolLM2-1.7B-Instruct`. Bundle verified
immutable model/tokenizer revisions; enum selection must not resolve a moving
branch or download weights. The only advanced override shared by all branches
is `model=Path(...)` for an already-local checkpoint. Paths are open resources,
not enum choices. Do not accept arbitrary model-name strings in this release.
This keeps an offline local-checkpoint example possible without a public
generator-builder API. A local directory is pinned by a verified asset manifest
rather than its path string; hashing that manifest is permitted during planning, loading weights
and scanning input text are not. An override is labeled as a modified recipe,
not as a reproduction of the paper's default generator results.

Do not expose `p.Generator`, per-strategy models, prompt strings, output-format
lists, temperature/top-p knobs, or variant counts in the first release. The
internal resolved generator definition still records model/tokenizer/template
pins, context/output limits, and decoding parameters. `run_key` and `seed` allow
another bounded realization without adding an output-variant API. Retain the
same key on retries; it is separate from request idempotency keys.

Execution placement, worker count, queue capacity, and progress preferences
belong to executor configuration. Record actual backend, device, dtype, and
runtime versions in execution receipts. Include settings in a realization's
fingerprint when they can change generated text; a seed alone does not guarantee
identical generations across hardware or kernels.

### 3.2 Strategy semantics and upstream recipe adoption

Treat a strategy as a versioned research recipe, not inherently as one prompt or
one inference call. The four initial presets happen to share a teacher and have
one output per input. Keep these as explicit first-release restrictions; do not
generalize them into an architecture that rules out retrieval, verification, or
several outputs in a later recipe family. Do not expose unimplemented presets.

Each preset has a fixed source label and versioned prompt recipe:

| Public value | Source label | Intended transformation |
| --- | --- | --- |
| `p.SynthesisStrategy.FINEPHRASE_FAQ` | `faq` | Present source information as standalone questions and answers. |
| `p.SynthesisStrategy.FINEPHRASE_TUTORIAL` | `tutorial` | Present source information as an instructional sequence. |
| `p.SynthesisStrategy.FINEPHRASE_TABLE` | `table` | Organize source information in a table followed by a question and answer. |
| `p.SynthesisStrategy.FINEPHRASE_MATH` | `math` | Derive a numerical word problem and worked solution from source relationships. |

All selected presets consume the same original training views in parallel. Each
unique selected view/preset combination is one task, with at most one captured
output document. Several QA pairs in a FAQ remain one document. A task can have
zero accepted outputs; persist terminal rejections so resume does not retry them
indefinitely. A math prompt is not proof of arithmetic correctness or source
faithfulness; evaluate those properties in the optional research harness.

Reject empty lists, duplicate presets, accidental strings/mappings/classes,
unknown enum members or recipe versions, and user attempts to rename a branch.
`original` is reserved and not a selectable generation preset. Copy caller lists
and mutable protobuf fields on entry. Preserve supplied order for presentation;
canonicalize branches by preset ID for identity, scheduling, and output ordering.
Reordering presets or adding another branch must not perturb existing tasks.

There are at most four selected branches in this release. Bound serialized
request size and require positive integer ceilings when supplied. Bound derived
task count after restoring the selected input, before model loading/inference;
it cannot always be known when registering a lazy recipe. Document actual limits.
Invalid policies must fail even when the input is empty.

Require actual `SynthesisStrategy` and `SynthesisModel` members at the fluent
SDK boundary; do not silently accept arbitrary strings, integers, another enum
class, or unknown values. Raw protobuf requests use typed enum fields with stable
explicit numeric mappings and an `UNSPECIFIED=0` value that cannot execute.
Never derive wire numbers from declaration order or persist Python repr/name.
Adding recipe versions does not change an existing member's semantic identity;
resolved recipe pins distinguish outcomes across versions.

Use `SynthesisSource` for `pool.source(...)`, source descriptors, previews, and
strategy-domain weight/bound keys. Normalize keys to `.value` before invoking
existing label-based allocation/serialization. Retain ordinary string keys for
existing open-domain mixtures and tuple labels. Do not coerce unrelated enums.
For synthesis-only source selectors require the correct enum type; diagnostic
messages may display the stable strings. Confirm enum-key and equivalent
canonical-string mixture recipes have equal identities. Source enum values
also provide the typed projection for `p.synthesis.strategy`.

Reuse `ExecutionStatus` for pool status. Use internal protobuf enums for finite
finish/rejection/exhaustion reasons and backend kinds, with explicit validation
of provider strings. Keep free text only for details. Budgets, paths, IDs, seeds,
and open-domain labels remain ordinary typed values. Avoid a public backend enum
or generator-tuning knobs unless execution placement actually needs them.

Adopt the published [FinePhrase production recipe](https://github.com/huggingface/datatrove/blob/main/examples/inference/finephrase.py)
and [research repository](https://github.com/huggingface/finephrase) as the source
of recipe assets. Pin upstream commits, retain required attribution/license
metadata, and package the four templates and resolved configuration. Do not
reconstruct prompts from their names or retrieve templates at execution time.
The inspected production script specifies an 8,192-token model context and
2,048-token completion limit. Its [generation driver](https://github.com/huggingface/datatrove/blob/main/examples/inference/generate_data.py)
loads decoding defaults from model assets and truncates long rendered prompts
using a character-budget approximation. Resolve those defaults from the pinned
assets and verify chat templating before freezing recipe version 1. Record
differences between the research experiments and production script.

Prefer the reusable framework dependency over reimplementing the paper's
inference infrastructure. FinePhrase itself imports DataTrove's
`InferenceConfig`/`InferenceRunner`. PremixDB already pins DataTrove 0.10.1 on
most platforms and 0.3.0 on Intel macOS. The inspected 0.10.1 source contains
[inference runner APIs](https://raw.githubusercontent.com/huggingface/datatrove/v0.10.1/src/datatrove/pipeline/inference/run_inference.py);
the locally installed 0.3.0 distribution has no inference package. Verify the
complete released dependency and extra requirements before changing the lockfile.

Use a thin adapter over released DataTrove inference for supported generation
runtimes. Delegate server management, concurrent requests, and supported request
caching to it. PremixDB still owns training-input selection, split safety, stable
task identity, final outcome publication, lineage, and mixing. Make the adapter
bridge DataTrove document IDs/output metadata to PremixDB task receipts. Bound
admitted work, preserve per-task seed semantics, and make upstream failures
observable; do not assume upstream caching alone satisfies pool immutability.

Keep the small local Transformers adapter for CPU/Intel macOS and tiny-checkpoint
validation. Resolve the backend through executor capabilities, record it in the
recipe/realization fingerprint before inference, and never silently substitute
backends after failure. Do not expose DataTrove rollout callbacks as user strategy
configuration. Supported backend imports are lazy and optional runtime support
must not break ordinary PremixDB imports on Intel macOS.

Do not add the entire `finephrase` package to PremixDB's base environment.
Its inspected [dependency manifest](https://github.com/huggingface/finephrase/blob/main/pyproject.toml)
pins `datasets==3.5.1` and `tokenizers==0.21.0`, conflicting with this checkout's
`datasets>=5.0.1` and tokenizers 0.22/0.23 platform ranges, and includes CUDA
inference/training packages. Reusing DataTrove plus pinned recipe assets is the
recommended dependency boundary. An exact research-stack reproduction can run in
an isolated worker environment; do not force its training stack into the SDK.

Add a recipe-parity gate: assert the packaged templates/configuration match the
pinned upstream assets, and compare tokenizer/chat-template rendering on small
fixtures. Audit every output rejection rule as a PremixDB deviation when absent
upstream. Rejecting malformed output must not become a rigid format-adherence
filter that silently changes the recipe. Research performance and H100
throughput remain unverified until the corresponding experiments actually run.

### 3.3 Pool and query surface

Use a new `SynthesisPool` resource rather than calling the pool a `DataMixture`.
It is a collection of training ingredients; a `DataMixture` remains a collection
of alternative complete dataset recipes.

Minimum pool surface:

| Operation | Behavior |
| --- | --- |
| `id`, `status`, bounded `repr` | Read saved plan/status metadata; no generation. |
| `sources` | Return immutable descriptors for `original` and each strategy; no generation. |
| `original` | Return the full selected input `Query`, including its eventual holdouts. |
| `source(SynthesisSource)` | Return a lazy training query for one branch, retaining evaluation context. |
| `query(...)` | Return a lazy query over original training rows plus synthetic rows. |
| `mix(...)` | Shortcut for `pool.query().mix(...)`, defaulting domains to strategy. |
| `wait(timeout=...)` | Materialize the pool and return the completed handle. |
| `profile()` | Materialize if necessary; report counts and work per branch. |
| `preview(source=..., limit=3, ...)` | Browse one source with bounded text; see execution policy below. |

`pool.query()` and `pool.source()` operate on **training rows**. The original
validation/test population travels as pinned evaluation context, not as rows
that a post-synthesis training filter can remove. Display that scope in their
representations and documentation. `pool.original` remains the way to inspect
the complete selected original population.

Implement these views as `SynthesisQuery` handles reusing ordinary query
operations and execution machinery, with a synthesis-backed input variant in
the query resource. Avoid an ordinary Python wrapper that loses its context
when `.mix()`, `_with_fields()`, or reopening is used.

Support curation before mixing:

```python
curated = pool.query(
    steps=[
        p.where(p.text.characters > 100),
        p.dedupe(),
    ],
)
mixture = curated.mix(
    weights={
        p.SynthesisSource.ORIGINAL: 0.6,
        p.SynthesisSource.FAQ: 0.2,
        p.SynthesisSource.TUTORIAL: 0.1,
        p.SynthesisSource.TABLE: 0.1,
        p.SynthesisSource.MATH: 0.0,
    },
    tokens=1_000_000,
    replacement=True,
)
```

Post-synthesis curation must read the retained selected input and generated text,
not replace the original selection with its whole source snapshots. New quality,
language, topic, and embedding fields are computed from generated text. Do not
inherit parent scores as if the generated output had been scored.

For the first release, `preview()` on an unresolved synthetic source materializes
the bounded pool before browsing it. Document this plainly: output-window limits
bound display, not generation work. Original-source previews may read only the
pinned training original selection. Partial generation previews that leave a
pool pending are a later optimization, not a requirement for release.

Reopen resources through `db.Synthesis(pool_id)` and synthesis-backed queries
through `db.Query(query_id)`, adding a public query reopen collection if needed.
The current query reopen helper is private (`db._query`). These are narrow typed
resource lookups, not a new generic string-based resource API. Completed pools
must reopen in read-only sessions with no generator model installed.

Do not expose `.synthesize()` again on a `SynthesisQuery` in the first release.
Recursive generation needs an explicit ancestry/depth contract and is deferred.

## 4. Split contract: implement before generation

The current `content_split()` hashes captured-content digests with `Splits.seed`.
`RetainedDocument.content` preserves the captured digest even when a query has
removed spans. Preserve that rule for original documents; use the selected,
retained text as the actual generation input.

A synthetic document has different content and cannot safely be assigned using
its own content hash. Assign splits to original content groups first, and pass
only training parents to generation.

At synthesis execution:

1. Execute/restore the original selection once, preserving IDs, occurrence order,
   captured-content groups, retained UTF-8 ranges, and existing provenance.
2. Resolve the supplied split policy, using existing explicit 80/10/10 defaults
   when omitted. Save the policy in the pool recipe before execution.
3. Persist membership receipts for original train/validation/test selections.
4. Generate only from distinct selected training document views. If query sampling
   repeated an occurrence, retain its multiplicity in the original training
   branch but synthesize once per unique `(document ID, retained-view digest)`.
5. Mark generated documents as training-only by their pool provenance. Never
   rehash them into validation/test later.
6. Construct every candidate's validation and test from the original membership
   receipts. Strategy weights, training sampling seeds, output filtering, and
   synthesis seeds must not change those selections.

`pool.mix(splits=None)` inherits the pool's pinned policy. An explicitly equal
policy is accepted; a different policy fails before generation or tokenization.
The caller must create another pool to change split proportions or split seed.
Checking only proportions is insufficient; seed is part of the policy.

Keep validation/test text, document order, profiles, and packed token sequences
identical across candidates when the training tokenizer and packing policy are
the same. Holdout tokens are outside `.mix(tokens=...)`, as in the existing API.
Packing still happens independently within each split.

The baseline `query.mix(splits=policy, ...)` and the pool's `original`-only training
branch must have identical original membership and retained text. If their
training occurrence order and packing settings are also equal, compare exact
token sequences as well. Do not compare sampled recipes with different order
policies and expect identical packed sequence geometry.

An all-training policy is supported. A policy with no training parents produces
completed empty synthetic branches without loading a model. Invalid strategies
still fail before this shortcut. Record zero-capacity branches explicitly.

Reject generated text whose exact content digest matches any original
validation/test document, including the digest of a retained holdout view.
This prevents an output that reproduces a heldout document from entering train.
Use existing explicit decontamination operations for stronger n-gram/reference
checks; neither parent disjointness nor exact matching guarantees freedom from
semantic contamination or teacher-model memorization.

For future multi-parent strategies, require every parent in a training task to
be a training member. Store all parent IDs. Do not create cross-split connected
components and repartition evaluation after generating data.

## 5. Mixture dimensions and optimization

### 5.1 Strategy is an actual domain

Add intrinsic metadata projection `p.synthesis.strategy` with values `original`
and the declared strategy names. It requires no model inference. Store labels in
bounded artifact metadata, not as a giant per-document assignment map in every
mix request.

Synthesis-backed `.mix()` defaults to this projection. Ordinary `Query.mix()`
continues defaulting to `p.source.corpus_id`. Validate labels directly; do not
decode `faq` or `original` as corpus IDs, and do not pretend a user label is an
immutable source identity.

Support explicit alternative domains using existing field projections, including
tuples such as `[p.synthesis.strategy, p.topic.label]`. Such tuples optimize the
joint distribution; they do not imply a new hierarchical factorization algorithm.
Preserve the existing canonical tuple-key encoding. A single strategy projection
uses plain names in weights, bounds, and public profiles.

`p.source.corpus_id` retains its normal meaning. Generated branches have real
derived corpus identities; parent corpus identities remain in lineage. Grouping
original input from several corpora under `original` is deliberate. Independent
optimization of original source domains remains available through explicit
domains; a special parent-corpus projection can be a follow-up feature.

### 5.2 Reuse the existing sampler

Use `engine.mixing.allocations`, `runtime.mixing.generate`, `MixturePool`, RegMix,
and existing packing. Supply a synthesis-aware training population and labels;
do not build another allocation/proposal implementation.

- Fixed mappings cover the declared domains, including explicit zero weights.
- Preserve empty strategy domains with zero inventory after curation.
- Positive fixed weight for an empty domain fails clearly. RegMix forces empty
  domains to zero and validates the remaining bounds/capacity constraints.
- If all training domains are empty, preserve the ordinary empty pass-through
  result when no positive training budget is requested; reject impossible
  positive allocations.
- A zero weight allocates exactly zero content tokens; it does not by itself
  avoid prior inventory generation.
- With no weights/budget, retain every available training occurrence in a stable
  order. With only a budget, use natural training-token proportions.
- Preserve current without-replacement capacity checks and repeated-pass
  replacement behavior. Report actual capacity under the training tokenizer.
- A default full-inventory budget without replacement leaves little freedom for
  RegMix. Examples exploring proportions must use a smaller budget or explicit
  replacement, as documented in [training.md](training.md).

For release simplicity, candidate resolution materializes every declared branch
before computing inventory, even when some fixed weights are zero. An
original-only baseline can use the ordinary query without synthesis. Selective
branch generation is a later feature requiring a different completeness contract.

Inventories may contain repeated text across strategies. Their identities and
generation provenance remain distinct. Do not automatically erase a branch by
globally deduplicating content at publication. Users can apply cross-branch
dedupe as a query operation; make survivor ordering deterministic and document
that it can change branch capacities. Handle duplicates within the same branch
using a deterministic acceptance rule and recorded rejection outcomes.

### 5.3 Validation and experiment records

Candidate IDs and profiles must retain the pool ID, outcome digest, strategy
definitions, final weight vector, train-token budget, sampling seed, split policy,
training tokenizer, and packing policy. A training script should log those along
with model initialization seed, optimizer, training compute, validation metric,
and chosen candidate. Generation work is accounted for separately.

RegMix's token prior is inventory-dependent. Fix generated inventory before
comparing candidate sets. Changing the generation recipe or its budget creates
a new experiment ingredient; it is not a weight-only ablation.

## 6. Resource and protobuf design

### 6.1 Public metadata

Add `proto/premixdb/v1/synthesis.proto` containing typed metadata equivalent to:

| Message | Required information |
| --- | --- |
| `Generator` (resolved metadata only) | Pinned model/tokenizer/template references and decoding semantics; not an exported Python policy. |
| `SynthesisStrategyDefinition` | Typed strategy enum plus recipe version; typed source enum; resolved prompt/template/parser and shared generator references. |
| `CreateSynthesisRequest` | Original query ID; canonical strategies; per-strategy ceilings; splits; seed; run key; git/runtime pin; request idempotency key. |
| `SynthesisPool` | Plan ID; original query ID; resolved recipe; status/error; source descriptors; completed outcome reference/digest; bounded summary. |
| `SynthesisSourceDescriptor` | Typed source enum plus stable domain label; definition ID; corpus ID where applicable; output snapshot/artifact ID after completion; counts. |
| `SynthesisProfile` | Per-source inventory, work, accepted/rejected counts, exhaustion reasons, and completion/cache information. |
| Create/Get/List request/response messages | Existing typed resource naming and parent-query listing conventions. |
| `SynthesisQueryInput` | Pool ID plus optional typed source selector, rather than embedded generated rows. |
| `SynthesisContext` | Pool ID and pinned splits; completed outcome digest when resolved. |

There is no service `.proto` in the current checkout. Implement these as typed
Coordinator/Catalog methods, matching existing Create/Get resource methods.
Do not add a gRPC server as part of synthesis.

Keep protobuf imports acyclic. `Splits` currently lives in
`data_mixture.proto`, which imports `query.proto`. Having `query.proto` import
`synthesis.proto` while synthesis imports data mixtures for `Splits` would create
a cycle. Extract `Splits` into a small shared `splits.proto`, preserving its
package, full message name, field numbers, and wire representation. Import it
publicly from `data_mixture.proto` so existing generated-module exports and
`p.Splits` remain available. Let synthesis depend on shared splits/storage/status
metadata, query depend on synthesis input metadata, and data mixtures depend on
query/synthesis/shared splits. Synthesis messages refer to queries by ID and
must not import their resource messages. Verify generated Python and `.pyi`
reexports and old split-message bytes explicitly.

Extend `Query` and `CreateQueryRequest` with a synthesis-backed input variant.
Require exactly one input mode: existing nonempty `snapshot_ids`, or
`SynthesisQueryInput`. A recipe cannot secretly combine both modes. Keep existing
query operations, sampling, decontamination, and field projections reusable.

Extend `Mix`, `CreateMixRequest`, `Dataset`, and `CreateDatasetRequest` with
optional `SynthesisContext`, using fresh field numbers. The authoritative context
is inherited from the query input; reject a forged, mismatched, or manually
detached context. Do not depend on transient Python attributes for holdout rules.

Mixture registration stays lazy. At candidate resolution, materialize the pool,
freeze the outcome digest in concrete dataset recipes, and then compute their
IDs. Synthesis-backed pass-through candidate resolution needs this materialization
too; ordinary pass-through indexing remains lazy as before.

Use explicit protobuf presence for optional budgets. Reject unknown fields,
invalid oneofs, IDs of incorrect length, unsupported strategy versions, and
policy inconsistencies at SDK and raw-request boundaries.

Do not fill `query.snapshot_ids` with execution-discovered snapshots and then
reinterpret them as the original recipe. Store resolved population references
and selection constraints in a separate execution receipt. Resolve enrichment
and index pins against that population at materialization time and save those
pins in the receipt; an unresolved synthesis plan cannot derive them from empty
snapshot IDs. The logical query plan retains its input mode and identity. Do not
change ordinary query identity encoding while adding this execution path.

### 6.2 Internal data-plane artifacts

Add private messages in `proto/premixdb/internal/synthesis.proto` for:

- Original selection/membership receipts with split, occurrence order, and
  retained-view identity. Reference existing saved selection shards where possible.
- Generation task definitions and input descriptors with parent IDs, retained
  ranges/view digests, strategy definition ID, internal format/output ordinal, and derived seed.
- Per-task output receipts with accepted text object references, completion token
  counts, terminal rejection reasons, attempts, and execution fingerprint.
- Branch manifests with ordered tasks/output descriptors and text/snapshot refs.
- Generated-document lineage rows and a bounded shard index.
- Pool outcome manifest binding all branches and original membership receipts.

Extend `internal/transport.proto` only if generation uses `PartitionPipeline`.
Add a typed synthesis request/result oneof and strict boundary validation in
`runtime/partition_types.py`; never put arbitrary object payloads into existing
feature messages. Keep document text in bounded worker messages or object refs,
not in catalog metadata.

## 7. Identity, caching, and durable publication

### 7.1 Separate intent from outcome

Use versioned canonical encodings consistent with `engine/identity.py`.
Do not use Python object hashes, repr strings, mutable model aliases, file paths,
or physical shard boundaries as logical identities.

Define and document these identities:

1. **Strategy definition ID:** preset ID/version, full prompt/template/parser
   definitions, pinned generator/tokenizer/template, and decoding parameters.
   The fixed source label belongs to the allowlisted preset definition.
2. **Pool plan ID:** selected input query ID, named strategy definitions, ceilings,
   split policy, synthesis seed, run key, and execution/code identity.
3. **Task/cache ID:** parent IDs and exact selected-view digests/ranges, semantic
   strategy ID, internal output ordinal, derived seed, and realization fingerprint.
   Do not include worker index or completion order.
4. **Generated document ID:** use the existing captured-document identity codec
   with a derived corpus ID, stable task/output source key, and actual text.
   Generated source keys include task ID and output ordinal, not a temporary path.
5. **Branch/pool outcome digest:** ordered logical receipts and actual output
   content digests, independent of physical sharding and timestamps.

Derive a task's random seed from the synthesis seed, semantic strategy definition,
parent view and internal output ordinal. Adding another strategy must not perturb the
existing strategy's seeds or generated tasks. Per-task inference randomness must
be isolated; shared global RNG state across concurrent jobs is unacceptable.

A seed identifies a requested realization, not a promise of bit-identical model
inference across environments. Persist actual outputs. The first fully committed
outcome binds the plan; all candidates and reopen operations use it. Never
silently regenerate a missing/corrupt published shard under an already completed
plan. Raise an integrity error and require explicit repair or a new run key.

### 7.2 Resume and publication

Use existing single-flight admission and metadata transactions. In-memory
single-flight only deduplicates within one coordinator; add an owned durable
claim/lock for branch/task publication when separate sessions or processes share
storage. Preserve a consistent lock order and release claims on failure.

Execution order:

1. Save a validated pool recipe as pending, with no source scan or model load.
2. Acquire the pool materialization claim; refresh published state after claiming.
3. Restore/materialize the parent query and publish original membership receipts.
4. Resolve pinned model assets and enumerate bounded deterministic tasks.
5. Load completed task receipts, then dispatch only missing work.
6. Commit complete task results atomically, including terminal zero-output results.
7. Assemble each branch's ordered accepted inventory and capture it as stored
   generated text using the normal snapshot/object infrastructure.
8. Publish verified lineage/manifests/profile objects.
9. Atomically bind the complete pool outcome and mark the resource completed.

Readers must see either pending/error or a complete verified outcome. A partial
branch is resumable internal state, not a complete public inventory.
Use `storage/publication.py` for durable immutable object publication and the
SQLite metadata store for resource state/outcome binding. On a race, validate
and reuse the winning outcome rather than overwriting it.

Crash after a remote/provider call but before a receipt would not imply exactly
once billing; remote providers are deferred. For local inference, a crash can
repeat an uncommitted task, but must never duplicate committed rows or alter the
winning completed artifact. Test every publication boundary.

Reuse per-task computation across compatible pools when the full semantic input
and realization fingerprint match. Recompute budget acceptance when assembling
a new branch; a receipt that was rejected only because a prior branch budget was
exhausted must not poison a larger-budget pool's computation cache.

Do not carry the whole corpus, all completion strings, or all lineage rows in a
Python dict. Bound task queues, input batches, output shards, model caches, and
completed receipt buffers. Generated text uses the existing object/frame stores.

## 8. Execution adapter and output acceptance

Place generation behind an internal typed protocol: pinned generator definition
and bounded input task -> typed completion with text, token IDs/counts, finish
reason, and provider/runtime receipt. Strategies construct prompts; the adapter
performs inference; the coordinator validates and commits outputs.

For the primary adapter, call released DataTrove inference APIs with the pinned
recipe's payload; keep its runtime dependency isolated behind capability checks.
For the local adapter, use `AutoTokenizer` and `AutoModelForCausalLM` with immutable
assets and the existing Torch/Transformers platform matrix. Do not enable remote
model code by default or require CUDA for ordinary tests. Persist the chat
template/tokenizer fingerprint; an alternate local model without a chat template
uses a defined, versioned text-only fallback.

Use isolated per-task RNG state in the local adapter. For upstream batching,
verify that request seeds preserve the declared task-seed semantics; do not
replace supported batching with a second homemade scheduling layer. Record batch and
execution settings that affect outcomes. Model reuse must not share unsafe
mutable generation state across workers.

Explicitly bound input context. Initially reject overlong inputs with a recorded
reason instead of silently truncating or chunking them. Chunking changes parent
views, task enumeration, budgets, and lineage and needs a separate versioned
policy. This is a declared deviation from the upstream production launcher's
character-based truncation, not an exact preprocessing reproduction. If matching
its truncation later, version that policy and persist the actual rendered input
and consumed parent ranges. Capture only completion text; exclude the input
prompt and special tokens. When upstream returns counts rather than token IDs,
validate usage fields and preserve their origin. Record server-reported inference
work separately from retokenized captured text; do not silently equate them.

Accept only valid nonempty UTF-8 output within per-document byte/token limits.
Reject EOS-less length-truncated outputs initially, prompt echoes, exact holdout
copies, and exact duplicates within a branch, recording the reason. Do not label
an educational prompt as a factuality guarantee. Quality/faithfulness scoring is
a separate curation operation or a future explicit strategy verifier.

Commit accepted outputs in canonical task order, not completion order. For a
token ceiling, accept a whole document only when it fits; stop at the first
otherwise-valid output that would exceed the remaining budget, record budget
exhaustion, and do not truncate the text. The result may underfill its ceiling.
For a document ceiling, stop after that many accepted documents. Bound speculative
dispatch so substantial extra inference does not continue after exhaustion.

Operational failures (asset mismatch, invalid provider response, unavailable
device, inference exception) fail the branch/pool after bounded retries.
Ordinary content rejections are terminal recorded outcomes. Default to three
attempts for transient operational failures; repeat the same task seed on retry.
Validate this retry policy and account for all inference work.

Use a cooperative stop signal at task boundaries for session shutdown and
internal interruption. Do not invent a public cancellation API that the current
resource system lacks. A wait timeout means the caller stopped waiting; it does
not silently cancel admitted work. Ensure executor shutdown, dependency waits,
and model teardown are bounded and do not leave background writes after storage
closure. Avoid nested materialization waits on the same single-worker pool: run
dependencies inline under single-flight or use the existing admission pattern
without occupying the only worker while waiting for a queued dependency.

## 9. Provenance and storage correctness

Retain original document IDs and source keys in the original branch. Never
recapture its selected rows under a new corpus just to give them the label
`original`; that would change identities and obscure their real sources.

Generated text is new captured content. Its token-byte alignment refers to that
generated text, not to parent byte ranges. Store a separate generation lineage
edge to each input view, with original document ID, input query ID, snapshot
membership, captured digest, retained-view digest/ranges, task/strategy ID, pool
ID, and output ordinal. Do not claim copied-byte correspondence for a paraphrase.

Add a typed optional generation reference to provenance contracts, referencing
bounded lineage shards. Extend public ID conversion and lineage readers. Avoid
duplicating a complete ancestor graph in every token-region record. Packed
`TokenRegion.document_id` identifies generated documents normally; generation
lineage can be followed separately through those IDs.

Save original selection receipts and resolved snapshot references needed for
restoration. A synthesis-backed query cannot be restored merely by loading every
document from the original snapshots: its pre-synthesis selection may have
filtered, sampled, deduplicated, or removed spans.

Persist the originating pool reference on generated snapshot metadata as well.
Planning an ordinary query over a generated snapshot must either inherit its
verified matching synthesis context or reject the unsupported composition.
Do not allow `GetSnapshot` followed by `.query().mix()` or `union()` to silently
rehash generated content into holdouts. In the initial implementation, reject
unions across incompatible pools/evaluation contexts and reject recursive
synthesis on generated inputs. Re-importing exported text as an unrelated corpus
is outside this lineage guarantee, so do not claim to detect arbitrary copies.

On read, verify manifests, digests, schema versions, exact label coverage, parent
membership, training roles, task/output keys, and all object bounds. Reject
unknown parents, cross-pool references, duplicate task outputs, and incomplete
lineage. Read-only reopening must not load weights, execute a missing query, or
fall back to unverified text.

## 10. Integration map in the current checkout

These are owning areas to inspect and extend, not a requirement to put all logic
in the coordinator. New modules should follow the existing separation of API,
schemas, runtime, engine, and storage.

| Area | Existing files and required changes |
| --- | --- |
| Public entry points | `api/query.py`: synthesis method; `api/synthesis.py` (new): pool and query handles; `api/database.py`, `api/collections.py`: typed creation/get/reopen/list support. |
| Policies/exports | `schemas/enums.py`: public strategy/model/source enums; `engine/synthesis.py`: resolved recipes; `api/__init__.py`, package `__init__.py`: typed exports, keeping generator configuration internal. |
| Request/schema boundary | `schemas/requests.py`, `schemas/validation.py`, `schemas/messages.py`: synthesis builder, context preservation, budgets and oneof validation. |
| Typed resources | `api/base.py`: closed resource unions, repr dispatch, execution kind/wait handling; `api/display.py`, `api/progress.py`: bounded display/progress and no accidental execution. |
| Shared schemas | `proto/premixdb/v1/splits.proto` (new): extract the existing split message without changing its wire type; public reexports avoid import cycles. |
| Planning | `runtime/planner.py`: synthesis query input, canonical identity, field resolution deferred until the generated population exists. |
| Generation runtime | `runtime/synthesis.py` and `synthesis/generation.py` (new): task enumeration, adapter, receipts, branch capture, completion. |
| Admission/catalog | `runtime/coordinator.py`, `runtime/materialization.py`, `storage/catalog.py`, `storage/metadata.py`: typed methods, caches, execution events, durable claims, catalog parent indexing. |
| Queries/selections | `runtime/coordinator.py`, `engine/queries.py`, `storage/selections.py`, `storage/lineage.py`: selected-view restoration, source roles, context and generation lineage. |
| Indexed queries/enrichment | `runtime/analytics.py`, `runtime/analytical_preparation.py`, `storage/analytics.py`, `engine/indexed.py`, `runtime/enrichment.py`, `runtime/catalog.py`: synthesis population receipts and generated-text fields. |
| Strategy projection | `fields/ids.py`, `fields/expressions.py`, `fields/selectors.py`, `fields/catalog.py`, `engine/curation.py`, `storage/profiles.py`, plus `intrinsic.proto`: metadata projection and tuple domains. |
| Mixing | `runtime/mixtures.py`, `runtime/mixing.py`, `runtime/datasets.py`, `engine/mixtures.py`, `api/mixture.py`: strategy labels/inventory and completed context pinning. |
| Splits/datasets | `schemas/splits.py`, `runtime/split_datasets.py`, `runtime/datasets.py`, `api/dataset.py`: fixed original holdouts, training-only generated membership, profiles and packing. |
| Worker execution | `runtime/pipeline.py`, `runtime/partitions.py`, `runtime/partition_types.py`, internal transport messages: bounded typed generation tasks if partition execution is used. |
| Durable storage | `storage/objects.py`, `storage/publication.py`, `engine/snapshots.py`, `engine/records.py`: ordinary generated text capture plus synthesis sidecars. |
| Training | `training/sequences.py`, `training/reader.py`, `training/torch.py`: verify inherited behavior and provenance; change only where new context requires it. |
| Build/docs | `scripts/generate_protos.py`, `src/_premixdb_build.py`, `pyproject.toml`, `docs/training.md`, examples and packaging tests. |

Important current assumptions to remove deliberately:

- Query compilation requires nonempty snapshot IDs. Add the explicit alternative
  source mode, retaining strict validation of ordinary query inputs.
- `_with_fields()` reconstructs requests from snapshot IDs and operations. Copy
  synthesis source and evaluation context too.
- Many intrinsic-field checks use `(1, 2, 3, 4)` or `<= FIELD_SOURCE_CORPUS_ID`.
  Replace numeric-range assumptions with a shared explicit intrinsic-field
  predicate before adding a synthesis field. Update SDK comparisons, planner
  projections, indexing, profiles, enrichment resolution, and domain routing.
- `_mix_pool()` and split packing partition every document with `content_split()`.
  Route synthesis-backed training and evaluation through pinned role/membership
  receipts; never call that function on generated text to assign a holdout.
- Indexed population preparation takes whole snapshots. Preserve the original
  selected-view membership as a first-class population constraint.
- Selection restoration validates provenance snapshots against query inputs.
  Persist the resolved source receipt so synthesis inputs pass meaningful checks
  rather than weakening validation or inventing snapshot IDs.
- Public profile formatting converts source-corpus keys into public IDs. Strategy
  labels stay plain strings; composite projections use normal tuple encoding.
- Metadata and execution dispatch use closed protobuf/type unions. Update every
  relevant union and parent listing index, including raw requests and reopen.

Prefer one shared synthesis-aware population resolver returning a verified
training view, original evaluation views, branch labels, and outcome context.
Use it in inventory, packing, profile, preview, enrichment, and restoration;
independent implementations of these rules will drift.

## 11. Implementation sequence and milestone gates

### A. Freeze contracts, dependency boundary, and validation

Audit the pinned DataTrove release and inference extras on the supported platform
matrix first. Record available APIs, upstream commit/version, inference-server
requirements, and the exact adapter entry points. Verify that adding required
extras does not upgrade Intel macOS's incompatible platform pins or import GPU
packages into ordinary SDK use. Keep resolver/build smoke checks alongside the
adapter contract tests; choose the released API, not a research-repo private fork.

Implement public enums, pinned FinePhrase recipe assets, protobuf messages,
request validation, canonical definitions, and public exports. Add API/type tests without execution.

Gate: invalid policies fail before input scanning/model loading; protobuf
roundtrips preserve definitions; policy/list mutation cannot affect recipes;
equivalent preset-list orderings have equal identities; ordinary query/mix IDs
and interfaces remain unchanged.

### B. Establish source membership and split context

Implement original selected-view receipts, synthesis query source mode, and the
shared population resolver using fixture-generated branch data before connecting
real inference. Add dataset/mix context propagation and immutable holdouts.

Gate: transformed inputs preserve original IDs/digests/ranges; generation sees
only training views; candidate holdouts match the ordinary baseline; changed
split policies and forged context fail; query curation cannot detach context.

### C. Build durable synthesis materialization

Implement pool planning, task identities, receipts, generated snapshot capture,
manifests, lineage, catalog integration, single-flight/durable claims, and resume.
Use a deterministic fake backend in tests.

Gate: no execution from registration/repr/listing; first completed outcome is
immutable; reopening restores identical text and IDs; interrupted runs reuse
committed tasks; concurrent callers observe one verified outcome; corruption
fails without regeneration.

### D. Add the real generation adapter and strategies

Connect DataTrove's released inference runner through a narrow typed adapter,
plus the small CPU/local adapter. Add pinned model assets, four upstream recipes,
task seeding, bounds, acceptance rules, and inference teardown. Complete the
recipe-parity gate from section 3.2. Reuse upstream batching and request caches
where their declared semantics satisfy the task/budget contracts; test the
adapter rather than duplicating those facilities.

Gate: actual adapter produces completion-only text; input/output/context limits
are enforced; prompt/tokenizer/model changes alter identity; fake-backed default
tests are offline; a tiny locally created causal-LM checkpoint can exercise the
local adapter without external downloads. A dependency-backed offline fixture
also exercises DataTrove using its supported dummy/custom server on platforms
with inference support; a GPU/model throughput test is a separate opt-in check.

### E. Integrate strategy domains and candidate recipes

Implement `p.synthesis.strategy`, default domain selection for synthesis queries,
fixed and RegMix inventories, bounds, empty-domain rules, profiles, and outcome
pinning on resolved dataset recipes. Reuse existing allocation and sampling.

Gate: zero weights allocate zero; fixed totals match budget; capacity/exposure
constraints hold; all candidates share exactly the same generation outcome;
candidate iteration/training never increases provider call counts.

### F. Complete curation, indexing, and training reads

Ensure query steps, decontamination, field inference, direct/indexed execution,
saved selections, bounded previews, byte/model tokenizers, packing, and PyTorch
adapters work with generated documents and original holdouts.

Gate: direct/indexed parity where supported; generated fields use generated text;
training token provenance resolves to correct synthetic bytes and parents;
read-only restores and adapters after database closure behave as existing tests
require. If an indexed path is unsupported initially, explicitly route to a
verified direct path; never return a different selection silently.

### G. Documentation, example, build, and handoff

Add an offline synthesis/strategy-ablation example and update public docs with
execution triggers, per-strategy versus training budgets, split-policy pinning,
source-label weights, RegMix's actual scope, and generation-work accounting.
Run the checks in section 13 and record their actual results.

Use checked-in `examples/data` inputs. The offline example accepts an explicit
already-local teacher checkpoint path and never downloads a model implicitly.
Its installed-package integration test creates a tiny local checkpoint/tokenizer
in a temporary directory and exercises the public `model=local_path` override. Keep
fixture generation helpers in test support or the example itself, not in a
production fake-provider switch. Any real-teacher/downloaded-model demonstration
is explicitly opt-in and separate from the offline smoke test.

Gate: all acceptance criteria below hold; fresh wheel/sdist include new schemas,
bindings, typing, and prompt assets; installed-package examples do not depend on
test-only provider injection or unpublished local files.

## 12. Validation cases

Use small fixture populations and counting deterministic fake backends. Reserve
larger fixtures for split/Unicode/budget/publication boundaries. Do not download
models or call paid providers in the ordinary test suite.

Suggested focused files are `tests/test_synthesis.py`,
`tests/test_synthesis_storage.py`, `tests/test_synthesis_splits.py`,
`tests/test_synthesis_mixing.py`, `tests/test_synthesis_generation.py`, and
`tests/test_synthesis_transport.py`. Add focused regressions to existing owning
test files rather than duplicating all their cases.

| Area | Required cases and observable assertions |
| --- | --- |
| Optionality | Ordinary queries never instantiate a generator; existing recipes, IDs, splits, previews, and packed tokens remain unchanged. |
| Request validation | Empty list, duplicate/forged/unknown presets, passed classes, wrong types, wrong-enum/model types and nonexistent local model paths, bool-as-int budgets, invalid internal decoding parameters, request overflow, raw oneof/context mismatches, unknown fields. Verify the default ceiling and bundled model pins without a user-supplied generator. Assert zero provider calls and zero source reads. Actual task-limit validation after parent restoration still precedes model loading. |
| Mutation/identity | Mutate caller lists/protobufs after registration; recipe bytes unchanged. Reorder strategies; same IDs. Change recipe version/model/tokenizer/template/seed/run key/splits/view; expected identity changes. Add a branch; existing task seeds unchanged. |
| Laziness | `synthesize`, `sources`, repr, query planning, mix registration, listing and get leave counters at zero. Provider calls begin at documented execution triggers. Pool/synthetic preview may execute; `limit=0` does not. |
| Source views | Original selected query excludes filtered rows; retained Unicode ranges survive. Repeated original occurrences stay repeated, but generation runs once per unique selected view. Branch labels survive curation/reopen. Reopening generated snapshots cannot detach context; incompatible pool unions and recursive synthesis fail. |
| Splits | Force parents in each split; provider receives only train IDs/text. A child whose own hash lands in validation remains train-only. Duplicate captured content across sources stays grouped. Mismatched split seed fails. Multi-parent invalid receipts fail. |
| Holdouts | Compare document IDs/order, byte-token sequences, model-token sequences, profiles and packed ranges across several candidate weights and sampling seeds. Compare to ordinary original-query baseline under equal packing. Synthetic exact holdout copies are rejected. |
| Budgets | Document ceiling, accepted-token ceiling, first-output-too-large, zero accepted outputs, finite exhaustion, Unicode byte ceiling, long input rejection, truncated completion, duplicate rejection, and bounded speculative work. Account for rejected/retried inference separately. |
| Generation adapter | Verify only completion is decoded, special/prompt tokens excluded, pinned chat template used, immutable asset digest checked, per-task RNG isolated, finite retries, model loaded once per supported execution scope and released on close. |
| Durability | Reopen with `cache_bytes=0`; text, IDs, profiles, and lineage match. Read-only reopen needs no model/network. Pending read-only execution fails clearly. Missing/corrupt text/receipt/manifest fails without provider invocation. |
| Resume | Crash after receipt publication, branch snapshot publication, lineage publication, and immediately before outcome binding. Committed tasks are reused; accepted rows are not duplicated; stable logical ordering survives different shard sizes/completion order. |
| Concurrency | Two handles and two independent sessions materializing the same pool race safely. Different branch completion orders produce the same canonical assembly with a deterministic backend. Materializer with one worker does not deadlock on parent queries. |
| Lifecycle | Wait timeouts use one deadline; underlying work remains correctly accounted for. Internal interruption prevents new tasks and preserves resumable receipts. Session close releases models/workers/locks and leaves no writes against a closed store. |
| Mixing | Fixed weights including zeros; unknown/missing labels; empty branches; infeasible positive weights/bounds; natural proportions; RegMix determinism; replacement/capacity/max-epoch constraints; alternative tuple domains. Compare existing allocation results, not a second sampler. |
| Inventory reuse | Resolve and pack multiple candidates/tokenizers/packing lengths; generation call count stays fixed. Changed training tokenizer re-inventories generated text without re-generating it. |
| Curation | Exact/line/similarity dedupe, language/quality filtering with fixture producers, retained-span decontamination, empty post-filter branches. Updated capacity is correct and evaluation context unchanged. |
| Provenance | Pack multi-document sequences; resolve every synthetic region to generated text then parent views. Byte ranges cover generated UTF-8 content, never fabricated parent alignment. Parent/public ID conversion roundtrips. |
| Indexed/direct parity | Equivalent selection/labels/stats from supported indexed and direct paths, including generated intrinsic strategy projection and transformed parent views. Restoration does not widen original selection. |
| Worker transport | Strict protobuf parse, size/count limits, unknown versions, bad IDs/ranges/parents, forged finish reasons, malformed responses. A real subprocess roundtrip uses tiny fixtures without loading external models. |
| Enums/dependency | All public enum members roundtrip through explicit protobuf maps; unknown/UNSPECIFIED/wrong-enum values fail. Source enum keys normalize to canonical labels with equal identities. DataTrove adapter works on the pinned release; SDK import succeeds without inference modules on Intel macOS. Offline runner tests verify task IDs, budgets, failure propagation, resume and shutdown. No full finephrase dependency conflict is introduced. |
| Recipe parity | Four strategy enum members require no constructors; labels and assets match pinned upstream definitions. Shared model override reaches every branch and changes identity. Fixture prompt rendering matches the declared recipe; acceptance-rule deviations are recorded. No public free-form prompt or per-branch generator API. |
| Packaging | New public imports, type annotations, acyclic proto compilation, unchanged `Splits` wire bytes/full name and generated-module reexports, prompt-template package data, installed wheel/sdist workflow, no source-tree-relative runtime paths. |

Additional useful property tests: allocation totals and exact zero allocations,
permutation invariance of preset lists, manifest digest independence from
shard boundaries, UTF-8 range validity, and split membership conservation. Avoid
tests that merely mirror implementation branches without checking an external
invariant.

### Research-level validation after implementation

Functional correctness does not establish that synthesis improves a model.
Provide a separate, opt-in ablation harness using fixed original holdouts and
recorded generation inventory. Compare:

1. Original-only training under a fixed training-token budget.
2. Repetition/upsampling of original data under the same budget.
3. Each synthesis strategy individually with explicit real/synthetic ratios.
4. Combined strategies and a bounded RegMix candidate sweep.
5. An original-only model given comparable total compute when generation work
   is included, where feasible.

Hold architecture, initialization policy, optimizer, training tokenizer, packing,
and evaluation protocol fixed. Run multiple model seeds for serious conclusions;
report variance and exact dataset/recipe IDs. Use validation to choose candidates,
and evaluate the selected candidate on test once. Report generation and training
compute separately. A small smoke ablation verifies the harness, not the research
claim or generality of a winning ratio.

## 13. Commands and completion checklist

The current Makefile and pytest configuration support these commands. Adjust the
new focused filenames if their responsibilities are consolidated during implementation:

```sh
uv run --locked python scripts/generate_protos.py
uv run --locked python scripts/generate_protos.py --check
uv run --locked pytest -n 0 tests/test_synthesis.py tests/test_synthesis_storage.py tests/test_synthesis_splits.py tests/test_synthesis_mixing.py tests/test_synthesis_generation.py tests/test_synthesis_transport.py
uv run --locked pytest tests/test_mix.py tests/test_mixture_pool.py tests/test_dataset_splits.py tests/test_materialization.py tests/test_analytics.py tests/test_training_reads.py tests/test_token_ranges.py tests/test_token_publication.py tests/test_requests.py tests/test_intrinsic_fields.py tests/test_field_expressions.py tests/test_resource_display.py tests/test_catalog_listings.py
make lint
make typecheck
make protos
make integration
make check
```

`make check` runs lint, type checks, generated-protobuf verification, coverage over
non-performance tests, and fresh wheel/sdist builds with strict Twine checks.
It does not publish. `make integration` isolates integration workflows. The
ordinary pytest defaults exclude integration/performance tests and use up to six
workers. Use `-n 0` when diagnosing stateful fake-backend or crash tests; do not
replace the normal full check with only the focused suite.

Completion criteria:

- [ ] Optional named parallel strategies generate separately mixable sources.
- [ ] API planning is lazy and the ordinary path is unchanged.
- [ ] Strategy/model/source choices use public enums with stable wire mappings.
- [ ] FinePhrase recipes bundle pinned defaults without `p.Generator`.
- [ ] Released DataTrove inference is reused through a tested platform-aware adapter.
- [ ] Recipe parity, bounded defaults, and the shared model override are validated.
- [ ] Only selected training views enter generation; holdouts remain original.
- [ ] Split/context metadata survives every raw request, query, mix, and dataset path.
- [ ] Generation outcomes are durable, immutable, resumable, and independently verifiable.
- [ ] Mixture candidates reuse inventory and support zeros, bounds, and capacity checks.
- [ ] Generated-text curation works and reports resulting branch capacities.
- [ ] Provenance distinguishes generated-byte alignment from parent derivation.
- [ ] Read-only and installed-package workflows need no regeneration.
- [ ] Offline tests cover failure, race, budget, Unicode, and split boundaries.
- [ ] New example and docs distinguish generation work from training budgets.
- [ ] Required checks pass, with exact commands/results recorded in the implementation handoff.
- [ ] Any opt-in live-model or research experiments are labeled as run or unrun;
      neither is implied by mocked/provider-contract test success.

## 14. Research basis and reading map

These sources informed the design discussion preceding this plan. The architecture
above is an engineering inference, not a public API prescribed by the papers.
The proposed first-release paradigm is FinePhrase: document-grounded rewriting
into separately mixable formats. Its
[paper](https://arxiv.org/abs/2604.13977) is accepted at COLM 2026 and includes
Colin Raffel, Guilherme Penedo, Leandro von Werra, and Thomas Wolf. It directly
compares prompt, model, source, and mix-in choices and supplies reusable recipes.

### 14.1 Compare the major paradigms before expanding implementation scope

There is no verified common-benchmark ranking across these methods. Select by
training objective, available organic data, and whether synthesis must adapt
during training. Do not equate a higher data-efficiency multiplier with lower
total compute, or user-response quality with language-model perplexity.

| Research recipe | Adoption implication |
| --- | --- |
| [FinePhrase](https://arxiv.org/html/2604.13977v2), COLM 2026 | Best fit for the initial fixed-corpus scope: four packaged transformations, reusable generated inventories, and ordinary causal-LM training. It is an engineering fit, not an overall research winner. |
| [FineInstructions](https://arxiv.org/html/2601.22146v3), Patel, Raffel, Callison-Burch | A strong candidate for instruction-oriented pretraining: retrieve user-derived templates, instantiate grounded answers, then judge/filter. Preserve separate instruction/answer records, template/index/model pins, and excerpt lineage. Verify training formatting and loss policy rather than assuming that plain FAQ text reproduces the method. |
| [SynPro](https://arxiv.org/html/2605.17849v2), Yu and Xiong, COLM 2026 | Adaptive rephrasing/reformatting with generator RL and student-dependent influence rewards. Requires training checkpoints, a separate reference set, and repeated synthesis rounds. A frozen generator can yield a reusable pool, but omits the defining adaptive loop and must not be presented as full SynPro. |
| [RePro](https://arxiv.org/abs/2510.10681), ICML 2026 | A useful intermediate option: a learned rephraser optimized for quality and faithfulness. Distinguish adopting a released pinned generator from implementing its training procedure. |

FineInstructions' [resource page](https://huggingface.co/fineinstructions) exposes
templates, generation/retrieval models, and data, but currently labels pipeline
code as "Coming soon". Verify implementation completeness before estimating an
adapter as a drop-in integration. Existing PremixDB sequence readers support
loss masks; that alone does not establish instruction-aware packing or reproduce
the paper's training setup.

Keep the public list-of-strategy-enums design across research families. Internally,
each implemented family resolves all its assets and output contract into a
portable recipe. A future instruction recipe needs bounded zero-to-many output
records and retrieval/verifier task receipts, not a hidden alternate prompt.
An adaptive recipe additionally needs immutable round IDs, generator/student
checkpoint references, and training orchestration outside ordinary `.mix()`.
Reference data used for generator optimization must be separate from final
validation/test data; optimization must never consume the latter.

Before claiming an advantage across families, run a controlled comparison on
common source data, matched student architecture and training budgets, and
appropriate evaluations. Include response-quality measures for instruction
recipes, ordinary held-out language modeling for rewriting, and total generation
plus training work. For adaptive synthesis, report both fixed-budget and
train-to-saturation comparisons and isolate the contribution of generator RL,
influence rewards, regeneration, and ordinary mixture optimization.

### 14.2 Broader reading map

This is a selected reading map, not an exhaustive or citation-ranked survey.
Recent preprints have less established evidence than replicated workflows.

| Source | Relevant finding or design implication |
| --- | --- |
| [TinyStories](https://arxiv.org/abs/2305.07759), 2023 | Synthetic constrained stories can support small-model learning; generation designs a distribution, not just a text edit. |
| [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644), 2023 | Filtered real data and synthetic educational text/exercises motivate distinct reusable ingredients. |
| [Phi-1.5](https://arxiv.org/abs/2309.05463), 2023 | Educational synthetic generation extends beyond code; topic and presentation choices matter. |
| [WRAP](https://aclanthology.org/2024.acl-long.757/), ACL 2024 | Rephrase web documents into different styles and jointly train on real and synthetic text. |
| [Instruction Pre-Training](https://aclanthology.org/2024.emnlp-main.148/), EMNLP 2024 | Generated instruction-response pairs expand the forms in which corpus knowledge is presented. |
| [FineInstructions](https://arxiv.org/abs/2601.22146), 2026 | Retrieved user-derived instruction templates and grounded answers target downstream response quality; this is a distinct recipe family from fixed-format rewriting. |
| [MIND](https://arxiv.org/abs/2410.12881), 2024 | Synthetic math conversations and raw-data formatting motivate explicit strategies and composition. |
| [Nemotron-CC](https://arxiv.org/abs/2412.02595), 2024 | Large-scale filtering plus rephrasing motivates integration with normal curation rather than an opaque mix callback. |
| [Phi-4](https://arxiv.org/abs/2412.08905), 2024 | Synthetic data is part of a broader training recipe; useful systems evidence rather than an isolated causal comparison. |
| [EntiGraph / Synthetic Continued Pretraining](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html), ICLR 2025 | Entity-based synthesis motivates future corpus-level tasks and multiple-parent lineage. |
| [MGA / Reformulation for Pretraining Data Augmentation](https://arxiv.org/abs/2502.04235), 2025 | Genre/audience reformulation motivates separately named strategy variants and repetition baselines. |
| [SynthLLM / Scaling Laws of Synthetic Data](https://arxiv.org/abs/2503.19551), COLM 2025 | Cross-document concept recombination and scaling experiments motivate bounded inventories and future cohort strategies. |
| [REWIRE / Recycling the Web](https://arxiv.org/abs/2506.04689), 2025 | Low-quality source material can be transformed; synthesis inputs and real mix-in sources need not be the same quality tier. |
| [BeyondWeb](https://arxiv.org/abs/2508.10975), 2025 | Generation strategy, source selection, and model family are separate experimental choices. |
| [Demystifying Synthetic Data in LLM Pre-training](https://arxiv.org/abs/2510.01631), 2025 | Synthetic type, model size, and budget affect useful mixing ratios; its favorable approximately 30% rephrased-data result is not a universal default. |
| [RePro](https://arxiv.org/abs/2510.10681), ICML 2026 | Quality and faithfulness rewards motivate recording rephraser semantics and keeping output validation explicit. |
| [FinePhrase](https://arxiv.org/abs/2604.13977), COLM 2026 | Proposed first-release recipe source: structured formats, original-data mix-in, and public prompts/code. |
| [WRAP++](https://arxiv.org/abs/2604.06829), 2026 preprint | Cross-document QA motivates multiple-parent contracts; do not implement this as an unnamed ordinary rephrase option. |
| [SynPro](https://arxiv.org/abs/2605.17849), COLM 2026, September v2 | Adaptive synthesis is a training-loop method; a fixed preset omits its student-dependent generator updates. |
| [Cosmopedia](https://huggingface.co/blog/cosmopedia), 2024 engineering report | Prompt/topic diversity and large-scale generation infrastructure provide practical implementation context. |
| [ProX](https://arxiv.org/abs/2409.17115) and [RefineX](https://arxiv.org/abs/2507.03253) | Programmatic refinement is an adjacent operation; preserve a distinction between editing captured text and generating new documents. |
| [Model collapse](https://www.nature.com/articles/s41586-024-07566-y) and [Accumulating real and synthetic data](https://arxiv.org/abs/2404.01413), 2024 | Recursive replacement and accumulation have different outcomes; preserve original sources and generation history rather than asserting synthetic data is uniformly harmful or safe. |
| [Scaling Data-Constrained Language Models](https://jmlr.org/papers/v26/24-1000.html) | Repetition is an essential controlled baseline when evaluating the benefit of additional synthetic inventory. |
| [Masked-input regularization](https://arxiv.org/abs/2606.06888), 2026 preprint | Training-time augmentation without a persisted corpus is a different feature and should not be hidden inside synthesis or mixing. |

Interpret reported efficiency gains carefully. Faster training to a target loss,
better benchmark accuracy at a fixed token budget, and lower total compute after
including generation are different claims. This implementation should make all
three measurable without treating any one paper's ratio or speedup as a default
guarantee.
