# Examples

Run from the repository root:

```bash
uv sync --locked
uv run python examples/01_tiny_shakespeare_snapshots.py
```

Edit the values near the top of each script: `INPUT`, `STORAGE`, `LIMIT`, and any
thresholds. Mixture examples use `WEB`, `PAPERS`, and `LITERATURE`. Lesson 11 groups
settings in configuration dataclasses. No CLI arguments.

| Example | Shows |
| --- | --- |
| [01 · Snapshots](01_tiny_shakespeare_snapshots.py) | Capture and reopen the same text |
| [02 · Filtering](02_c4_filtering.py) | Compare lengths and Gopher word-count limits on C4 |
| [03 · Dedupe](03_c4_deduplication.py) | Remove a copied page; explain T5/Falcon dedupe differences |
| [04 · Distributions](04_s2orc_distributions.py) | Inspect word counts |
| [05 · Packing](05_tiny_shakespeare_packing.py) | Read a PyTorch batch |
| [06 · Decontamination](06_tiny_shakespeare_decontamination.py) | Remove evaluation overlap |
| [07 · Quality](07_c4_quality_scores.py) | Cache scores; distinguish GPT-3/LLaMA classifiers |
| [08 · Source mixtures](08_pile_source_mixture.py) | Allocate a toy budget; reference published model mixtures |
| [09 · Candidates](09_regmix_candidates.py) | Compare seeded recipes |
| [10 · Resume](10_tiny_shakespeare_resume.py) | Restore a consumed sequence position |
| [11 · RegMix training search](11_c4_pretraining_ablation.py) | Train proposed topic/content-type mixtures and automatically select the best measured weights |

The Shakespeare lessons use the nine-block excerpt bundled in the installed
`premixdb` package and work offline.
It comes from [char-rnn's Tiny Shakespeare](https://github.com/karpathy/char-rnn).

## Data for the other examples

```bash
uv run python scripts/prepare_c4.py
uv run python scripts/prepare_s2orc.py --limit 100
uv run python scripts/prepare_s2orc.py --split train --limit 100
```

These write a C4 training shard, a peS2o validation sample, and a peS2o training
sample under `.cache/`. Use training papers for mixtures. Your own JSONL can
replace them: C4 needs `text`; papers need `id` and `text`.

The quality example downloads QuRater on first use and starts with eight pages.
Change `MINIMUM` to reuse the scores with a different cutoff.

## Published model recipes on your own data

[recipes/](recipes/README.md) contains predefined preprocessing/query functions
and source mixtures for **T5/C4, Falcon/RefinedWeb, Gopher, LLaMA 1, and GPT-3**.
Each includes paper references and states which original stages need upstream
processing or classifier assets.

Edit the settings to apply the Falcon adaptation to extracted crawl JSONL:

```bash
uv run python -m examples.recipes.apply
```

Use `examples.recipes.mix` for the published source weights on prepared corpora.
The numbered lessons remain bounded demonstrations; they do not reconstruct the
original training datasets or mixtures.

## Find mixture weights with RegMix and real training

```bash
uv run python scripts/prepare_c4.py
uv run --locked python examples/11_c4_pretraining_ablation.py
```

The training loop is the first function. `MIX = MixConfig(domains=p.Topic)` groups
nonempty, deduplicated C4 documents by their predicted topic. Change it to
`MixConfig(domains=p.ContentType)` to search content-type proportions instead.

The script trains six mixtures proposed by `p.RegMix()`. Every candidate gets a
fresh causal transformer with the same initialization, learning rate, and
**20,480 input-byte budget in 40 updates**. Candidate 0 is the comparison reference;
all training runs belong to the RegMix search. Replacement allows labels to be
upsampled. Topic/content-type classifiers download on first use and cache their
outputs; they require PyTorch 2.5+. The byte tokenizer and training model need no
downloads.

The central workflow is:

```python
candidates = population.mix(
    domains=MIX.domains,
    weights=p.RegMix(),
    n_candidates=MIX.candidates,
    tokens=TRAINING.steps * TRAINING.batch_size * MODEL.sequence_length,
    replacement=True,
    tokenizer=p.ByteTokenizer(),
    sequence_length=MODEL.sequence_length,
    packing=p.Concat(),
)
validation = evaluation_batches(candidates[0].validation)
losses = []
for dataset in candidates:
    run = train(dataset)
    losses.append(evaluate(run.model, validation))
best_index = min(range(len(losses)), key=losses.__getitem__)
best_weights = candidates.weights[best_index]
best_dataset = candidates[best_index]
```

RegMix proposals, document sampling, and the default 80/10/10 split policy already
use seed `0`. The trainer also defaults to seed `0`, resetting it for each fresh
model. Fixed defaults make the recipes and initialization reproducible within the
same environment; the example needs no explicit data seed arguments.

All models score the **same cached validation bytes from candidate 0**. Mixture
weights apply to training only, so held-out documents keep their natural
proportions. Content-based splits assign whole documents before sampling and
packing. After selecting by validation loss, the script compares the winner and
candidate 0 on the same test bytes.

Edit settings in the dataclasses near the top:

| Configuration | Settings |
| --- | --- |
| `DATA` / `DataConfig` | Input path, storage path, document limit |
| `MODEL` / `ModelConfig` | Context length, width, layers, attention heads |
| `TRAINING` / `TrainingConfig` | Updates, batch size, learning rate, model seed |
| `MIX` / `MixConfig` | Topic/content-type domains, candidate count |
| `EVALUATION` / `EvaluationConfig` | Held-out batches per split |
| `RUNTIME` / `RuntimeConfig` | Device and CPU threads |

For example, `TRAINING = TrainingConfig(steps=10, batch_size=4)` shortens every run,
and `MIX = MixConfig(domains=p.ContentType, candidates=8)` searches eight
content-type recipes. Training results, the search report, and the winner
checkpoint are also dataclasses; they are converted to dictionaries when saved.

The script prints each recipe, its loss, and the change from candidate 0, then
saves `report.json` and `winner.pt` in
`.cache/tutorials/pretraining-ablation/results/`. The report's `selected_weights`
and `selected_dataset_id` make the choice usable programmatically. Reuse the
weights in a later training recipe:

```python
report = json.loads((DATA.storage / "results/report.json").read_text())
selected = population.mix(
    domains=MIX.domains,
    weights=report["selected_weights"],
    tokens=TRAINING.steps * TRAINING.batch_size * MODEL.sequence_length,
    replacement=True,
    tokenizer=p.ByteTokenizer(),
    sequence_length=MODEL.sequence_length,
    packing=p.Concat(),
)[0]
```

`RegMix` supplies proposals; the real training loop and validation objective
supply selection. This finds the best measured candidate for this model and
budget. The full RegMix paper's loss-prediction stage is a further extension.

## Tests

```bash
uv run --locked pytest tests/test_readme_workflows.py tests/test_tutorials.py tests/test_pretraining_ablation.py -m 'not performance'
```

These execute the README's Python blocks and all eleven lessons with offline
inputs. Classifier outputs are controlled; the final ablation trains real transformers
with a reduced budget and checks every candidate, automatic selection, reusable
weights, default-seed reproducibility, causal masking, and disjoint splits.
Package tests also run the snapshot lesson
using the data bundled in a built wheel. Run `make test-all` for the full suite.
