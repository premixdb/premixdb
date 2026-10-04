# Examples

Run from the repository root:

```bash
uv sync --locked
uv run python examples/01_tiny_shakespeare_snapshots.py
```

Edit the values near the top of each script: `INPUT`, `STORAGE`, `LIMIT`, and any
thresholds. Mixture examples use `WEB`, `PAPERS`, and `LITERATURE`. No CLI arguments.

| Example | Shows |
| --- | --- |
| [01 · Snapshots](01_tiny_shakespeare_snapshots.py) | Capture and reopen the same text |
| [02 · Filtering](02_c4_filtering.py) | Compare length thresholds |
| [03 · Dedupe](03_c4_deduplication.py) | Remove a copied page |
| [04 · Distributions](04_s2orc_distributions.py) | Inspect word counts |
| [05 · Packing](05_tiny_shakespeare_packing.py) | Read a PyTorch batch |
| [06 · Decontamination](06_tiny_shakespeare_decontamination.py) | Remove evaluation overlap |
| [07 · Quality](07_c4_quality_scores.py) | Cache scores and change a cutoff |
| [08 · Source mixtures](08_pile_source_mixture.py) | Allocate a token budget |
| [09 · Candidates](09_regmix_candidates.py) | Compare seeded recipes |
| [10 · Resume](10_tiny_shakespeare_resume.py) | Restore a consumed sequence position |

The Shakespeare examples use the nine-block excerpt bundled in the installed
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

## Tests

```bash
uv run --locked pytest tests/test_readme_workflows.py tests/test_tutorials.py -m 'not performance'
```

These execute the README's Python blocks and all ten lessons with small offline
inputs and controlled model scores. Package tests also run the snapshot lesson
using the data bundled in a built wheel. Run `make test-all` for the full suite.
