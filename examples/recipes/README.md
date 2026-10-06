# Model data recipes

Reusable Python preprocessing and premixdb queries for **Falcon/RefinedWeb,
T5/C4, Gopher/MassiveWeb, LLaMA 1, and GPT-3**. These are documented adaptations
for custom data, not reproductions of the original training datasets. Each
module separates text preprocessing (`sources`) from a lazy saved query (`query`)
and provides published source fractions (`WEIGHTS`).

## Apply Falcon filtering to the local C4 sample

[apply.py](apply.py) defaults to the checked-in 64-page C4 sample. Run from
the repository root without preparing a dataset:

```bash
uv sync --locked
uv run python -m examples.recipes.apply
```

The default is an English Falcon adaptation on the local C4 pages. Set
`LIMIT = None` to read the entire file. Capture currently materializes sources in
memory; begin with a bounded sample. The first language query downloads fastText.
T5 additionally downloads the C4 English blocked-word list. DataTrove's spaCy
tokenizer handles word/sentence splitting without a separate model download.

You can also apply the predefined query in Python:

```python
from examples._tutorial import C4
import premixdb as p
from examples.recipes import falcon

with p.PremixDB(storage=".cache/my-crawl") as db:
    crawl = p.Source.read_jsonl(C4, limit=100)
    prepared = db.Corpus("custom/falcon-web", falcon.sources(crawl))
    selected = falcon.query(prepared)
    print(selected.profile())
    print(selected.preview())
    dataset = selected.mix(sequence_length=64)[0]
```

`sources()` runs before capture so line edits become captured text. `query()`
alone assumes that preprocessing has already happened. Rerun preprocessing when
changing its settings; changing query settings produces a new query over the
same immutable snapshot. Use the original source keys when constructing sources
yourself, or `key_column="id"` when the JSONL supplies unique string IDs.

## Decisions and implementation boundaries

| Model | Published decision | Supplied adaptation and remaining work |
| --- | --- | --- |
| [T5/C4](t5.py) | Heuristic line cleanup; at least 5 words per line and 3 sentences per page; English probability ≥0.99; blocked words; deduplicate three-sentence spans. Cleaned C4 beat unfiltered C4 in the corpus ablation. [§2.2, §3.4.1/Table 8](https://arxiv.org/abs/1910.10683). | DataTrove C4 cleanup with explicit paper thresholds and blocked-word filtering. Query uses **fastText**, replacing **langdetect**, and exact **document** dedupe, replacing **span** dedupe. Sentence splitting also differs. Unlabeled C4 only; supervised T5 task mixing is outside this recipe. |
| [Falcon/RefinedWeb](falcon.py) | Reuse MassiveWeb quality/repetition heuristics, fastText threshold 0.65, then MinHash and exact-substring dedupe. Production Falcon uses 76% English web + 8% European web, with smaller curated additions and no upsampling. [RefinedWeb §3/Appendix G](https://arxiv.org/abs/2306.01116), [Falcon §4.2, §5.1/Table 15](https://arxiv.org/abs/2311.16867). | DataTrove Gopher heuristics, English query ≥0.65, premixdb MinHash and exact-document dedupe. **Upstream:** URL blocklists/scoring and revisited-URL removal, main-text extraction/formatting, Appendix G.2 line corrections, and ≥50-token exact repeated-substring removal. The index configuration differs; 0.8 similarity is an example choice, not a reported Falcon cutoff. English preprocessing must not be reused blindly for European-language sources. |
| [Gopher](gopher.py) | MassiveWeb: 50–100,000 words, average word length 3–10, symbol/bullet/ellipsis/stop-word rules and Table A1 repetition thresholds. Choose source weights using held-out losses. [Appendix A.1.1, A.3.1; Table 2](https://arxiv.org/abs/2112.11446). | DataTrove quality/repetition filters, English labels, exact/MinHash document dedupe. Optional exact-document evaluation exclusion differs from the paper's overlap procedure. **Upstream:** HTML extraction, SafeSearch content filtering, and original test-overlap filtering. Tokenization/index settings differ; apply web heuristics to MassiveWeb, not books/code. |
| [LLaMA 1](llama.py) | Combine CCNet-processed Common Crawl (67%) with differently cleaned C4 (15%). Add a Wikipedia-reference classifier to CCNet; use source-specific processing for code, books, papers, etc. [§2.1/Table 1](https://arxiv.org/abs/2302.13971). | Caller supplies already CCNet-processed pages and a trained reference-page classifier. Query adds English/exact-document selection. Use T5 preparation separately for C4. **Upstream:** CCNet line dedupe/n-gram quality filtering and every other source's cleaning. No invented replacement classifier or universal quality threshold. |
| [GPT-3](gpt3.py) | Train a logistic curated-vs-crawl classifier; keep pages when Pareto(9) > 1−score. Upsample curated sources; fuzzy dedupe within sources and remove WebText overlap from Common Crawl. [§2.2/Table 2.2, Appendix A/C](https://arxiv.org/abs/2005.14165). | Caller supplies classifier scores in [0,1]. Implements the published acceptance probability with deterministic per-key draws. Query adapts fuzzy dedupe to premixdb. **Upstream:** classifier training, cross-source WebText removal and benchmark span removal. The original classifier is not bundled; premixdb's index differs from Spark's ten-hash index. |

The historical classifier hooks are intentionally explicit:

```python
from examples.recipes import gpt3, llama

# quality_classifier(source) must return a calibrated score in [0, 1].
prepared_gpt3 = gpt3.sources(crawl, score=quality_classifier, seed=42)
# reference_classifier(source) must return bool. ccnet_pages is already processed.
prepared_llama = llama.sources(ccnet_pages, is_wikipedia_reference=reference_classifier)
```

Implement these callables using your own trained classifiers. QuRater educational
scores are not the GPT-3 or LLaMA classifiers. Source iterators are single-use;
create a fresh iterator for each preparation/capture.

## Apply a published mixture

[mix.py](mix.py) defaults to LLaMA 1 fractions with checked-in toy inputs for
all seven categories. C4 stands in for Common Crawl and C4; training peS2o papers
stand in for arXiv; the other categories use original toy text. These demonstrate
allocation and have not undergone the original source-specific pipelines. See
[the data notes](../data/README.md). Run:

```bash
uv run python -m examples.recipes.mix
```

To try another model, edit `MODEL` and map every category in its `WEIGHTS` to
a checked-in sample file. The runner validates source categories instead of silently
renormalizing an incomplete selection. It captures each category separately,
unions them, and binds fixed weights to corpus IDs using `source_weights`:

```python
from examples.recipes import llama, source_weights

# snapshots maps every label in llama.WEIGHTS to a distinct local sample corpus.
first, *rest = snapshots.values()
population = first.union(*rest).query()
mixture = population.mix(
    weights=source_weights(llama.WEIGHTS, snapshots),
    tokens=256,
    replacement=False,
    sequence_length=64,
)
```

Use the model's tokenizer asset for meaningful token fractions. The runner's
GPT-2 tokenizer is a runnable default, not a substitute for T5/LLaMA/Gopher
tokenizers. Inspect the realized token allocation in the dataset profile.
Falcon's production mixture is distinct from the web-only RefinedWeb experiments.
GPT-3's rounded Table 2.2 weights sum to 101%; `REPORTED_WEIGHTS` preserves those
values and `WEIGHTS` explicitly normalizes them to sum to one.

`REPLACEMENT = False` defaults to no source upsampling. A small sample may not
have enough tokens to satisfy a fixed budget; lower `TOKENS` or supply more data.
Set replacement deliberately for custom experiments that repeat sources. This
does not reproduce the original per-source epoch schedules.

## Offline checks

```bash
uv run --locked pytest tests/test_model_recipes.py -n 0
```

Checks use real local heuristic preprocessing and premixdb execution with
controlled language scores and blocked-word assets, avoiding downloads.
