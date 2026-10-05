# Prepared query profiling

The existing query API uses prepared analytical indexes for ordinary filters.
Snapshot capture prepares intrinsic byte, character, URI and corpus columns.
Enrichment prepares scalar columns, classifier probabilities and top-class labels
for its complete, immutable snapshot population. Each new threshold or combination
of filters reuses those columns; no cross-product of possible queries is stored.

Columns retain their original numeric types. Sorted dictionaries and Roaring rank
bitplanes support all six comparison operators without rounding int64 values into
float64. Coverage masks exclude nulls from ordinary comparisons, including `!=`.
Numeric profiles aggregate actual selected values, gathering the smaller of the
selection and its complement. Histogram memberships share population-wide bounds
across blocks; their counts are exact and their ranges remain conservative.

Completed queries store compressed membership and the survivors after each step.
They share the population's document descriptors. Preview, training and explicit
lineage inspection expand records when needed, including after eviction or restart.
Queries avoid publishing a second copy of every selected document or a full JSON
lineage map. Arrow objects and bitmap objects are verified before use; missing or
corrupt completed selections raise errors rather than rerunning the query.

Preparation belongs to the exact deduplicated snapshot union. A previously unseen
union or an older store without indexes requires preparation on its first use.
New enrichment still requires its original computation. Sampling, deduplication,
decontamination and vector component predicates retain the existing executor.
Dataset tokenization, packing and content split planning retain their existing
execution and caching behavior. This change accelerates query selection and query
profiling; it does not make arbitrary computations or dataset materialization
constant-time. Populations currently support at most 2^32 canonical ordinals.

## End-to-end measurement

Run the reproducible benchmark with:

```sh
uv run python scripts/benchmark_query_profiles.py \
  --documents 100000 --trials 7 --reference \
  --output reports/query-profiles-100k.json
```

The October 4, 2026 local run used six independent numeric enrichment columns,
four intrinsic columns, about 1% computed nulls, and a 512 MiB application cache.
Two predicates used previously unseen thresholds in each trial. Every timed call
included public query planning, execution, all ten field profiles, bounded inline
preview and durable result publication. Counts and numeric moments were checked
against independent NumPy selection. One warmup was excluded per workload.

| Selection | Selected documents | Median | p95 |
| --- | ---: | ---: | ---: |
| Narrow | 247 | 46 ms | 47 ms |
| Medium | 2,927 | 123 ms | 126 ms |
| Half | 49,467 | 42 ms | 43 ms |
| Broad | 80,126 | 42 ms | 44 ms |

A separate, previously unseen half-selection took 30,114 ms through the existing
executor. A fresh half-selection after reopening with a zero-byte application
cache took 214 ms. Filesystem caches were warm. There were seven measured indexed
trials per workload and one reference trial; the reference predicates differed
slightly to prevent completed-result reuse.

Capture plus intrinsic preparation took 182 seconds. Synthetic enrichment plus
field preparation took 62 seconds. The synthetic producer replaced model inference;
all storage and execution paths were real. Larger production populations and
classifier fanout require their own end-to-end measurements. Earlier in-memory
Roaring measurements do not establish production latency at those scales.

Correctness coverage includes typed comparison boundaries above 2^53, nulls,
adaptive aggregation, aligned blocks, large-integer variance, executor equivalence,
step order, classifier projections, bounded previews, cache eviction, read-only
reopening and corrupted artifacts. See `tests/test_analytics.py`.
