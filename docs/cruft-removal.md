# Repository cleanup

Completed October 3, 2026. The original audit's 23 items are resolved below.
Saved-data compatibility and the default full-feature installation remain supported.

| Item | Resolution |
| --- | --- |
| 1. Comparison enum | Removed unused `ComparisonUnit`. |
| 2. Hugging Face extra | Removed the duplicate dependency; retained an empty compatibility extra. Hub capture remains included in the base installation. |
| 3. Timeout plumbing | Removed inert executor, submit, get, and page arguments. Public wait budgets, deadlines, and future timeouts remain active. |
| 4. Inspector | Removed the test-only inspection module. Ported useful coverage to preview, profile, projection, lineage, and packing checks. |
| 5. Reference API | Moved the independent engine harness to `tests/_reference.py`; removed shipped `_api.py` and `local.py`. |
| 6. Field catalogs | Generate typed Python enums and language attributes from `intrinsic.proto`, preserving numeric IDs and keyword spellings. |
| 7. State grafting | Added typed completed-query construction and normal dataset initialization for partition outputs. Restoring a selection does not execute its recipe. |
| 8. Saved-data compatibility | Retained legacy `.ref` imports, inline snapshot inventories, missing-profile recovery, deprecated selector spellings, and checkpoint compatibility. These remain supported formats; removing them would require migrations. |
| 9. Shakespeare data | Examples and shell tests use the packaged excerpt; removed the duplicate example file. The full shell-demo corpus remains separate. |
| 10. README | Corrected Python fences and the qualified constructor; verified installation and sequential offline workflows. Examples retain the shell's database binding. |
| 11. Dependencies | Retained the full-feature installation contract, including Hub capture, model enrichment, and PyTorch. Splitting extras would change that contract. |
| 12. Default policies | Removed the two unused no-op conversion methods; retained the public default markers. |
| 13. Expressions | Intrinsic and derived fields share operators, predicates, ordering, and selector conversion. Intrinsic counts remain uint64; derived integers remain int64. |
| 14. Summaries | Removed the private SDK summary bridge and unused summary types. Service tests assert protobuf profiles; engine count records remain independent. |
| 15. Hidden conveniences | Removed `_list_document` and `_describe`; tests use preview/profile. Kept `_with_fields`, which still plans projected fields and mix domains. |
| 16. Mix naming | The public collection is `DataMixture`; `Query.mix()` is the only fluent dataset constructor. Removed the `Datasets` alias. |
| 17. Protobuf indexing | Traverse cache rows and columns once with strict alignment; removed the repeated linear `at` helper. |
| 18. Catalog paging | SQLite parent indexes supply filtering, state precedence, public ordering, and windows before payload loading. Bounded caches reuse paginated membership and token scopes; revisions invalidate them after local or external writes. |
| 19. Empty namespaces | ObjectStore creates blob directories when publishing. Fresh initialization creates metadata only; legacy fixtures construct their layout explicitly. |
| 20. Second formatter | Removed the reference adapter's display implementation and local plan vocabulary. |
| 21. Record codecs | Shared `engine/records.py` owns captured-document decoding and structural validation. Selections require profiled frames; legacy snapshots remain readable. |
| 22. Tokenizer label | Documented the v1 tokenizer label as a historical codec identity. Preserved its bytes and saved dataset IDs; execution fingerprints separately include installed package versions. |
| 23. Packaging smoke tests | Retained lightweight checks for every build mode and full GPT-2/PyTorch training validation on the installed wheel. |

The retired `premixdb.local` adapter was documented as a reference testing API.
Use the main `PremixDB` API in applications; tests still compare it independently
with the engine. The retired inspector had no application callers in this repo;
interactive workflows use `.preview()`, `.profile()`, and resource collections.

SQLite metadata keeps its existing resource payloads and schema version. The
additional parent index is derived from those payloads, backfilled transactionally
on writable opening, and included in metadata backups. Read-only catalogs created
before this index still work. Payload digests remain verified when resources load.

Validation includes numeric boundaries, sampled occurrences, cold restoration,
legacy catalogs, cross-connection continuation invalidation, state precedence,
metadata-only reads, and bounded payload loading. A 260-query fixture verifies
that a three-item dataset window decodes three dataset payloads and no queries;
continuation pages reuse their membership snapshot.

The final complete non-performance suite passed **799 tests with one platform
skip in 25.86 seconds**, compared with **29.52 seconds** before cleanup on this
machine. This includes the additional concurrent-publication regression.
`make check` passed lint, formatting, type checking, generated-code
validation, the full branch-coverage suite (**799 passed, one skip in 47.16 seconds;
91.84% coverage**), and wheel/sdist validation. Coverage instrumentation is separate
from normal suite timing. These are local measurements, not CI guarantees.

A manual catalog check with **1,500 queries and 6,000 datasets** loaded a five-item
window in **47.87 ms**, decoding five payloads. A microbenchmark of the real protobuf
cache's **64 rows and eight columns** took **1,398.3 µs** with repeated linear indexing
and **83.9 µs** with one traversal. This measures row iteration, not complete model
inference or catalog creation.

Keep the six bounded workers, deferred heavy imports, offline fixtures, and single
installed training smoke test described in [development](development.md).
