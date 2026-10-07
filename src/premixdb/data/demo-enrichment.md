# Demo enrichment

`demo-enrichment.json` freezes real CPU inference for the eight original Tiny
Shakespeare speeches used by the shell demo. It contains all four QuRating
scores and all 24 WebOrganizer topic logits for each speech.

The artifact records content SHA-256 hashes, field schema IDs, complete producer
policies, pinned model revisions, package versions, and inference settings.
Runtime manifests retain that generation provenance and record the artifact's
SHA-256 and original computation cohort. Saved values are independent of the
user's machine; they are not recomputed to reproduce platform rounding.

Queries reuse these outputs when every source document matches a saved content
hash and the producer policy and field schemas match. Subsets and renamed source
occurrences can reuse them. Changed text, mixed populations containing other
text, other models, and other inference settings use normal enrichment.
Thresholds, class probabilities, and topic labels remain computed from the
complete saved scores and logits. Other field families still run normally.

Regenerate from the catalog's pinned models with:

```bash
uv run python scripts/prepare_demo_enrichment.py
```

Generation may download model assets and runs on CPU. It writes the artifact
only after both models finish successfully. The eight input speeches are taken
directly from `_demo_sources()`; their text remains in `tiny_shakespeare.txt`.
