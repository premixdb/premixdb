"""Cache educational-value scores and try a different cutoff."""

from __future__ import annotations

import math

from _tutorial import C4, DEFAULT_STORAGE, check_inputs

import premixdb as p

INPUT = C4
STORAGE = DEFAULT_STORAGE
LIMIT = 8
MINIMUM = 1.0


def main() -> None:
    check_inputs(INPUT, limit=LIMIT)
    if not math.isfinite(MINIMUM):
        raise ValueError("MINIMUM must be finite")
    with p.PremixDB(storage=STORAGE) as db:
        snapshot = db.Corpus("tutorial/c4-quality", p.Source.read_jsonl(INPUT, limit=LIMIT))
        # GPT-3 (Appendix A) trained a curated-vs-crawl classifier and used Pareto
        # resampling: https://arxiv.org/abs/2005.14165. LLaMA 1 (§2.1) trained a
        # Wikipedia-reference classifier: https://arxiv.org/abs/2302.13971.
        # QuRater educational scores and this cutoff are a separate demonstration;
        # recipes/gpt3.py and recipes/llama.py accept the appropriate classifiers.
        selected = snapshot.query(steps=[p.where(p.quality.educational_value >= MINIMUM)])
        print(
            "Captured → selected documents:",
            snapshot.profile().documents,
            "→",
            selected.profile().output_documents,
        )
        print(selected.profile())
        print(selected.preview())


if __name__ == "__main__":
    main()
