# Local example inputs

These small teaching fixtures are checked into the repository. Running the
examples does not fetch corpus data. Output databases and model caches are
separate from these inputs. Model-based enrichment can download model assets.

| File | Contents | Origin |
| --- | --- | --- |
| `c4.jsonl` | First 64 English training rows; unchanged text, URL, timestamp | [AllenAI C4](https://huggingface.co/datasets/allenai/c4), revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, `en/c4-train.00000-of-01024.json.gz` |
| `s2orc-train.jsonl` | First four full-text S2ORC training papers; unchanged ID, text, source | [AllenAI peS2o v2](https://huggingface.co/datasets/allenai/peS2o), revision `636a503e44a3ca1b58e01fb61eab0825cd574de0`, `data/v2/train-00010-of-00020.json.gz` |
| `s2orc-validation.jsonl` | First four full-text S2ORC validation papers; unchanged ID, text, source | Same peS2o revision, `data/v2/validation-00001-of-00002.json.gz` |
| `code.jsonl` | Word-count function and explanatory comments | Original toy code written for these examples |
| `reference.jsonl` | Telescope reference paragraph | Original toy prose; stands in for Wikipedia |
| `literature.jsonl` | Fictional observatory passage | Original toy prose; stands in for books |
| `questions.jsonl` | File-comparison question and answer | Original toy prose; stands in for Stack Exchange |

[sources.json](sources.json) records pinned upstream URLs, selection rules, row
counts, and SHA-256 digests for the C4 and peS2o samples. Both upstream dataset
cards specify ODC-BY; their original content terms also apply. Attribution is to
AllenAI's C4 and peS2o dataset releases and the source documents. C4 retains
source URLs. The peS2o IDs identify the original S2ORC papers in the pinned shards.
The Shakespeare lessons use the checked-in `src/premixdb/data/tiny_shakespeare_excerpt.txt`,
packaged with premixdb; it comes from [char-rnn's Tiny Shakespeare](https://github.com/karpathy/char-rnn).

These prefix samples are not representative. Keep peS2o validation papers for
inspection and use training papers in mixtures. In `recipes/mix.py`, C4 also
stands in for Common Crawl and peS2o for arXiv. The category names illustrate
published weights, not the original datasets or their preparation pipelines.
