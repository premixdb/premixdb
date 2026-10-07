# Local Shakespeare benchmark fixture

`benchmark.jsonl` contains one example from
[Flower Labs' LEAF Shakespeare release](https://huggingface.co/datasets/flwrlabs/shakespeare),
revision `2cfe3a5ba0d9b34634ade086803a479b2c2b8e11`. LEAF is a next-character
prediction benchmark built from Shakespeare's Complete Works. Each original
example has an 80-character input `x`, a next character `y`, and a `character_id`.
The local `text` is `x + y`, unchanged. Original fields remain in the file.

This example is selected for local decontamination: it shares an exact 13-word
sequence with `speech/5385` in the bundled `tiny_shakespeare.txt`. That speech is
one of the eight documents in the shell's `demo` corpus. The other seven speeches
have no overlap at this threshold. Matching is case-sensitive and uses
whitespace-delimited words. The `demo_overlap` field records this selection.

The shell demo uses speeches 9, 13, 15, 16, 17, 22, 23, and 5385 from the full
text, retaining their original keys and text. Each is at least 100 characters.
The nine-block tutorial excerpt is a separate teaching input.

The source exposes only a `train` split. These rows are held out as local
references; they are not an official test split. This single example is not
representative and should not be used to report LEAF benchmark scores.

Writable shells publish it as `db.Corpus("benchmark")` when the name is absent,
and replace the previous built-in 128-row fixture. Custom corpora are preserved.
Outside the shell, capture the packaged input directly:

```python
from importlib.resources import as_file, files

with as_file(files("premixdb").joinpath("data/benchmark.jsonl")) as path:
    benchmark = db.Corpus(
        "benchmark",
        p.Source.read_jsonl(path, key_column="id"),
    )
```

`benchmark-source.json` records the upstream revision and file digest, downloaded
byte ranges and their digests, selection rule, and local SHA-256. Each local ID
encodes the source CSV record's zero-based byte offset. `benchmark-LICENSE.txt`
retains LEAF's BSD-2-Clause license and TalwalkarLab's 2018 copyright notice.
Attribution: William Shakespeare; LEAF, Caldas et al. (2018),
[LEAF: A Benchmark for Federated Settings](https://arxiv.org/abs/1812.01097);
Flower Labs' dataset conversion.
