# Fields

```python
query = snapshot.query(
    steps=[
        p.where(p.language.en >= 0.75),
        p.where(p.quality.educational_value >= 1.0),
        p.where(p.topic.label == p.Topic.SCIENCE_AND_TECH),
    ]
)
query.preview()
query.profile()
```

Fields compute when the query runs and reuse cached results when thresholds change.
Models download on first use and run on CPU. Quality, topic, and embedding models
require PyTorch 2.5+.

| Field | Value |
| --- | --- |
| `text.bytes`, `text.characters` | Text size |
| `datatrove.n_words`, `datatrove.*` | Word counts and text statistics |
| `language.en`, `language.label` | Language probability and label |
| `quality.educational_value`, `quality.*` | Regression scores |
| `topic.label`, `topic.science_and_tech` | Topic label and probability |
| `content_type.label` | Content-type label |
| `embedding.harrier` | Normalized document vector |

Quality scores aren't probabilities. Word counts aren't tokenizer counts.
Missing values are null; use `field.is_null()` to select them.

Built-in fields use [DataTrove](https://github.com/huggingface/datatrove),
[QuRating](https://github.com/princeton-nlp/QuRating),
[WebOrganizer](https://github.com/CodeCreator/WebOrganizer), and
[Harrier](https://huggingface.co/microsoft/harrier-oss-v1-0.6b).
See [curation](curation.md) for similarity dedupe and mixture domains.
