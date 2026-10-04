"""Offline semantics plus installed-dependency integration tests (no model downloads)."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING, Unpack, cast
from unittest.mock import patch

from _type_support import (
    EncodeOptions,
    Vectors,
    numeric,
)

if TYPE_CHECKING:
    from torch import Tensor

from premixdb.enrichment import (
    DataTroveFields,
    Document,
    DupekitIndex,
    Embeddings,
    ModelPin,
    QuRating,
    WebOrganizer,
    from_datatrove,
    probabilities,
    top_class,
)
from premixdb.enrichment.interfaces import ModelTokenizer, SequenceModel
from premixdb.enrichment.models import QUALITY_DIMENSIONS
from premixdb.enrichment.types import field
from premixdb.runtime.enrichment import numeric_vector
from premixdb.v1 import field_pb2 as fields

REVISION = "a" * 40


class ClassificationTests(unittest.TestCase):
    def test_softmax_preserves_label_order_and_handles_large_logits(self) -> None:
        spec = field("topic", width=3, classes=("Z", "A", "B"))
        result = probabilities(spec, [10000, 10001, -10000])
        self.assertEqual(list(result), ["Z", "A", "B"])
        self.assertAlmostEqual(sum(result.values()), 1)
        self.assertEqual(top_class(spec, [10000, 10001, -10000])[0], "A")
        spec.classification.temperature = 2
        self.assertLess(probabilities(spec, [10000, 10001, -10000])["A"], result["A"])

    def test_regression_is_not_classification_and_invalid_values_fail(self) -> None:
        with self.assertRaises(ValueError):
            probabilities(field("quality.educational_value"), [2])
        spec = field("type", width=2, classes=("one", "two"))
        for values in ([1], [math.nan, 0], [0, math.inf]):
            with self.assertRaises(ValueError):
                probabilities(spec, values)
        for temperature in (0, -1, math.nan):
            spec.classification.temperature = temperature
            with self.assertRaises(ValueError):
                probabilities(spec, [0, 1])

    def test_sigmoid_is_independent_and_has_no_top_class(self) -> None:
        spec = field("tags", width=2, classes=("a", "b"))
        spec.classification.transform = fields.PROBABILITY_TRANSFORM_SIGMOID
        self.assertEqual(probabilities(spec, [1000, 1000]), {"a": 1, "b": 1})
        with self.assertRaises(ValueError):
            top_class(spec, [1000, 1000])


def classifier_tokenizer() -> ModelTokenizer:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "1": 1, "[PAD]": 2, "3": 3, "[BOS]": 4}))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 4)])
    return cast(
        ModelTokenizer,
        PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]"
        ),
    )


class ModelTests(unittest.TestCase):
    def test_builtin_metadata_and_empty_batches_do_not_load_models(self) -> None:
        from premixdb.enrichment.language import LanguageScores
        from premixdb.runtime.catalog import recipe
        from premixdb.runtime.enrichment import producer

        with patch(
            "premixdb.enrichment.models._sequence_model", side_effect=AssertionError("model load")
        ):
            for name in (
                "quality.educational_value",
                "weborganizer.topic",
                "weborganizer.content_type",
                "embedding.harrier",
                "language.en",
            ):
                worker = producer(recipe(name))
                assert not isinstance(worker, DupekitIndex)
                assert worker.fields
                assert worker.definition
                assert worker.compute([]) == []
            quality = producer(recipe("quality.educational_value"))
            assert quality.compute([Document("empty", " ")])[0]["quality.educational_value"] is None
            language = producer(recipe("language.en"))
            with patch.object(LanguageScores, "_lid", new_callable=property):
                assert language.compute([Document("empty", "")])[0]["language.en"] is None

    def test_quality_covers_tail_and_uses_content_weighted_windows(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(num_labels=4))
        windows_seen = []
        tokenizer = classifier_tokenizer()

        def scores(
            model: SequenceModel, windows: list[dict[str, list[int]]], rows: int, width: int
        ) -> list[list[float]]:
            windows_seen.extend(windows)
            return [[float(window["input_ids"][1])] * 4 for window in windows]

        with (
            patch("premixdb.enrichment.models._sequence_model", return_value=(tokenizer, model)),
            patch.object(tokenizer, "pad", side_effect=lambda windows, **kwargs: windows),
            patch("premixdb.enrichment.models._logits", side_effect=scores),
        ):
            provider = QuRating(ModelPin("princeton-nlp/QuRater-1.3B", REVISION), batch_size=2)
            docs = [Document("a", " ".join(["1"] * 511 + ["3"] * 2)), Document("empty", "")]
            rows = provider.compute(docs)
            for name in QUALITY_DIMENSIONS:
                self.assertAlmostEqual(numeric(rows[0][f"quality.{name}"]), 517 / 513)
                self.assertIsNone(rows[1][f"quality.{name}"])
            self.assertEqual([len(w["input_ids"]) for w in windows_seen], [512, 3])
            self.assertTrue(all(not f.HasField("classification") for f in provider.fields))
            self.assertEqual(rows, provider.compute(docs[:1]) + provider.compute(docs[1:]))

    def test_quality_runs_with_installed_transformers_without_model_downloads(self) -> None:
        from transformers.utils import is_torch_available

        if not is_torch_available():
            self.skipTest("model inference requires PyTorch >=2.5")
        from transformers import BertConfig, BertForSequenceClassification

        config = BertConfig(
            vocab_size=5,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            pad_token_id=2,
        )
        config.num_labels = 4
        model = BertForSequenceClassification(config).eval()
        provider = QuRating(ModelPin("princeton-nlp/QuRater-1.3B", REVISION))
        provider._model_pair = (classifier_tokenizer(), model)
        rows = provider.compute([Document("short", "1 3"), Document("long", "1 " * 513)])
        self.assertEqual([row["id"] for row in rows], ["short", "long"])
        for row in rows:
            self.assertTrue(
                all(math.isfinite(numeric(row[f"quality.{name}"])) for name in QUALITY_DIMENSIONS)
            )

    def test_weborganizer_numeric_label_order_url_and_raw_logits(self) -> None:
        labels = {str(i): f"class {i}" for i in reversed(range(24))}
        model = SimpleNamespace(config=SimpleNamespace(id2label=labels))
        calls = []

        def tokenize(texts: list[str], **kwargs: str | bool | int) -> dict[str, Tensor]:
            calls.extend(texts)
            return {}

        with (
            patch("premixdb.enrichment.models._sequence_model", return_value=(tokenize, model)),
            patch("premixdb.enrichment.models._logits", return_value=[list(range(24))]),
        ):
            provider = WebOrganizer(
                ModelPin("WebOrganizer/TopicClassifier", REVISION), task="topic"
            )
            spec = provider.fields[0]
            self.assertEqual(list(spec.classification.classes), [f"class {i}" for i in range(24)])
            result = provider.compute([Document("a", "web text", "https://example.org")])
            self.assertEqual(calls, ["https://example.org\n\nweb text"])
            self.assertEqual(result[0][spec.name], list(range(24)))
            self.assertEqual(top_class(spec, numeric_vector(result[0][spec.name]))[0], "class 23")
            with self.assertRaises(ValueError):
                provider.compute([Document("b", "missing URL")])
            no_url = WebOrganizer(
                ModelPin("WebOrganizer/FormatClassifier-NoURL", REVISION), task="content_type"
            )
            no_url.compute([Document("a", "plain text")])
            self.assertEqual(calls[-1], "plain text")

    def test_embedding_document_and_query_prompts(self) -> None:
        calls = []

        class Encoder:
            def __init__(self, *args: str, **kwargs: str) -> None:
                self.max_seq_length = None

            def get_embedding_dimension(self) -> int:
                return 2

            def encode(self, texts: list[str], **kwargs: Unpack[EncodeOptions]) -> Vectors:
                calls.append(kwargs)
                return cast(Vectors, SimpleNamespace(tolist=lambda: [[1, 0] for _ in texts]))

        with patch.dict(
            "sys.modules", {"sentence_transformers": SimpleNamespace(SentenceTransformer=Encoder)}
        ):
            for family, repository, prompt in (
                ("harrier", "microsoft/harrier-oss-v1-0.6b", "web_search_query"),
            ):
                provider = Embeddings(ModelPin(repository, REVISION), family=family)
                self.assertEqual(provider.compute([]), [])
                provider.compute([Document("a", "text")])
                provider.encode_queries(["query"])
                self.assertEqual(calls[-2]["prompt"], "")
                self.assertEqual(calls[-1]["prompt_name"], prompt)
                self.assertTrue(calls[-1]["normalize_embeddings"])

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            ModelPin("repo", "main")
        with self.assertRaises(ValueError):
            DupekitIndex(num_perms=10, num_bands=3)
        with self.assertRaises(ValueError):
            DupekitIndex(seed=-1)


class DupekitIntegrationTests(unittest.TestCase):
    def test_partition_equivalence_complete_membership_and_empty_schema(self) -> None:
        from blake3 import blake3

        provider = DupekitIndex(num_perms=16, num_bands=4)
        docs = [
            Document("a", "a useful repeated document"),
            Document("b", "a useful repeated document"),
            Document("c", "different useful text"),
            Document("short", "é"),
            Document("empty", ""),
        ]
        batch = provider.compute(docs)
        rows = batch.to_pylist()
        self.assertEqual(batch.schema, provider.compute([]).schema)
        self.assertEqual(
            rows, provider.compute(docs[:2]).to_pylist() + provider.compute(docs[2:]).to_pylist()
        )
        self.assertEqual(len(rows), len(docs))
        self.assertEqual(rows[0]["exact_hash"], blake3(docs[0].text.encode()).digest())
        self.assertEqual(len(rows[0]["exact_hash"]), 32)
        self.assertEqual(rows[0]["minhash"], rows[1]["minhash"])
        self.assertEqual(rows[0]["lsh_buckets"], rows[1]["lsh_buckets"])
        self.assertNotEqual(rows[0]["minhash"], rows[2]["minhash"])
        self.assertIsNone(rows[3]["minhash"])
        self.assertIsNone(rows[4]["lsh_buckets"])
        with self.assertRaises(ValueError):
            provider.compute([docs[0], docs[0]])


class DataTroveIntegrationTests(unittest.TestCase):
    def test_real_fields_preserve_empty_documents(self) -> None:
        provider = DataTroveFields()
        docs = [Document("a", "the cat is here."), Document("empty", ""), Document("space", " ")]
        rows = provider.compute(docs)
        self.assertEqual(rows[0]["datatrove.length"], 16)
        self.assertGreater(numeric(rows[0]["datatrove.n_words"]), 0)
        self.assertGreater(numeric(rows[0]["datatrove.stop_word_ratio"]), 0)
        self.assertEqual(rows[1]["datatrove.n_words"], 0)
        self.assertIsNone(rows[1]["datatrove.avg_word_length"])
        self.assertEqual(rows[2]["datatrove.white_space_ratio"], 1)
        self.assertEqual(rows, [row for doc in docs for row in provider.compute([doc])])
        self.assertEqual(set(rows[0]) - {"id"}, {f.name for f in provider.fields})

    def test_datatrove_reader_bridge(self) -> None:
        from datatrove.data import Document as DTDocument

        source = DTDocument(id="42", text="Hello", metadata={"url": "https://example.org"})
        result = list(from_datatrove([source], namespace="crawl"))[0]
        self.assertEqual(result.url, "https://example.org")
        self.assertNotEqual(result.id, list(from_datatrove([source], namespace="other"))[0].id)


if __name__ == "__main__":
    unittest.main()
