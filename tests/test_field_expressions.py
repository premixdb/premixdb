"""The public field API is statically typed and compiles without model runtimes."""

from __future__ import annotations

import operator
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _type_support import invalid_call

import premixdb
from premixdb import (
    ContentType,
    DataTroveMetric,
    Language,
    Quality,
    Topic,
    content_type,
    datatrove,
    embedding,
    language,
    quality,
    topic,
    where,
)
from premixdb.v1 import query_pb2 as q


class ExpressionTests(unittest.TestCase):
    def test_requested_syntax_and_enum_access(self) -> None:
        expression = where(language.en > 0.8)
        self.assertEqual(expression.field_where.field, q.FIELD_LANGUAGE_EN)
        self.assertFalse(expression.field_where.field_name)
        self.assertEqual(expression.field_where.number, 0.8)
        self.assertEqual(expression, where(language[Language.EN] > 0.8))
        self.assertEqual(
            where(quality.educational_value >= 2), where(quality[Quality.EDUCATIONAL_VALUE] >= 2)
        )
        self.assertEqual(
            where(topic.science_and_tech > 0.5), where(topic[Topic.SCIENCE_AND_TECH] > 0.5)
        )
        label = where(content_type.label == ContentType.TUTORIAL).field_where
        self.assertEqual(label.projection, q.FieldComparison.TOP_CLASS)
        self.assertEqual(label.text, "Tutorial")
        self.assertEqual(where(embedding.harrier.component(4) > 0.2).field_where.component, 4)
        self.assertEqual(
            where(quality.writing_style.is_null()).field_where.projection, q.FieldComparison.IS_NULL
        )

    def test_catalog_is_complete(self) -> None:
        from premixdb._field_ids import DERIVED_FIELD_NAMES, FIELD_IDS, field_id, field_name

        self.assertEqual(set(FIELD_IDS), {member.value for member in premixdb.IntrinsicField})
        self.assertEqual(len(DERIVED_FIELD_NAMES), 200)
        self.assertEqual(len(q.IntrinsicField.keys()), 205)
        self.assertEqual(
            [
                q.FIELD_TEXT_BYTES,
                q.FIELD_TEXT_CHARACTERS,
                q.FIELD_OBJECT_URI,
                q.FIELD_SOURCE_CORPUS_ID,
            ],
            [1, 2, 3, 4],
        )
        for member in premixdb.IntrinsicField:
            self.assertEqual(field_name(field_id(member.value)), member.value)
        self.assertEqual(len(Language), 176)
        self.assertEqual(len(Topic), 24)
        self.assertEqual(len(ContentType), 24)
        from premixdb.enrichment.datatrove import DOC_METRICS, WORD_METRICS

        self.assertEqual({m.value for m in DataTroveMetric}, set(DOC_METRICS + WORD_METRICS))
        for member in Language:
            self.assertEqual(language[member].name, "language." + member.value)
        self.assertEqual(language.as_.name, "language.as")
        self.assertEqual(language.is_.name, "language.is")
        self.assertEqual(language.or_.name, "language.or")
        for topic_member in Topic:
            named = getattr(topic, topic_member.name.lower())
            self.assertEqual(where(named > 0.5), where(topic[topic_member] > 0.5))
            self.assertEqual(named.class_name, topic_member.value)
        for content_member in ContentType:
            named = getattr(content_type, content_member.name.lower())
            self.assertEqual(where(named > 0.5), where(content_type[content_member] > 0.5))
            self.assertEqual(named.class_name, content_member.value)
        self.assertIs(datatrove.n_words.value_type, int)
        self.assertIs(datatrove[DataTroveMetric.N_WORDS].value_type, int)
        for namespace in (language, topic, content_type, quality, datatrove, embedding):
            from premixdb._field_expr import ScalarField, VectorField

            for selector in vars(type(namespace)).values():
                if isinstance(selector, (ScalarField, VectorField)):
                    self.assertIn(selector.name, DERIVED_FIELD_NAMES)

    def test_request_names_fields_without_build_definitions(self) -> None:
        operation = where(language.en > 0.8)
        request = premixdb.query(b"s" * 32, steps=[operation])
        self.assertEqual(request.operations[0].field_where.field, q.FIELD_LANGUAGE_EN)
        self.assertFalse(request.field_snapshot_ids)
        self.assertFalse(request.operations[0].field_where.field_snapshot_id)
        self.assertFalse(hasattr(premixdb.Snapshot, "enrich"))
        self.assertFalse(hasattr(premixdb, "model_fields"))

    def test_invalid_python_expressions_are_rejected(self) -> None:
        for expression in (
            lambda: invalid_call(operator.gt, language.en, "0.8"),
            lambda: quality.writing_style > True,
            lambda: invalid_call(operator.getitem, topic, ContentType.TUTORIAL),
            lambda: topic.label == "Hardware",
            lambda: bool(language.en > 0.8),
            lambda: getattr(language, "zz"),
            lambda: where(language.en > float("nan")),
        ):
            with self.assertRaises((TypeError, ValueError, AttributeError)):
                expression()

    def test_importing_typed_fields_loads_no_workers(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from premixdb import where, language; import sys; "
                "assert where(language.en > .8); "
                "assert not {'torch','datatrove','dupekit','premixdb.engine.datasets','transformers'} & sys.modules.keys()",
            ],
            check=True,
        )

    @unittest.skipUnless(Path(sys.executable).with_name("ty").exists(), "install ty")
    def test_type_checker_accepts_fields_and_rejects_wrong_value_types(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typed_fields.py"
            prelude = (
                "from typing import assert_type\n"
                "from premixdb import where, language, topic, content_type, ContentType, datatrove, DataTroveMetric\n"
                "from premixdb.v1.query_pb2 import Operation\n"
            )
            path.write_text(
                prelude
                + "\n".join(
                    [
                        "assert_type(where(language.en > 0.8), Operation)",
                        "assert_type(where(topic.science_and_tech > 0.5), Operation)",
                        "assert_type(where(content_type.label == ContentType.TUTORIAL), Operation)",
                        "assert_type(where(datatrove[DataTroveMetric.N_WORDS] >= 10), Operation)",
                        "assert_type(where(datatrove[DataTroveMetric.DIGIT_RATIO] < 0.5), Operation)",
                    ]
                )
            )
            command = [
                str(Path(sys.executable).with_name("ty")),
                "check",
                "--project",
                str(root),
                str(path),
            ]
            valid = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
            path.write_text(
                prelude
                + 'where(language.en > "English")\nwhere(content_type.label == "Tutorial")\n'
            )
            invalid = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("English", invalid.stdout)
            self.assertIn("Tutorial", invalid.stdout)


if __name__ == "__main__":
    unittest.main()
