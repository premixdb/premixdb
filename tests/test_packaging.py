"""Exercise package builds without any pre-generated protobuf bindings."""

from __future__ import annotations

import configparser
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
class PackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "project"
        self.root.mkdir()
        for name in ("pyproject.toml", "MANIFEST.in", "README.md"):
            shutil.copy2(ROOT / name, self.root / name)
        shutil.copytree(ROOT / "proto", self.root / "proto")
        shutil.copytree(
            ROOT / "src/premixdb",
            self.root / "src/premixdb",
            ignore=shutil.ignore_patterns("*_pb2.py", "*_pb2.pyi", "*_pb2_grpc.py", "__pycache__"),
        )
        shutil.copy2(ROOT / "src/_premixdb_build.py", self.root / "src")
        (self.root / "scripts").mkdir()
        for name in ("generate_protos.py", "prepare_s2orc.py"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts")
        (self.root / "examples").mkdir()
        shutil.copy2(ROOT / "examples/01_tiny_shakespeare_snapshots.py", self.root / "examples")
        shutil.copy2(ROOT / "examples/04_s2orc_distributions.py", self.root / "examples")
        shutil.copy2(ROOT / "examples/_tutorial.py", self.root / "examples")
        shutil.copy2(ROOT / "examples/README.md", self.root / "examples")

    def run_python(
        self, code: str, *, root: Path | None = None, success: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root or self.root,
            text=True,
            capture_output=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def expected_bindings(self) -> set[str]:
        return {
            str(p.relative_to(self.root / "proto").with_name(f"{p.stem}_pb2{suffix}"))
            for p in (self.root / "proto/premixdb").rglob("*.proto")
            for suffix in (".py", ".pyi")
        }

    def assert_importable(self, path: Path, *, training: bool = False) -> None:
        code = f"""
import importlib.abc
import sys
from pathlib import Path

class BlockBuildTools(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('grpc_tools', 'grpc', 'setuptools'):
            raise ModuleNotFoundError('build tool unavailable: ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockBuildTools())
sys.path.insert(0, {str(path)!r})
import premixdb
from premixdb._cli import _demo_sources
from premixdb.v1 import corpus_pb2
from premixdb.internal import derivation_pb2
assert Path(premixdb.__file__).is_relative_to({str(path)!r})
message = premixdb.corpus('build-smoke-test')
assert corpus_pb2.CreateCorpusRequest.FromString(message.SerializeToString()) == message
assert "grpc" not in sys.modules
assert derivation_pb2.DESCRIPTOR
assert premixdb.__version__ == '0.1.1'
sources = _demo_sources()
assert len(sources) == 7222
assert sources[0].text.startswith('First Citizen:')
import os
import tempfile
os.environ.pop('PREMIXDB_GIT_COMMIT', None)
with tempfile.TemporaryDirectory() as storage:
    with premixdb.PremixDB(storage=storage) as db:
        snapshot = db.Corpus('wheel-workflow', [premixdb.Source('a', 'hello')])
        assert db.Corpus('wheel-workflow').id == snapshot.id
        query = snapshot.query()
        assert query.profile().output_documents == 1
        assert query.preview()[0]['text'] == 'hello'
        dataset = query.mix(tokenizer=premixdb.ByteTokenizer(), sequence_length=8)[0]
        assert dataset[0].tokens == list(b'hello') + [256, 257, 257]
        assert dataset[0].mask == [True] * 6 + [False] * 2
"""
        if training:
            code += """
from torch.utils.data import DataLoader
with tempfile.TemporaryDirectory() as storage:
    with premixdb.PremixDB(storage=storage) as db:
        dataset = db.Corpus('training', [premixdb.Source('a', 'hello')]).query().mix(sequence_length=8)[0]
        assert dataset[0].tokens == [31373] + [50256] * 7
        data = dataset.torch()
    batch = next(iter(DataLoader(data, batch_size=1)))
    assert batch['input_ids'].tolist() == [[31373] + [50256] * 7]
    assert batch['attention_mask'].tolist() == [[1, 1] + [0] * 6]
    assert batch['labels'].tolist() == [[31373, 50256] + [-100] * 6]
"""
        self.run_python(code)

    def assert_wheel(self, root: Path, *, training: bool = False) -> None:
        wheel = next((root / "dist").glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            self.assertTrue(self.expected_bindings() <= names)
            self.assertIn("premixdb/data/gpt2-tokenizer.json", names)
            self.assertIn("premixdb/data/gpt2-LICENSE.txt", names)
            self.assertFalse(any("obsolete_pb2" in name for name in names))
            metadata = archive.read("premixdb-0.1.1.dist-info/METADATA").decode()
            self.assertNotIn("Requires-Dist: grpcio-tools", metadata)
            self.assertNotIn("Requires-Dist: grpcio;", metadata)
            self.assertFalse(any(name.endswith("_pb2_grpc.py") for name in names))
            self.assertNotIn("Requires-Dist: setuptools", metadata)
            self.assertNotIn("Provides-Extra: grain", metadata)
            self.assertNotIn("Requires-Dist: apache-beam", metadata)
            self.assertIn("Provides-Extra: huggingface", metadata)
            self.assertIn("Requires-Dist: ipython<10,>=9.17.1", metadata)
            self.assertNotIn("Requires-Dist: boto3", metadata)
            self.assertNotIn("Requires-Dist: google-cloud-storage", metadata)
            self.assertEqual(
                archive.read("premixdb/data/tiny_shakespeare.txt"),
                (ROOT / "src/premixdb/data/tiny_shakespeare.txt").read_bytes(),
            )
            self.assertFalse(any(name.endswith(".html") for name in names))
            self.assertNotIn("premixdb/server/inspector.py", names)
            self.assertIn("premixdb/__main__.py", names)
            entry_points = configparser.ConfigParser()
            entry_points.read_string(
                archive.read("premixdb-0.1.1.dist-info/entry_points.txt").decode()
            )
            self.assertEqual(entry_points["console_scripts"]["premixdb"], "premixdb._cli:main")
            installed = root / "installed"
            archive.extractall(installed)
        self.assert_importable(installed, training=training)
        self.run_python(
            f"""
import runpy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, {str(installed)!r})
sys.path.insert(0, {str(root / "examples")!r})
from _tutorial import TINY, tiny_sources
assert TINY.is_relative_to({str(installed)!r})
assert len(tiny_sources()) == 9
lesson = runpy.run_path({str(root / "examples/01_tiny_shakespeare_snapshots.py")!r})
with tempfile.TemporaryDirectory() as storage:
    lesson['main'].__globals__['STORAGE'] = Path(storage)
    lesson['main']()
""",
            root=root,
        )

    def test_sdist_contains_schemas_and_rebuilds_bindings(self) -> None:
        # Even a developer's ignored local outputs must stay out of the sdist.
        source = self.root / "src/premixdb/v1"
        (source / "corpus_pb2.py").write_text("raise AssertionError('stale source binding')")
        (source / "obsolete_pb2.pyi").write_text("stale: str")
        self.run_python("from setuptools.build_meta import build_sdist; build_sdist('dist')")
        with tarfile.open(next((self.root / "dist").glob("*.tar.gz"))) as archive:
            names = archive.getnames()
            self.assertFalse(any("_pb2" in name for name in names))
            prefix = "premixdb-0.1.1/"
            for name in (
                "src/_premixdb_build.py",
                "src/premixdb/data/gpt2-tokenizer.json",
                "src/premixdb/data/gpt2-LICENSE.txt",
                "proto/premixdb/v1/corpus.proto",
                "scripts/prepare_s2orc.py",
                "examples/01_tiny_shakespeare_snapshots.py",
                "examples/04_s2orc_distributions.py",
                "examples/_tutorial.py",
                "examples/README.md",
            ):
                self.assertIn(prefix + name, names)
            archive.extractall(self.root / "unpacked", filter="data")
        unpacked = self.root / "unpacked/premixdb-0.1.1"
        self.run_python(
            "from setuptools.build_meta import build_wheel; build_wheel('dist')", root=unpacked
        )
        self.assert_wheel(unpacked)
        self.assertFalse(list((unpacked / "src").rglob("*_pb2.py")))

    def test_wheel_replaces_stale_build_outputs(self) -> None:
        for directory in ("src/premixdb/v1", "build/lib/premixdb/v1"):
            source = self.root / directory
            source.mkdir(parents=True, exist_ok=True)
            (source / "corpus_pb2.py").write_text("raise AssertionError('stale binding')")
            (source / "obsolete_pb2.py").write_text("raise AssertionError('obsolete binding')")
        removed = self.root / "build/lib/premixdb/server"
        removed.mkdir(parents=True)
        (removed / "inspector.html").write_text("stale browser interface")
        (removed / "inspector.py").write_text("raise AssertionError('removed module')")
        self.run_python("from setuptools.build_meta import build_wheel; build_wheel('dist')")
        self.assert_wheel(self.root, training=True)

    def test_editable_builds_generate_local_bindings(self) -> None:
        for mode in ("default", "strict"):
            with self.subTest(mode=mode):
                for path in (self.root / "src").rglob("*_pb2*"):
                    path.unlink()
                settings = {"editable_mode": "strict"} if mode == "strict" else None
                self.run_python(
                    "from setuptools.build_meta import build_editable; "
                    f"build_editable('dist-{mode}', config_settings={settings!r})"
                )
                for path in self.expected_bindings():
                    self.assertTrue((self.root / "src" / path).exists(), path)
                if mode == "strict":
                    installed = next((self.root / "build").glob("__editable__.*"))
                else:
                    installed = self.root / "src"
                self.assert_importable(installed)

    def test_invalid_schema_fails_wheel_and_editable_builds(self) -> None:
        with (self.root / "proto/premixdb/v1/corpus.proto").open("a") as schema:
            schema.write("\ninvalid syntax;\n")
        for command in (
            "build_wheel('dist')",
            "build_editable('dist')",
            "build_editable('dist', config_settings={'editable_mode': 'strict'})",
        ):
            with self.subTest(command=command):
                self.run_python(
                    "from setuptools.build_meta import build_wheel, build_editable; " + command,
                    success=False,
                )
        self.assertFalse(list((self.root / "dist").glob("*.whl")))


if __name__ == "__main__":
    unittest.main()
