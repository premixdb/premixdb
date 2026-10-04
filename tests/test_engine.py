"""Pinned PremixDB mixture vectors and planning boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _type_support import invalid_call

from premixdb.engine import execution as native

ROOT = Path(__file__).resolve().parents[1]


class KernelBoundaryTests(unittest.TestCase):
    def test_mixture_identity_tokens_and_provenance_match_golden_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code = native.CodeVersion("local://mixture-test", "a" * 40, "09" * 32)
            snapshot = native.Store(directory).capture(
                "01" * 16, [native.Source("a", "abcd"), native.Source("b", "XYZ")], code
            )
            query = native.execute([snapshot], [], code)
            pool = query.mixture_pool("object.uri", {})
            for expected in json.loads(
                (Path(__file__).parent / "fixtures/mixture-v1.json").read_text()
            ):
                args = (
                    {"a": 6, "b": 3},
                    "04" * 32,
                    expected["seed"],
                    True,
                    2,
                    expected["length"],
                    256,
                    expected["padding"],
                )
                ds = pool.dataset(*args)
                self.assertEqual(ds.id, expected["id"])
                self.assertEqual(ds.summary(), expected["summary"])
                self.assertEqual(ds.occurrences(), expected["occurrences"])
                self.assertEqual([s.tokens for s in ds.reader((0, 1, 0, 1))], expected["tokens"])
                profile = pool.profile(*args)
                self.assertEqual(profile["stratum_tokens"], pool.exposure(ds))
                for topology in ((0, 1, 0, 1), (1, 3, 2, 4), (0, 2**32 - 1, 0, 2**32 - 1)):
                    reader = ds.reader(topology)
                    while True:
                        state = json.loads(json.dumps(reader.checkpoint()))
                        self.assertEqual(
                            [s.ordinal for s in ds.reader(topology, state)],
                            list(range(state["next_ordinal"], len(ds), topology[1] * topology[3]))
                            if state["next_ordinal"] is not None
                            else [],
                        )
                        if next(reader, None) is None:
                            break
                for invalid in (
                    {"version": 2},
                    {"dataset": "00" * 32},
                    {"next_ordinal": len(ds)},
                    {"next_ordinal": True},
                ):
                    with self.assertRaises(ValueError):
                        invalid_call(
                            ds.reader, (0, 1, 0, 1), ds.reader((0, 1, 0, 1)).checkpoint() | invalid
                        )
            for args in (
                ({"a": 5, "b": 0}, False, None),
                ({"a": 9, "b": 0}, True, 2),
                ({"a": 0}, True, None),
                ({"a": 0, "b": 0}, True, 0),
            ):
                allocations, replacement, cap = args
                with self.assertRaises(ValueError):
                    pool.dataset(allocations, "04" * 32, 0, replacement, cap, 1, None, None)

    def test_query_planning_does_not_import_execution(self) -> None:
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('premixdb.engine.execution', 'premixdb.engine.datasets'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import premixdb
from premixdb.runtime import compile_query
p = compile_query(premixdb.query('01'*32, steps=[premixdb.where(premixdb.text.bytes > 1)]))
assert len(p.id) == 32 and len(p.git_commit) in (20,32)
"""
        subprocess.run([sys.executable, "-c", code], check=True)
