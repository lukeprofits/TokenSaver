"""Retention: old or excess request bodies go, rows stay, shared blobs survive."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import vault  # noqa: E402


class PruneTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.saved = (vault.VAULT_DIR, vault.DB_PATH, vault.BLOB_DIR)
        vault.VAULT_DIR, vault.DB_PATH, vault.BLOB_DIR = self.dir, self.dir / "vault.db", self.dir / "blobs"

    def tearDown(self):
        vault.VAULT_DIR, vault.DB_PATH, vault.BLOB_DIR = self.saved

    def _rec(self, body, kind="api_request"):
        return vault.record(kind, content=body, summary="s", meta={"k": 1}, index=False)

    def test_age_and_size_cap(self):
        big = "x" * 10_000
        ids = [self._rec(big + str(i)) for i in range(5)]
        keep_other = vault.record("tool_result", content=big, summary="t")
        # nothing old, 5 x ~10K under a 100K cap: nothing goes
        self.assertEqual(vault.prune(7, 100_000)["pruned"], 0)
        # cap of 25K: the three oldest go
        done = vault.prune(7, 25_000)
        self.assertEqual(done["pruned"], 3)
        self.assertEqual(vault.read(ids[0])[1], vault.PRUNED)
        self.assertTrue(vault.read(ids[4])[1].startswith("xxx"))
        self.assertTrue(vault.read(keep_other)[1].startswith("xxx"), "other kinds untouched")
        # everything older than "now + 1 day" goes
        done = vault.prune(1, 10**9, now=time.time() + 2 * 86400)
        self.assertEqual(done["pruned"], 2)

    def test_shared_blob_survives(self):
        body = "y" * 10_000
        a = self._rec(body)
        b = vault.record("tool_result", content=body, summary="same bytes")
        vault.prune(7, 0)
        self.assertEqual(vault.read(a)[1], vault.PRUNED)
        self.assertEqual(vault.read(b)[1], body, "blob still referenced by a kept row")

    def test_metadata_kept(self):
        rid = self._rec("z" * 10_000)
        vault.prune(7, 0)
        self.assertEqual(vault.read_meta(rid).get("k"), 1)


if __name__ == "__main__":
    unittest.main()
