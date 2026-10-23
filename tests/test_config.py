"""config: precedence, bounds, provenance, and a broken file never breaking anything."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import config  # noqa: E402


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.saved = config.PATH
        config.PATH = self.dir / "config.json"
        config._cache.update(mtime=None, data={}, checked=0.0)
        self.env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("TOKENSAVER_")}

    def tearDown(self):
        config.PATH = self.saved
        config._cache.update(mtime=None, data={}, checked=0.0)
        os.environ.update(self.env)

    def test_default_when_nothing_set(self):
        self.assertEqual(config.resolve("bg_after_min"), (15.0, "default"))

    def test_precedence_env_user_tuned_default(self):
        config.set_value("bg_after_min", 30, "tuned", why="test")
        self.assertEqual(config.resolve("bg_after_min"), (30.0, "tuned"))
        config.set_value("bg_after_min", 20, "user")
        self.assertEqual(config.resolve("bg_after_min"), (20.0, "user"))
        os.environ["TOKENSAVER_BG_AFTER_MIN"] = "45"
        try:
            self.assertEqual(config.resolve("bg_after_min"), (45.0, "env"))
        finally:
            del os.environ["TOKENSAVER_BG_AFTER_MIN"]
        config.unset("bg_after_min", "user")
        self.assertEqual(config.resolve("bg_after_min"), (30.0, "tuned"))

    def test_only_tuned_values_are_clamped(self):
        config.set_value("bg_after_min", 500, "tuned")
        self.assertEqual(config.get("bg_after_min"), 61.0)
        config.set_value("bg_after_min", 500, "user")
        self.assertEqual(config.get("bg_after_min"), 500.0)
        self.assertTrue(any(n == "bg_after_min" and note for n, _, _, note in config.report()))

    def test_provenance_recorded(self):
        config.set_value("bg_index", 2000, "tuned", why="clear win", evidence={"gain_pp": 1.4})
        slot = json.loads(config.PATH.read_text())["settings"]["bg_index"]["tuned"]
        self.assertEqual((slot["value"], slot["why"], slot["evidence"]["gain_pp"]), (2000, "clear win", 1.4))

    def test_corrupt_file_is_quarantined_not_fatal(self):
        config.PATH.write_text("{not json")
        self.assertEqual(config.resolve("port"), (8787, "default"))
        self.assertTrue(list(self.dir.glob("config.json.corrupt.*")))

    def test_bool_and_bad_values(self):
        os.environ["TOKENSAVER_BG_COMPACT"] = "1"
        try:
            self.assertIs(config.get("bg_compact"), True)
        finally:
            del os.environ["TOKENSAVER_BG_COMPACT"]
        os.environ["TOKENSAVER_PROXY_PORT"] = "not-a-number"
        try:
            self.assertEqual(config.resolve("port"), (8787, "default"))
        finally:
            del os.environ["TOKENSAVER_PROXY_PORT"]

    def test_concurrent_writes_lose_nothing(self):
        names = ["audit_every", "bg_index", "bg_keep_tokens", "vault_keep_days"]
        values = {"audit_every": 30, "bg_index": 1500, "bg_keep_tokens": 40000, "vault_keep_days": 9}
        threads = [threading.Thread(target=config.set_value, args=(n, values[n], "user")) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        config._cache.update(mtime=None, checked=0.0)
        for n in names:
            self.assertEqual(config.get(n), values[n])


if __name__ == "__main__":
    unittest.main()
