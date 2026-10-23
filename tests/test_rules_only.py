"""Compaction without a model: prepare needs no ollama, the quality signals can
turn it off, and setup gives every machine compaction (models only with --with-prune)."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands
from tests import REAL_HOME  # noqa: E402

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokensaver import background, config, proxy, setup

REAL = REAL_HOME / ".claude" / "settings.json"
_BEFORE = REAL.read_bytes() if REAL.exists() else None


def tearDownModule():
    assert (REAL.read_bytes() if REAL.exists() else None) == _BEFORE, "a test touched the real settings.json"


def history(n=80):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": [{"type": "text", "text": f"step {i}"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": f"r{i}", "name": "Bash",
                                                   "input": {"command": f"run {i}"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"r{i}",
                                              "content": "\n".join(f"log {i}.{k} ok" for k in range(300))}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"step {i} done"}]}]
    return out + [{"role": "user", "content": [{"type": "text", "text": "the newest request"}]}]


class PrepareTest(unittest.TestCase):
    def test_no_model_call(self):
        with mock.patch.object(background, "ollama", side_effect=AssertionError("model called")):
            got = background.prepare(history())
        self.assertTrue(got and got["gate"]["passed"])
        self.assertLess(got["new_tokens"], got["old_tokens"])
        self.assertEqual(got["text"], background._extract(history(), got["cut"])[0]["text"],
                         "same text as the rules-only build")


class QualityOffTest(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.db = d / "vault.db"
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, kind TEXT, meta TEXT)")
        conn.commit()
        conn.close()
        self.saved_cfg = config.PATH
        config.PATH = d / "config.json"
        config.load(force=True)
        self.recorded = []
        self.p = [mock.patch.object(proxy.vault, "DB_PATH", self.db),
                  mock.patch.object(proxy.vault, "record", lambda kind, **k: self.recorded.append(kind)),
                  mock.patch.object(proxy, "BG_COMPACT", True)]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        config.PATH = self.saved_cfg
        config.load(force=True)

    def follow(self, n, before, after):
        conn = sqlite3.connect(self.db)
        for _ in range(n):
            conn.execute("INSERT INTO events (kind, meta) VALUES ('bg_followup', ?)", (json.dumps(
                {"before_requests": 30, "after_requests": 30, "before_rereads": before, "after_rereads": after}),))
        conn.commit()
        conn.close()

    def test_too_few_or_small_rise_keeps_it_on(self):
        self.follow(4, 0, 10)
        proxy._quality_verdict()
        self.assertTrue(config.get("bg_compact") or config.resolve("bg_compact")[1] == "default")
        self.assertEqual(self.recorded, [])
        self.follow(1, 1, 3)              # 5 now, but (10*4+3)/150 - 1/150 is a real rise: next test
        self.assertIsNotNone(proxy.quality_rise([{"before_requests": 30, "after_requests": 30}] * 5))
        self.assertAlmostEqual(proxy.quality_rise([{"before_requests": 30, "after_requests": 30,
                                                   "before_rereads": 1, "after_rereads": 3}] * 5), 2 / 30)

    def test_clear_rise_turns_it_off_with_reason(self):
        config.set_value("bg_compact", True, "user", why="setup")
        self.follow(5, 0, 6)              # +0.2 per request
        proxy._quality_verdict()
        self.assertFalse(config.get("bg_compact"))
        self.assertFalse(proxy.BG_COMPACT)
        self.assertIn("bg_quality_off", self.recorded)
        self.assertTrue(config.load()["settings"]["bg_compact"]["user"]["why"].startswith("quality signals"))

    def test_hand_set_value_wins(self):
        config.set_value("bg_compact", True, "user", why="I want it")
        self.follow(5, 0, 6)
        proxy._quality_verdict()
        self.assertTrue(config.get("bg_compact"))
        self.assertEqual(self.recorded, [])


class SetupTierTest(unittest.TestCase):
    def setUp(self):
        self.saved_cfg = config.PATH
        config.PATH = Path(tempfile.mkdtemp()) / "config.json"
        config.load(force=True)

    def tearDown(self):
        config.PATH = self.saved_cfg
        config.load(force=True)

    def plan(self, gb, platform, with_prune):
        from tokensaver import machine, prune, service
        with mock.patch.object(machine, "ram_bytes", return_value=gb * 2 ** 30), \
             mock.patch.object(service, "platform", return_value=platform), \
             mock.patch.object(setup, "ollama_models", return_value=None), \
             mock.patch.object(prune, "installed", return_value={}), \
             mock.patch.object(setup.shutil, "which", side_effect=lambda c: f"/usr/bin/{c}"), \
             mock.patch.object(setup, "billing_type", return_value="test"):
            steps = setup.plan(route="settings", service=False, statusbar="no", with_prune=with_prune)
        return {s.name: s for s in steps}

    def test_every_machine_gets_compaction_without_models(self):
        for gb, plat in ((8, "macos"), (16, "linux"), (24, "macos"), (64, "macos"), (64, "linux")):
            steps = self.plan(gb, plat, with_prune=False)
            self.assertIn("bg_compact=True", steps["config"].detail)
            self.assertNotIn("local_model", steps["config"].detail)
            self.assertNotIn("ollama", steps)
            self.assertNotIn("prune", steps)

    def test_with_prune_sizes_the_models(self):
        small = self.plan(16, "macos", with_prune=True)
        self.assertEqual(small["prune"].status, "skip", "Nimble needs 24 GB+")
        mid = self.plan(32, "macos", with_prune=True)
        self.assertEqual(mid["prune"].status, "todo")
        self.assertNotIn("local_model", mid["config"].detail, "no 35B under 48 GB")
        big = self.plan(64, "macos", with_prune=True)
        self.assertIn("local_model=qwen3.6:35b-mlx", big["config"].detail)
        self.assertIn("qwen3.6:35b-mlx", big["ollama"].detail)
        linux = self.plan(64, "linux", with_prune=True)
        self.assertNotIn("local_model", linux["config"].detail, "the 35B check is measured on macOS only")


if __name__ == "__main__":
    unittest.main()
