"""Editing a user's Claude settings without disturbing anything else they set up."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands
from tests import REAL_HOME  # noqa: E402

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokensaver import claude_settings, config, routing, setup

REAL = REAL_HOME / ".claude" / "settings.json"
_BEFORE = REAL.read_bytes() if REAL.exists() else None


def tearDownModule():
    assert (REAL.read_bytes() if REAL.exists() else None) == _BEFORE, "a test touched the real settings.json"


class WriteTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_symlink_stays_a_link_and_mode_is_kept(self):
        real = self.d / "dotfiles" / "settings.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"model": "opus", "note": "café — ünïcode"}, ensure_ascii=False))
        os.chmod(real, 0o600)
        link = self.d / "settings.json"
        link.symlink_to(real)
        claude_settings.write_json(link, dict(json.loads(real.read_text()), env={"A": "1"}))
        self.assertTrue(link.is_symlink(), "a dotfiles link must survive")
        self.assertEqual(stat.S_IMODE(real.stat().st_mode), 0o600)
        text = real.read_text()
        self.assertIn("café — ünïcode", text, "no \\u escapes for the user's own characters")
        self.assertEqual(json.loads(text)["env"], {"A": "1"})
        self.assertTrue(list(real.parent.glob("settings.json.bak.tokensaver.*")), "backup next to the target")

    def test_never_writes_invalid_json(self):
        p = self.d / "settings.json"
        p.write_text('{"model": "opus"}')
        with self.assertRaises(ValueError):
            claude_settings.write_text(p, "{broken")
        self.assertEqual(p.read_text(), '{"model": "opus"}')
        self.assertEqual([x.name for x in self.d.iterdir()], ["settings.json"], "no temp file left behind")

    def test_routing_leaves_other_keys_and_formatting_intact(self):
        p = self.d / "settings.json"
        original = {"model": "opus", "permissions": {"allow": ["Bash(ls)"]}, "env": {"MY_VAR": "x"},
                    "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "say done"}]}]}}
        p.write_text(json.dumps(original, indent=2))
        with mock.patch.object(routing, "SETTINGS", p), mock.patch.object(routing, "STATE", self.d / "r.json"):
            routing.enable("test", backup=False)
            routing.disable("test", backup=False, forget=True)
        self.assertEqual(json.loads(p.read_text()), original, "on then off leaves the file as it was")


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.saved = config.PATH
        config.PATH = self.d / "config.json"
        config.load(force=True)

    def tearDown(self):
        config.PATH = self.saved
        config.load(force=True)

    def plan_with(self, settings_env=None, managed_env=None):
        s, m = self.d / "settings.json", self.d / "managed-settings.json"
        s.write_text(json.dumps({"env": settings_env or {}}))
        m.write_text(json.dumps({"env": managed_env or {}}))
        with mock.patch.object(setup, "SETTINGS", s), \
             mock.patch.object(claude_settings, "managed_paths", return_value=[m]), \
             mock.patch.object(setup.shutil, "which", side_effect=lambda c: f"/usr/bin/{c}"), \
             mock.patch.dict(os.environ, {}, clear=False):
            for k in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "ANTHROPIC_BASE_URL"):
                os.environ.pop(k, None)
            return setup.plan(route="settings", service=False, statusbar="no")

    def test_bedrock_in_settings_env_is_skipped(self):
        steps = self.plan_with(settings_env={"CLAUDE_CODE_USE_BEDROCK": "1"})
        self.assertEqual([(s.name, s.status) for s in steps], [("preflight", "skip")])
        self.assertIn("Bedrock", steps[0].detail)

    def test_managed_base_url_is_skipped(self):
        steps = self.plan_with(managed_env={"ANTHROPIC_BASE_URL": "https://gw.corp.example"})
        self.assertEqual([(s.name, s.status) for s in steps], [("preflight", "skip")])
        self.assertIn("managed settings", steps[0].detail)

    def test_user_gateway_in_settings_is_chained(self):
        steps = self.plan_with(settings_env={"ANTHROPIC_BASE_URL": "https://gw.corp.example"})
        pre = next(s for s in steps if s.name == "preflight")
        self.assertTrue(any("chaining" in n for n in pre.notes))
        self.assertIn("upstream=https://gw.corp.example", next(s for s in steps if s.name == "config").detail)


if __name__ == "__main__":
    unittest.main()
