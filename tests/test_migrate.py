"""The one-time rename migration: names rewritten, data moved with a link left, idempotent."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands
from tests import REAL_HOME  # noqa: E402

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokensaver import migrate, names, setup

REAL = REAL_HOME / ".claude" / "settings.json"
_BEFORE = REAL.read_bytes() if REAL.exists() else None


def tearDownModule():
    assert (REAL.read_bytes() if REAL.exists() else None) == _BEFORE, "a test touched the real settings.json"


class MigrateTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        R = migrate.REPO
        self.settings = self.d / "settings.json"
        self.settings.write_text(json.dumps({
            "model": "opus",
            "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": str(R / "everytoken-hook")}]}]},
            "statusLine": {"type": "command", "command": f"bash {R / 'everytoken-statusline'}"}}))
        self.rc = self.d / ".zshrc"
        self.rc.write_text("export A=1\n" + migrate.OLD_RC_BEGIN + "\n"
                           f"alias claude-et='X=1 {R / 'everytoken-claude'} --model opus'\n"
                           f"alias claude-mcp-et='{R / 'everytoken-claude'}'\n" + migrate.OLD_RC_END + "\nalias ll='ls'\n")
        self.p = [mock.patch.object(migrate, "SETTINGS", self.settings),
                  mock.patch.object(migrate, "_rc_files", return_value=[self.rc])]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()

    def test_names_rewritten_once(self):
        self.assertEqual(len(migrate.rewrite_names(apply=False)), 2)
        migrate.rewrite_names()
        s = json.loads(self.settings.read_text())
        self.assertEqual(s["model"], "opus")
        self.assertTrue(s["hooks"]["PreToolUse"][0]["hooks"][0]["command"].endswith("tokensaver-hook"))
        self.assertTrue(s["statusLine"]["command"].endswith("tokensaver-statusline"))
        rc = self.rc.read_text()
        self.assertIn(setup.RC_BEGIN, rc)
        self.assertIn("alias claude-ts='X=1 ", rc)
        self.assertIn("alias claude-mcp-ts=", rc)
        self.assertNotIn("everytoken-", rc)          # the repo folder itself may keep its name
        self.assertTrue(rc.startswith("export A=1\n") and rc.endswith("alias ll='ls'\n"))
        self.assertEqual(migrate.rewrite_names(apply=False), [], "second run changes nothing")

    def test_data_moved_with_link_and_marker(self):
        old, new = self.d / ".everytoken", self.d / ".tokensaver"
        old.mkdir()
        (old / "vault.db").write_text("x")
        with mock.patch.object(names, "HOME_OLD", old), mock.patch.object(names, "HOME_NEW", new):
            self.assertEqual(names.home(), old)
            migrate._data_step(apply=True)
            self.assertTrue(old.is_symlink() and (old / "vault.db").read_text() == "x")
            self.assertEqual(names.home(), new)
            self.assertEqual(migrate._data_step(apply=True), [], "idempotent")

    def test_both_folders_full_refuses(self):
        old, new = self.d / ".everytoken", self.d / ".tokensaver"
        old.mkdir()
        new.mkdir()
        (new / "vault.db").write_text("y")
        with mock.patch.object(names, "HOME_OLD", old), mock.patch.object(names, "HOME_NEW", new):
            with self.assertRaises(RuntimeError):
                migrate._data_step(apply=True)

    def test_old_env_names_still_work(self):
        with mock.patch.dict("os.environ", {"EVERYTOKEN_DISABLE": "1"}, clear=False):
            self.assertEqual(names.env("DISABLE"), "1")
        with mock.patch.dict("os.environ", {"EVERYTOKEN_DISABLE": "1", "TOKENSAVER_DISABLE": "0"}):
            self.assertEqual(names.env("DISABLE"), "0", "the new name wins")


if __name__ == "__main__":
    unittest.main()
