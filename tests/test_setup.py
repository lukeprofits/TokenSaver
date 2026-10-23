"""setup: aliases composed not overwritten, the rc block idempotent and removable."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands
from tests import REAL_HOME  # noqa: E402

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import config, setup  # noqa: E402

# Every test here must leave the real home alone. Fail loudly if one would not.
REAL_SETTINGS = REAL_HOME / ".claude" / "settings.json"
_BEFORE = REAL_SETTINGS.read_bytes() if REAL_SETTINGS.exists() else None


def tearDownModule():
    after = REAL_SETTINGS.read_bytes() if REAL_SETTINGS.exists() else None
    assert after == _BEFORE, "a setup test modified the real ~/.claude/settings.json"

L = str(setup.LAUNCHER)


class AliasTest(unittest.TestCase):
    def test_flags_and_env_survive(self):
        got = setup.compose_aliases({
            "claude": "ENABLE_CLAUDEAI_MCP_SERVERS=false command claude --model opus --strict-mcp-config",
            "claude-mcp": "command claude --model opus",
            "claude-local": "ollama launch claude -- qwen3.6:35b-mlx",
            "ll": "ls -la"}, "replace")
        self.assertEqual(got["claude"], f"ENABLE_CLAUDEAI_MCP_SERVERS=false {L} --model opus --strict-mcp-config")
        self.assertEqual(got["claude-mcp"], f"{L} --model opus")
        self.assertNotIn("claude-local", got)
        self.assertNotIn("ll", got)

    def test_try_mode_leaves_originals_alone(self):
        got = setup.compose_aliases({"claude": "ENABLE_CLAUDEAI_MCP_SERVERS=false command claude --model opus",
                                     "claude-mcp": "command claude --model opus"}, "try")
        self.assertEqual(set(got), {"claude-ts", "claude-mcp-ts"})
        self.assertEqual(got["claude-ts"], f"ENABLE_CLAUDEAI_MCP_SERVERS=false {L} --model opus")

    def test_replace_mode_keeps_a_way_around(self):
        got = setup.compose_aliases({"claude": "ENABLE_CLAUDEAI_MCP_SERVERS=false command claude --model opus"},
                                    "replace")
        self.assertEqual(got["claude-direct"], "ENABLE_CLAUDEAI_MCP_SERVERS=false command claude --model opus")
        self.assertIn(L, got["claude"])

    def test_plain_alias_when_none(self):
        self.assertEqual(setup.compose_aliases({}, "try"), {"claude-ts": L})
        self.assertEqual(setup.compose_aliases({}, "replace"), {"claude": L, "claude-direct": "command claude"})

    def test_already_routed_alias_left_alone(self):
        self.assertEqual(setup.compose_aliases({"claude": f"{L} --model opus"}, "replace"), {})


class RcBlockTest(unittest.TestCase):
    def test_block_is_idempotent_and_removable(self):
        home = Path(tempfile.mkdtemp())
        rc = home / ".zshrc"
        rc.write_text("export A=1\nalias claude='command claude --model opus'\n")
        saved = config.PATH
        config.PATH = home / "config.json"
        try:
            with mock.patch.object(setup, "rc_file", return_value=rc), \
                 mock.patch.object(setup, "user_aliases", return_value={"claude": "command claude --model opus"}):
                for _ in range(2):
                    steps = setup.plan(route="alias", shell_rc="yes", service=False)
                    shell = next(s for s in steps if s.name == "shell")
                    if shell.status == "todo":
                        shell.apply()
                text = rc.read_text()
                self.assertEqual(text.count(setup.RC_BEGIN), 1, "second run must not add a second block")
                self.assertTrue(text.startswith("export A=1\nalias claude='command claude --model opus'\n"))
                self.assertEqual(next(s for s in setup.plan(route="alias", shell_rc="yes", service=False)
                                      if s.name == "shell").status, "already")
                # install.uninstall edits the real ~/.claude/settings.json: never in a test.
                from tokensaver import install
                # uninstall also restores the status line in settings.json: temp copies only.
                # ...and must never stop the real service or edit the real routing.
                from tokensaver import migrate, routing, service
                with mock.patch.object(setup.sys, "platform", "linux"), \
                     mock.patch.object(install, "uninstall", return_value="(hooks: skipped in test)"), \
                     mock.patch.object(service, "remove", return_value=[]), \
                     mock.patch.object(routing, "disable", return_value="nothing of ours to remove"), \
                     mock.patch.object(migrate, "rewrite_names", return_value=[]), \
                     mock.patch.object(setup, "SETTINGS", home / "settings.json"), \
                     mock.patch.object(setup, "BIN_DIR", home / "bin"), \
                     mock.patch.object(setup, "_mcp_remove", return_value=None), \
                     mock.patch.object(setup, "STATUS_ORIG", home / "statusline-orig"), \
                     redirect_stdout(io.StringIO()):
                    setup.uninstall()
                self.assertEqual(rc.read_text().rstrip("\n"),
                                 "export A=1\nalias claude='command claude --model opus'")
        finally:
            config.PATH = saved


class NonInteractiveTest(unittest.TestCase):
    def test_without_tty_or_yes_nothing_changes(self):
        home = Path(tempfile.mkdtemp())
        saved = config.PATH
        config.PATH = home / "config.json"
        try:
            with mock.patch.object(setup, "rc_file", return_value=home / ".zshrc"), \
                 mock.patch.object(setup, "user_aliases", return_value={}), \
                 mock.patch.object(setup.sys.stdin, "isatty", return_value=False), \
                 redirect_stdout(io.StringIO()):
                code = setup.run(service=False)
            self.assertEqual(code, 3)
            self.assertFalse((home / ".zshrc").exists())
            self.assertFalse(config.PATH.exists())
        finally:
            config.PATH = saved


if __name__ == "__main__":
    unittest.main()


class ServiceTest(unittest.TestCase):
    def test_unit_files_pin_the_interpreter_and_the_cli(self):
        from tokensaver import service
        for text in (service.plist_text(), service.watchdog_plist_text(), service.unit_text(),
                     service.watchdog_unit_text()):
            self.assertIn(service.python(), text)
            self.assertIn(str(service.CLI), text)
        self.assertIn("Restart=always", service.unit_text())
        self.assertIn(f"OnUnitActiveSec={service.WATCH_EVERY_S}s", service.watchdog_timer_text())
        self.assertIn("<key>StartInterval</key>", service.watchdog_plist_text())
        self.assertIn("<key>KeepAlive</key>", service.plist_text())

    def test_install_writes_service_files_not_another_file(self):
        """Regression: a shared variable once made it write ~/.zshrc into the plist."""
        from tokensaver import service
        home = Path(tempfile.mkdtemp())
        ok = mock.Mock(returncode=0, stdout="OK", stderr="")
        with mock.patch.object(service, "PLIST", home / "a.plist"), \
             mock.patch.object(service, "WATCH_PLIST", home / "w.plist"), \
             mock.patch.object(service, "platform", return_value="macos"), \
             mock.patch.object(service, "_run", return_value=ok), \
             mock.patch.object(service, "_loaded", return_value=False):
            service.install()
            self.assertEqual((home / "a.plist").read_text(), service.plist_text())
            self.assertEqual((home / "w.plist").read_text(), service.watchdog_plist_text())


class StatusbarTest(unittest.TestCase):
    def test_wraps_and_restores_existing_statusline(self):
        home = Path(tempfile.mkdtemp())
        settings = home / "settings.json"
        settings.write_text(json.dumps({"model": "opus", "statusLine": {"type": "command",
                                         "command": "bash ~/.claude/statusline-command.sh", "padding": 0}}))
        with mock.patch.object(setup, "SETTINGS", settings), \
             mock.patch.object(setup, "STATUS_ORIG", home / "statusline-orig"), \
             mock.patch.object(setup, "STATUS_REFRESH_ADDED", home / "refresh-added"):
            step = setup.statusline_step("yes")
            step.apply()
            d = json.loads(settings.read_text())
            self.assertEqual(d["statusLine"]["command"], setup._our_status_command())
            self.assertEqual(d["statusLine"]["padding"], 0)
            self.assertEqual(d["statusLine"]["refreshInterval"], setup.STATUS_REFRESH_S)
            self.assertEqual(d["model"], "opus", "other keys untouched")
            self.assertEqual((home / "statusline-orig").read_text(), "bash ~/.claude/statusline-command.sh")
            self.assertEqual(setup.statusline_step("yes").status, "already")
            setup._restore_statusline()
            d = json.loads(settings.read_text())
            self.assertEqual(d["statusLine"]["command"], "bash ~/.claude/statusline-command.sh")
            self.assertNotIn("refreshInterval", d["statusLine"], "the refresh we added goes too")
            self.assertEqual(d["model"], "opus")

    def test_existing_install_gets_refresh_and_user_interval_wins(self):
        home = Path(tempfile.mkdtemp())
        settings = home / "settings.json"
        with mock.patch.object(setup, "SETTINGS", settings), \
             mock.patch.object(setup, "STATUS_ORIG", home / "statusline-orig"), \
             mock.patch.object(setup, "STATUS_REFRESH_ADDED", home / "refresh-added"):
            settings.write_text(json.dumps({"statusLine": {"type": "command",
                                                           "command": setup._our_status_command()}}))
            step = setup.statusline_step("yes")
            self.assertEqual(step.status, "todo", "an older install is upgraded")
            step.apply()
            self.assertEqual(json.loads(settings.read_text())["statusLine"]["refreshInterval"],
                             setup.STATUS_REFRESH_S)
            self.assertEqual(setup.statusline_step("yes").status, "already")
            settings.write_text(json.dumps({"statusLine": {"type": "command", "command": "mine.sh",
                                                           "refreshInterval": 30}}))
            (home / "refresh-added").unlink()
            setup.statusline_step("yes").apply()
            d = json.loads(settings.read_text())["statusLine"]
            self.assertEqual(d["refreshInterval"], 30, "the user's own interval is kept")
            setup._restore_statusline()
            self.assertEqual(json.loads(settings.read_text())["statusLine"].get("refreshInterval"), 30)

    def test_unparseable_settings_left_alone(self):
        home = Path(tempfile.mkdtemp())
        settings = home / "settings.json"
        settings.write_text("{broken")
        with mock.patch.object(setup, "SETTINGS", settings):
            self.assertEqual(setup.statusline_step("yes").status, "skip")
        self.assertEqual(settings.read_text(), "{broken")
