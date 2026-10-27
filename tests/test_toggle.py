"""`tokensaver on|off`: off stays off (the watchdog never undoes it), on needs a live proxy."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands
from tests import REAL_HOME  # noqa: E402

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tokensaver import __main__ as cli, routing, service, watchdog

REAL = REAL_HOME / ".claude" / "settings.json"
_BEFORE = REAL.read_bytes() if REAL.exists() else None


def tearDownModule():
    assert (REAL.read_bytes() if REAL.exists() else None) == _BEFORE, "a test touched the real settings.json"


class ToggleTest(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.settings = d / "settings.json"
        self.settings.write_text(json.dumps({"model": "opus", "env": {"MINE": "1"}}, indent=2) + "\n")
        self.before = self.settings.read_bytes()
        self.p = [mock.patch.object(routing, "SETTINGS", self.settings),
                  mock.patch.object(routing, "STATE", d / "routing.json"),
                  mock.patch.object(watchdog, "STATE", d / "watchdog.json"),
                  mock.patch.object(watchdog.vault, "record", lambda *a, **k: None)]
        for x in self.p:
            x.start()
        routing.enable("setup", backup=False)

    def tearDown(self):
        for x in self.p:
            x.stop()

    def run_cli(self, cmd, healthy=True):
        with mock.patch.object(service, "health", return_value={"ok": True} if healthy else None), \
             redirect_stdout(io.StringIO()) as out:
            code = cli.main([cmd])
        return code, out.getvalue()

    def test_off_stays_off_through_a_healthy_watchdog_check(self):
        code, _ = self.run_cli("off")
        self.assertEqual(code, 0)
        self.assertFalse(routing.is_on())
        self.assertEqual(json.loads(self.settings.read_text())["env"], {"MINE": "1"})
        for _ in range(3):
            watchdog.once(check=lambda: True)
        self.assertFalse(routing.is_on(), "the watchdog must not undo the user's off")

    def test_on_again_and_round_trip_keeps_other_settings(self):
        self.run_cli("off")
        code, _ = self.run_cli("on")
        self.assertEqual(code, 0)
        self.assertTrue(routing.is_on())
        self.assertEqual(json.loads(self.settings.read_text())["env"]["MINE"], "1")
        self.run_cli("off")
        self.assertEqual(json.loads(self.settings.read_text()), json.loads(self.before))

    def test_on_refuses_while_the_proxy_is_down(self):
        self.run_cli("off")
        code, out = self.run_cli("on", healthy=False)
        self.assertEqual(code, 1)
        self.assertFalse(routing.is_on(), "a dead proxy must never be routed to")
        self.assertIn("not answering", out)

    def test_pause_lasts_until_local_midnight_then_the_watchdog_turns_it_on(self):
        code, _ = self.run_cli("pause")
        self.assertEqual(code, 0)
        st = routing.state()
        self.assertFalse(routing.is_on())
        self.assertEqual(st["until"], routing.next_midnight())
        _, out = self.run_cli("status")
        self.assertIn("paused until", out)
        watchdog.once(check=lambda: True)
        self.assertFalse(routing.is_on(), "still today: stays paused")
        with mock.patch.object(routing.time, "time", return_value=st["until"] + 1):
            watchdog.once(check=lambda: False)
            watchdog.once(check=lambda: False)
            self.assertFalse(routing.is_on(), "after midnight but the proxy is down: never route to it")
            watchdog.once(check=lambda: True)
        self.assertTrue(routing.is_on(), "after midnight with a healthy proxy: back on")
        self.assertNotIn("until", routing.state())

    def test_on_ends_a_pause_and_off_is_not_ended_at_midnight(self):
        self.run_cli("pause")
        self.run_cli("on")
        self.assertTrue(routing.is_on())
        self.assertNotIn("until", routing.state())
        self.run_cli("off")
        with mock.patch.object(routing.time, "time", return_value=routing.next_midnight() + 3600):
            watchdog.once(check=lambda: True)
        self.assertFalse(routing.is_on(), "a plain off has no end time")

    def test_next_midnight(self):
        import datetime as dt
        t = dt.datetime(2026, 10, 27, 23, 59, 30).timestamp()
        self.assertEqual(dt.datetime.fromtimestamp(routing.next_midnight(t)), dt.datetime(2026, 10, 28, 0, 0))
        t = dt.datetime(2026, 10, 27, 0, 0, 1).timestamp()
        self.assertEqual(dt.datetime.fromtimestamp(routing.next_midnight(t)), dt.datetime(2026, 10, 28, 0, 0))

    def test_help_lists_every_everyday_command(self):
        for argv in (["help"], []):
            _, out = self.run_cli(*argv) if argv else self.run_cli_raw([])
            for cmd in ("status", "off", "pause", "on", "doctor", "--benchmark", "update", "uninstall"):
                self.assertIn(f"tokensaver {cmd}", out)

    def run_cli_raw(self, argv):
        with redirect_stdout(io.StringIO()) as out:
            code = cli.main(argv)
        return code, out.getvalue()

    def test_status(self):
        _, out = self.run_cli("status")
        self.assertIn("TokenSaver is on", out)
        self.run_cli("off")
        _, out = self.run_cli("status")
        self.assertIn("TokenSaver is off", out)


if __name__ == "__main__":
    unittest.main()
