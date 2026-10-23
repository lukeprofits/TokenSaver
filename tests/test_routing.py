"""Routing through settings.json env: only our keys, put back what we replaced; the watchdog."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands
from tests import REAL_HOME  # noqa: E402

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokensaver import background, machine, routing, watchdog

REAL = REAL_HOME / ".claude" / "settings.json"
_BEFORE = REAL.read_bytes() if REAL.exists() else None


def tearDownModule():
    assert (REAL.read_bytes() if REAL.exists() else None) == _BEFORE, "a test touched the real settings.json"


class RoutingTest(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.settings, self.state = d / "settings.json", d / "routing.json"
        self.p = [mock.patch.object(routing, "SETTINGS", self.settings),
                  mock.patch.object(routing, "STATE", self.state),
                  mock.patch.object(watchdog, "STATE", d / "watchdog.json"),
                  mock.patch.object(watchdog.vault, "record", lambda *a, **k: None)]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()

    def read(self):
        return json.loads(self.settings.read_text())

    def test_enable_and_disable_touch_only_our_keys(self):
        self.settings.write_text(json.dumps({"model": "opus", "env": {"FOO": "1"}}))
        routing.enable("test", backup=False)
        env = self.read()["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], routing.wanted()["ANTHROPIC_BASE_URL"])
        self.assertEqual(env["FOO"], "1")
        self.assertTrue(routing.is_on())
        routing.disable("test", backup=False, forget=True)
        self.assertEqual(self.read(), {"model": "opus", "env": {"FOO": "1"}})

    def test_a_replaced_gateway_is_put_back(self):
        self.settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://corp.example/gw"}}))
        routing.enable("test", backup=False)
        self.assertEqual(routing.state()["replaced"]["ANTHROPIC_BASE_URL"], "https://corp.example/gw")
        routing.disable("test", backup=False)
        self.assertEqual(self.read()["env"]["ANTHROPIC_BASE_URL"], "https://corp.example/gw")

    def test_a_key_changed_by_someone_else_is_left_alone(self):
        routing.enable("test", backup=False)
        d = self.read()
        d["env"]["ANTHROPIC_BASE_URL"] = "http://elsewhere:1"
        self.settings.write_text(json.dumps(d))
        routing.disable("test", backup=False)
        self.assertEqual(self.read()["env"]["ANTHROPIC_BASE_URL"], "http://elsewhere:1")

    def test_users_tool_search_choice_wins(self):
        self.settings.write_text(json.dumps({"env": {"ENABLE_TOOL_SEARCH": "false"}}))
        routing.enable("test", backup=False)
        self.assertEqual(self.read()["env"]["ENABLE_TOOL_SEARCH"], "false")

    def test_unparseable_settings_never_touched(self):
        self.settings.write_text("{not json")
        with self.assertRaises(RuntimeError):
            routing.enable("test", backup=False)
        self.assertEqual(self.settings.read_text(), "{not json")

    def test_watchdog_bypasses_after_two_failures_and_restores(self):
        routing.enable("setup", backup=False)
        self.assertEqual(watchdog.once(lambda: False), "no change", "one failure is not enough")
        self.assertTrue(watchdog.once(lambda: False).startswith("bypassed"))
        self.assertFalse(routing.is_on())
        self.assertTrue(watchdog.once(lambda: True).startswith("restored"))
        self.assertTrue(routing.is_on())

    def test_watchdog_does_nothing_unless_setup_routed(self):
        self.assertEqual(watchdog.once(lambda: False), "not managing routing")
        self.assertFalse(self.settings.exists())

    def test_watchdog_does_not_undo_an_uninstall(self):
        routing.enable("setup", backup=False)
        routing.disable("uninstall", backup=False, forget=True)
        self.assertEqual(watchdog.once(lambda: True), "not managing routing")
        self.assertFalse(routing.is_on())


class MachineTest(unittest.TestCase):
    def test_linux_meminfo_and_battery(self):
        d = Path(tempfile.mkdtemp())
        (d / "meminfo").write_text("MemTotal:       16303428 kB\nMemAvailable:    8151714 kB\n"
                                   "SwapTotal:       2097148 kB\nSwapFree:        1048574 kB\n")
        mi = machine._meminfo(str(d / "meminfo"))
        self.assertEqual(mi["MemTotal"], 16303428 * 1024)
        self.assertEqual(mi["MemAvailable"], 8151714 * 1024)
        (d / "psi").write_text("some avg10=12.50 avg60=3.00 avg300=1.00 total=1\n")
        self.assertEqual(machine._linux_pressure(str(d / "psi")), 2)
        bat = d / "BAT0"
        bat.mkdir()
        (bat / "status").write_text("Discharging\n")
        (bat / "capacity").write_text("20\n")
        self.assertTrue(machine._linux_battery(str(d)))
        (bat / "status").write_text("Charging\n")
        self.assertFalse(machine._linux_battery(str(d)))

    def test_gate_tier_by_memory_and_platform(self):
        gb = 2 ** 30
        self.assertEqual(background.gate_model_for(64 * gb, "macos"), "qwen3.6:35b-mlx")
        self.assertNotEqual(background.gate_model_for(64 * gb, "linux"), "qwen3.6:35b-mlx", "MLX is macOS only")
        self.assertIsNone(background.gate_model_for(4 * gb, "macos"), "too small: compaction off")


if __name__ == "__main__":
    unittest.main()
