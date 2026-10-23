"""The proxy is in the request path of the session that may be running setup:
bootstrap waits for the old job and retries; nothing stops the proxy from a
routed Claude session; uninstall from one stops it only after a delay."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokensaver import routing, service


def done(code=0, err=""):
    return subprocess.CompletedProcess([], code, "", err)


class ReloadTest(unittest.TestCase):
    def test_waits_for_the_old_job_then_retries_error_5(self):
        calls = []
        prints = iter([True, True, False])          # still loaded twice, then gone
        boots = iter([done(5, "Bootstrap failed: 5: Input/output error"), done(0)])

        def run(args, timeout=20):
            calls.append(args[1])
            if args[1] == "print":
                return done(0 if next(prints, False) else 113)
            if args[1] == "bootstrap":
                return next(boots)
            return done(0)
        slept = []
        with mock.patch.object(service, "_run", side_effect=run):
            service._reload_agent(Path("/x.plist"), service.LABEL, sleep=slept.append)
        self.assertEqual(calls, ["bootout", "print", "print", "print", "bootstrap", "print", "bootstrap"])
        self.assertEqual(len(slept), 3, "two waits for the old job, one before the retry")

    def test_other_bootstrap_errors_are_reported_at_once(self):
        with mock.patch.object(service, "_run", side_effect=lambda a, timeout=20:
                               done(113) if a[1] == "print" else done(1, "Bootstrap failed: 37: bad plist")
                               if a[1] == "bootstrap" else done(0)):
            with self.assertRaisesRegex(RuntimeError, "37"):
                service._reload_agent(Path("/x.plist"), service.LABEL, sleep=lambda s: None)

    def test_unchanged_proxy_is_not_touched(self):
        home = Path(tempfile.mkdtemp())
        with mock.patch.object(service, "PLIST", home / "p.plist"), \
             mock.patch.object(service, "WATCH_PLIST", home / "w.plist"), \
             mock.patch.object(service, "platform", return_value="macos"), \
             mock.patch.object(service, "_loaded", return_value=True), \
             mock.patch.object(service, "_install_agent") as inst:
            (home / "p.plist").write_text(service.plist_text())
            service.install()
        self.assertEqual([c.args[2] for c in inst.call_args_list], [service.WATCH_LABEL],
                         "only the changed watchdog is replaced")


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.p = [mock.patch.object(routing, "SETTINGS", self.d / "settings.json"),
                  mock.patch.object(service, "ALLOW_PROXY_STOP", False)]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()

    def test_signals(self):
        self.assertIsNone(service.stopping_proxy_is_unsafe())
        with mock.patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}):
            self.assertIn("ANTHROPIC_BASE_URL", service.stopping_proxy_is_unsafe())
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}):
            self.assertIn("Claude Code session", service.stopping_proxy_is_unsafe())
        (self.d / "settings.json").write_text('{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}')
        with mock.patch("sys.stdin.isatty", return_value=False):
            self.assertIn("no terminal", service.stopping_proxy_is_unsafe())
        with mock.patch("sys.stdin.isatty", return_value=True):
            self.assertIsNone(service.stopping_proxy_is_unsafe(), "routed, but run by hand in a terminal")
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}), mock.patch.object(service, "ALLOW_PROXY_STOP", True):
            self.assertIsNone(service.stopping_proxy_is_unsafe(), "the explicit override")

    def test_restart_and_proxy_replace_refuse(self):
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}), mock.patch.object(service, "_run") as run:
            with self.assertRaisesRegex(RuntimeError, "normal terminal"):
                service.restart()
            run.assert_not_called()
            home = self.d
            with mock.patch.object(service, "PLIST", home / "p.plist"), \
                 mock.patch.object(service, "WATCH_PLIST", home / "w.plist"), \
                 mock.patch.object(service, "platform", return_value="macos"), \
                 mock.patch.object(service, "_loaded", return_value=True), \
                 mock.patch.object(service, "_install_agent") as inst:
                (home / "p.plist").write_text("old proxy file")
                with self.assertRaises(RuntimeError):
                    service.install()
                self.assertEqual(inst.call_count, 0, "nothing replaced")

    def test_uninstall_from_a_session_leaves_the_running_proxy(self):
        """An open session does not go back to direct when routing is removed
        (measured), so the proxy must keep running for it until logout."""
        home = self.d
        (home / "p.plist").write_text("x")
        (home / "w.plist").write_text("x")
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}), \
             mock.patch.object(service, "PLIST", home / "p.plist"), \
             mock.patch.object(service, "WATCH_PLIST", home / "w.plist"), \
             mock.patch.object(service, "AGENTS", home), \
             mock.patch.object(service, "platform", return_value="macos"), \
             mock.patch.object(service, "_loaded", return_value=True), \
             mock.patch.object(service, "LEFT_RUNNING", False), \
             mock.patch.object(service, "_run", return_value=done(0)) as run:
            out = service.remove()
            self.assertTrue(service.LEFT_RUNNING)
        self.assertFalse((home / "w.plist").exists() or (home / "p.plist").exists(),
                         "both files go: nothing starts at the next login")
        self.assertTrue(any("stays until you log out" in x for x in out))
        boots = [c.args[0][2] for c in run.call_args_list if c.args[0][:2] == ["launchctl", "bootout"]]
        self.assertEqual(boots, [f"{service._domain()}/{service.WATCH_LABEL}"], "the proxy job is not booted out")

    def test_uninstall_from_a_terminal_stops_everything(self):
        home = self.d
        (home / "p.plist").write_text("x")
        with mock.patch.object(service, "PLIST", home / "p.plist"), \
             mock.patch.object(service, "WATCH_PLIST", home / "w.plist"), \
             mock.patch.object(service, "AGENTS", home), \
             mock.patch.object(service, "platform", return_value="macos"), \
             mock.patch.object(service, "_loaded", return_value=True), \
             mock.patch.object(service, "LEFT_RUNNING", False), \
             mock.patch.object(service, "_run", return_value=done(0)) as run:
            service.remove()
            self.assertFalse(service.LEFT_RUNNING)
        self.assertIn(["launchctl", "bootout", f"{service._domain()}/{service.LABEL}"],
                      [c.args[0] for c in run.call_args_list])


class ReviveTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_unloaded_job_is_reloaded_and_counts_as_healthy(self):
        from tokensaver import watchdog
        (self.d / "p.plist").write_text("x")
        with mock.patch.object(service, "PLIST", self.d / "p.plist"), \
             mock.patch.object(service, "platform", return_value="macos"), \
             mock.patch.object(service, "_loaded", return_value=False), \
             mock.patch.object(service, "_run", return_value=done(0)) as run, \
             mock.patch.object(watchdog.routing, "state", return_value={"managed": True, "on": True}), \
             mock.patch.object(watchdog, "STATE", self.d / "wd.json"), \
             mock.patch.object(watchdog, "service_wait_health", return_value=True), \
             mock.patch.object(watchdog.routing, "disable") as off, \
             mock.patch.object(watchdog.vault, "record"):
            for _ in range(3):
                watchdog.once(check=lambda: False)
        self.assertEqual(run.call_args.args[0][:2], ["launchctl", "bootstrap"])
        off.assert_not_called()

    def test_uninstalled_on_purpose_is_left_alone(self):
        from tokensaver import watchdog
        with mock.patch.object(service, "PLIST", self.d / "missing.plist"), \
             mock.patch.object(service, "platform", return_value="macos"), \
             mock.patch.object(service, "_run") as run:
            self.assertIsNone(watchdog.revive())
        run.assert_not_called()

    def test_loaded_but_crashed_is_left_to_launchd(self):
        from tokensaver import watchdog
        (self.d / "p.plist").write_text("x")
        with mock.patch.object(service, "PLIST", self.d / "p.plist"), \
             mock.patch.object(service, "platform", return_value="macos"), \
             mock.patch.object(service, "_loaded", return_value=True), \
             mock.patch.object(service, "_run") as run:
            self.assertIsNone(watchdog.revive())
        run.assert_not_called()


class SetupPlanTest(unittest.TestCase):
    def test_service_step_skipped_from_a_routed_session(self):
        from tokensaver import setup
        with mock.patch.object(service, "up_to_date", return_value=False), \
             mock.patch.object(service, "proxy_would_restart", return_value=True), \
             mock.patch.object(service, "stopping_proxy_is_unsafe", return_value="inside a Claude Code session"), \
             mock.patch.object(setup.shutil, "which", side_effect=lambda c: f"/usr/bin/{c}"):
            steps = {s.name: s for s in setup.plan(route="settings", statusbar="no")}
        self.assertEqual(steps["service"].status, "skip")
        self.assertTrue(any("normal terminal" in n for n in steps["service"].notes))


if __name__ == "__main__":
    unittest.main()
