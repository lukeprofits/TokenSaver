"""The proxy re-tunes weekly, and reads the stored UTC time correctly."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import time
import unittest
from unittest import mock

from tokensaver import proxy


def stamp(seconds_ago: float, now: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - seconds_ago))


class RetuneTest(unittest.TestCase):
    def run_check(self, last):
        now = time.time()
        started = []
        proxy._retune_checked = 0.0
        with mock.patch.object(proxy.config, "section", return_value={"last_run": last(now)} if last else {}), \
             mock.patch.object(proxy.threading, "Thread",
                               side_effect=lambda **kw: mock.Mock(start=lambda: started.append(1))):
            proxy._maybe_retune(now)
        return bool(started)

    def test_weekly(self):
        self.assertEqual(proxy.RETUNE_EVERY_S, 7 * 86400)

    def test_eight_days_old_runs(self):
        self.assertTrue(self.run_check(lambda now: stamp(8 * 86400, now)))

    def test_six_days_old_waits(self):
        self.assertFalse(self.run_check(lambda now: stamp(6 * 86400, now)))

    def test_never_run_runs(self):
        self.assertTrue(self.run_check(None))

    def test_age_is_read_as_utc(self):
        now = time.time()
        with mock.patch.object(proxy.config, "section", return_value={"last_run": stamp(3600, now)}):
            self.assertAlmostEqual(proxy._last_tune_age(now), 3600, delta=2)


if __name__ == "__main__":
    unittest.main()
