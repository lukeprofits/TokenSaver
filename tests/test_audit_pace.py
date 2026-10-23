"""audit.pace_gap: follow Anthropic's own rate-limit headers, never find the limit by a 429."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import audit  # noqa: E402


class PaceTest(unittest.TestCase):
    def setUp(self):
        self.saved = dict(audit._state), audit.MIN_GAP_S
        audit.MIN_GAP_S = 1.0

    def tearDown(self):
        audit._state.clear()
        audit._state.update(self.saved[0])
        audit.MIN_GAP_S = self.saved[1]

    def gap(self, remaining, reset_in):
        audit._state.update(remaining=remaining, reset_at=None if reset_in is None else 1000.0 + reset_in)
        return audit.pace_gap(1000.0)

    def test_no_headers_uses_floor(self):
        self.assertEqual(self.gap(None, None), 1.0)

    def test_plenty_left_uses_floor(self):
        self.assertEqual(self.gap(4000, 60), 1.0)

    def test_spreads_what_is_left_until_reset(self):
        jobs = 50 // audit.CALLS_PER_JOB - 1
        self.assertAlmostEqual(self.gap(50, 60), 60 / jobs)

    def test_out_of_room_waits_for_the_window(self):
        self.assertEqual(self.gap(2, 30), 31.0)

    def test_headers_parsed(self):
        audit._note_limits({"anthropic-ratelimit-requests-limit": "4000",
                            "anthropic-ratelimit-requests-remaining": "3999",
                            "anthropic-ratelimit-requests-reset": "2026-10-15T18:30:00Z"})
        self.assertEqual((audit._state["limit"], audit._state["remaining"]), (4000, 3999))
        self.assertIsNotNone(audit._state["reset_at"])


if __name__ == "__main__":
    unittest.main()
