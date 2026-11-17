"""bill.py: which warm cache misses are charged to TokenSaver in the live bill."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import unittest
import uuid

from tokensaver import bill, vault


def _req(session: str, write: int, bytes_in: int, bytes_out: int) -> None:
    vault.record("api_request", tool="proxy", session_id=session, index=False, meta={
        "status": 200, "path": "/v1/messages", "model": "claude-opus-5-5", "request_class": "main",
        "rewrite": {"bytes_in": bytes_in, "bytes_out": bytes_out},
        "usage": {"input_tokens": 2, "cache_read_input_tokens": 300_000 - write, "output_tokens": 10,
                  "cache_creation_input_tokens": write,
                  "cache_creation": {"ephemeral_1h_input_tokens": write}}})


class MissCauseTest(unittest.TestCase):
    def misses(self, first_out: int, second_out: int):
        s = "bill-" + uuid.uuid4().hex[:8]
        _req(s, 1_000, 1_000_000, first_out)
        _req(s, 300_000, 1_000_000, second_out)            # warm, and the whole prefix written again
        return bill.compute(session_prefix=s).misses

    def test_unchanged_bytes_are_not_ours(self):
        # TokenSaver sent both requests exactly as Claude Code did: plain Claude Code misses the same.
        (m,) = self.misses(1_000_000, 1_000_000)
        self.assertFalse(m.ours, m.cause)

    def test_changed_bytes_stay_ours(self):
        # The earlier request was rewritten, this one not: the prefix changed, so it could be ours.
        (m,) = self.misses(700_000, 1_000_000)
        self.assertTrue(m.ours, m.cause)


if __name__ == "__main__":
    unittest.main()
