"""Fixes from a light Sonnet 5.5 user's benchmark (2026-11-25): the audit's thinking-free
counts, the reason a refused count is recorded with, and trailing thinking in histories."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import unittest

from tokensaver import audit, benchmark


def _raw():
    return json.dumps({"model": "claude-sonnet-5-5", "thinking": {"type": "adaptive"}, "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"},
                                          {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
    ]}).encode()


class PayloadTest(unittest.TestCase):
    def test_thinking_free_count_sends_thinking_off(self):
        p = audit.payload(_raw(), no_thinking=True)
        self.assertNotIn("thinking", p)
        self.assertEqual([b["type"] for b in p["messages"][1]["content"]], ["tool_use"])

    def test_full_count_keeps_thinking(self):
        p = audit.payload(_raw())
        self.assertEqual(p["thinking"], {"type": "adaptive"})
        self.assertEqual(p["messages"][1]["content"][0]["type"], "thinking")


class ReasonTest(unittest.TestCase):
    def setUp(self):
        self.saved = dict(audit._state)

    def tearDown(self):
        audit._state.clear()
        audit._state.update(self.saved)

    def test_400_reason_recorded(self):
        audit._state.update(disabled=None, last_error="messages.3.content.0.type: Expected `thinking`")
        audit._on_error(400, {}, {"session": "s"})
        self.assertIn("Expected `thinking`", audit._state["disabled"])

    def test_auth_error_unchanged(self):
        audit._state.update(disabled=None, last_error="whatever")
        audit._on_error(401, {}, {"session": "s"})
        self.assertEqual(audit._state["disabled"], "count_tokens returned 401")


class TrailingThinkingTest(unittest.TestCase):
    def test_trailing_thinking_dropped(self):
        # The shape the counter refused: thinking, a tool call, then thinking again.
        hist = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "a", "signature": "s"},
                                              {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
                                              {"type": "thinking", "thinking": "b", "signature": "s"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
        ]
        out, rep = benchmark.countable(hist)
        self.assertEqual([b["type"] for b in out[1]["content"]], ["thinking", "tool_use"])
        self.assertEqual(rep.get("trailing_thinking"), 1)

    def test_thinking_only_message_goes(self):
        hist = [{"role": "user", "content": "a"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "s"}]},
                {"role": "user", "content": "b"}]
        out, _ = benchmark.countable(hist)
        self.assertEqual([m["role"] for m in out], ["user"])

    def test_normal_history_untouched(self):
        out, rep = benchmark.countable(json.loads(_raw())["messages"])
        self.assertNotIn("trailing_thinking", rep)
        self.assertEqual(out[1]["content"][0]["type"], "thinking")


if __name__ == "__main__":
    unittest.main()
