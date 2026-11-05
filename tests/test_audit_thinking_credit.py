"""The plain side of the audit includes thinking TokenSaver removed (priced at the lowest
measured rate), and the free-point decision's counts are made acceptable to count_tokens."""

import tests  # noqa: F401  isolation first (tests/__init__.py)

import json
import unittest
from unittest import mock

from tokensaver import audit, proxy


def thinking(sig_len):
    return {"type": "thinking", "thinking": "t", "signature": "s" * sig_len}


def body(msgs):
    return json.dumps({"model": "claude-opus-5-5", "messages": msgs}).encode()


class ThinkingCreditTest(unittest.TestCase):
    def setUp(self):
        audit._LOW_CACHE.clear()

    def test_nothing_removed_nothing_added(self):
        m = [{"role": "user", "content": "a"}, {"role": "assistant", "content": [thinking(1000)]}]
        self.assertEqual(audit._stripped_thinking({"body": body(m), "sent": body(m)}), 0)

    def test_removed_thinking_priced_at_the_lowest_measured_rate(self):
        orig = [{"role": "user", "content": "a"}, {"role": "assistant", "content": [thinking(10_000)]}]
        sent = [{"role": "user", "content": "a"}, {"role": "assistant", "content": [{"type": "text", "text": "x"}]}]
        with mock.patch.object(audit, "low_signature_factor", return_value=0.12):
            self.assertEqual(audit._stripped_thinking({"body": body(orig), "sent": body(sent)}), 1200)

    def test_low_rate_never_above_the_default(self):
        rows = [({"model": "claude-opus-5-5", "factor": f},) for f in (0.4, 0.3)]
        fake = mock.MagicMock()
        fake.execute.return_value = [(json.dumps(r[0]),) for r in rows]
        with mock.patch("sqlite3.connect", return_value=fake):
            from tokensaver.background import TOKENS_PER_SIGNATURE_CHAR
            self.assertEqual(audit.low_signature_factor("claude-opus-5-5"), TOKENS_PER_SIGNATURE_CHAR)


class DecisionCountBodyTest(unittest.TestCase):
    def test_fields_count_tokens_refuses_are_dropped_and_failures_recorded(self):
        seen = []

        def fake_count(path, headers, b):
            seen.append(b)
            return (None, 400, {}) if "display" in json.dumps(b.get("thinking")) else (100, 200, {})
        req = {"model": "claude-opus-5-5", "thinking": {"type": "adaptive", "display": "updates"},
               "messages": [{"role": "user", "content": "hi", "output_config": {"effort": "medium"}}]}
        with mock.patch.object(audit, "_count", side_effect=fake_count):
            got = proxy._count_all({"now": req}, {}, "/v1/messages")
        self.assertEqual(got, {"now": 100})
        self.assertNotIn("output_config", seen[0]["messages"][0])
        with mock.patch.object(audit, "_count", return_value=(None, 429, {})):
            self.assertEqual(proxy._count_all({"now": req}, {}, "/v1/messages"), {})
        self.assertIn("429", proxy._COUNT_WHY.get("now", ""))


if __name__ == "__main__":
    unittest.main()
