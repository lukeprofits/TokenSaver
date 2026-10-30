"""The thinking rate: measured on sent requests, never disabling the audit, fitted with enough data."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import sqlite3
import time
import unittest
from unittest import mock

from tokensaver import audit, background, vault

SIG = "x" * 30_000


def setUpModule():
    vault.record("test_init", index=False)      # creates the fake home's vault


def body(display=True):
    msgs = [{"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "output_config": {"effort": "high"},
             "content": [{"type": "thinking", "thinking": "", "signature": SIG}, {"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "text", "text": "next"}]}]
    th = {"type": "adaptive", "display": "updates"} if display else {"type": "adaptive"}
    return {"model": "claude-opus-5-5", "messages": msgs, "thinking": th}


def counter(rate=0.25, refuse=None):
    """count(body, beta): 1,000 + rate per signature char while thinking is present.
    `refuse`: (status, message) for every count that keeps thinking."""
    seen = []

    def count(b, beta):
        seen.append((b, beta))
        sig = background._signature_chars(b["messages"])
        if sig and refuse:
            return None, f"{refuse[0]} {refuse[1]}"
        return 1000 + round(rate * sig), ""
    return count, seen


class MeasureTest(unittest.TestCase):
    def test_measured_with_binding_check_on_a_clean_copy(self):
        b = body()
        count, seen = counter(0.25)
        got = audit.measure_thinking(b, count)
        self.assertAlmostEqual(got["factor"], 0.25, places=3)
        self.assertEqual(got["sig_chars"], len(SIG))
        with_t = seen[0][0]
        self.assertEqual(with_t["thinking"]["block_binding"], {"prefix_mismatch_behavior": "error"})
        self.assertNotIn("display", with_t["thinking"])
        self.assertTrue(all(set(m) == {"role", "content"} for x, _ in seen for m in x["messages"]))
        self.assertTrue(all(beta == audit.BINDING_BETA for _, beta in seen))
        self.assertIn("output_config", b["messages"][1], "the caller's body is unchanged")
        self.assertEqual(b["thinking"]["display"], "updates")

    def test_refused_zero_and_implausible_are_not_values(self):
        got = audit.measure_thinking(body(), counter(refuse=(400, "thinking bound to a different conversation"))[0])
        self.assertIsNone(got["factor"])
        self.assertIn("bound to a different conversation", got["why"])
        self.assertIn("0 tokens", audit.measure_thinking(body(), counter(0.0)[0])["why"])
        self.assertIn("implausible", audit.measure_thinking(body(), counter(0.9)[0])["why"])
        off = dict(body(), thinking={"type": "disabled"})
        self.assertIn("thinking off", audit.measure_thinking(off, counter()[0])["why"])


class JobTest(unittest.TestCase):
    def setUp(self):
        audit._sig_last.clear()
        audit._state.update(sig_day="", sig_today=0, calls_today=0, disabled=None)

    def job(self, session="s1"):
        return {"path": "/v1/messages", "headers": {"anthropic-beta": "oauth-2025-04-20"},
                "sent": json.dumps(body()).encode(), "session": session}

    def rows(self):
        conn = sqlite3.connect(vault.DB_PATH)
        try:
            return [json.loads(m) for (m,) in conn.execute(
                "SELECT meta FROM events WHERE kind='thinking_factor' ORDER BY id")]
        finally:
            conn.close()

    def test_a_binding_400_is_recorded_and_never_disables_the_audit(self):
        before = len(self.rows())
        with mock.patch.object(audit, "_count", return_value=(None, 400, {})):
            audit._measure_job(self.job("s400"))
        self.assertIsNone(audit._state["disabled"], "the audit stays on")
        rows = self.rows()
        self.assertEqual(len(rows), before + 1)
        self.assertIsNone(rows[-1]["factor"])
        self.assertTrue(rows[-1]["why"].startswith("count refused: 400"))

    def test_rate_limit_records_nothing(self):
        before = len(self.rows())
        with mock.patch.object(audit, "_count", return_value=(None, 429, {})):
            audit._measure_job(self.job("s429"))
        self.assertEqual(len(self.rows()), before)
        self.assertIsNone(audit._state["disabled"])

    def test_beta_merged_and_once_per_session_per_30_min(self):
        calls = []

        def fake(path, h, b):
            calls.append(h["anthropic-beta"])
            sig = background._signature_chars(b["messages"])
            return 1000 + round(0.25 * sig), 200, {}
        with mock.patch.object(audit, "_count", side_effect=fake):
            audit._measure_job(self.job("s-once"))
            audit._measure_job(self.job("s-once"))
        self.assertEqual(len(calls), 2, "the second job within 30 min measured nothing")
        self.assertTrue(all(c == f"oauth-2025-04-20,{audit.BINDING_BETA}" for c in calls))
        self.assertAlmostEqual(self.rows()[-1]["factor"], 0.25, places=3)

    def test_daily_cap(self):
        audit._state.update(sig_day=audit._day(), sig_today=audit.SIG_MAX_PER_DAY)
        with mock.patch.object(audit, "_count", side_effect=AssertionError("no call")):
            audit._measure_job(self.job("s-cap"))


class FitTest(unittest.TestCase):
    def setUp(self):
        conn = sqlite3.connect(vault.DB_PATH)
        conn.execute("DELETE FROM events WHERE kind='thinking_factor'")
        conn.commit()
        conn.close()

    def add(self, factor, model="claude-opus-5-5", session="a"):
        vault.record("thinking_factor", session_id=session, index=False,
                     meta={"factor": factor, "model": model, "why": "", "sig_chars": 30000})

    def test_default_until_enough(self):
        for f in (0.25, 0.26):
            self.add(f)
        v, why, info = audit.signature_factor("claude-opus-5-5")
        self.assertEqual(v, background.TOKENS_PER_SIGNATURE_CHAR)
        self.assertEqual(info["source"], "default")
        self.assertIn("2/5 measured", why)

    def test_needs_two_sessions_then_model_median(self):
        for f in (0.24, 0.25, 0.26, 0.27, 0.28):
            self.add(f, session="a")
        self.assertEqual(audit.signature_factor("claude-opus-5-5")[2]["source"], "default", "one session only")
        self.add(0.25, session="b")
        v, why, info = audit.signature_factor("claude-opus-5-5")
        self.assertEqual(info["source"], "measured")
        self.assertAlmostEqual(v, 0.26, places=3)
        self.assertEqual((info["n"], info["sessions"]), (6, 2))

    def test_pooled_when_the_model_has_too_few(self):
        for i, f in enumerate((0.20, 0.21, 0.22, 0.23, 0.24)):
            self.add(f, model="claude-opus-5", session=str(i))
        v, why, info = audit.signature_factor("claude-opus-5-5")
        self.assertEqual(info["source"], "measured_all")
        self.assertIn("all models", why)

    def test_failed_measurements_are_not_values(self):
        for i in range(6):
            self.add(None, session=str(i))
        self.assertEqual(audit.signature_factor()[2]["source"], "default")


class SizeTest(unittest.TestCase):
    def test_size_tokens_follows_the_hook(self):
        msgs = body()["messages"]
        saved = background.SIG_FACTOR
        try:
            background.SIG_FACTOR = lambda: 0.1
            low = background.size_tokens(msgs)
            background.SIG_FACTOR = lambda: 0.3
            high = background.size_tokens(msgs)
        finally:
            background.SIG_FACTOR = saved
        self.assertEqual(high - low, round(0.2 * len(SIG)))


if __name__ == "__main__":
    unittest.main()
