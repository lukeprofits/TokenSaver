"""The thinking rate: a small store of sent requests, measured weekly, fitted per model."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import sqlite3
import time
import unittest
from unittest import mock

from tokensaver import audit, background, thinking, vault

SIG = "x" * 30_000


def setUpModule():
    vault.record("test_init", index=False)      # creates the fake home's vault


def body(model="claude-opus-5-5", extra=""):
    msgs = [{"role": "user", "content": [{"type": "text", "text": "q" + extra}]},
            {"role": "assistant", "output_config": {"effort": "high"},
             "content": [{"type": "thinking", "thinking": "", "signature": SIG}, {"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "text", "text": "next"}]}]
    return {"model": model, "messages": msgs, "system": [{"type": "text", "text": "sys"}],
            "tools": [{"name": "Bash", "input_schema": {}}], "thinking": {"type": "adaptive", "display": "updates"}}


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


def rows():
    conn = sqlite3.connect(vault.DB_PATH)
    try:
        return [json.loads(m) for (m,) in conn.execute(
            "SELECT meta FROM events WHERE kind='thinking_factor' ORDER BY id")]
    finally:
        conn.close()


def clear():
    import shutil
    shutil.rmtree(thinking.DIR, ignore_errors=True)
    conn = sqlite3.connect(vault.DB_PATH)
    conn.execute("DELETE FROM events WHERE kind='thinking_factor'")
    conn.commit()
    conn.close()


class MeasureTest(unittest.TestCase):
    def test_measured_with_binding_check_on_a_clean_copy(self):
        b = body()
        count, seen = counter(0.25)
        got = audit.measure_thinking(b, count)
        self.assertAlmostEqual(got["factor"], 0.25, places=3)
        with_t = seen[0][0]
        self.assertEqual(with_t["thinking"]["block_binding"], {"prefix_mismatch_behavior": "error"})
        self.assertNotIn("display", with_t["thinking"])
        self.assertTrue(all(set(m) == {"role", "content"} for x, _ in seen for m in x["messages"]))
        self.assertTrue(all(beta == audit.BINDING_BETA for _, beta in seen))
        self.assertIn("output_config", b["messages"][1], "the caller's body is unchanged")

    def test_refused_zero_and_implausible_are_not_values(self):
        got = audit.measure_thinking(body(), counter(refuse=(400, "thinking bound to a different conversation"))[0])
        self.assertIsNone(got["factor"])
        self.assertIn("bound to a different conversation", got["why"])
        self.assertIn("0 tokens", audit.measure_thinking(body(), counter(0.0)[0])["why"])
        self.assertIn("implausible", audit.measure_thinking(body(), counter(0.9)[0])["why"])


class StoreTest(unittest.TestCase):
    def setUp(self):
        clear()

    def test_one_sample_per_session_newest_wins_after_refresh(self):
        t = 1_000_000.0
        self.assertTrue(thinking.store("s1", body(extra="1"), now=t))
        self.assertFalse(thinking.store("s1", body(extra="2"), now=t + 60), "refresh limit")
        self.assertTrue(thinking.store("s1", body(extra="3"), now=t + thinking.REFRESH_S + 1))
        self.assertEqual(thinking.load("s1")["messages"][0]["content"][0]["text"], "q3")
        self.assertEqual(len(list(thinking.DIR.glob("*.json.gz"))), 1)

    def test_shared_parts_stored_once_and_rebuilt_exactly(self):
        thinking.store("s1", body(), now=1.0)
        thinking.store("s2", body(extra="x"), now=2.0)
        self.assertEqual(len(list((thinking.DIR / "shared").glob("*.json.gz"))), 1)
        self.assertEqual(thinking.load("s2"), body(extra="x"))

    def test_too_little_thinking_or_none_is_not_stored(self):
        small = body()
        small["messages"][1]["content"][0]["signature"] = "x" * 100
        self.assertFalse(thinking.store("s", small))
        self.assertFalse(thinking.store("s", dict(body(), thinking={"type": "disabled"})))
        self.assertFalse(thinking.store(None, body()))

    def test_pruned_by_age_and_per_model_and_orphans_go(self):
        now = 10_000_000.0
        thinking.store("old", body(model="claude-x"), now=now - 31 * 86400)
        for i in range(thinking.PER_MODEL + 2):
            thinking.store(f"s{i}", body(), now=now + i)
        idx = thinking._index()["samples"]
        self.assertNotIn("old", idx, "past 30 days")
        self.assertEqual(len(idx), thinking.PER_MODEL)
        self.assertNotIn("s0", idx, "oldest of the model dropped first")
        self.assertEqual(len(list(thinking.DIR.glob("*.json.gz"))), thinking.PER_MODEL)
        shared = {e["shared"] for e in idx.values()}
        self.assertEqual({p.name[:-8] for p in (thinking.DIR / "shared").glob("*.json.gz")}, shared,
                         "the old model's shared file went with it")


class WeeklyTest(unittest.TestCase):
    def setUp(self):
        clear()
        audit._state.update(calls_today=0, disabled=None)

    def test_measures_waiting_samples_once_a_week(self):
        thinking.store("s1", body(), now=time.time())
        self.assertTrue(thinking.due())
        self.assertEqual(thinking.measure_due(counter(0.25)[0]), 1)
        self.assertAlmostEqual(rows()[-1]["factor"], 0.25, places=3)
        self.assertFalse(thinking.due(), "measured: nothing waits, and the week restarts")
        thinking.store("s2", body(extra="new"), now=time.time())
        self.assertFalse(thinking.due(), "a new sample waits for next week")

    def test_rate_limit_leaves_samples_for_the_next_job(self):
        thinking.store("s1", body(), now=time.time())
        self.assertEqual(thinking.measure_due(counter(refuse=(429, "rate limited"))[0]), 0)
        self.assertEqual(rows(), [])
        self.assertTrue(thinking.due(), "still waiting")

    def test_a_binding_400_in_the_worker_never_disables_the_audit(self):
        job = {"path": "/v1/messages", "headers": {"anthropic-beta": "oauth-2025-04-20"},
               "sent": json.dumps(body()).encode(), "session": "s400"}
        calls = []

        def fake(path, h, b):
            calls.append(h["anthropic-beta"])
            return None, 400, {}
        with mock.patch.object(audit, "_count", side_effect=fake):
            audit._samples_job(job)
        self.assertIsNone(audit._state["disabled"], "the audit stays on")
        self.assertTrue(rows()[-1]["why"].startswith("count refused: 400"))
        self.assertTrue(all(c == f"oauth-2025-04-20,{audit.BINDING_BETA}" for c in calls))


class FitTest(unittest.TestCase):
    def setUp(self):
        clear()

    def add(self, factor, model="claude-opus-5-5", session="a"):
        vault.record("thinking_factor", session_id=session, index=False,
                     meta={"factor": factor, "model": model, "why": "", "sig_chars": 30000})

    def test_default_until_enough_results_and_sessions(self):
        for i in range(9):
            self.add(0.2, session=str(i % 3))
        v, why, info = audit.signature_factor("claude-opus-5-5")
        self.assertEqual((v, info["source"]), (background.TOKENS_PER_SIGNATURE_CHAR, "default"))
        self.assertIn("9/10 measured", why)
        self.add(0.2, session="0")
        self.assertEqual(audit.signature_factor("claude-opus-5-5")[2]["source"], "measured")

    def test_needs_three_sessions(self):
        for i in range(12):
            self.add(0.2, session=str(i % 2))
        self.assertEqual(audit.signature_factor("claude-opus-5-5")[2]["source"], "default")

    def test_never_another_models_rate(self):
        for i in range(12):
            self.add(0.2, model="claude-opus-5", session=str(i))
        v, _, info = audit.signature_factor("claude-opus-5-5")
        self.assertEqual((v, info["source"]), (background.TOKENS_PER_SIGNATURE_CHAR, "default"))
        self.assertAlmostEqual(audit.signature_factor("claude-opus-5")[0], 0.2)
        self.assertEqual(audit.signature_factor(None)[2]["source"], "default")

    def test_failed_measurements_are_not_values(self):
        for i in range(12):
            self.add(None, session=str(i))
        self.assertEqual(audit.signature_factor("claude-opus-5-5")[2]["source"], "default")


class SizeTest(unittest.TestCase):
    def test_size_tokens_uses_the_sessions_model(self):
        msgs = body()["messages"]
        saved = background.SIG_FACTOR
        try:
            background.SIG_FACTOR = lambda model=None: {"m-low": 0.1, "m-high": 0.3}.get(model, 0.2)
            token = background.MODEL.set("m-low")
            low = background.size_tokens(msgs)
            background.MODEL.set("m-high")
            high = background.size_tokens(msgs)
            background.MODEL.reset(token)
            default = background.size_tokens(msgs)
        finally:
            background.SIG_FACTOR = saved
        self.assertEqual(high - low, round(0.2 * len(SIG)))
        self.assertEqual(default - low, round(0.1 * len(SIG)), "no model set: the default")


if __name__ == "__main__":
    unittest.main()
