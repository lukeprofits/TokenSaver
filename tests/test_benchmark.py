
import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands
import json
import tempfile
import unittest
from pathlib import Path

from tokensaver import benchmark, transcripts


def rec(role, content, mid=None, side=False, usage=True, **kw):
    m = {"role": role, "content": content}
    if mid:
        m["id"], m["model"] = mid, "claude-opus-5-5"
        if usage:
            m["usage"] = {"input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 5}
    return dict({"message": m, "isSidechain": side, "timestamp": "2026-10-22T10:00:00Z"}, **kw)


class HistoryTest(unittest.TestCase):
    def write(self, records):
        p = Path(tempfile.mkdtemp()) / "s.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records))
        return p

    def test_sidechain_left_out_and_requests_line_up(self):
        p = self.write([rec("user", "hi"), rec("assistant", [{"type": "text", "text": "a"}], "m1"),
                        rec("user", "sub", side=True), rec("assistant", "x", "s1", side=True),
                        rec("user", "next"), rec("assistant", [{"type": "text", "text": "b"}], "m2")])
        msgs, reqs = transcripts.api_history(p)
        self.assertEqual(len(reqs), 2, "subagent request not counted")
        self.assertEqual([m["content"] for m in transcripts.request_messages(msgs, reqs[1])][-1], [{"type": "text", "text": "next"}])

    def test_history_restarts_at_claude_codes_compaction(self):
        p = self.write([rec("user", "old"), rec("assistant", "a", "m1"),
                        {"subtype": "compact_boundary", "compactMetadata": {}},
                        rec("user", "summary"), rec("assistant", "b", "m2")])
        msgs, reqs = transcripts.api_history(p)
        sent = transcripts.request_messages(msgs, reqs[1])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["content"][0]["text"], "summary")

    def test_tool_references_become_text_for_counting(self):
        m = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                           "content": [{"type": "tool_reference", "tool_name": "WebSearch"}]}]}]
        out = benchmark._countable(m)
        self.assertEqual(out[0]["content"][0]["content"][0], {"type": "text", "text": "[tool WebSearch]"})
        self.assertEqual(m[0]["content"][0]["content"][0]["type"], "tool_reference", "original untouched")


class ReliabilityTest(unittest.TestCase):
    def test_warns_with_reasons_when_many_counts_fail(self):
        from tokensaver import benchmark as bm
        count = type("C", (), {"calls": 2084, "failed": 410,
                               "errors": {"400 messages.43: `tool_use` ids were found without `tool_result`": 66}})()
        rows = [{"requests": 1280, "model": "claude-opus-5-5", "count_failures": 160,
                 "ours_cost": 140.0, "plain_cost": 141.95},
                {"requests": 840, "model": "claude-opus-5", "count_failures": 0, "ours_cost": 231.0, "plain_cost": 280.0}]
        text = "\n".join(bm.reliability(rows, count))
        self.assertIn("410 of 2,084 token counts failed (20%)", text)
        self.assertIn("probably TOO LOW", text)
        self.assertIn("1280 requests", text)
        self.assertNotIn(" 840 requests", text, "only sessions that had failures are listed")
        self.assertIn("tokensaver update", text)

    def test_quiet_when_counts_mostly_succeed(self):
        from tokensaver import benchmark as bm
        count = type("C", (), {"calls": 736, "failed": 20, "errors": {}})()
        self.assertEqual(bm.reliability([], count), [])


SIG = "x" * 30_000      # a thinking signature big enough to measure


def thinking_msgs(n=2):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": [{"type": "text", "text": f"q{i}"}]},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": SIG},
                                                  {"type": "text", "text": f"a{i}"}]}]
    return out + [{"role": "user", "content": [{"type": "text", "text": "next"}]}]


class FakeCounter:
    """count_body: 1,000 tokens without thinking; with valid thinking +0.15/char;
    "invalid" bodies 400 under the binding check, as the API does."""
    def __init__(self, valid=True):
        self.valid, self.seen = valid, []

    def count_body(self, body, beta=None):
        self.seen.append((body, beta))
        has = any(b.get("type") == "thinking" for m in body["messages"] for b in m["content"]
                  if isinstance(b, dict))
        if not has:
            return 1000, ""
        if (body.get("thinking") or {}).get("block_binding", {}).get("prefix_mismatch_behavior") == "error" \
                and not self.valid:
            return None, "400 thinking block bound to a different conversation"
        sig = benchmark.bg._signature_chars(body["messages"])
        return 1000 + round(0.15 * sig), ""


class ThinkingFactorTest(unittest.TestCase):
    def body(self):
        return {"model": "claude-opus-5-5", "messages": thinking_msgs(), "system": [{"type": "text", "text": "s"}],
                "thinking": {"type": "adaptive"}}

    def test_versions_are_separate_models(self):
        self.assertEqual(benchmark._version("claude-opus-5-5"), "opus 5.5")
        self.assertEqual(benchmark._version("claude-opus-5"), "opus 5")
        self.assertEqual(benchmark._version("claude-haiku-4-5-20251001"), "haiku 4.5")
        self.assertEqual(benchmark._version(""), "?")

    def test_sent_request_measured_with_the_binding_check(self):
        b = self.body()
        c = FakeCounter()
        f, why = benchmark._measure_sent(b, benchmark.bg._signature_chars(b["messages"]), c)
        self.assertAlmostEqual(f, 0.15, places=3)
        self.assertEqual(why, "")
        checked = [x for x, beta in c.seen if (x.get("thinking") or {}).get("block_binding")]
        self.assertEqual(len(checked), 1)
        self.assertEqual(checked[0]["thinking"]["block_binding"], {"prefix_mismatch_behavior": "error"})
        self.assertTrue(all(beta == benchmark.BINDING_BETA for _, beta in c.seen))
        self.assertNotIn("block_binding", b["thinking"], "the caller's body is not changed")

    def test_message_fields_count_tokens_rejects_are_dropped(self):
        b = self.body()
        b["messages"][1] = dict(b["messages"][1], output_config={"effort": "high"})
        b["thinking"] = {"type": "adaptive", "display": "updates"}
        c = FakeCounter()
        f, _ = benchmark._measure_sent(b, benchmark.bg._signature_chars(b["messages"]), c)
        self.assertIsNotNone(f)
        self.assertTrue(all(set(m) == {"role", "content"} for x, _ in c.seen for m in x["messages"]))
        self.assertTrue(all("display" not in (x.get("thinking") or {}) for x, _ in c.seen))
        self.assertEqual(b["thinking"]["display"], "updates", "the caller's body is not changed")
        self.assertIn("output_config", b["messages"][1], "the caller's body is not changed")

    def test_invalid_thinking_is_never_measured(self):
        b = self.body()
        f, why = benchmark._measure_sent(b, 60_000, FakeCounter(valid=False))
        self.assertIsNone(f)
        self.assertIn("bound to a different conversation", why)

    def test_source_order_sent_then_rebuilt_then_fallback(self):
        from unittest import mock
        b = self.body()
        with mock.patch.object(benchmark, "_sent_requests",
                               return_value={"claude-opus-5-5": [(b, benchmark.bg._signature_chars(b["messages"]))]}):
            got = benchmark.thinking_factors([], FakeCounter())
        self.assertEqual(got["claude-opus-5-5"]["source"], "sent")
        self.assertIn("verified valid", got["claude-opus-5-5"]["how"])
        with mock.patch.object(benchmark, "_sent_requests",
                               return_value={"claude-opus-5-5": [(b, 60_000)]}):
            got = benchmark.thinking_factors([], FakeCounter(valid=False))
        fa = got["claude-opus-5-5"]
        self.assertEqual(fa["source"], "fallback")
        self.assertIn("1 sent request(s) tried", fa["how"])
        self.assertIn("count refused: 400", fa["how"], "the reason is shown, not swallowed")

    def test_sent_requests_read_from_ledger_and_vault(self):
        from tokensaver import vault
        sent = thinking_msgs()
        state = vault.VAULT_DIR / "proxy_state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "sess-t1.bg.sent.json").write_text(json.dumps({"epoch": "v2:x", "fp": [], "sent": sent, "t": 1}))
        original = dict(self.body(), messages=[dict(m) for m in sent])
        original["messages"][1] = {"role": "assistant", "content": [{"type": "text", "text": "rewritten away"}]}
        vault.record("api_request", content=json.dumps(original), session_id="sess-t1",
                     meta={"request_class": "main", "status": 200, "rewrite": {"bytes_in": 1, "bytes_out": 1}},
                     index=False)
        got = benchmark._sent_requests()
        self.assertIn("claude-opus-5-5", got)
        body, sig = got["claude-opus-5-5"][0]
        self.assertEqual(body["messages"], sent, "the ledger's sent messages, not the stored original")
        self.assertEqual(body["system"], original["system"])
        self.assertEqual(sig, benchmark.bg._signature_chars(sent))

    def test_sensitivity_line_only_with_fallback(self):
        rows = [{"factor_source": "fallback", "thinking_chars_removed": 1000, "plain_cost": 100.0,
                 "ours_cost": 80.0, "ours_cost_factor_lo": 81.0, "ours_cost_factor_hi": 79.0},
                {"factor_source": "sent", "thinking_chars_removed": 1000, "plain_cost": 100.0,
                 "ours_cost": 80.0, "ours_cost_factor_lo": 85.0, "ours_cost_factor_hi": 75.0}]
        text = "\n".join(benchmark.factor_sensitivity(rows))
        self.assertIn("1 session(s)", text)
        self.assertIn("at most 0.5 points", text, "only the fallback session's swing counts")
        self.assertEqual(benchmark.factor_sensitivity(rows[1:]), [])


if __name__ == "__main__":
    unittest.main()
