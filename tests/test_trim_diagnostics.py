"""The benchmark explains its trim decisions: why a break got no trim, what each applied trim
cut, and what each warm one paid (or would have) on what really followed."""

import json
import tempfile
import time as _t
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from tokensaver import benchmark, pricing

COUNT = lambda model, msgs, **k: 1000 + len(json.dumps(msgs)) // 4


def _session(gaps_min, n=60, ttl="1h"):
    """Bash calls with big outputs, so the old part is worth trimming."""
    key = "ephemeral_1h_input_tokens" if ttl == "1h" else "ephemeral_5m_input_tokens"
    d = Path(tempfile.mkdtemp())
    t, rows = _t.mktime((2026, 10, 20, 9, 0, 0, 0, 0, -1)), []
    for i in range(n):
        t += gaps_min.get(i, 1) * 60
        stamp = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))
        rows.append({"type": "user", "timestamp": stamp, "uuid": f"u{i}",
                     "message": {"role": "user", "content": f"step {i}"}})
        rows.append({"type": "assistant", "timestamp": stamp, "uuid": f"a{i}", "message": {
            "id": f"m{i}", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": f"running {i}"},
                        {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"ls {i}"}}],
            "usage": {"input_tokens": 2, "cache_read_input_tokens": 200_000, "output_tokens": 10,
                      "cache_creation_input_tokens": 100, "cache_creation": {key: 100}}}})
        rows.append({"type": "user", "timestamp": stamp, "uuid": f"r{i}", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "\n".join(f"line {i}.{j} " + "x" * 60
                                                                               for j in range(300))}]}})
        rows.append({"type": "assistant", "timestamp": stamp, "uuid": f"b{i}", "message": {
            "id": f"m{i}b", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": f"done {i}"}],
            "usage": {"input_tokens": 2, "cache_read_input_tokens": 200_000, "output_tokens": 10,
                      "cache_creation_input_tokens": 100, "cache_creation": {key: 100}}}})
    path = d / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return NS(session=NS(path=path, primary_model="claude-opus-5-5"),
              price=pricing.price_for("claude-opus-5-5"), stem="s")


class ReplayTrimsTest(unittest.TestCase):
    def setUp(self):
        from tokensaver import keepalive as ka
        p = mock.patch.object(ka, "idle_compact_at", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_decisions_recorded_and_priced(self):
        r = benchmark.replay(_session({30: 20, 45: 90}), COUNT, 8, 900, 70, 0.264)
        trims = r["trims"]
        self.assertTrue(trims, "both breaks priced")
        self.assertEqual(len(trims) + sum(r["no_trim"].values()), r["breaks"], "every break accounted for")
        warm = [t for t in trims if not t["cold"]]
        cold = [t for t in trims if t["cold"]]
        self.assertTrue(warm and cold)
        p = pricing.price_for("claude-opus-5-5")
        for t in warm:
            if t["s_new"] < t["s_now"]:
                want = (t["s_now"] - t["s_new"]) * p.read * (t["after"] - t["reps"]) / 1e6
                self.assertAlmostEqual(t["value"], want, delta=0.01)  # reps is stored rounded
        self.assertNotIn("value", cold[0], "a cold trim is free: nothing to weigh")
        self.assertEqual(warm[0]["after"], 29, "counted up to the next free point (the cold break)")
        for t in trims:
            if t["applied"]:
                self.assertEqual(max(t["parts"], key=t["parts"].get), "tool output")
                self.assertLess(t["kept_chars"], sum(t["parts"].values()))

    def test_no_text_in_the_record(self):
        r = benchmark.replay(_session({30: 20, 45: 90}), COUNT, 8, 900, 70, 0.264)
        self.assertNotIn("line 3.", json.dumps({"trims": r["trims"], "no_trim": r["no_trim"],
                                                "short_trims": r["short_trims"]}))

    def test_shorter_breaks_priced_not_applied(self):
        p = pricing.price_for("claude-opus-5-5")
        for ttl, cold in (("5m", True), ("1h", False)):
            base = benchmark.replay(_session({}, ttl=ttl), COUNT, 8, 900, 70, 0.264)
            r = benchmark.replay(_session({30: 8, 32: 8}, ttl=ttl), COUNT, 8, 900, 70, 0.264)
            self.assertEqual((r["compactions"], base["compactions"]), (0, 0), "an 8-minute break never trims")
            self.assertEqual(len(r["short_trims"]), 2 if cold else 1,
                             "on a 5m cache each expired break is its own free point" if cold
                             else "warm: both breaks share one stretch; only the first counts")
            t = r["short_trims"][0]
            self.assertEqual(t["cold"], cold, ttl)
            cut = t["s_now"] - t["s_new"]
            if cold:
                want = cut * (p.write_5m + p.read * t["after"]) / 1e6
                self.assertAlmostEqual(t["value"], want, delta=0.01)
                self.assertGreater(t["value"], 0, "free on an expired cache")


class CostSplitTest(unittest.TestCase):
    def test_billed_and_modelled_tokens(self):
        with mock.patch("tokensaver.keepalive.idle_compact_at", return_value=None):
            r = benchmark.replay(_session({30: 20}), COUNT, 8, 900, 70, 0.264)
        n = r["requests"]
        self.assertEqual(r["real_tokens"], {"input": 2 * n, "read": 200_000 * n, "write_5m": 0,
                                            "write_1h": 100 * n, "output": 10 * n})
        m = r["model_tokens"]
        self.assertEqual(m["read"] + m["write"], (2 + 200_000 + 100) * n, "every billed token, split")
        self.assertGreater(m["read"], m["write"])

    def test_line(self):
        rows = [{"real_tokens": {"input": 3_000_000, "read": 90_000_000, "write_1h": 5_000_000, "write_5m": 0,
                                 "output": 1_000_000},
                 "model_tokens": {"read": 88_000_000, "write": 10_000_000}}]
        self.assertEqual(benchmark.cost_split_line(rows),
                         ["   billed: 90.0M read, 5.0M written, 3.0M uncached input, 1.0M output; "
                          "modelled: 88.0M read, 10.0M written"])
        self.assertEqual(benchmark.cost_split_line([{}]), [])


class TrimsLinesTest(unittest.TestCase):
    def row(self, *trims, no_trim=None):
        return {"trims": list(trims), "no_trim": no_trim or {}}

    def warm(self, reps, value, applied=False):
        return {"cold": False, "applied": applied, "reps": reps, "after": 0, "s_now": 200, "s_new": 100,
                "value": value, **({"parts": {"tool output": 80, "user text": 20}, "kept_chars": 30}
                                   if applied else {})}

    def test_best_limit_and_counts(self):
        rows = [self.row(self.warm(40, 2.0, applied=True), self.warm(90, 3.0), self.warm(180, -5.0),
                         no_trim={"too little old history": 4})]
        lines = benchmark.trims_lines(rows, 70)
        self.assertIn("3 priced, 1 applied (0 cold, 1 warm); 2 warm skipped", lines[0])
        self.assertIn("too little old history x4", lines[0])
        self.assertIn("best limit here 100 requests (+$3.00 vs 70)", lines[1])
        self.assertIn("skipped ones would have made +$3.00 (1) and -$5.00 (1)", lines[1])
        self.assertIn("tool output 80%, user text 20%; the trim keeps 30%", lines[2])

    def test_shorter_breaks_line(self):
        short = [{"cold": True, "reps": 0, "after": 5, "s_now": 9, "s_new": 1, "value": 1.5},
                 {"cold": False, "reps": 40, "after": 5, "s_now": 9, "s_new": 1, "value": -0.5},
                 {"cold": False, "reps": 400, "after": 5, "s_now": 9, "s_new": 1, "value": -9.0}]
        lines = benchmark.trims_lines([{"trims": [], "no_trim": {}, "short_trims": short}], 70)
        self.assertEqual(lines[-1], " trims at shorter breaks (5-15 min, priced, not applied; estimate): "
                                    "on an expired cache +$1.50 (1 breaks), on a warm one that pays back -$0.50 (1)")

    def test_quiet_without_breaks(self):
        self.assertEqual(benchmark.trims_lines([{"trims": [], "no_trim": {}}], 70), [])


class PredictionTest(unittest.TestCase):
    """Cache reads predicted from reply times and from request times."""

    def reqs(self, rows):
        stamp = lambda s: _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(1_800_000_000 + s))
        return [NS(timestamp=stamp(end), start=stamp(start) if start is not None else "", read=read,
                   context=100_000, write_5m=w5, write_1h=0) for start, end, read, w5 in rows]

    def test_request_times_fix_a_long_reply_at_the_edge(self):
        # A 5-minute cache. Request 2 starts 4 min after request 1 started (a hit), but its reply
        # took almost 3 minutes: reply times say 5 min 40 s (a miss).
        rq = self.reqs([(0, 60, 0, 100_000), (240, 400, 99_000, 1_000)])
        p = __import__("tokensaver.cachettl", fromlist=["x"]).prediction(rq)
        self.assertEqual(p["reply"], [0, 1])
        self.assertEqual(p["request"], [1, 1])
        self.assertEqual((p["edge_reply"], p["edge_request"]), ([0, 1], [1, 1]))

    def test_missing_start_falls_back_to_reply_time(self):
        rq = self.reqs([(None, 60, 0, 100_000), (None, 120, 99_000, 1_000)])
        from tokensaver import cachettl
        p = cachettl.prediction(rq)
        self.assertEqual((p["reply"], p["request"]), ([1, 1], [1, 1]))

    def test_line(self):
        rows = [{"prediction": {"reply": [9, 10], "request": [10, 10], "edge_reply": [1, 2], "edge_request": [2, 2]}}]
        self.assertEqual(benchmark.prediction_line(rows),
                         ["   cache reads predicted: from reply times 90.0%, from request times 100.0%; "
                          "near the cache's expiry (2 gaps): 50.0% vs 100.0%"])

    def test_transcript_start_times(self):
        from tokensaver import transcripts
        d = _session({})
        reqs = [e[1] for e in transcripts.api_history(d.session.path)[1]]
        self.assertTrue(all(r.start and r.start <= r.timestamp for r in reqs))
        self.assertEqual([r.start for r in transcripts.read_session(d.session.path).requests],
                         [r.start for r in reqs], "both loaders agree")


class MachineStateTest(unittest.TestCase):
    def test_no_proxy_and_lines(self):
        with mock.patch("tokensaver.service.health", return_value=None):
            st = benchmark.machine_state()
        self.assertIsNone(st["proxy"])
        self.assertIn("bill", st)
        self.assertEqual(benchmark.machine_line(st)[0], " TokenSaver here: proxy not answering")
        blob = json.dumps(st)
        self.assertNotIn("http", blob, "no URLs in a shared file")

    def test_running_proxy_line(self):
        st = {"proxy": {"code": "a", "code_on_disk": "b", "audit": {"on": False, "disabled": "401 from count_tokens"},
                        "fallback": False},
              "bill": {"last_7_days": {"real": 90.0, "plain": 100.0, "requests": 10},
                       "all": {"real": 0, "plain": 0, "requests": 0}}}
        lines = benchmark.machine_line(st)
        self.assertEqual(lines[0], " TokenSaver here: proxy code a (older than the code on disk: restart), "
                                   "audit off (401 from count_tokens), fallback off")
        self.assertEqual(lines[1], "   live bill: last 7 days $90.00 vs $100.00 plain (10.0% saved, 10 requests)")

    def test_cost_by_model(self):
        rows = [{"model": "claude-opus-5-5", "plain_cost": 10.0, "real_cost": 9.0},
                {"model": "claude-sonnet-5-5", "plain_cost": 2.0, "real_cost": 2.0}]
        self.assertEqual(benchmark.cost_by_model_line(rows),
                         ["   by model: opus 5.5 $10.00 vs $9.00 (1), sonnet 5.5 $2.00 vs $2.00 (1)"])
        self.assertEqual(benchmark.cost_by_model_line(rows[:1]), [])


if __name__ == "__main__":
    unittest.main()
