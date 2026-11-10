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


def _session(gaps_min, n=60):
    """Bash calls with big outputs, so the old part is worth trimming."""
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
                      "cache_creation_input_tokens": 100, "cache_creation": {"ephemeral_1h_input_tokens": 100}}}})
        rows.append({"type": "user", "timestamp": stamp, "uuid": f"r{i}", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "\n".join(f"line {i}.{j} " + "x" * 60
                                                                               for j in range(300))}]}})
        rows.append({"type": "assistant", "timestamp": stamp, "uuid": f"b{i}", "message": {
            "id": f"m{i}b", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": f"done {i}"}],
            "usage": {"input_tokens": 2, "cache_read_input_tokens": 200_000, "output_tokens": 10,
                      "cache_creation_input_tokens": 100, "cache_creation": {"ephemeral_1h_input_tokens": 100}}}})
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
        self.assertNotIn("line 3.", json.dumps({"trims": r["trims"], "no_trim": r["no_trim"]}))


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

    def test_quiet_without_breaks(self):
        self.assertEqual(benchmark.trims_lines([{"trims": [], "no_trim": {}}], 70), [])


if __name__ == "__main__":
    unittest.main()
