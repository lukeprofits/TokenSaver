"""Giant pasted logs in old history (`paste`, `bg_trim_pastes`): what qualifies, what a cut
keeps, that the developer's own words and recent history are never touched, and the
benchmark's value and quality lines for it."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import tempfile
import time as _t
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from tokensaver import background as bg, benchmark, paste, pricing


def log(n: int, every_error: int = 97) -> str:
    out = []
    for i in range(n):
        if i % every_error == 13:
            out.append(f"2026-11-17 10:{i // 60 % 60:02d}:{i % 60:02d}.123 ERROR [main] c.e.Checkout - "
                       f"payment declined order={i} code=card_declined")
        else:
            out.append(f"2026-11-17 10:{i // 60 % 60:02d}:{i % 60:02d}.123 INFO  [main] c.e.Service - "
                       f"handled request id={i} took={i % 7}ms")
    return "\n".join(out)


CODE = "\n".join(f"    def handler_{i}(self, req):\n        return self.service.run(req, {i})" for i in range(250))
PROSE = "\n".join(f"Paragraph {i}: we decided to keep the old behaviour because the migration "
                  f"would break the reporting team's nightly job, which still reads it." for i in range(300))


class RegionTest(unittest.TestCase):
    def test_small_logs_are_left_alone(self):
        for n in (10, 150):
            text = "why?\n" + log(n)
            self.assertEqual(paste.cut(text, set()), (text, 0), f"{n} lines")

    def test_a_giant_log_is_cut_and_the_question_kept(self):
        text = "Why does checkout fail? Here's the log:\n\n" + log(600) + "\n\nThanks!"
        out, gone = paste.cut(text, set())
        self.assertGreater(gone, 400)
        self.assertLess(len(out), len(text) / 4)
        self.assertTrue(out.startswith("Why does checkout fail? Here's the log:\n\n"))
        self.assertTrue(out.endswith("\n\nThanks!"))
        self.assertIn(paste.MARK, out)
        self.assertIn("id=0 ", out, "first lines kept")
        self.assertIn("id=599 ", out, "last lines kept")
        self.assertIn("payment declined", out, "an error kind kept")
        self.assertIn("this kind of line", out, "with how often it repeated")

    def test_a_line_referred_to_later_is_kept(self):
        text = log(600)
        self.assertNotIn("id=300 ", paste.cut(text, set())[0])
        self.assertIn("id=300 ", paste.cut(text, {"id=300"} | {"300"})[0])

    def test_code_prose_and_mixed_are_left_alone(self):
        for name, text in (("code", CODE), ("prose", PROSE)):
            self.assertEqual(paste.cut(text, set())[1], 0, name)
        mixed = "\n".join(a + "\n" + b for a, b in zip(log(300).split("\n"), PROSE.split("\n")))
        self.assertEqual(paste.cut(mixed, set())[1], 0, "half explanation, half log")
        traced = "\n".join(f"2026-11-17 10:00:{i % 60:02d} DEBUG calling" if i % 3 == 0
                           else f"    return self.service.run(req, {i})" for i in range(600))
        self.assertEqual(paste.cut(traced, set())[1], 0, "code with log lines in it")
        stamped = "\n".join(f'    return check("2026-11-17 10:00:{i % 60:02d}", {i})' for i in range(300))
        self.assertIn("reads like code", paste.regions(stamped)[1], "code full of timestamps")
        self.assertEqual(paste.cut(stamped, set())[1], 0)

    def test_stack_trace_and_json(self):
        trace = "java.lang.IllegalStateException: boom\n" + "\n".join(
            f"\tat com.example.Service.call{i}(Service.java:{i})" for i in range(400))
        self.assertGreater(paste.cut(trace, set())[1], 300)
        data = json.dumps([{"id": i, "name": f"item {i}", "price": i * 3} for i in range(400)], indent=2)
        self.assertGreater(paste.cut(data, set())[1], 1000)

    def test_deterministic(self):
        text = "see:\n" + log(800)
        self.assertEqual(paste.cut(text, {"id=5"}), paste.cut(text, {"id=5"}))


def _hist(paste_text: str, n_after: int = 40) -> list:
    """A pasted log early on, then enough work after it for a trim to have an old part."""
    out = [{"role": "user", "content": "Why does checkout fail? Here's the log:\n" + paste_text}]
    for i in range(n_after):
        if i and i % 10 == 0:                          # a trim cuts only where the developer typed
            out.append({"role": "user", "content": f"next, part {i}"})
            out.append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
        out.append({"role": "assistant", "content": [{"type": "text", "text": f"step {i}"},
                    {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"ls {i}"}}]})
        out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}",
                    "content": "\n".join(f"file{i}_{j}.txt " + "x" * 60 for j in range(300))}]})
    out.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    out.append({"role": "user", "content": "thanks"})
    return out


class TrimTest(unittest.TestCase):
    def test_off_by_default_and_unchanged_when_off(self):
        h = _hist(log(600))
        self.assertFalse(bg.TRIM_PASTES)
        same = lambda r: {k: v for k, v in r.items() if k != "made"}
        self.assertEqual(same(bg.build(h)), same(bg.build(h, pastes=False)))
        self.assertNotIn(paste.MARK, bg.build(h)["text"])

    def test_on_cuts_the_paste_in_the_old_part_only(self):
        h = _hist(log(600))
        off, on = bg.build(h, pastes=False), bg.build(h, pastes=True)
        self.assertEqual(off["cut"], on["cut"])
        self.assertIn(paste.MARK, on["text"])
        self.assertIn("very large pasted logs", on["text"])
        self.assertLess(len(on["text"]), len(off["text"]) - 20_000)
        self.assertIn("Why does checkout fail? Here's the log:", on["text"], "the developer's words kept")
        # The recent history the trim keeps verbatim is never touched.
        # (`apply` puts the trim text at the head of the first kept message, then that message.)
        applied = bg.apply(h, dict(on, tail_end=len(h)))
        self.assertEqual(applied[1:], h[on["cut"] + 1:])
        self.assertEqual(applied[0]["content"][1:], bg._items(h[on["cut"]]))

    def test_a_paste_in_the_recent_part_is_kept_whole(self):
        h = _hist(log(10), n_after=40) + [{"role": "user", "content": "and now this:\n" + log(600)}]
        on = bg.build(h, pastes=True)
        applied = bg.apply(h, dict(on, tail_end=len(h)))
        self.assertEqual(applied[-1], h[-1])

    def test_a_detail_the_developer_mentions_later_is_kept(self):
        h = _hist(log(600))
        h[-1] = {"role": "user", "content": "what about request id=300 ?"}
        self.assertIn("id=300 ", bg.build(h, pastes=True)["text"])


COUNT = lambda model, msgs, **k: 1000 + len(json.dumps(msgs)) // 4


def _session(paste_lines: int):
    """A transcript: a pasted log, Bash work, a 20-minute break, more work."""
    d = Path(tempfile.mkdtemp())
    t, rows = _t.mktime((2026, 10, 20, 9, 0, 0, 0, 0, -1)), []
    rows.append({"type": "user", "timestamp": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t)), "uuid": "u0",
                 "message": {"role": "user", "content": "here's the log:\n" + log(paste_lines)}})
    for i in range(60):
        t += 90 * 60 if i == 40 else 60                 # a lunch break: the cache expired
        stamp = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))
        if i and i % 10 == 0:                            # a trim cuts only where the developer typed
            rows.append({"type": "user", "timestamp": stamp, "uuid": f"q{i}",
                         "message": {"role": "user", "content": f"next, part {i}"}})
        elif i:
            rows.append({"type": "user", "timestamp": stamp, "uuid": f"u{i}",
                         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i - 1}",
                                     "content": "\n".join(f"line {i}.{j} " + "x" * 60 for j in range(200))}]}})
        text = "the declined order was order=13" if i == 50 else f"running {i}"
        rows.append({"type": "assistant", "timestamp": stamp, "uuid": f"a{i}", "message": {
            "id": f"m{i}", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text},
                        {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"ls {i}"}}],
            "usage": {"input_tokens": 2, "cache_read_input_tokens": 200_000, "output_tokens": 10,
                      "cache_creation_input_tokens": 100, "cache_creation": {"ephemeral_1h_input_tokens": 100}}}})
    path = d / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return NS(session=NS(path=path, primary_model="claude-opus-5-5"),
              price=pricing.price_for("claude-opus-5-5"), stem="s")


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        from tokensaver import keepalive as ka
        p = mock.patch.object(ka, "idle_compact_at", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_cutting_pastes_saves_and_is_judged_on_what_came_next(self):
        d = _session(3000)
        off = benchmark.replay(d, COUNT, 8, 900, 70, 0.264)
        on = benchmark.replay(d, COUNT, 8, 900, 70, 0.264, pastes=True)
        self.assertEqual(off["plain_cost"], on["plain_cost"], "plain Claude Code is the same")
        self.assertLess(on["ours_cost"], off["ours_cost"])
        lost = [t["lost"] for t in on["trims"] if t.get("lost")]
        self.assertTrue(lost)
        self.assertGreater(lost[0]["paste_cut"], 0)
        self.assertNotIn("lost", json.dumps(off["trims"]), "only recorded when pastes are cut")
        self.assertNotIn("handled request", json.dumps(on), "counts only, no text")

    def test_scan_and_lines(self):
        d = _session(3000)
        scan = benchmark.paste_scan(__import__("tokensaver").transcripts.api_history(d.session.path)[0])
        self.assertEqual(scan["qualifying"], 1)
        off = dict(benchmark.replay(d, COUNT, 8, 900, 70, 0.264), session="aa")
        on = benchmark.replay(d, COUNT, 8, 900, 70, 0.264, pastes=True)
        lines = benchmark.pastes_lines([off], {"aa": on}, {"aa": scan})
        self.assertIn("1 in 1 sessions", lines[0])
        self.assertIn("off by default", lines[1])
        self.assertIn("quality, on what was used after each break", lines[2])
        none = benchmark.pastes_lines([off], {}, {"bb": {"qualifying": 0, "left_alone": {"reads like code": 2}}})
        self.assertEqual(none, [" giant pastes (200+ lines of logs or data): none here; "
                                "big pastes left alone: reads like code x2"])

    def test_lost_counts_a_detail_used_again(self):
        old = [{"role": "user", "content": "log:\n" + log(600)}]
        kept = paste.cut(old[0]["content"], set())[0]
        later = [{"role": "assistant", "content": [{"type": "text", "text": "request id=300 failed"}]}]
        got = benchmark._lost(old, kept, later)
        self.assertGreater(got["paste_cut"], 0)
        self.assertGreaterEqual(got["paste_lost"], 1)
        self.assertEqual(benchmark._lost(old, kept, [])["paste_lost"], 0)


class RunTest(unittest.TestCase):
    """The whole `tokensaver --benchmark` path on a transcript with a giant paste, with the
    token counter answered locally: the new lines are printed and saved, nothing crashes."""

    def test_end_to_end(self):
        from tokensaver import keepalive as ka, sim
        src = _session(3000).session.path
        root = Path(tempfile.mkdtemp())
        (root / "proj").mkdir()
        (root / "proj" / "s.jsonl").write_text(src.read_text())
        real_sessions = sim.sessions
        out = Path(tempfile.mkdtemp()) / "result.json"

        def body_count(self, body, beta=None, budget=None):
            self.calls += 1
            return 1000 + len(json.dumps(body.get("messages"))) // 4, ""

        with mock.patch.object(benchmark, "credentials", return_value=({"x-api-key": "k"}, "test")), \
                mock.patch.object(benchmark.Counter, "count_body", body_count), \
                mock.patch.object(sim, "sessions", lambda **k: real_sessions(root=root, **k)), \
                mock.patch.object(benchmark, "machine_state", return_value={"proxy": None}), \
                mock.patch.object(ka, "idle_compact_at", return_value=None), \
                mock.patch("builtins.print") as printed:
            code = benchmark.run(yes=True, out=out)
        text = "\n".join(" ".join(str(a) for a in c.args) for c in printed.call_args_list)
        self.assertEqual(code, 0, text[-2000:])
        self.assertIn("giant pastes (200+ lines of logs or data): 1 in 1 sessions", text)
        self.assertIn("cutting them in trims", text)
        self.assertIn("quality, on what was used after each break", text)
        self.assertNotIn("not reported", text)
        saved = json.loads(out.read_text())
        self.assertEqual(sum(s["qualifying"] for s in saved["pastes"]["scans"].values()), 1)
        self.assertTrue(saved["pastes"]["with"])
        self.assertNotIn("handled request", out.read_text(), "numbers only: no text from the session")


if __name__ == "__main__":
    unittest.main()
