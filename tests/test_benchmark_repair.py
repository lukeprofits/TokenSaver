"""Rebuilt histories count_tokens refused (a coworker's run: 1,102 of 1,544 counts) are made
countable the same way on both sides, and rate limits are waited out instead of failing.
No network beyond a stub count_tokens server on 127.0.0.1."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import http.server
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tokensaver import benchmark, passes, transcripts


def violations(messages: list) -> list[str]:
    """The API's rules the benchmark tripped on: block types count_tokens takes without
    tools, and each tool call's result leading the very next user message."""
    out = []
    for i, m in enumerate(messages):
        for b in m["content"]:
            t = b.get("type")
            if t not in benchmark.COUNTABLE_TYPES:
                out.append(f"messages.{i}: {t} not accepted")
            if t == "tool_result" and isinstance(b.get("content"), list):
                for x in b["content"]:
                    if x.get("type") not in benchmark.COUNTABLE_TYPES:
                        out.append(f"messages.{i}: {x.get('type')} inside a tool_result")
        if m["role"] == "assistant":
            ids = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
            nxt = messages[i + 1]["content"] if i + 1 < len(messages) and messages[i + 1]["role"] == "user" else []
            lead = [b.get("tool_use_id") for b in nxt[:len(ids)] if b.get("type") == "tool_result"]
            if ids and sorted(lead) != sorted(ids):
                out.append(f"messages.{i}: `tool_use` ids were found without `tool_result` blocks immediately after")
        if m["role"] == "user":
            prev = messages[i - 1]["content"] if i and messages[i - 1]["role"] == "assistant" else []
            calls = {b.get("id") for b in prev if b.get("type") == "tool_use"}
            for b in m["content"]:
                if b.get("type") == "tool_result" and b.get("tool_use_id") not in calls:
                    out.append(f"messages.{i}: tool_result without a call")
    return out


def U(*blocks):
    return {"role": "user", "content": list(blocks)}


def A(*blocks):
    return {"role": "assistant", "content": list(blocks)}


TXT = lambda s: {"type": "text", "text": s}
USE = lambda i, name="Bash": {"type": "tool_use", "id": i, "name": name, "input": {"command": "ls"}}
RES = lambda i, c="ok": {"type": "tool_result", "tool_use_id": i, "content": c}
SRV = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_fetch", "input": {"url": "https://x"}}
SRV_RES = {"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_1",
           "content": {"type": "web_fetch_result", "url": "https://x", "content": {"type": "document",
                       "source": {"type": "text", "media_type": "text/plain", "data": "page"}}}}


class RepairTest(unittest.TestCase):
    def check(self, msgs, expect_repairs):
        self.assertTrue(violations(msgs), "the case must fail as rebuilt")
        out, rep = benchmark.countable(msgs)
        self.assertEqual(violations(out), [])
        for k in expect_repairs:
            self.assertGreater(rep.get(k, 0), 0, k)
        return out

    def test_mixed_server_and_client_tools_in_one_turn(self):
        # Docs: the server tool's result opens the NEXT assistant message, after the client result.
        msgs = [U(TXT("fetch it and run ls")), A(SRV, USE("t1")), U(RES("t1")), A(SRV_RES, TXT("done")),
                U(TXT("thanks"))]
        out = self.check(msgs, ["converted"])
        self.assertIn("page", json.dumps(out), "the fetched content is still counted, as text")

    def test_interrupted_call_gets_a_placeholder(self):
        out = self.check([U(TXT("go")), A(USE("t1")), U(TXT("[Request interrupted by user for tool use]"))],
                         ["placeholder"])
        self.assertEqual(out[2]["content"][0]["content"], benchmark.PLACEHOLDER)

    def test_result_after_text_is_moved_first(self):
        self.check([U(TXT("go")), A(USE("t1")), U(TXT("note"), RES("t1"))], [])

    def test_result_written_later_is_moved_back(self):
        out = self.check([U(TXT("go")), A(USE("t1")), U(TXT("wait")), A(TXT("waiting")), U(RES("t1"))], ["moved"])
        self.assertEqual(out[2]["content"][0]["tool_use_id"], "t1")

    def test_stray_result_and_mcp_blocks(self):
        mcp = {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "q", "server_name": "s", "input": {}}
        self.check([U(RES("ghost")), A(mcp, TXT("x")), U(TXT("y"))], ["stray", "converted"])

    def test_nothing_changes_a_clean_history(self):
        msgs = [U(TXT("go")), A(USE("t1"), USE("t2")), U(RES("t1"), RES("t2")), A(TXT("done"))]
        out, rep = benchmark.countable(msgs)
        self.assertEqual(out, msgs)
        self.assertEqual({k: v for k, v in rep.items() if not k.startswith("_")}, {})

    def test_the_same_on_both_sides(self):
        msgs = [U(TXT("go")), A(SRV, USE("t1")), U(RES("t1", "line\n" * 50)), A(SRV_RES, TXT("done")), U(TXT("ok"))]
        ours = passes.rewrite_messages(msgs, aliases="off")
        a, ra = benchmark.countable(msgs)
        b, rb = benchmark.countable(ours)
        self.assertEqual(violations(b), [])
        self.assertEqual({k: v for k, v in ra.items() if k != "_origin"},
                         {k: v for k, v in rb.items() if k != "_origin"})

    def test_from_a_real_shaped_transcript(self):
        d = Path(tempfile.mkdtemp())
        rows = [
            {"type": "user", "message": {"role": "user", "content": "fetch and run"}},
            {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [SRV, USE("t1")],
                                              "usage": {"input_tokens": 5, "output_tokens": 5}}},
            {"type": "user", "message": {"role": "user", "content": [RES("t1")]}},
            {"type": "assistant", "message": {"id": "m2", "role": "assistant", "content": [SRV_RES, TXT("done")],
                                              "usage": {"input_tokens": 5, "output_tokens": 5}}},
            {"type": "user", "message": {"role": "user", "content": "thanks"}},
        ]
        p = d / "s.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        messages, _ = transcripts.api_history(p)
        self.assertTrue(violations(messages))
        self.assertEqual(violations(benchmark.countable(messages)[0]), [])


class Stub(http.server.BaseHTTPRequestHandler):
    plan: list = []          # statuses to return in order; then 200
    retry_after = "0.01"
    strict_api = True

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        status = Stub.plan.pop(0) if Stub.plan else 200
        if status == 200 and Stub.strict_api:
            v = violations(body["messages"])
            if v:
                status = 400
                msg = v[0]
        if status == 200:
            data = json.dumps({"input_tokens": len(json.dumps(body))}).encode()
            self.send_response(200)
            self.send_header("anthropic-ratelimit-requests-remaining", "1000")
        else:
            data = json.dumps({"error": {"message": msg if status == 400 else "slow down"}}).encode()
            self.send_response(status)
            if status == 429:
                self.send_header("retry-after", Stub.retry_after)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class CounterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        mock.patch.object(benchmark, "COUNT_URL", f"http://127.0.0.1:{self.server.server_port}/v1/messages/count_tokens").start()
        Stub.plan, Stub.strict_api = [], True
        self.c = benchmark.Counter({"x-api-key": "k"}, rate_per_s=1000, sleep=lambda s: None)

    def tearDown(self):
        mock.patch.stopall()

    def test_refused_shapes_are_counted_after_repair(self):
        msgs = [U(TXT("go")), A(SRV, USE("t1")), U(RES("t1")), A(SRV_RES, TXT("done")), U(TXT("ok"))]
        self.assertIsNotNone(self.c("claude-opus-5-5", msgs))
        self.assertEqual((self.c.failed, self.c.rate_limited), (0, 0))
        self.assertGreater(self.c.repairs.get("converted", 0), 0)

    def test_rate_limits_are_waited_out_not_failed(self):
        Stub.plan = [429, 429, 529]
        self.assertIsNotNone(self.c("claude-opus-5-5", [U(TXT("hi"))]))
        self.assertEqual((self.c.failed, self.c.rate_limited, self.c.attempts), (0, 0, 4))

    def test_a_limit_that_never_clears_is_reported_as_rate_limited(self):
        Stub.plan = [429] * 50
        with mock.patch.object(benchmark, "SAMPLE_BUDGET_S", 0.05):
            c = benchmark.Counter({"x-api-key": "k"}, rate_per_s=1000)
            self.assertIsNone(c("claude-opus-5-5", [U(TXT("hi"))]))
        self.assertEqual((c.failed, c.rate_limited), (0, 1))
        self.assertTrue(c.busy(), "optional samples are skipped for a while")

    def test_a_deciding_count_waits_longer_than_a_sample(self):
        waited = []
        Stub.retry_after = "1"                       # longer than the sample budget below
        try:
            with mock.patch.object(benchmark, "SAMPLE_BUDGET_S", 0.5), \
                    mock.patch.object(benchmark, "RATE_BUDGET_S", 100):
                c = benchmark.Counter({"x-api-key": "k"}, rate_per_s=1000, sleep=waited.append)
                Stub.plan = [429] * 3
                self.assertIsNone(c("claude-opus-5-5", [U(TXT("hi"))]), "a sample gives up")
                c = benchmark.Counter({"x-api-key": "k"}, rate_per_s=1000, sleep=waited.append)
                Stub.plan = [429] * 3
                self.assertIsNotNone(c("claude-opus-5-5", [U(TXT("hi"))], patient=True), "a deciding count waits")
        finally:
            Stub.retry_after = "0.01"
        self.assertEqual(c.rate_limited, 0)

    def test_a_new_refusal_names_its_message_and_is_retried_as_text(self):
        Stub.strict_api = False
        weird = {"type": "text", "text": "x", "citations": [{"type": "some_new_thing"}]}
        seen = []

        def first_refuses(handler):
            body = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
            seen.append(body)
            status = 400 if any("citations" in b for m in body["messages"] for b in m["content"]) else 200
            data = json.dumps({"error": {"message": "messages.1.content.0.citations: unknown"}}
                              if status == 400 else {"input_tokens": 7}).encode()
            handler.send_response(status)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        with mock.patch.object(Stub, "do_POST", first_refuses):
            self.assertEqual(self.c("claude-opus-5-5", [U(TXT("a")), A(weird), U(TXT("b"))]), 7)
            self.assertEqual(self.c.failed, 0)
            self.assertEqual(len(seen), 2, "refused once, then accepted")
            self.assertIn("some_new_thing", seen[1]["messages"][1]["content"][0]["text"], "kept as text, still counted")
            self.assertEqual(len(self.c.strict), 1, "remembered: later requests send it as text at once")
            self.assertEqual(self.c("claude-opus-5-5", [U(TXT("a")), A(weird), U(TXT("b")), A(TXT("c")),
                                                        U(TXT("d"))]), 7)
            self.assertEqual(len(seen), 3, "the next request of the session goes through first time")

    def test_rejection_shapes_carry_no_session_text(self):
        Stub.strict_api = False
        Stub.plan = [400, 400]

        def refuse(handler):
            handler.rfile.read(int(handler.headers["Content-Length"]))
            data = json.dumps({"error": {"message": "messages.1: something new"}}).encode()
            handler.send_response(400)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        secret = "SENTINEL-do-not-share"
        with mock.patch.object(Stub, "do_POST", refuse):
            self.c("claude-opus-5-5", [U(TXT(secret)), A(USE("t1", "mcp__corp__lookup")), U(RES("t1", secret))])
        self.assertEqual(self.c.failed, 1)
        dumped = json.dumps(self.c.rejections)
        self.assertNotIn(secret, dumped)
        self.assertNotIn("corp", dumped, "MCP tool names are not shared")
        self.assertIn("tool_use", dumped)


class KeepAliveReplayTest(unittest.TestCase):
    """The benchmark replays keep-alive with the proxy's own rule: a lunch break kept warm
    is a cache read instead of a rewrite, and every bump is paid for, including the ones
    after the session's last request."""

    def session(self):
        import time as _t
        from types import SimpleNamespace
        from tokensaver import pricing
        d = Path(tempfile.mkdtemp())
        t0 = _t.mktime((2026, 10, 20, 9, 0, 0, 0, 0, -1))
        rows, t = [], t0
        for i in range(60):
            t += 90 * 60 if i == 30 else 60                      # a 90-minute break in the middle
            stamp = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))
            rows.append({"type": "user", "timestamp": stamp, "message": {"role": "user", "content": f"step {i}"}})
            rows.append({"type": "assistant", "timestamp": stamp, "message": {
                "id": f"m{i}", "role": "assistant", "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": f"done {i}"}],
                "usage": {"input_tokens": 2, "cache_read_input_tokens": 200_000, "output_tokens": 10,
                          "cache_creation_input_tokens": 100,
                          "cache_creation": {"ephemeral_1h_input_tokens": 100}}}})
        path = d / "s.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        sess = SimpleNamespace(path=path, primary_model="claude-opus-5-5")
        ends = {}
        from tokensaver import keepalive as ka
        for day in range(10, 20):                                # earlier days ending at 18:00
            e = _t.mktime((2026, 10, day, 18, 0, 0, 0, 0, -1))
            ends[ka._day(e)] = (ka._weekday(e), ka._hour(e))
        return SimpleNamespace(session=sess, price=pricing.price_for("claude-opus-5-5"), stem="s"), ends

    def test_a_kept_warm_break_is_cheaper_and_bumps_are_paid(self):
        d, ends = self.session()
        count = lambda model, msgs, **k: 1000 + len(json.dumps(msgs)) // 4
        off = benchmark.replay(d, count, 8, 900, 70, 0.264)
        on = benchmark.replay(d, count, 8, 900, 70, 0.264, keepalive_ends=ends)
        self.assertEqual(on["keepalive_kept"], 1, "the 90-minute break came back to a warm cache")
        self.assertGreaterEqual(on["keepalive_bumps"], 2, "one for the break, more after the last request")
        self.assertGreater(on["keepalive_cost"], 0)
        self.assertLess(on["ours_cost"], off["ours_cost"], "a rewrite of ~200K saved for a few reads")
        self.assertEqual(on["plain_cost"], off["plain_cost"], "plain Claude Code has no keep-alive")


if __name__ == "__main__":
    unittest.main()
