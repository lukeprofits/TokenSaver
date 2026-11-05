"""Transcripts that hold the same record or tool call twice (another machine's run: 210 of
1,776 counts refused with "`tool_use` ids must be unique"), every other rule count_tokens
enforces, checked before counting, and refusals of shapes never seen before: retried with the
named message made stricter until it counts. Also: keep-alive lets go of a session Claude
Code compacts itself. No network beyond a stub count_tokens server on 127.0.0.1."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import calendar
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests import test_benchmark_repair as R
from tests.test_benchmark_repair import A, RES, TXT, U, USE, Stub, violations
from tokensaver import benchmark, keepalive, transcripts


def write(rows) -> Path:
    p = Path(tempfile.mkdtemp()) / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def asst(uid, mid, blocks, usage=True):
    m = {"id": mid, "role": "assistant", "content": blocks}
    if usage:
        m["usage"] = {"input_tokens": 5, "output_tokens": 5}
    return {"type": "assistant", "uuid": uid, "message": m}


def user(uid, blocks):
    return {"type": "user", "uuid": uid, "message": {"role": "user", "content": blocks}}


class TranscriptDuplicatesTest(unittest.TestCase):
    def test_repeated_records_and_calls_are_left_out(self):
        rows = [
            user("u0", "go"),
            asst("a1", "m1", [USE("t1")]),
            asst("a1", "m1", [USE("t1")]),                  # the same record line twice
            user("u1", [RES("t1")]),
            asst("a2", "m2", [USE("t2")]),
            asst("a3", "m2", [USE("t3")]),                  # parallel calls of one reply
            user("u2", [RES("t2")]),
            asst("a4", "m2", [USE("t2")]),                  # t2 again, under a new uuid
            user("u3", [RES("t3")]),
            user("u4", [RES("t2")]),                        # its result again
            asst("a5", "m3", [TXT("done")]),
        ]
        info = {}
        messages, reqs = transcripts.api_history(write(rows), info)
        self.assertEqual(violations(messages), [])
        self.assertEqual(info["duplicates"], 3)
        ids = [b["id"] for m in messages for b in m["content"] if b.get("type") == "tool_use"]
        self.assertEqual(ids, ["t1", "t2", "t3"])
        self.assertEqual(len(reqs), 3, "one request per reply, as before")

    def test_a_clean_transcript_is_unchanged(self):
        rows = [user("u0", "go"), asst("a1", "m1", [USE("t1"), USE("t2")]), user("u1", [RES("t1"), RES("t2")]),
                asst("a2", "m2", [TXT("done")])]
        info = {}
        messages, _ = transcripts.api_history(write(rows), info)
        self.assertEqual(info["duplicates"], 0)
        self.assertEqual(messages, [U(TXT("go")), A(USE("t1"), USE("t2")), U(RES("t1"), RES("t2")), A(TXT("done"))])

    def test_a_new_history_after_compaction_may_reuse_nothing_but_starts_clean(self):
        rows = [user("u0", "go"), asst("a1", "m1", [USE("t1")]), user("u1", [RES("t1")]),
                {"type": "system", "subtype": "compact_boundary", "timestamp": "2026-11-05T18:47:58Z"},
                user("u2", "summary"), asst("a2", "m2", [USE("t1")]), user("u3", [RES("t1")])]
        info = {}
        messages, _ = transcripts.api_history(write(rows), info)
        self.assertEqual(info["duplicates"], 0, "ids are unique per history, which restarts here")
        self.assertEqual(info["boundaries"], [(3, "2026-11-05T18:47:58Z")])


class CountableRulesTest(unittest.TestCase):
    def check(self, msgs, repair):
        self.assertTrue(violations(msgs), "the case must fail as rebuilt")
        out, rep = benchmark.countable(msgs)
        self.assertEqual(violations(out), [])
        self.assertGreater(rep.get(repair, 0), 0, repair)
        return out

    def test_the_shapes_that_were_refused(self):
        # From the result file: the same call twice in one reply, its result paired twice.
        self.check([U(TXT("go")), A(USE("t1"), USE("t1")), U(RES("t1"), RES("t1"))], "duplicate")
        self.check([U(TXT("go")), A(USE("t1")), U(RES("t1")),
                    A(USE("t2"), TXT("a"), USE("t3"), TXT("b"), USE("t1")), U(RES("t2"), RES("t3"), RES("t1"))],
                   "duplicate")

    def test_empty_text_blocks(self):
        self.check([U(TXT("go")), A(TXT(""), TXT("ok")), U(TXT("  "), TXT("next"))], "empty")

    def test_roles_alternate(self):
        out = self.check([U(TXT("a")), U(TXT("b")), A(TXT("c"))], "merged")
        self.assertEqual(out, [U(TXT("a"), TXT("b")), A(TXT("c"))])

    def test_a_message_left_empty_merges_its_neighbours(self):
        self.check([U(TXT("a")), A(TXT("b")), U(TXT("")), A(TXT("c"))], "merged")

    def test_thinking_outside_the_assistant(self):
        self.check([U({"type": "thinking", "thinking": "x", "signature": "s"}, TXT("a")), A(TXT("b"))], "converted")

    def test_same_rules_both_sides(self):
        msgs = [U(TXT("go")), A(USE("t1"), USE("t1")), U(RES("t1", "line\n" * 40), RES("t1"))]
        from tokensaver import passes
        a, ra = benchmark.countable(msgs)
        b, rb = benchmark.countable(passes.rewrite_messages(msgs, aliases="off"))
        self.assertEqual(violations(b), [])
        self.assertEqual({k: v for k, v in ra.items() if k != "_origin"},
                         {k: v for k, v in rb.items() if k != "_origin"})


class EscalationTest(R.CounterTest):
    def test_an_unknown_refusal_escalates_until_counted(self):
        Stub.strict_api = False
        seen = []

        def refuse_until_text(handler):
            body = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
            seen.append(body)
            # Refuses message 1 while it holds any tool_use block (level 1 keeps them).
            bad = any(b.get("type") == "tool_use" for b in body["messages"][1]["content"])
            data = json.dumps({"error": {"message": "messages.1.content.0: something never seen"}}
                              if bad else {"input_tokens": 9}).encode()
            handler.send_response(400 if bad else 200)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        with mock.patch.object(Stub, "do_POST", refuse_until_text):
            self.assertEqual(self.c("claude-opus-5-5", [U(TXT("a")), A(USE("t1")), U(RES("t1")), A(TXT("b"))]), 9)
        self.assertEqual(self.c.failed, 0)
        self.assertEqual(self.c.escalated, {1: 1, 2: 1})
        self.assertEqual(len(seen), 3, "as is, level 1, level 2")
        last = seen[-1]["messages"]
        self.assertEqual(violations(last), [])
        self.assertIn("tool_use", last[1]["content"][0]["text"], "the call is still counted, as text")

    def test_a_refusal_that_names_no_message_is_reported(self):
        Stub.strict_api = False

        def refuse(handler):
            handler.rfile.read(int(handler.headers["Content-Length"]))
            data = json.dumps({"error": {"message": "model: unknown"}}).encode()
            handler.send_response(400)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        with mock.patch.object(Stub, "do_POST", refuse):
            self.assertIsNone(self.c("claude-opus-5-5", [U(TXT("a"))]))
        self.assertEqual(self.c.failed, 1)


class DuplicatesThroughTheStubTest(R.CounterTest):
    def test_duplicate_calls_count_first_time(self):
        msgs = [U(TXT("go")), A(USE("t1"), USE("t1")), U(RES("t1"), RES("t1"))]
        self.assertIsNotNone(self.c("claude-opus-5-5", msgs))
        self.assertEqual((self.c.failed, self.c.attempts), (0, 1), "fixed before sending, no refusal")


class KeepAliveCompactionTest(unittest.TestCase):
    def setUp(self):
        keepalive._SESSIONS.clear()

    def tearDown(self):
        keepalive._SESSIONS.clear()

    def held(self, bumps=0):
        keepalive._SESSIONS["s1"] = {"path": "/v1/messages", "sent": b"{}", "headers": {}, "started": 1000.0,
                                     "size": 400_000, "model": "claude-opus-5-5", "bumps": bumps, "spent": 0.0,
                                     "last_bump": 1000.0 + 58 * 60 if bumps else None, "misses": 0,
                                     "inflight": False}

    def test_claude_codes_own_compaction_ends_the_hold(self):
        self.held()
        with mock.patch.object(keepalive, "_record_break") as rec:
            keepalive.compacted("s1", 1000.0 + 55 * 60, {"cache_read_input_tokens": 400_000})
        self.assertNotIn("s1", keepalive._SESSIONS, "never bumped: its history starts over")
        rec.assert_not_called()

    def test_after_bumps_the_compaction_settles_the_break(self):
        self.held(bumps=1)
        with mock.patch.object(keepalive, "_record_break") as rec, mock.patch.object(keepalive, "_prove") as prove:
            keepalive.compacted("s1", 1000.0 + 100 * 60, {"cache_read_input_tokens": 400_000})
        self.assertEqual(rec.call_args.kwargs["kept"], True, "typed /compact read the kept cache")
        prove.assert_called_once_with(True)


class KeepAliveReplayCompactionTest(R.KeepAliveReplayTest):
    def test_no_bumps_when_claude_code_compacted_the_break(self):
        count = lambda model, msgs, **k: 1000 + len(json.dumps(msgs)) // 4
        d, ends = self.session()
        base = benchmark.replay(d, count, 8, 900, 70, 0.264, keepalive_ends=ends)
        rows = [json.loads(x) for x in d.session.path.read_text().splitlines()]
        # Claude Code's own compaction 55 minutes into the 90-minute break (after reply 29).
        brk = rows[59]["timestamp"]
        t = calendar.timegm(time.strptime(brk, "%Y-%m-%dT%H:%M:%SZ")) + 55 * 60
        rows.insert(60, {"type": "system", "subtype": "compact_boundary",
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))})
        d.session.path.write_text("\n".join(json.dumps(r) for r in rows))
        on = benchmark.replay(d, count, 8, 900, 70, 0.264, keepalive_ends=ends)
        self.assertEqual(on["keepalive_kept"], 0, "a new history is written whatever the cache holds")
        self.assertEqual(base["keepalive_kept"], 1)
        self.assertLess(on["keepalive_bumps"], base["keepalive_bumps"], "none in the compacted break")


if __name__ == "__main__":
    unittest.main()
