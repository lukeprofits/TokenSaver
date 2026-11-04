"""Cache keep-alive: the learned day end (no lookahead), when a bump is due, which sessions
may be bumped at all, what a bump sends, and how each outcome is handled. No real requests:
a stub upstream on 127.0.0.1, fake times."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import http.server
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from tokensaver import config, keepalive as ka

H = 3600
UA = "claude-cli/2.1.287 (external, cli)"
SENT = json.dumps({"model": "claude-opus-5-5", "max_tokens": 128000, "stream": True,
                   "thinking": {"type": "adaptive"},
                   "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                   "messages": [{"role": "user", "content": [{"type": "text", "text": "x",
                                 "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]}).encode()
USAGE = {"cache_read_input_tokens": 300_000, "cache_creation_input_tokens": 900,
         "cache_creation": {"ephemeral_1h_input_tokens": 900}}


def at(day: int, hour: float, month: int = 10) -> float:
    return time.mktime((2026, month, day, int(hour), int(hour % 1 * 60), 0, 0, 0, -1))


def session(started, size=300_000, **kw):
    s = {"path": "/v1/messages?beta=true", "sent": SENT,
         "headers": {"authorization": "Bearer SECRET", "User-Agent": UA}, "started": started,
         "size": size, "model": "claude-opus-5-5", "bumps": 0, "spent": 0.0, "last_bump": None,
         "misses": 0, "inflight": False}
    s.update(kw)
    return s


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved_cfg, self.saved_state = config.PATH, ka.STATE
        config.PATH = self.tmp / "config.json"
        config.load(force=True)
        ka.STATE = self.tmp / "keepalive.json"
        ka._CACHED.update(at=0.0, st={})
        for d in (ka._SESSIONS, ka._COUNTS, ka._WARM, ka._POLLED):
            d.clear()
        ka._CRED[0] = None
        ka._STARTED[0] = True                     # no background loop in tests
        ka.UPSTREAM[0] = "https://api.anthropic.com"
        self.rec = mock.patch.object(ka.vault, "record").start()

    def tearDown(self):
        mock.patch.stopall()
        for d in (ka._SESSIONS, ka._COUNTS, ka._WARM, ka._POLLED):
            d.clear()
        config.PATH, ka.STATE = self.saved_cfg, self.saved_state
        config.load(force=True)

    def on(self):
        config.set_value("cache_keepalive", "on", "user")

    def remember(self, sid="s", headers=None, usage=None, sent=SENT, model="claude-opus-5-5", t=None, rl=None):
        ka.remember(sid, "/v1/messages?beta=true", sent, headers or {"authorization": "Bearer SECRET",
                                                                       "User-Agent": UA},
                    t or time.time(), usage or USAGE, model, rl)


class DayEndTest(Base):
    def test_warmup_then_p75_minus_margin(self):
        ends = {ka._day(at(d, 16)): (ka._weekday(at(d, 16)), 16.0) for d in (5, 6, 7, 8)}
        self.assertIsNone(ka.stop_hour(ends, at(9, 12)), "4 past days: still learning")
        ends[ka._day(at(9, 18))] = (ka._weekday(at(9, 18)), 18.0)
        self.assertAlmostEqual(ka.stop_hour(ends, at(12, 12)), 16.0 - ka.MARGIN_H)

    def test_no_lookahead(self):
        ends = {ka._day(at(d, 16)): (ka._weekday(at(d, 16)), 16.0) for d in range(1, 9)}
        ends[ka._day(at(9, 23))] = (ka._weekday(at(9, 23)), 23.0)
        self.assertAlmostEqual(ka.stop_hour(ends, at(9, 12)), 15.0, msg="today is not used")

    def test_weekday_pattern_wins_with_enough_samples(self):
        ends = {ka._day(at(d, 20)): (ka._weekday(at(d, 20)), 20.0) for d in (5, 12, 19, 26)}   # Mondays
        ends.update({ka._day(at(d, 13)): (ka._weekday(at(d, 13)), 13.0) for d in (6, 7, 8, 13, 14, 15)})
        self.assertAlmostEqual(ka.stop_hour(ends, at(2, 12, month=11)), 19.0, msg="a Monday")
        self.assertAlmostEqual(ka.stop_hour(ends, at(3, 12, month=11)), 19.0, msg="a Tuesday: all days")

    def test_late_night_counts_toward_the_day(self):
        self.assertEqual(ka._day(at(9, 1)), ka._day(at(8, 22)))
        self.assertGreater(ka._hour(at(9, 1)), 24)


class DueTest(Base):
    def setUp(self):
        super().setUp()
        self.ends = {ka._day(at(d, 17)): (ka._weekday(at(d, 17)), 17.0) for d in range(1, 9)}

    def due(self, s, now, spent=0.0, cap=2.0):
        return ka.due(s, now, self.ends, spent, cap, ka.BUMP_AFTER_S)

    def test_timing_at_58_min_with_a_hard_stop(self):
        t = at(9, 12)
        self.assertEqual(self.due(session(t), t + 57 * 60), "not yet")
        self.assertIsNone(self.due(session(t), t + 58 * 60))
        self.assertIn("too late", self.due(session(t), t + 59.5 * 60), "inside the last minute: never")
        big = session(t, sent=b"x" * 30_000_000)              # 30 MB: 90 s to upload at the assumed speed
        self.assertIn("too late", self.due(big, t + 58.6 * 60), "the margin grows with the upload")

    def test_two_at_any_hour_more_only_before_day_end(self):
        t = at(9, 15.5)
        self.assertIsNone(self.due(session(t), t + 58 * 60))
        self.assertIsNone(self.due(session(t, bumps=1, last_bump=t + 58 * 60), t + 116 * 60))
        self.assertIn("end of the day", self.due(session(t, bumps=2, last_bump=t + 116 * 60), t + 174 * 60))
        e = at(9, 11)
        self.assertIsNone(self.due(session(e, bumps=2, last_bump=e + 116 * 60), e + 174 * 60), "13:54: before")
        self.ends = {}
        self.assertIn("learning", self.due(session(e, bumps=2, last_bump=e + 116 * 60), e + 174 * 60))

    def test_stops(self):
        self.assertIn("cap", self.due(session(at(9, 12)), at(9, 12) + 58 * 60, spent=2.0))
        self.assertIn("bumps this break", self.due(session(at(9, 9), bumps=ka.MAX_BUMPS), at(9, 10)))
        self.assertEqual(self.due(session(at(9, 12), dropped="x"), at(9, 13)), "x")
        self.assertIn("held too long", self.due(session(at(9, 0)), at(9, 13)))


class EligibleTest(Base):
    def why(self, **kw):
        args = dict(path="/v1/messages", sent=SENT, headers={"User-Agent": UA}, usage=USAGE,
                    model="claude-opus-5-5", rate_limits={})
        args.update(kw)
        return ka.eligible(**args)

    def test_only_interactive_claude_code(self):
        self.assertIsNone(self.why())
        self.assertIn("interactive", self.why(headers={"User-Agent": "claude-cli/2.1 (external, sdk-cli)"}))
        self.assertIn("interactive", self.why(headers={"User-Agent": "Python-urllib/3.12"}))
        self.assertIsNone(self.why(headers={"User-Agent": "claude-cli/2.1 (external, claude-vscode)"}))

    def test_only_anthropic_one_hour_caches_known_models(self):
        ka.UPSTREAM[0] = "https://gateway.example.com"
        self.assertIn("upstream", self.why())
        ka.UPSTREAM[0] = "https://api.anthropic.com"
        five = SENT.replace(b', "ttl": "1h"', b"", 1)
        self.assertIn("1-hour", self.why(sent=five), "a 5-minute marker: every bump would be a write")
        self.assertIn("unknown model", self.why(model="claude-new-thing-9"))
        self.assertIn("no 1-hour cache", self.why(usage={"cache_read_input_tokens": 0}))

    def test_rate_limit_headroom(self):
        rl = {"anthropic-ratelimit-requests-limit": "100", "anthropic-ratelimit-requests-remaining": "10"}
        self.assertIn("rate limit", self.why(rate_limits=rl))
        rl["anthropic-ratelimit-requests-remaining"] = "60"
        self.assertIsNone(self.why(rate_limits=rl))


class RememberTest(Base):
    def test_big_sessions_only(self):
        self.on()
        self.remember(usage={"cache_read_input_tokens": 50_000})
        self.assertNotIn("s", ka._SESSIONS, "a new 50K session is not big")
        self.remember()
        self.assertIn("s", ka._SESSIONS, "150K+ is big")
        self.remember("r", usage={"cache_read_input_tokens": 50_000}, t=time.time() - 2 * H)
        self.remember("r", usage={"cache_read_input_tokens": 50_000})
        self.assertIn("r", ka._SESSIONS, "returned to after a break: big")

    def test_a_different_login_drops_everything_held(self):
        self.on()
        self.remember("a")
        self.remember("b", headers={"authorization": "Bearer OTHER", "User-Agent": UA})
        self.assertEqual(ka._SESSIONS["a"]["headers"], {})
        self.assertIn("different login", ka._SESSIONS["a"]["dropped"])
        self.assertTrue(ka._SESSIONS["b"]["headers"])

    def test_return_settles_kept_and_resets_warm(self):
        self.on()
        ka._SESSIONS["s"] = session(at(9, 12), bumps=2, spent=0.12, last_bump=at(9, 13.9))
        ka._WARM["s"] = at(9, 13.9)
        self.remember(t=at(9, 14.5))
        b = ka.load_state()["breaks"][-1]
        self.assertTrue(b["kept"])
        self.assertAlmostEqual(b["saved"], 300_000 * (8 - 0.2) / 1e6, places=3)
        self.assertIsNone(ka.last_warm("s"), "a real request: the proxy's own clock takes over")

    def test_login_never_on_disk(self):
        self.on()
        self.remember(headers={"authorization": "Bearer SECRET-TOKEN", "User-Agent": UA})
        self.assertIn("s", ka._SESSIONS)
        ka.tick(is_open=lambda s: True)
        for p in self.tmp.rglob("*"):
            if p.is_file():
                self.assertNotIn("SECRET-TOKEN", p.read_text(errors="replace"), p.name)


class Stub(http.server.BaseHTTPRequestHandler):
    reply = {}
    status = 200
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        Stub.seen.append((dict(self.headers.items()), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        data = json.dumps(Stub.reply if Stub.status == 200 else {"error": {"message": "nope"}}).encode()
        self.send_response(Stub.status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class BumpTest(Base):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        super().setUp()
        ka.UPSTREAM[0] = f"http://127.0.0.1:{self.server.server_port}"
        Stub.seen, Stub.status = [], 200
        Stub.reply = {"usage": {"cache_read_input_tokens": 300_000, "cache_creation_input_tokens": 0,
                                "output_tokens": 0}}
        self.on()

    def due_session(self, sid="s", **kw):
        ka._SESSIONS[sid] = session(time.time() - 58.2 * 60, **kw)
        return ka._SESSIONS[sid]

    def tick(self, is_open=lambda s: True):
        with mock.patch.object(ka, "stop_hour", return_value=99.0):
            return ka.tick(is_open=is_open, wait=True)

    def test_sends_the_same_request_with_max_tokens_0(self):
        s = session(time.time() - 58 * 60)
        res = ka.bump("sess", s)
        hdrs, body = Stub.seen[0]
        sent = json.loads(s["sent"])
        self.assertEqual((body["max_tokens"], body["stream"]), (0, False))
        self.assertEqual({k: v for k, v in body.items() if k not in ("max_tokens", "stream")},
                         {k: v for k, v in sent.items() if k not in ("max_tokens", "stream")})
        self.assertEqual(hdrs.get("authorization") or hdrs.get("Authorization"), "Bearer SECRET")
        self.assertTrue(res["hit"])

    def test_a_hit_makes_the_return_warm(self):
        self.due_session()
        self.tick()
        self.assertEqual(ka._SESSIONS["s"]["bumps"], 1)
        self.assertIsNotNone(ka.last_warm("s"), "the proxy must treat the return as warm")

    def test_on_time_miss_backs_off_then_a_second_turns_it_off(self):
        Stub.reply = {"usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 300_000}}
        self.due_session("a")
        self.tick()
        self.assertIn("already gone", ka._SESSIONS["a"]["dropped"])
        self.assertEqual(ka._SESSIONS["a"]["headers"], {})
        self.assertTrue(ka.load_state().get("backoff"))
        self.assertEqual(ka.bump_after(ka.load_state()), ka.BACKOFF_AFTER_S)
        st = ka.load_state()
        self.assertGreater(st["today"]["spent"], 2.0, "a miss costs a full write ...")
        ka._update_state(today={})            # ... which the daily cap would also block; a new day
        config.set_value("cache_keepalive", "auto", "user")
        ka._SESSIONS["b"] = session(time.time() - 55.2 * 60)
        with mock.patch.object(ka, "VERIFIED", True):
            ka._CACHED.update(at=0.0)
            self.tick()
        self.assertIn("twice", ka.load_state()["auto_off"])

    def test_output_tokens_turn_it_off(self):
        Stub.reply = {"usage": {"cache_read_input_tokens": 300_000, "output_tokens": 12}}
        self.due_session()
        self.tick()
        self.assertIn("generated output", ka.load_state()["auto_off"])

    def test_refusals_and_limits_stop_without_retrying(self):
        for status, expect in ((401, "login refused"), (429, "stopped for this break"), (529, "stopped")):
            Stub.status, Stub.seen = status, []
            self.due_session(f"s{status}")
            self.tick()
            self.tick()
            self.assertIn(expect, ka._SESSIONS[f"s{status}"]["dropped"])
            self.assertEqual(len(Stub.seen), 1, f"{status}: never retried")
        Stub.status = 400
        self.due_session("x1")
        self.tick()
        self.assertNotIn("auto_off", ka.load_state(), "one refused body stops that session only")
        self.due_session("x2")
        self.tick()
        self.assertIn("refused", ka.load_state()["auto_off"])

    def test_closed_sessions_are_not_bumped(self):
        self.due_session()
        self.tick(is_open=lambda s: False)
        self.assertEqual(Stub.seen, [])
        self.assertEqual(ka._SESSIONS["s"]["headers"], {})

    def test_open_comes_from_the_status_line(self):
        ka.polled("s")
        self.assertTrue(ka._open("s"))
        ka._POLLED["s"] = time.time() - ka.OPEN_POLL_S - 1
        self.assertFalse(ka._open("s"), "no poll for 2 minutes: the session was closed")

    def test_many_sessions_due_together_all_go_out(self):
        for i in range(6):
            self.due_session(f"p{i}")
        self.tick()
        self.assertEqual(len(Stub.seen), 6)


class SettleTest(Base):
    def test_auto_waits_for_verification(self):
        st = {"ends": {f"2026-10-0{d}": [d % 7, 17.0] for d in range(1, 9)}}
        with mock.patch.object(ka, "VERIFIED", False):
            self.assertFalse(ka.enabled(st)[0])
        with mock.patch.object(ka, "VERIFIED", True):
            self.assertTrue(ka.enabled(st)[0])
            on, why = ka.enabled({"ends": {}})
            self.assertTrue(on, "no history needed for the first two bumps")
            self.assertIn("learns your hours", why)

    def test_nobody_came_back_is_wasted_and_auto_off(self):
        mock.patch.object(ka, "VERIFIED", True).start()
        config.set_value("cache_keepalive", "auto", "user")
        st = {"breaks": [{"t": time.time(), "session": "x", "bumps": 1, "spent": 0.08, "saved": 0.0,
                          "kept": False, "misses": 0, "size": 1} for _ in range(9)]}
        ka._save_state(st)
        ka._SESSIONS["s"] = session(time.time() - 3 * H, bumps=1, spent=0.08, last_bump=time.time() - H - 60)
        ka.tick(is_open=lambda s: True)
        self.assertNotIn("s", ka._SESSIONS)
        st = ka.load_state()
        self.assertFalse(st["breaks"][-1]["kept"])
        self.assertIn("spent", st["auto_off"])
        self.assertFalse(ka.enabled(st)[0])
        config.set_value("cache_keepalive", "on", "user")
        self.assertTrue(ka.enabled(st)[0], "`on` overrides the automatic stop")

    def test_off_holds_nothing(self):
        self.on()
        self.remember()
        config.set_value("cache_keepalive", "off", "user")
        ka.tick(is_open=lambda s: True)
        self.assertEqual(ka._SESSIONS, {})


if __name__ == "__main__":
    unittest.main()
