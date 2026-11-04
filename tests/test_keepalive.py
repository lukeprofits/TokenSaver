"""Cache keep-alive: the learned day end (no lookahead), when a bump is due, what a
bump sends, and how results settle and turn it off. No real requests: a stub
upstream on 127.0.0.1, fake times."""

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


def at(day: int, hour: float, month: int = 10) -> float:
    """A local time: 2026-<month>-<day> at <hour> (fractional)."""
    return time.mktime((2026, month, day, int(hour), int(hour % 1 * 60), 0, 0, 0, -1))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved_cfg, self.saved_state = config.PATH, ka.STATE
        config.PATH = self.tmp / "config.json"
        config.load(force=True)
        ka.STATE = self.tmp / "keepalive.json"
        ka._CACHED.update(at=0.0, st={})
        ka._SESSIONS.clear()
        self.rec = mock.patch.object(ka.vault, "record").start()
        mock.patch.object(ka, "learn", side_effect=lambda: ka.load_state()).start()

    def tearDown(self):
        mock.patch.stopall()
        ka._SESSIONS.clear()
        config.PATH, ka.STATE = self.saved_cfg, self.saved_state
        config.load(force=True)

    def workdays(self, ends: dict):
        """Pretend these days ended at these hours (learned state)."""
        st = ka.load_state()
        st["ends"] = {ka._day(at(d, h)): [ka._weekday(at(d, h)), ka._hour(at(d, h))] for d, h in ends.items()}
        st["learned_at"] = time.time() + 10 * 86400       # never re-learn during a test
        ka._save_state(st)


class DayEndTest(Base):
    def test_warmup_then_p75_minus_margin(self):
        ends = {}
        for d in (5, 6, 7, 8):                            # Mon-Thu 2026-10-05..08
            ends[ka._day(at(d, 16))] = (ka._weekday(at(d, 16)), 16.0)
        self.assertIsNone(ka.stop_hour(ends, at(9, 12)), "4 past days: still learning")
        ends[ka._day(at(9, 18))] = (ka._weekday(at(9, 18)), 18.0)
        self.assertAlmostEqual(ka.stop_hour(ends, at(12, 12)), 16.0 - ka.MARGIN_H)

    def test_no_lookahead(self):
        ends = {ka._day(at(d, 16)): (ka._weekday(at(d, 16)), 16.0) for d in range(1, 9)}
        ends[ka._day(at(9, 23))] = (ka._weekday(at(9, 23)), 23.0)     # today's own late finish
        self.assertAlmostEqual(ka.stop_hour(ends, at(9, 12)), 15.0, msg="today is not used")

    def test_weekday_pattern_wins_with_enough_samples(self):
        ends = {}
        days = [5, 12, 19, 26]                            # Mondays in October 2026: long days
        for d in days:
            ends[ka._day(at(d, 20))] = (ka._weekday(at(d, 20)), 20.0)
        for d in (6, 7, 8, 13, 14, 15):                   # other days end early
            ends[ka._day(at(d, 13))] = (ka._weekday(at(d, 13)), 13.0)
        self.assertAlmostEqual(ka.stop_hour(ends, at(2, 12, month=11)), 19.0, msg="a Monday: Mondays run late")
        self.assertAlmostEqual(ka.stop_hour(ends, at(3, 12, month=11)), 19.0,
                               msg="a Tuesday: too few Tuesdays, all days' p75")

    def test_late_night_counts_toward_the_day(self):
        self.assertEqual(ka._day(at(9, 1)), ka._day(at(8, 22)), "1 am belongs to the day before")
        self.assertGreater(ka._hour(at(9, 1)), 24)


def session(started, size=300_000, **kw):
    s = {"path": "/v1/messages?beta=true", "sent": json.dumps({"model": "claude-opus-5-5", "max_tokens": 128000,
         "stream": True, "thinking": {"type": "adaptive"}, "messages": [{"role": "user", "content": "x"}]}).encode(),
         "headers": {"authorization": "Bearer SECRET", "anthropic-beta": "a,b"}, "started": started,
         "size": size, "model": "claude-opus-5-5", "bumps": 0, "spent": 0.0, "last_bump": None, "misses": 0}
    s.update(kw)
    return s


class DueTest(Base):
    def setUp(self):
        super().setUp()
        self.ends = {ka._day(at(d, 17)): (ka._weekday(at(d, 17)), 17.0) for d in range(1, 9)}

    def due(self, s, now, spent=0.0, cap=2.0):
        return ka.due(s, now, self.ends, spent, cap)

    def test_timing(self):
        t = at(9, 12)
        self.assertEqual(self.due(session(t), t + 49 * 60), "not yet")
        self.assertIsNone(self.due(session(t), t + 50 * 60))
        self.assertIn("too late", self.due(session(t), t + 59 * 60))
        self.assertIsNone(self.due(session(t, bumps=1, last_bump=t + 50 * 60), t + 100 * 60), "every 50 min")

    def test_stops(self):
        t = at(9, 15.5)
        self.assertIn("end of the day", self.due(session(t), t + 50 * 60), "16:20 is after 16:00 (p75 17 - 1 h)")
        self.assertIsNone(self.due(session(at(9, 14.5)), at(9, 14.5) + 50 * 60), "15:20 is before")
        self.assertIn("cap", self.due(session(at(9, 12)), at(9, 12) + 50 * 60, spent=2.0))
        self.assertIn("bumps this break", self.due(session(at(9, 9), bumps=ka.MAX_BUMPS), at(9, 10)))
        self.assertEqual(self.due(session(at(9, 12), dropped="x"), at(9, 13)), "x")


class Stub(http.server.BaseHTTPRequestHandler):
    reply = {"usage": {"cache_read_input_tokens": 300_000, "cache_creation_input_tokens": 0}}
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
        Stub.reply = {"usage": {"cache_read_input_tokens": 300_000, "cache_creation_input_tokens": 0}}

    def test_sends_the_same_request_with_max_tokens_0(self):
        s = session(at(9, 12))
        res = ka.bump("sess", s)
        hdrs, body = Stub.seen[0]
        sent = json.loads(s["sent"])
        self.assertEqual(body["max_tokens"], 0)
        self.assertFalse(body["stream"])
        self.assertEqual({k: v for k, v in body.items() if k not in ("max_tokens", "stream")},
                         {k: v for k, v in sent.items() if k not in ("max_tokens", "stream")},
                         "everything else exactly as sent: same cache key")
        self.assertEqual(hdrs.get("authorization") or hdrs.get("Authorization"), "Bearer SECRET")
        self.assertTrue(res["hit"])
        self.assertAlmostEqual(res["cost"], 300_000 * 0.20 / 1e6, places=4)

    def test_a_miss_stops_the_session_and_drops_the_login(self):
        Stub.reply = {"usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 300_000}}
        config.set_value("cache_keepalive", "on", "user")
        ka._SESSIONS["s"] = session(time.time() - 51 * 60)
        with mock.patch.object(ka, "stop_hour", return_value=99.0):
            ka.tick(running=lambda: True)
        s = ka._SESSIONS["s"]
        self.assertEqual((s["misses"], s["headers"]), (1, {}))
        self.assertIn("already gone", s["dropped"])

    def test_login_refused_and_bump_refused(self):
        config.set_value("cache_keepalive", "on", "user")
        with mock.patch.object(ka, "stop_hour", return_value=99.0):
            Stub.status = 401
            ka._SESSIONS["s"] = session(time.time() - 51 * 60)
            ka.tick(running=lambda: True)
            self.assertEqual(ka._SESSIONS["s"]["headers"], {})
            Stub.status = 400
            ka._SESSIONS["t"] = session(time.time() - 51 * 60)
            ka.tick(running=lambda: True)
        self.assertIn("refused", ka.load_state()["auto_off"])

    def test_no_claude_running_means_no_bump(self):
        config.set_value("cache_keepalive", "on", "user")
        ka._SESSIONS["s"] = session(time.time() - 51 * 60)
        with mock.patch.object(ka, "stop_hour", return_value=99.0):
            ka.tick(running=lambda: False)
        self.assertEqual(Stub.seen, [])
        self.assertEqual(ka._SESSIONS["s"]["headers"], {})


class SettleTest(Base):
    def test_return_kept_warm_is_saved(self):
        config.set_value("cache_keepalive", "on", "user")
        ka._SESSIONS["s"] = session(at(9, 12), bumps=2, spent=0.12, last_bump=at(9, 13.6))
        ka.remember("s", "/v1/messages", b"{}", {}, at(9, 14), {"cache_read_input_tokens": 300_000,
                                                                "cache_creation_input_tokens": 900}, "claude-opus-5-5")
        b = ka.load_state()["breaks"][-1]
        self.assertTrue(b["kept"])
        self.assertAlmostEqual(b["saved"], 300_000 * (8 - 0.2) / 1e6, places=3)

    def test_auto_waits_for_verification(self):
        st = {"ends": {f"2026-10-0{d}": [d % 7, 17.0] for d in range(1, 9)}}
        with mock.patch.object(ka, "VERIFIED", False):
            self.assertIn("verified", ka.enabled(st)[1])
            self.assertFalse(ka.enabled(st)[0])
        with mock.patch.object(ka, "VERIFIED", True):
            self.assertTrue(ka.enabled(st)[0])
            self.assertIn("learning", ka.enabled({"ends": {}})[1])

    def test_nobody_came_back_is_wasted_and_auto_off(self):
        mock.patch.object(ka, "VERIFIED", True).start()
        config.set_value("cache_keepalive", "auto", "user")
        st = ka.load_state()
        st["ends"] = {f"2026-10-0{d}": [d % 7, 17.0] for d in range(1, 9)}
        st["learned_at"] = time.time() + 86400
        st["breaks"] = [{"t": time.time(), "session": "x", "bumps": 1, "spent": 0.08, "saved": 0.0, "kept": False,
                         "misses": 0, "size": 1} for _ in range(9)]
        ka._save_state(st)
        ka._SESSIONS["s"] = session(time.time() - 3 * H, bumps=1, spent=0.08, last_bump=time.time() - H - 60)
        ka.tick(running=lambda: True)
        self.assertNotIn("s", ka._SESSIONS, "settled and forgotten")
        st = ka.load_state()
        self.assertFalse(st["breaks"][-1]["kept"])
        self.assertIn("spent", st["auto_off"])
        self.assertFalse(ka.enabled(st)[0])
        config.set_value("cache_keepalive", "on", "user")
        self.assertTrue(ka.enabled(st)[0], "`on` overrides the automatic stop")

    def test_small_sessions_and_off_hold_nothing(self):
        config.set_value("cache_keepalive", "on", "user")
        ka.remember("s", "/v1/messages", b"{}", {"authorization": "Bearer X"}, time.time(),
                    {"cache_read_input_tokens": 5_000}, "claude-opus-5-5")
        self.assertNotIn("s", ka._SESSIONS)
        ka.remember("s", "/v1/messages", b"{}", {"authorization": "Bearer X"}, time.time(),
                    {"cache_read_input_tokens": 50_000}, "claude-opus-5-5")
        self.assertIn("s", ka._SESSIONS)
        config.set_value("cache_keepalive", "off", "user")
        ka.tick(running=lambda: True)
        self.assertEqual(ka._SESSIONS, {}, "off: the held login goes at the next tick")

    def test_login_never_on_disk(self):
        config.set_value("cache_keepalive", "on", "user")
        ka.remember("s", "/v1/messages", b"{}", {"authorization": "Bearer SECRET-TOKEN"}, time.time(),
                    {"cache_read_input_tokens": 50_000}, "claude-opus-5-5")
        ka.tick(running=lambda: True)
        for p in self.tmp.rglob("*"):
            if p.is_file():
                self.assertNotIn("SECRET-TOKEN", p.read_text(errors="replace"), p.name)


if __name__ == "__main__":
    unittest.main()
