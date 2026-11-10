"""Cache lifetime (5 minutes or 1 hour), read per request: `cachettl`, and everything that
used to assume 1 hour (the benchmark replay, sim, the proxy's cold test and fallback canary)."""

import json
import tempfile
import time as _t
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from tokensaver import benchmark, cachettl, pricing


def _cc(ttl=None):
    return {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}


class CacheTtlTest(unittest.TestCase):
    def test_body(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "a", "cache_control": _cc("1h")}]}]
        self.assertEqual(cachettl.of_request({"messages": msgs}), "1h")
        msgs[0]["content"][0]["cache_control"] = _cc()
        self.assertEqual(cachettl.of_request({"messages": msgs}), "5m", "no ttl field is 5 minutes")
        self.assertIsNone(cachettl.of_request({"messages": [{"role": "user", "content": "a"}]}))
        sys_only = {"system": [{"type": "text", "text": "s", "cache_control": _cc("1h")}],
                    "messages": [{"role": "user", "content": "a"}]}
        self.assertEqual(cachettl.of_request(sys_only), "1h")
        mixed = dict(sys_only, messages=[{"role": "user", "content": [
            {"type": "text", "text": "a", "cache_control": _cc()}]}])
        self.assertEqual(cachettl.of_request(mixed), "5m", "the last message marker decides")

    def test_usage(self):
        r = lambda w5, w1: NS(write_5m=w5, write_1h=w1)
        self.assertEqual(cachettl.of_usage(r(0, 500)), "1h")
        self.assertEqual(cachettl.of_usage(r(500, 0)), "5m")
        self.assertEqual(cachettl.of_usage(r(1944, 165030)), "1h", "both: where most of it went")
        self.assertIsNone(cachettl.of_usage(r(0, 0)))
        self.assertEqual(cachettl.per_request([r(0, 0), r(9, 0), r(0, 0), r(0, 7), r(0, 0)]),
                         ["5m", "5m", "5m", "1h", "1h"], "carried forward; the start takes the first seen")
        self.assertEqual(cachettl.per_request([r(0, 0)]), ["1h"], "nothing reported: the default")

    def test_prices(self):
        p = pricing.price_for("claude-sonnet-5-5")
        self.assertEqual(cachettl.write_price(p, "5m"), p.write_5m)
        self.assertEqual(cachettl.write_price(p, "1h"), p.write_1h)
        self.assertEqual(cachettl.write_price(p, None), p.write_1h)
        self.assertEqual((cachettl.seconds("5m"), cachettl.seconds("1h"), cachettl.seconds(None)), (300, 3600, 3600))


def _session(ttl: str, gaps_min, n=30, model="claude-sonnet-5-5"):
    """A transcript whose requests read ~200K and write a little at `ttl`."""
    d = Path(tempfile.mkdtemp())
    key = "ephemeral_1h_input_tokens" if ttl == "1h" else "ephemeral_5m_input_tokens"
    t, rows = _t.mktime((2026, 10, 20, 9, 0, 0, 0, 0, -1)), []
    for i in range(n):
        t += gaps_min.get(i, 1) * 60
        stamp = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))
        rows.append({"type": "user", "timestamp": stamp, "message": {"role": "user", "content": f"step {i}"}})
        rows.append({"type": "assistant", "timestamp": stamp, "message": {
            "id": f"m{i}", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": f"done {i}"}],
            "usage": {"input_tokens": 2, "cache_read_input_tokens": 200_000, "output_tokens": 10,
                      "cache_creation_input_tokens": 100, "cache_creation": {key: 100}}}})
    path = d / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return NS(session=NS(path=path, primary_model=model), price=pricing.price_for(model), stem="s")


def _ends():
    from tokensaver import keepalive as ka
    ends = {}
    for day in range(10, 20):
        e = _t.mktime((2026, 10, day, 18, 0, 0, 0, 0, -1))
        ends[ka._day(e)] = (ka._weekday(e), ka._hour(e))
    return ends


COUNT = lambda model, msgs, **k: 1000 + len(json.dumps(msgs)) // 4


class BenchmarkTtlTest(unittest.TestCase):
    def setUp(self):
        from tokensaver import keepalive as ka
        p = mock.patch.object(ka, "idle_compact_at", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def replay(self, d, **k):
        return benchmark.replay(d, COUNT, 8, 900, 70, 0.264, **k)

    def test_a_10_minute_pause_misses_a_5_minute_cache_only(self):
        for ttl, misses in (("5m", True), ("1h", False)):
            steady = self.replay(_session(ttl, {}))["plain_cost"]
            paused = self.replay(_session(ttl, {15: 10}))["plain_cost"]
            if misses:
                self.assertGreater(paused, steady, ttl)
            else:
                self.assertEqual(paused, steady, ttl)

    def test_writes_priced_at_their_own_lifetime(self):
        short, long_ = self.replay(_session("5m", {})), self.replay(_session("1h", {}))
        self.assertLess(short["plain_cost"], long_["plain_cost"], "a 5m write costs less than a 1h one")
        self.assertEqual((short["write_5m_tokens"], short["write_1h_tokens"]), (3000, 0))

    def test_keepalive_only_on_a_1_hour_cache(self):
        on = self.replay(_session("1h", {15: 90}), keepalive_ends=_ends())
        self.assertGreater(on["keepalive_bumps"], 0)
        off = self.replay(_session("5m", {15: 90}), keepalive_ends=_ends())
        self.assertEqual((off["keepalive_bumps"], off["keepalive_kept"], off["keepalive_cost"]), (0, 0, 0))

    def test_summary_line(self):
        self.assertEqual(benchmark.cache_lifetime_line([{"write_5m_tokens": 0, "write_1h_tokens": 9}]),
                         " cache lifetime: 1 hour")
        line = benchmark.cache_lifetime_line([{"write_5m_tokens": 9, "write_1h_tokens": 0},
                                              {"write_5m_tokens": 0, "write_1h_tokens": 9}])
        self.assertIn("5 minutes on 1 of 2 sessions", line)


class SimTtlTest(unittest.TestCase):
    """A 20-minute break is warm on a 1h cache (payback decides) and cold on a 5m one."""

    def data(self, ttl):
        from tokensaver import sim
        t, rq = 1_800_000_000, []
        for g in [0, 1, 1, 20, 1, 1]:
            t += g * 60
            stamp = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))
            w5, w1 = (995, 0) if ttl == "5m" else (0, 995)
            rq.append(NS(timestamp=stamp, context=400_000, inp=5, read=399_000, write=995, write_5m=w5,
                         write_1h=w1, out=500))
        price = NS(inp=4.0, out=20.0, write_5m=5.0, write_1h=8.0, read=0.2)
        saved = sim._charge_rereads
        self.addCleanup(setattr, sim, "_charge_rereads", saved)
        sim._charge_rereads = lambda *a, **k: 0
        return sim, NS(price=price, requests=rq, stem="s", session=NS(floor=50_000), blocks=[])

    def test_cold_only_on_5m(self):
        for ttl, apps in (("1h", 0), ("5m", 1)):
            sim, d = self.data(ttl)
            self.assertEqual(sim.price_session(d, 0.6, 15 * 60, 100_000, 120_000, 1.0, payback_max=0).applications,
                             apps, ttl)


class ProxyTtlTest(unittest.TestCase):
    def setUp(self):
        from tokensaver import proxy
        self.proxy = proxy
        self.addCleanup(proxy._TTL.clear)
        self.addCleanup(proxy._WATCH.clear)
        self.addCleanup(proxy._FELL_BACK.clear)

    def body(self, ttl):
        return {"messages": [{"role": "user", "content": [{"type": "text", "text": "a", "cache_control": _cc(ttl)}]}]}

    def test_expired_follows_the_previous_request(self):
        p = self.proxy
        p._note_ttl("s1", self.body(None))            # 5m
        self.assertEqual(p._expired_s("s1"), 3660, "the first request: the previous one is unknown")
        self.assertEqual(p._expired_s("s1", next_request=True), 360)
        p._note_ttl("s1", self.body("1h"))            # a subscriber leaving extra usage
        self.assertEqual(p._expired_s("s1"), 360, "this gap ends a 5m entry")
        self.assertEqual(p._expired_s("s1", next_request=True), 3660)

    def watch(self, session, ttl, gaps_s):
        p, now = self.proxy, [1_900_000_000.0]
        with mock.patch.object(p.time, "time", lambda: now[0]), \
                mock.patch.object(p.vault, "record", lambda *a, **k: None), \
                mock.patch.object(p.config, "section", lambda *a: {}), \
                mock.patch.object(p.config, "set_section", lambda *a: None):
            for g in gaps_s:
                now[0] += g
                p._note_ttl(session, self.body(ttl))
                p._watch(session, "main", "/v1/messages", {"cache_read_input_tokens": 0}, {"bytes_in": 1})
        return session in p._FELL_BACK

    def test_5m_returns_do_not_trip_the_fallback(self):
        self.assertFalse(self.watch("s5", None, [0, 600, 600, 600]),
                         "three 10-minute pauses on a 5m cache read nothing, as expected")
        self.assertTrue(self.watch("s1h", "1h", [0, 600, 600, 600]),
                        "negative control: on a 1h cache the same reads are misses")


if __name__ == "__main__":
    unittest.main()
