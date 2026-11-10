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

    def test_shared_prefix(self):
        stamp = lambda m: _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(1_800_000_000 + m * 60))
        r = lambda m, read, ctx: NS(timestamp=stamp(m), read=read, context=ctx, write_5m=0, write_1h=ctx - read)
        reqs = [r(0, 24_000, 80_000), r(1, 80_000, 82_000), r(100, 25_000, 90_000), r(101, 90_000, 91_000)]
        self.assertEqual(cachettl.shared_prefix(reqs), 25_000, "median read at the first request and the 99-min gap")
        resumed = [r(0, 70_000, 80_000), r(1, 80_000, 81_000)]
        self.assertEqual(cachettl.shared_prefix(resumed), 0, "a first request reading most of it is a resume, not shared")
        p = NS(read=0.2, write_1h=8.0, write_5m=5.0, out=0)
        for x in reqs:
            x.out = 0
        cost = cachettl.price_history(reqs, p, "1h")
        want = (24_000 * 0.2 + 56_000 * 8 + 80_000 * 0.2 + 2_000 * 8
                + 25_000 * 0.2 + 65_000 * 8 + 90_000 * 0.2 + 1_000 * 8) / 1e6
        self.assertAlmostEqual(cost, want, places=6)

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


def _history(root: Path, sessions: int, n: int, gap_min: float, ctx: int, ttl: str = "5m"):
    """`sessions` transcripts of `n` requests, context growing 2K per request, `gap_min` apart."""
    key = "ephemeral_1h_input_tokens" if ttl == "1h" else "ephemeral_5m_input_tokens"
    t = _t.mktime((2026, 10, 20, 9, 0, 0, 0, 0, -1))
    for s in range(sessions):
        rows = []
        for i in range(n):
            t += gap_min * 60
            stamp = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))
            size = ctx + 2000 * i
            rows.append({"type": "user", "uuid": f"u{s}-{i}", "timestamp": stamp,
                         "message": {"role": "user", "content": f"step {s} {i}"}})
            rows.append({"type": "assistant", "uuid": f"a{s}-{i}", "timestamp": stamp, "message": {
                "id": f"m{s}-{i}", "role": "assistant", "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 2, "cache_read_input_tokens": size - 2002 if i else 0,
                          "output_tokens": 10, "cache_creation_input_tokens": 2000 if i else size - 2,
                          "cache_creation": {key: 2000 if i else size - 2}}}})
        d = root / "proj"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"s{s}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return t


class ChooseTest(unittest.TestCase):
    def choose(self, *a, **k):
        root = Path(tempfile.mkdtemp())
        end = _history(root, *a, **k)
        return cachettl.choose(root=root, now=end + 86400)

    def test_bursty_short_sessions_leave_claude_codes_default(self):
        d = self.choose(6, 15, 1, 50_000)
        self.assertFalse(d["set"], d["why"])
        self.assertLess(d["cost_5m"], d["cost_1h"])

    def test_pauses_on_big_contexts_set_1h(self):
        d = self.choose(6, 15, 20, 200_000, ttl="1h")         # as run on 1h: the reads are real
        self.assertTrue(d["set"], d["why"])
        self.assertLess(d["cost_1h"], d["cost_5m"])

    def test_thin_history_defaults_to_1h(self):
        d = self.choose(2, 15, 1, 50_000)
        self.assertTrue(d["set"])
        self.assertIn("only 2 sessions", d["why"])

    def test_an_unreliable_model_falls_back_to_the_default(self):
        # 20-minute pauses that still read a 5-minute cache: impossible, so the model is no guide here
        d = self.choose(6, 15, 20, 200_000, ttl="5m")
        self.assertTrue(d["thin"] and d["set"], d)
        self.assertIn("cache model", d["why"])

    def test_model_accuracy_against_real_reads(self):
        root = Path(tempfile.mkdtemp())
        end = _history(root, 6, 15, 1, 50_000)            # every read is real and the model agrees
        self.assertEqual(cachettl.choose(root=root, now=end + 86400)["accuracy"], 1.0)


class SettingTest(unittest.TestCase):
    def setUp(self):
        from tokensaver import claude_settings
        self.path = claude_settings.SETTINGS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for p in (self.path, cachettl._state_path()):
            self.addCleanup(lambda p=p: p.exists() and p.unlink())
        self.decision = {"set": True, "cost_1h": 1.0, "cost_5m": 2.0, "why": "test"}

    def write(self, data):
        self.path.write_text(json.dumps(data) if not isinstance(data, str) else data)

    def read(self):
        return json.loads(self.path.read_text())

    def test_apply_keeps_other_keys_and_remove_takes_only_ours(self):
        self.write({"model": "opus", "env": {"A": "1"}})
        cachettl.apply(self.decision, backup=False)
        self.assertEqual(self.read(), {"model": "opus", "env": {"A": "1"}, "promptCacheTtl": "1h"})
        self.assertEqual(cachettl.current(), ("1h", True))
        cachettl.remove(backup=False)
        self.assertEqual(self.read(), {"model": "opus", "env": {"A": "1"}})

    def test_a_users_own_value_is_never_touched(self):
        self.write({"promptCacheTtl": "5m"})
        self.assertIn("left alone", cachettl.apply(self.decision, backup=False))
        self.assertEqual(self.read(), {"promptCacheTtl": "5m"})
        self.assertIsNone(cachettl.remove(backup=False))
        self.assertEqual(self.read(), {"promptCacheTtl": "5m"})

    def test_a_value_changed_since_stays(self):
        self.write({})
        cachettl.apply(self.decision, backup=False)
        self.write({"promptCacheTtl": "5m"})                 # the user changed it afterwards
        cachettl.remove(backup=False)
        self.assertEqual(self.read(), {"promptCacheTtl": "5m"})

    def test_unparseable_file_untouched(self):
        self.write("{not json")
        with self.assertRaises(RuntimeError):
            cachettl.apply(self.decision, backup=False)
        self.assertEqual(self.path.read_text(), "{not json")

    def test_recheck_flips_only_past_the_band_twice_running(self):
        self.write({})
        cachettl.apply(self.decision, backup=False)
        now = [2_000_000_000.0]
        cheap5 = {"set": False, "cost_1h": 100.0, "cost_5m": 85.0, "sessions": 9, "accuracy": 0.99,
                  "margin": 0.01, "switch_1h": 1.0, "switch_5m": 1.0, "thin": False, "why": "x"}
        close = dict(cheap5, cost_5m=99.5)                  # inside the model's 1% margin
        for d, removed in ((close, False), (cheap5, False), (close, False), (cheap5, False), (cheap5, True)):
            now[0] += cachettl.RECHECK_S + 1
            with mock.patch.object(cachettl, "choose", lambda **k: d):
                cachettl.recheck(now=now[0])
            self.assertEqual("promptCacheTtl" not in self.read(), removed, (d, removed))

    def test_a_small_saving_does_not_pay_for_the_switch(self):
        d = {"cost_1h": 100.0, "cost_5m": 98.0, "margin": 0.01, "switch_5m": 1.0}
        self.assertFalse(cachettl.worth_switching(d, "5m"), "2% is past the margin, but a week of it "
                                                           "($0.47) does not pay one rewrite ($1)")
        self.assertTrue(cachettl.worth_switching(dict(d, cost_5m=80.0), "5m"))
        self.assertFalse(cachettl.worth_switching(dict(d, cost_5m=95.0, margin=0.10), "5m"),
                         "a model that misses 10% of reads needs a 10% margin")

    def test_margin_follows_the_models_accuracy(self):
        self.assertEqual(cachettl.margin(0.996), cachettl.MIN_MARGIN)
        self.assertAlmostEqual(cachettl.margin(0.85), 0.15)
        self.assertEqual(cachettl.margin(None), 0.10)

    def test_thin_history_is_rechecked_daily(self):
        self.write({})
        thin = {"set": True, "cost_1h": 1.0, "cost_5m": 1.0, "sessions": 1, "accuracy": None, "margin": 0.1,
                "switch_1h": 0.0, "switch_5m": 0.0, "thin": True, "why": "thin"}
        cachettl.save_state({"decision": thin, "checked_at": 1_000_000.0})
        calls = []
        with mock.patch.object(cachettl, "choose", lambda **k: calls.append(1) or thin):
            cachettl.recheck(now=1_000_000.0 + 86400 + 1)
        self.assertEqual(len(calls), 1, "a day later, not a week")

    def test_off_sticks(self):
        self.write({})
        cachettl.apply(self.decision, backup=False)
        with mock.patch.object(cachettl, "choose", lambda **k: self.decision):
            cachettl.set_mode("off")
            self.assertEqual(self.read(), {})
            self.assertIsNone(cachettl.recheck(now=_t.time() + 30 * 86400))
        self.assertEqual(self.read(), {})


class BenchmarkMainTtlTest(unittest.TestCase):
    def setUp(self):
        from tokensaver import keepalive as ka
        p = mock.patch.object(ka, "idle_compact_at", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_ours_priced_at_1h_plain_unchanged(self):
        d = _session("5m", {10: 20, 20: 40})                 # pauses a 5m cache does not survive
        base = benchmark.replay(d, COUNT, 8, 900, 70, 0.264)
        with1h = benchmark.replay(d, COUNT, 8, 900, 70, 0.264, main_ttl="1h")
        self.assertEqual(with1h["plain_cost"], base["plain_cost"], "plain keeps the lifetime it ran")
        self.assertLess(with1h["plain_cost_ours_ttl"], with1h["plain_cost"], "1h survives the pauses")
        self.assertLess(with1h["ours_cost"], base["ours_cost"])

    def test_keepalive_follows_ours(self):
        d = _session("5m", {15: 90})
        on = benchmark.replay(d, COUNT, 8, 900, 70, 0.264, keepalive_ends=_ends(), main_ttl="1h")
        self.assertGreater(on["keepalive_bumps"], 0, "with 1h set, keep-alive runs on this session")


class LiveFactsTest(unittest.TestCase):
    """A TokenSaver session's "as it ran" explains itself from what the proxy recorded."""

    def test_counts_from_the_vault(self):
        from tokensaver import vault
        sid = "live-facts-test-session"
        main = {"path": "/v1/messages", "request_class": "main", "code": "abc"}
        vault.record("api_request", content="{}", tool="proxy", session_id=sid, summary="r",
                     meta=dict(main, rewrite={"bytes_in": 100, "bytes_out": 90, "bg_applied": True}), index=False)
        vault.record("api_request", content="{}", tool="proxy", session_id=sid, summary="r",
                     meta=dict(main, rewrite={"bytes_in": 100, "bytes_out": 100,
                                              "bg_skipped": "payback 80 > 70 requests"}), index=False)
        vault.record("api_request", content="{}", tool="proxy", session_id=sid, summary="r",
                     meta=dict(main, rewrite={"fallback": "session"}), index=False)
        vault.record("bg_dropped_pending", tool="proxy", session_id=sid, summary="d", meta={}, index=False)
        vault.record("token_audit", tool="proxy", session_id=sid, summary="a",
                     meta={"tokens_original": 1000, "tokens_sent": 900}, index=False)
        f = benchmark.live_facts(sid)
        self.assertEqual((f["requests"], f["rewritten"], f["applied"], f["fallback"], f["dropped_pending"]),
                         (3, 1, 1, 1, 1))
        self.assertEqual(f["skipped"], {"payback 80 > 70 requests": 1})
        self.assertEqual((f["audit_original"], f["audit_sent"]), (1000, 900))
        self.assertEqual(f["codes"], [("abc", 3)])
        lines = benchmark.live_line([{"through_tokensaver": True, "live": f, "compactions": 2}])
        self.assertIn("1 compactions applied live vs 2 in today's replay", lines[0])
        self.assertTrue(any("10.0% fewer" in x for x in lines))
        self.assertEqual(benchmark.live_facts("no-such-session")["requests"], 0)


if __name__ == "__main__":
    unittest.main()
