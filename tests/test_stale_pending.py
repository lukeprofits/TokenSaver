"""A prepared compaction the history no longer matches is dropped (it used to block every
later compaction of that session), and one that still matches is kept."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import unittest
from unittest import mock

from tokensaver import background, proxy


def history(n, tag="a"):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": [{"type": "text", "text": f"{tag} step {i}"}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"{tag} done {i}"}]}]
    return out


def pending_for(msgs, cut):
    return {"cut": cut, "prefix_fp": background.fingerprint(msgs[:cut]), "total_tokens": 10 ** 6, "text": "x"}


class StalePendingTest(unittest.TestCase):
    def setUp(self):
        self.rec = mock.patch.object(proxy.vault, "record").start()

    def tearDown(self):
        mock.patch.stopall()

    def test_normal_growth_keeps_it_without_hashing(self):
        msgs = history(10)
        st = {"pending": pending_for(msgs, 8)}
        self.assertFalse(proxy._drop_stale_pending(st, msgs))      # first sight: full check, matches
        real = background.fingerprint

        def small_only(x):
            assert len(x) <= 2, "hashed the whole prefix on normal growth"
            return real(x)
        with mock.patch.object(background, "fingerprint", side_effect=small_only):
            self.assertFalse(proxy._drop_stale_pending(st, msgs + history(1, "b")))
        self.assertIsNotNone(st["pending"])

    def test_compact_drops_it(self):
        msgs = history(10)
        st = {"pending": pending_for(msgs, 8)}
        proxy._drop_stale_pending(st, msgs)
        rebuilt = history(2, "summary")                         # Claude Code's /compact: a new, short history
        self.assertTrue(proxy._drop_stale_pending(st, rebuilt))
        self.assertIsNone(st["pending"])
        self.rec.assert_called_once()

    def test_rewind_that_still_matches_is_kept_and_rebased(self):
        msgs = history(10)
        st = {"pending": pending_for(msgs, 6)}
        proxy._drop_stale_pending(st, msgs)
        rewound = msgs[:14]                                    # Esc Esc back a few turns, prefix intact
        self.assertFalse(proxy._drop_stale_pending(st, rewound))
        self.assertLess(st["pending"]["total_tokens"], 10 ** 6, "rebased to the current size")

    def test_rewind_past_the_cut_drops_it(self):
        msgs = history(10)
        st = {"pending": pending_for(msgs, 16)}
        proxy._drop_stale_pending(st, msgs)
        self.assertTrue(proxy._drop_stale_pending(st, msgs[:10]))

    def test_same_length_but_edited_start_is_checked(self):
        msgs = history(10)
        st = {"pending": pending_for(msgs, 8)}
        proxy._drop_stale_pending(st, msgs)
        edited = history(11, "c")
        self.assertTrue(proxy._drop_stale_pending(st, edited))


class KeptWarmReturnTest(unittest.TestCase):
    """A return after a keep-alive bump read the cache is warm: the proxy must not commit
    a compaction as if the cache had died ("cache expired"), which would pay the write
    the bump was there to avoid."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        from tests.test_rules_only import history
        from tokensaver import keepalive
        self.keepalive = keepalive
        self.dir = Path(tempfile.mkdtemp())
        self.msgs = history()
        pend = background.prepare(self.msgs[:-1])
        self.assertIsNotNone(pend)
        self.pend = dict(pend, made=1.0, total_tokens=10 ** 6)
        mock.patch.object(proxy, "_bg_path", lambda s: self.dir / f"{s}.bg.json").start()
        mock.patch.object(proxy.vault, "record").start()
        mock.patch.object(proxy, "BG_COMPACT", True).start()
        mock.patch.object(proxy, "_BG_STARTED", True).start()
        mock.patch.object(proxy.audit, "enabled", return_value=False).start()
        proxy._LIVE.pop("w", None)
        keepalive._WARM.clear()

    def tearDown(self):
        mock.patch.stopall()
        self.keepalive._WARM.clear()
        proxy._LIVE.pop("w", None)

    def run_return(self, warm_minutes_ago):
        import time
        now = time.time()
        proxy._bg_save("w", {"pending": dict(self.pend), "last": now - 100 * 60,
                             "n_msgs": len(self.msgs), "bg_start": proxy._hist_fp(self.msgs)})
        if warm_minutes_ago is not None:
            self.keepalive._WARM["w"] = now - warm_minutes_ago * 60
        return proxy._bg(self.msgs, "w")[1]

    def test_without_a_bump_the_cold_return_compacts(self):
        self.assertTrue(self.run_return(None).get("bg_applied"))

    def test_after_a_bump_it_is_warm(self):
        info = self.run_return(42)
        self.assertFalse(info.get("bg_applied"), "the bump kept the cache: no cold-cache compaction")
        self.assertLess(info["gap_s"], 50 * 60, "the gap is measured from the bump")


class BumpValueTest(unittest.TestCase):
    """keepalive asks the proxy what a warm return is worth before each bump."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.dir = Path(tempfile.mkdtemp())
        mock.patch.object(proxy, "_bg_path", lambda s: self.dir / f"{s}.bg.json").start()

    def tearDown(self):
        mock.patch.stopall()

    def value(self, pending):
        proxy._bg_save("v", {"pending": pending} if pending else {})
        return proxy.keepalive_value("v", 400_000, "claude-opus-5-5")

    def test_no_compaction_prepared_is_the_whole_rewrite(self):
        self.assertAlmostEqual(self.value(None), 400_000 * (8 - 0.2) / 1e6, places=4)

    def test_a_compaction_the_warm_return_applies_anyway_is_worth_nothing(self):
        with mock.patch.object(proxy.config, "get", side_effect=lambda k: 70.0 if k == "bg_warm_payback_max"
                               else proxy.config.SPECS[k].default):
            self.assertEqual(self.value({"gate": {"passed": True}, "payback_requests": 20}), 0.0)

    def test_a_cold_only_compaction_counts_its_smaller_rewrite_and_lost_reads(self):
        got = self.value({"gate": {"passed": True}, "payback_requests": 500, "sent_new": 250_000})
        n = proxy.config.get("bg_warm_payback_max")
        self.assertAlmostEqual(got, max(0.0, (250_000 * 8 - 400_000 * 0.2 - 150_000 * 0.2 * n) / 1e6), places=4)

    def test_keepalive_skips_a_worthless_bump(self):
        from tokensaver import keepalive as ka
        import time
        ka._SESSIONS.clear()
        ka._SESSIONS["v"] = {"path": "/v1/messages", "sent": b"{}", "headers": {"a": "b"}, "size": 400_000,
                             "model": "claude-opus-5-5", "bumps": 0, "spent": 0.0, "last_bump": None,
                             "misses": 0, "inflight": False, "started": time.time() - 58.2 * 60}
        sent = []
        with mock.patch.object(ka, "VALUE", lambda *a: 0.0), \
                mock.patch.object(ka, "enabled", return_value=(True, "on")):
            ka.tick(send=lambda k, v: sent.append(k) or {"status": 200}, is_open=lambda s: True, wait=True)
        self.assertEqual(sent, [])
        self.assertIn("trim will run", ka._SESSIONS["v"]["dropped"])
        ka._SESSIONS.clear()


if __name__ == "__main__":
    unittest.main()
