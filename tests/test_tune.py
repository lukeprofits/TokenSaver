"""tune.decide: each 'measuring is hard' rule falls back to the default."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import config, tune  # noqa: E402

OK = {"sessions": 50, "days": 30, "moments": 40, "sessions_with": 12}
WIN = {10.0: 30.0, 15.0: 27.0, 20.0: 25.0, 30.0: 23.0, 45.0: 19.0, 61.0: 16.0}


class DecideTest(unittest.TestCase):
    def d(self, table=WIN, halves=(WIN, WIN), current=15.0, prev=None, counts=OK):
        return tune.decide(table, halves, current, 15.0, prev, counts)

    def test_too_few_sessions_days_moments(self):
        for counts, word in ((dict(OK, sessions=3), "sessions"), (dict(OK, days=2), "days"),
                             (dict(OK, moments=4), "idle moments"), (dict(OK, sessions_with=2), "idle moments")):
            r = self.d(counts=counts)
            self.assertEqual((r.value, r.source), (15.0, "default"))
            self.assertIn(word, r.why)

    def test_small_gain_is_noise(self):
        r = self.d(table={**WIN, 10.0: 27.5})
        self.assertEqual((r.value, r.source), (15.0, "default"))

    def test_must_win_in_both_halves(self):
        r = self.d(halves=(WIN, {**WIN, 10.0: 26.0}))
        self.assertEqual((r.value, r.source), (15.0, "default"))
        self.assertIn("halves", r.why)

    def test_clear_stable_win_moves_one_step(self):
        flipped = {10.0: 16.0, 15.0: 20.0, 20.0: 23.0, 30.0: 25.0, 45.0: 27.0, 61.0: 30.0}
        r = self.d(table=flipped, halves=(flipped, flipped))
        self.assertEqual((r.value, r.source), (20.0, "tuned"), "one step from 15 toward 61")

    def test_disagreeing_runs_hold(self):
        r = self.d(prev=61.0)
        self.assertEqual((r.value, r.source), (15.0, "held"))

    def test_already_best(self):
        flat = {c: 20.0 for c in WIN}
        flat[15.0] = 21.5
        r = self.d(table=flat, halves=(flat, flat))
        self.assertEqual(r.value, 15.0)

    def test_bounds_hold_in_candidates(self):
        self.assertTrue(all(10 <= c <= 61 for c in tune.CANDIDATES))


class PaybackTest(unittest.TestCase):
    """bg_warm_payback_max is not fitted: the default, a user's own value, and an earlier
    version's fitted value removed."""

    def test_fitted_value_is_removed(self):
        config.set_value("bg_warm_payback_max", 49.0, "tuned")
        self.assertEqual(config.resolve("bg_warm_payback_max"), (49.0, "tuned"))
        d = tune._payback(apply=True)
        self.assertEqual((d.value, d.source), (70.0, "default"))
        self.assertEqual(config.resolve("bg_warm_payback_max"), (70.0, "default"))

    def test_dry_run_leaves_it(self):
        config.set_value("bg_warm_payback_max", 49.0, "tuned")
        tune._payback(apply=False)
        self.assertEqual(config.resolve("bg_warm_payback_max"), (49.0, "tuned"))

    def test_user_value_kept(self):
        config.set_value("bg_warm_payback_max", 40.0, "user")
        d = tune._payback(apply=True)
        self.assertEqual((d.value, d.source), (40.0, "user"))
        self.assertEqual(config.resolve("bg_warm_payback_max"), (40.0, "user"))

    def tearDown(self):
        for name in ("bg_warm_payback_max", "bg_after_min", "bg_prep_idle_min"):
            for src in ("tuned", "user"):
                config.unset(name, src)


class AfterMinNotFittedTest(unittest.TestCase):
    """bg_after_min is reported, never fitted: a fitted value (and its derived prep time) is
    removed by a real run, left by a dry run; a user's own value is kept."""

    def tearDown(self):
        for name in ("bg_warm_payback_max", "bg_after_min", "bg_prep_idle_min"):
            for src in ("tuned", "user"):
                config.unset(name, src)

    def run_tune(self, apply):
        from unittest import mock
        with mock.patch.object(tune, "_audited", return_value=0), mock.patch.object(tune, "_record"):
            return tune.run(apply=apply)

    def test_fitted_values_removed(self):
        config.set_value("bg_after_min", 10.0, "tuned")
        config.set_value("bg_prep_idle_min", 6.0, "tuned")
        self.run_tune(apply=False)
        self.assertEqual(config.resolve("bg_after_min"), (10.0, "tuned"), "dry run changes nothing")
        d = self.run_tune(apply=True)
        self.assertEqual((d.value, d.source), (15.0, "default"))
        self.assertEqual(config.resolve("bg_after_min"), (15.0, "default"))
        self.assertEqual(config.resolve("bg_prep_idle_min")[1], "default")

    def test_user_value_kept(self):
        config.set_value("bg_after_min", 30.0, "user")
        d = self.run_tune(apply=True)
        self.assertEqual((d.value, d.source), (30.0, "user"))
        self.assertEqual(config.resolve("bg_after_min"), (30.0, "user"))


class SimRulesTest(unittest.TestCase):
    """sim.price_session follows the proxy's rules: marginal re-compaction and payback."""

    def session(self, gaps_min, ctx=400_000):
        from types import SimpleNamespace as NS
        from tokensaver import sim
        t, rq = 1_800_000_000, []
        for g in [0] + gaps_min:
            t += g * 60
            stamp = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(t))
            rq.append(NS(timestamp=stamp, context=ctx, inp=5, read=ctx - 1000, write=995, write_5m=0,
                         write_1h=995, out=500))
        price = NS(inp=4.0, out=20.0, write_5m=5.0, write_1h=8.0, read=0.2)
        data = NS(price=price, requests=rq, stem="s", session=NS(floor=50_000), blocks=[])
        saved = sim._charge_rereads
        self.addCleanup(setattr, sim, "_charge_rereads", saved)
        sim._charge_rereads = lambda *a, **k: 0
        return sim, data

    def test_second_compaction_is_marginal(self):
        sim, data = self.session([1] * 5 + [70] + [1] * 5 + [70] + [1] * 5)
        r = sim.price_session(data, 0.6, 15 * 60, 100_000, 120_000, 1.0, payback_max=1e9)
        self.assertEqual(r.applications, 1, "same history again: nothing beyond the first reduction")

    def test_warm_compaction_needs_payback(self):
        sim, data = self.session([1] * 5 + [20] + [1] * 5)
        self.assertEqual(sim.price_session(data, 0.6, 15 * 60, 100_000, 120_000, 1.0,
                                           payback_max=20).applications, 0)
        self.assertEqual(sim.price_session(data, 0.6, 15 * 60, 100_000, 120_000, 1.0,
                                           payback_max=1e9).applications, 1)


if __name__ == "__main__":
    unittest.main()
