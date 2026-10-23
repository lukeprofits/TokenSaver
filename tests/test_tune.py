"""tune.decide: each 'measuring is hard' rule falls back to the default."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import tune  # noqa: E402

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
    """tune.decide_payback: the warm-apply limit from requests made after breaks."""

    def test_too_few_breaks_is_default(self):
        r = tune.decide_payback([50] * 5, 2, 70.0, 70.0)
        self.assertEqual((r.value, r.source), (70.0, "default"))

    def test_fits_p40_and_clamps(self):
        r = tune.decide_payback(sorted(range(0, 400, 10)), 8, 70.0, 70.0)
        self.assertEqual((r.value, r.source), (160.0, "tuned"))
        r = tune.decide_payback([1000] * 30, 8, 70.0, 70.0)
        self.assertEqual(r.value, 200.0, "clamped to the setting's upper bound")
        r = tune.decide_payback([1] * 30, 8, 70.0, 70.0)
        self.assertEqual(r.value, 20.0, "clamped to the lower bound")

    def test_small_change_holds(self):
        r = tune.decide_payback([75] * 30, 8, 70.0, 70.0)
        self.assertEqual((r.value, r.source), (70.0, "held"))


class SimRulesTest(unittest.TestCase):
    """sim.price_session follows the proxy's rules: marginal re-compaction and payback."""

    def session(self, gaps_min, ctx=400_000):
        from types import SimpleNamespace as NS
        from tokensaver import sim
        t, rq = 1_800_000_000, []
        for g in [0] + gaps_min:
            t += g * 60
            stamp = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(t))
            rq.append(NS(timestamp=stamp, context=ctx, inp=5, read=ctx - 1000, write=995, out=500))
        price = NS(inp=4.0, out=20.0, write_1h=8.0, read=0.2)
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
