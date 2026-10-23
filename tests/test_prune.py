"""prune + machine: staged pruning that never strains the machine and never loses work."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import machine, prune  # noqa: E402


def history():
    """An old attempt that failed, then the fix, then later work."""
    msgs = []
    for i, (cmd, out, err) in enumerate((("mvn -q test -f /repo/pom.xml", "BUILD FAILURE: 401 Unauthorized fetching groovy-bom " * 8, True),
                                          ("registry-login && mvn -q test -f /repo/pom.xml", "BUILD SUCCESS Tests run: 212 " * 8, False))):
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash",
                                                       "input": {"command": cmd}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}",
                                                  "content": out, "is_error": err}]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "Fixed: the build passes after login. " * 10}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": "great, next"}]})
    return msgs


class YieldTest(unittest.TestCase):
    def test_eases_off_in_steps(self):
        clock = [0.0]
        y = machine.Yield(sleep=lambda s: None, now=lambda: clock[0], check=lambda: (True, "pressure"))
        with mock.patch.object(machine.config, "get", return_value=True):
            self.assertEqual(y.step(), "go")                     # step 1: a short pause, then go
            clock[0] = machine.PAUSE_S[1] + 1
            self.assertEqual(y.step(), "small")                  # step 2: drop the big model
            clock[0] = machine.PAUSE_S[2] + 1
            self.assertEqual(y.step(), "stop")                   # step 3: stop for now

    def test_toggle_off_never_pauses(self):
        y = machine.Yield(sleep=lambda s: self.fail("slept"), check=lambda: (True, "pressure"))
        with mock.patch.object(machine.config, "get", return_value=False):
            self.assertEqual(y.step(), "go")

    def test_calm_resets(self):
        y = machine.Yield(sleep=lambda s: None, check=lambda: (False, ""))
        with mock.patch.object(machine.config, "get", return_value=True):
            self.assertEqual(y.step(), "go")
            self.assertIsNone(y.since)


class FitsTest(unittest.TestCase):
    def test_headroom(self):
        with mock.patch.object(machine, "available_bytes", return_value=40 * 2 ** 30):
            self.assertTrue(machine.fits(28 * 2 ** 30)[0])
            self.assertFalse(machine.fits(36 * 2 ** 30)[0])
        with mock.patch.object(machine, "available_bytes", return_value=None):
            self.assertFalse(machine.fits(1)[0], "unknown memory never loads a model")
        self.assertTrue(machine.fits(10 ** 12, loaded=True)[0])


class TierTest(unittest.TestCase):
    def tier(self, want, models, ram):
        cfg = {"prune_tier": want, "prune_judge": "nimble"}
        with mock.patch.object(prune, "installed", return_value={m: 1 for m in models}), \
                mock.patch("tokensaver.config.get", side_effect=lambda k: cfg[k]), \
                mock.patch.object(machine, "ram_bytes", return_value=ram * 2 ** 30):
            return prune.tier()[0]

    def test_auto(self):
        big = prune.bg.LOCAL_MODEL
        self.assertEqual(self.tier("auto", ["nimble:latest", big], 64), "deep")
        self.assertEqual(self.tier("auto", ["nimble:latest", big], 16), "fast", "deep needs 32 GB+")
        self.assertEqual(self.tier("auto", ["nimble:latest"], 64), "fast")
        self.assertEqual(self.tier("auto", [], 64), "off")
        self.assertEqual(self.tier("off", ["nimble:latest", big], 64), "off")


class StageTest(unittest.TestCase):
    def setUp(self):
        self.msgs = history()
        self.cut = 4                        # the two attempts are old; the summary is recent
        self.cache = prune.Cache(Path(tempfile.mkdtemp()) / "c.json")
        self.patches = [mock.patch.object(prune, "can_load", return_value=(True, "ok")),
                        mock.patch.object(prune, "unload"), mock.patch.object(prune, "remember_size")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def go(self):
        return mock.Mock(step=mock.Mock(return_value="go"))

    def test_deletes_at_threshold_and_caches(self):
        with mock.patch.object(prune, "_nimble", return_value=0.95) as nim:
            drop, info = prune.run_stage(self.msgs, self.cut, "fast", judge_model="nimble", min_p=0.8,
                                         cache=self.cache, yielder=self.go())
        self.assertTrue(drop and info["deleted"] >= 1)
        calls = nim.call_count
        with mock.patch.object(prune, "_nimble", side_effect=AssertionError("judged twice")):
            drop2, info2 = prune.run_stage(self.msgs, self.cut, "fast", judge_model="nimble", min_p=0.8,
                                           cache=prune.Cache(self.cache.path), yielder=self.go())
        self.assertEqual(info2["judged"], 0)
        self.assertEqual(info2["cached"], calls, "verdicts persist across runs")

    def test_below_threshold_keeps(self):
        with mock.patch.object(prune, "_nimble", return_value=0.5):
            drop, _ = prune.run_stage(self.msgs, self.cut, "fast", judge_model="nimble", min_p=0.8,
                                      cache=self.cache, yielder=self.go())
        self.assertEqual(drop, set())

    def test_strain_stops_and_keeps_work(self):
        y = mock.Mock(step=mock.Mock(return_value="stop"))
        with mock.patch.object(prune, "_nimble", return_value=0.95):
            drop, info = prune.run_stage(self.msgs, self.cut, "fast", judge_model="nimble", min_p=0.8,
                                         cache=self.cache, yielder=y)
        self.assertIsNone(drop)
        self.assertIn("strained", info["stopped"])

    def test_user_text_never_a_piece(self):
        ps = prune.pieces(self.msgs, self.cut)
        self.assertTrue(all(p.kind in ("assistant", "tool") for p in ps))
        self.assertFalse(any("great, next" in p.text for p in ps))

    def test_same_history_and_verdicts_same_bytes(self):
        """The compacted text must come out byte-identical from a copy of the
        same history with the same cached verdicts (ids differ between copies)."""
        import copy
        from tokensaver import background as bg
        with mock.patch.object(prune, "_nimble", return_value=0.95):
            d1, _ = prune.run_stage(self.msgs, self.cut, "fast", judge_model="nimble", min_p=0.8,
                                    cache=self.cache, yielder=self.go())
        twin = copy.deepcopy(self.msgs)
        with mock.patch.object(prune, "_nimble", side_effect=AssertionError("not cached")):
            d2, _ = prune.run_stage(twin, self.cut, "fast", judge_model="nimble", min_p=0.8,
                                    cache=prune.Cache(self.cache.path), yielder=self.go())
        t1 = bg._extract(self.msgs, self.cut, d1)[0]["text"]
        t2 = bg._extract(twin, self.cut, d2)[0]["text"]
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, bg._extract(self.msgs, self.cut)[0]["text"], "something was pruned")

    def test_memory_refusal_stops_before_loading(self):
        with mock.patch.object(prune, "can_load", return_value=(False, "needs 18 GB")), \
                mock.patch.object(prune, "_nimble", side_effect=AssertionError("loaded anyway")):
            drop, info = prune.run_stage(self.msgs, self.cut, "fast", judge_model="nimble", min_p=0.8,
                                         cache=self.cache, yielder=self.go())
        self.assertIsNone(drop)
        self.assertIn("18 GB", info["stopped"])


class LinesTest(unittest.TestCase):
    def test_picks_cached_and_bytes_stable(self):
        cache = prune.LineCache(Path(tempfile.mkdtemp()) / "l.json")
        text = "\n".join(f"line {i}" for i in range(10))
        go = mock.Mock(step=mock.Mock(return_value="go"))
        with mock.patch.object(prune, "_pick", return_value={2, 3}) as pick:
            fn, st = prune.line_picker("cur", set(), cache, go)
            out = fn("result", text)
        self.assertEqual(pick.call_count, 1)
        self.assertEqual(out, "line 0\n[... 1 lines left out]\nline 2\nline 3\n[... 6 lines left out]")
        cache.save()
        with mock.patch.object(prune, "_pick", side_effect=AssertionError("picked twice")):
            fn2, st2 = prune.line_picker("other digest", set(), prune.LineCache(cache.path), go)
            self.assertEqual(fn2("result", text), out, "same text, same bytes")
        self.assertEqual(st2["cached"], 1)

    def test_detail_lines_always_kept_and_strain_stops(self):
        cache = prune.LineCache(None)
        text = "\n".join(["head", "noise", "uses FooBarService here", "noise", "noise"])
        go = mock.Mock(step=mock.Mock(return_value="go"))
        with mock.patch.object(prune, "_pick", return_value=set()):
            fn, _ = prune.line_picker("cur", {"FooBarService"}, cache, go)
            self.assertIn("uses FooBarService here", fn("result", text))
        stop = mock.Mock(step=mock.Mock(return_value="stop"))
        fn3, st3 = prune.line_picker("cur", set(), prune.LineCache(None), stop)
        self.assertEqual(fn3("result", text), text)
        self.assertIn("strained", st3["stopped"])


if __name__ == "__main__":
    unittest.main()
