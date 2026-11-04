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


if __name__ == "__main__":
    unittest.main()
