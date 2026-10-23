"""Replay smart-prune decisions at different confidence thresholds, no model calls.

smart_prune.py logs nimble's p(yes) for every candidate pair. This rebuilds the
same pieces and candidates for each moment and, for each threshold, deletes the
pairs at or above it, then reports extra shrink and used-later loss (body only,
the honest count). Local, free, seconds.
Usage: python3 experiments/prune_sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import background as bg, prune  # noqa: E402
from bg_compact import moments  # noqa: E402
from smart_prune import OUT, body_of  # noqa: E402
from stack import indep_ident  # noqa: E402

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def main() -> int:
    rows = {}
    for line in OUT.open():
        r = json.loads(line)
        if r["model"] in ("nimble", "deep"):
            rows[(r["session"], r["model"])] = r   # latest run per session and mode
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl")):
      for mode in ("nimble", "deep"):
        if (path.stem[:8], mode) not in rows:
            continue
        r = rows.pop((path.stem[:8], mode))
        for before, after in moments(path):
            cut = bg.choose_cut(before)
            if not cut:
                continue
            old, recent = before[:cut], before[cut:]
            ps = prune.pieces(before, cut, bg.latest_doc_views(old))
            if len(ps) != r.get("pieces", len(ps)) or len(prune.candidates(ps)) != r["pairs"]:
                continue                            # a different moment of this session
            by_n = {p.n: p for p in ps}
            p_of = {}
            for x in r["log"]:
                if "earlier" in x and x.get("votes") and str(x["votes"][0]).startswith("p="):
                    p_of[x["earlier"]] = max(p_of.get(x["earlier"], 0.0), float(x["votes"][0][2:]))
            base, _ = bg._extract(before, cut)
            later = set().union(*(indep_ident(json.dumps(m)) for m in after if m["role"] == "assistant"))
            need = ((set().union(*(indep_ident(json.dumps(m)) for m in old))
                     - set().union(*(indep_ident(json.dumps(m)) for m in recent))) & later)
            base_body = indep_ident(body_of(base["text"]))
            print(f"\n{path.stem[:8]} [{mode}]: today {base['new_tokens']:,} tokens; {len(p_of)} judged pieces; "
                  f"{len(need & base_body)} identifiers used later are in today's text")
            print("  p(yes) >=   deleted   tokens     extra shrink   lost (used later)")
            for t in THRESHOLDS:
                drop = set()
                for n_, pv in p_of.items():
                    if pv >= t and n_ in by_n:
                        drop.update(id(x) for x in by_n[n_].items)
                res, _ = bg._extract(before, cut, drop)
                lost = (need & base_body) - indep_ident(body_of(res["text"]))
                n = sum(1 for pv in p_of.values() if pv >= t)
                print(f"  {t:8.2f}   {n:7d}   {res['new_tokens']:8,}   "
                      f"{100 * (1 - res['new_tokens'] / base['new_tokens']):10.1f}%   {len(lost):4d} {sorted(lost)[:5]}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
