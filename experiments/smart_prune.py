"""Smart prune on real breaks: extra shrink vs what Claude actually needed later.

For each real moment (a break of more than 61 min with 100K+ of context, from
bg_compact.moments) and each judge model, this builds today's extractive
compaction and the pruned one, then reports:
- extra shrink of the compacted text over today's version;
- used-later loss: identifiers unique to the old part that Claude used after
  the break, present in today's compacted text, gone from the pruned one.
  Counted twice: in the whole text, and in the body without the identifier
  index (which still lists names from deleted pieces, so it softens a loss);
- the deleted pieces that carried a lost identifier, printed to read by eye;
- the QA gate, optional (TOKENSAVER_PRUNE_GATE=1; ~90 s per moment);
- time per moment, and per judged pair.

Local only: ollama, zero API calls. Slow with the 35B model.
Usage: python3 experiments/smart_prune.py [max_moments] [model ...]
       (models default to qwen3:4b and the configured local model)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import names  # noqa: E402
from tokensaver import background as bg, prune  # noqa: E402
from bg_compact import moments  # noqa: E402
from stack import indep_ident  # noqa: E402

OUT = names.home() / "experiments" / "smart_prune.jsonl"


def body_of(text: str) -> str:
    return text.split("\n\n## Identifiers from the earlier part\n\n")[0]


# The two whole-pass runners, moved here from tokensaver/prune.py (2026-10-22):
# production runs the stages through `prune.run_stage`.

def fast_prune(messages: list, cut: int, model: str | None, protect: set[str] | None = None,
          log: list | None = None) -> set[int]:
    """ids of content items to leave out of the compacted old part. With no
    model, T1 only: nothing is deleted on similarity alone."""
    ps = prune.pieces(messages, cut, protect)
    cands = prune.candidates(ps)
    drop: set[int] = set()
    if not model:
        return drop
    for a, b, score in cands:
        t = time.time()
        ok, votes = prune.judge(a, b, model)
        if log is not None:
            log.append({"earlier": a.n, "later": b.n, "kind": a.kind, "score": score, "votes": votes,
                        "deleted": ok, "seconds": round(time.time() - t, 2),
                        "earlier_text": a.text[:300], "later_text": b.text[:300]})
        if ok:
            drop.update(id(x) for x in a.items)
    return drop


def deep_prune(messages: list, cut: int, spotter: str, judge_model: str,
               protect: set[str] | None = None, log: list | None = None,
               already: set[int] | None = None) -> set[int]:
    """The deep pass: pairs the spotter proposes, judged by `judge_model`,
    added to `already` (the fast pass's deletions)."""
    ps = prune.pieces(messages, cut, protect)
    drop = set(already or ())
    seen = set()
    for a, b in prune.spot(ps, spotter, log) or []:
        if (a.n, b.n) in seen or id(a.items[0]) in drop:
            continue
        seen.add((a.n, b.n))
        t = time.time()
        ok, votes = prune.judge(a, b, judge_model)
        if prune.PROGRESS:
            prune.PROGRESS(f"deep judge #{a.n}->#{b.n}: {votes} in {time.time() - t:.1f}s")
        if log is not None:
            log.append({"earlier": a.n, "later": b.n, "kind": a.kind, "score": None, "votes": votes,
                        "deleted": ok, "source": "deep", "seconds": round(time.time() - t, 2),
                        "earlier_text": a.text[:300], "later_text": b.text[:300]})
        if ok:
            drop.update(id(x) for x in a.items)
    return drop



def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    models = sys.argv[2:] or ["qwen3:4b", bg.LOCAL_MODEL]
    with_gate = os.environ.get("TOKENSAVER_PRUNE_GATE") == "1"
    prune.PROGRESS = lambda msg: print(f"    {time.strftime('%H:%M:%S')} {msg}", flush=True)
    OUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    seen, done = set(), 0
    only = set(filter(None, os.environ.get("TOKENSAVER_BG_SESSIONS", "").split(",")))
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: p.stat().st_size):
        if only and path.stem[:8] not in only:
            continue
        for before, after in moments(path):
            if done >= limit:
                return 0
            fp = bg.fingerprint(before[:50])
            if fp in seen:
                continue
            seen.add(fp)
            cut = bg.choose_cut(before)
            if not cut:
                continue
            done += 1
            old, recent = before[:cut], before[cut:]
            base, _ = bg._extract(before, cut)
            old_ids = set().union(*(indep_ident(json.dumps(m)) for m in old))
            recent_ids = set().union(*(indep_ident(json.dumps(m)) for m in recent))
            later = set().union(*(indep_ident(json.dumps(m)) for m in after if m["role"] == "assistant"))
            need = (old_ids - recent_ids) & later
            base_ids, base_body = indep_ident(base["text"]), indep_ident(body_of(base["text"]))
            ps = prune.pieces(before, cut, bg.latest_doc_views(old))
            t = time.time()
            cands = prune.candidates(ps)
            print(f"\n{path.stem[:8]}: old {base['old_tokens']:,} -> today {base['new_tokens']:,} tokens; "
                  f"{len(ps)} pieces, {len(cands)} candidate pairs ({time.time() - t:.1f}s); "
                  f"identifiers needed later: {len(need)}", flush=True)
            for model in models:
                log: list = []
                t = time.time()
                if model == "deep":     # fast pass (nimble on overlap pairs), then 35B spotter + nimble
                    fast = fast_prune(before, cut, "nimble", bg.latest_doc_views(old), log)
                    drop = deep_prune(before, cut, bg.LOCAL_MODEL, "nimble", bg.latest_doc_views(old),
                                            log, already=fast)
                else:
                    drop = fast_prune(before, cut, model, bg.latest_doc_views(old), log)
                secs = time.time() - t
                res, cand_for = bg._extract(before, cut, drop)
                ids, body = indep_ident(res["text"]), indep_ident(body_of(res["text"]))
                lost = (need & base_ids) - ids
                lost_body = (need & base_body) - body
                extra = 1 - res["new_tokens"] / max(base["new_tokens"], 1)
                gate = bg.gate(old, cand_for) if with_gate else None
                row = {"session": path.stem[:8], "model": model, "pairs": len(cands), "pieces": len(ps),
                       "deleted": sum(1 for x in log if x.get("deleted")), "base_tokens": base["new_tokens"],
                       "pruned_tokens": res["new_tokens"], "extra_shrink": round(extra, 4),
                       "need": len(need), "lost": sorted(lost), "lost_body": sorted(lost_body),
                       "seconds": round(secs), "gate": {k: v for k, v in (gate or {}).items() if k != "items"},
                       "log": log}
                with OUT.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                deep_n = sum(1 for x in log if x.get("source") == "deep")
                print(f"  {model:18s} deleted {row['deleted']:3d}/{len(cands) + deep_n} pairs"
                      f"{f' ({deep_n} proposed by the spotter)' if deep_n else ''}; "
                      f"{base['new_tokens']:,} -> {res['new_tokens']:,} tokens ({100 * extra:.1f}% smaller); "
                      f"lost {len(lost)} (body {len(lost_body)}) of {len(need & base_body)} used later; "
                      f"{secs:.0f}s ({secs / max(len(cands), 1):.1f}s/pair)"
                      + (f"; gate {row['gate']}" if gate else ""), flush=True)
                for ident in sorted(lost_body)[:8]:
                    for x in log:
                        if x.get("deleted") and ident in x.get("earlier_text", ""):
                            print(f"      LOST {ident!r} in deleted: {x['earlier_text'][:140]!r}")
                            print(f"           judged superseded by: {x['later_text'][:140]!r}")
                            break
                    else:
                        print(f"      LOST {ident!r} (deleted piece beyond the first 300 chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
