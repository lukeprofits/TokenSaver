"""Can a small local model run the compaction quality gate as well as the 35B?

The gate (`background.gate`) decides whether a compaction is applied: the
local model writes questions about each old stretch, answers them from the
original and from the compacted text, and grades both. Most coworkers have no
22 GB model, so this measures smaller ones against the 35B on this machine's
real idle breaks (`bg_compact.moments`), with four candidates per break:

- good:          today's extractive compaction (the 35B passed 8 of 9 real breaks);
- heads_only:    tool results cut to 40 chars, nothing Claude cited kept, no index;
- no_assistant:  every piece of Claude's own text removed;
- half:          only the first half of each stretch.

The three broken ones must fail. A model ships for a RAM tier only if it fails
every broken candidate the 35B fails and agrees with the 35B on the good ones
(at most one disagreement). Local only (ollama), $0; resumable: finished
(break, model, variant) rows in OUT are skipped.

Usage: python3 experiments/gate_models.py [model ...]    (default: the reference and the candidates)
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
from tokensaver import background as bg  # noqa: E402
from bg_compact import moments  # noqa: E402

OUT = names.home() / "experiments" / "gate_models.jsonl"
REFERENCE = "qwen3.6:35b-mlx"
CANDIDATES = ["qwen3:14b", "qwen3:8b", "gemma3:4b", "qwen3:4b"]
# Which sessions to draw breaks from: TOKENSAVER_SESSIONS=<id prefix>,<id prefix>; default all.
SESSIONS = tuple(filter(None, os.environ.get("TOKENSAVER_SESSIONS", "").split(",")))
MAX_BREAKS = 8


def variants(before: list, cut: int) -> dict:
    old = before[:cut]
    good = bg._extract(before, cut)[1]
    assistant_text = {id(x) for m in old if m.get("role") == "assistant"
                      for x in bg._items(m) if isinstance(x, dict) and x.get("type") == "text"}
    return {
        "good": good,
        "heads_only": lambda i, j: "\n\n".join(bg.render(old[i:j], 40, 60, drop_routine=True)),
        "no_assistant": lambda i, j: "\n\n".join(bg.render(old[i:j], bg.EXTRACT_RESULT_CHARS, 200,
                                                          drop_routine=True, skip=assistant_text)),
        "half": lambda i, j: good(i, i + max(1, (j - i) // 2)),
    }


def breaks():
    seen = set()
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: p.stat().st_size):
        if SESSIONS and path.stem[:8] not in SESSIONS:
            continue
        for before, _ in moments(path):
            fp = bg.fingerprint(before[:50])
            if fp in seen:
                continue
            seen.add(fp)
            cut = bg.choose_cut(before)
            if cut:
                yield path.stem[:8], before, cut
                break


def main() -> int:
    models = sys.argv[1:] or [REFERENCE] + CANDIDATES
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["session"], r["model"], r["variant"]))
    todo = [(session, before, cut, variants(before, cut)) for session, before, cut in list(breaks())[:MAX_BREAKS]]
    print(f"{len(todo)} breaks x {len(models)} models x 4 candidates (model by model: no reloads)", flush=True)
    for model in models:
        unusable = 0
        for session, before, cut, vs in todo:
            if unusable >= 2:
                print(f"  {model}: cannot run the gate (no usable questions on 2 real breaks); skipping the rest",
                      flush=True)
                break
            for name, cand in vs.items():
                if (session, model, name) in done:
                    continue
                if name == "heads_only" and model != REFERENCE:
                    continue    # the reference passes it too: it tests nothing (measured 2026-10-22)
                t = time.time()
                try:
                    g = bg.gate(before[:cut], cand, model=model)
                except Exception as exc:
                    g = {"passed": None, "error": f"{type(exc).__name__}: {exc}"[:200]}
                row = {"session": session, "model": model, "variant": name, "passed": g.get("passed"),
                       "questions": g.get("questions"), "original_right": g.get("original_right"),
                       "replacement_right": g.get("replacement_right"), "error": g.get("error"),
                       "seconds": round(time.time() - t)}
                with OUT.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {session} {model:18s} {name:12s} passed={row['passed']} "
                      f"{row['original_right']}->{row['replacement_right']} of {row['questions']} ({row['seconds']}s)",
                      flush=True)
                if name == "good" and (row["original_right"] or 0) < 3:
                    unusable += 1
                    break          # the original itself answers nothing: the broken ones say nothing more
    return report()


def report() -> int:
    rows = [json.loads(x) for x in OUT.read_text().splitlines()] if OUT.exists() else []
    ref = {(r["session"], r["variant"]): r["passed"] for r in rows if r["model"] == REFERENCE}
    print(f"\n{'model':18s} {'good passed':>12s} {'agree w/ 35B':>13s} {'broken caught':>14s} {'s/gate':>7s}")
    for model in sorted({r["model"] for r in rows}, key=lambda m: (m != REFERENCE, m)):
        mine = [r for r in rows if r["model"] == model and r["passed"] is not None]
        good = [r for r in mine if r["variant"] == "good"]
        broken = [r for r in mine if r["variant"] != "good"]
        agree = sum(1 for r in good if ref.get((r["session"], "good")) == r["passed"])
        must = [r for r in broken if ref.get((r["session"], r["variant"])) is False]
        caught = sum(1 for r in must if r["passed"] is False)
        secs = sum(r["seconds"] for r in mine) / max(len(mine), 1)
        print(f"{model:18s} {sum(r['passed'] for r in good):5d} of {len(good):<4d} {agree:5d} of {len(good):<5d} "
              f"{caught:5d} of {len(must):<6d} {secs:7.0f}")
    return 0


if __name__ == "__main__":
    if "--report" in sys.argv:
        raise SystemExit(report())
    raise SystemExit(main())
