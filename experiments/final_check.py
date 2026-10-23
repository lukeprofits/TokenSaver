"""The staged prune with one final check, through the proxy's own code, on real breaks.

For each real break (bg_compact.moments, the sessions gate_models.py uses) this
prepares the rules-only compaction exactly as the proxy does, then calls
`proxy._prune_step` until the prune is done (fast -> deep -> lines -> final
check, with the real local models), and reports:
- extra shrink of the result over the rules-only compaction;
- details Claude used after the break that the rules-only text had and the
  result lost (the conversation-wide extractor, as in smart_prune.py);
- the final check's verdict and scores, and seconds per step.

State goes to a temp dir and vault records are captured, so nothing real
changes. Local only (ollama), $0; slow (the 35B and Nimble). Resumable: breaks
already in OUT are skipped.
Usage: python3 experiments/final_check.py [max_breaks] | --report
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import names  # noqa: E402
from tokensaver import background as bg, proxy  # noqa: E402
from gate_models import breaks  # noqa: E402
from bg_compact import moments  # noqa: E402
from stack import indep_ident  # noqa: E402

OUT = names.home() / "experiments" / "final_check.jsonl"


def flat(m: dict) -> str:
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for k, x in v.items():
                if k not in ("signature", "id", "tool_use_id", "type", "cache_control", "name"):
                    walk(x)
    walk(m["content"])
    return "\n".join(out)


def after_of(session: str, before: list) -> list:
    for path in Path.home().glob(f".claude/projects/*/{session}*.jsonl"):
        for b, a in moments(path):
            if len(b) == len(before):
                return a
    return []


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    done = {json.loads(x)["session"] for x in OUT.read_text().splitlines()} if OUT.exists() else set()
    state = Path(tempfile.mkdtemp())
    proxy._bg_path = lambda s: state / f"{s}.bg.json"
    events: list = []
    proxy.vault.record = lambda kind, **k: events.append(dict(k, kind=kind))
    for n, (session, before, cut) in enumerate(breaks()):
        if n >= limit:
            break
        if session in done:
            continue
        pending = bg.prepare(before)
        if not pending:
            continue
        pending.update(made=time.time(), total_tokens=bg.size_tokens(before))
        proxy._bg_save(session, {"pending": pending})
        live = {"messages": before, "last": time.time() - 3600, "skeleton": {}}
        steps, t0 = [], time.time()
        events.clear()
        for _ in range(40):
            t = time.time()
            heavy = proxy._prune_step(session, live)
            ev = events[-1] if events else {}
            steps.append({"seconds": round(time.time() - t), "summary": ev.get("summary", ""),
                          "stage": (ev.get("meta") or {}).get("stage")})
            print(f"  {session} {steps[-1]['seconds']:5d}s  {steps[-1]['summary']}", flush=True)
            got = proxy._bg_load(session)["pending"]
            if got.get("prune_done") or not heavy:
                break
            if "paused" in steps[-1]["summary"]:
                time.sleep(30)
        got = proxy._bg_load(session)["pending"]
        after = after_of(session, before)
        old, recent = before[:cut], before[cut:]
        old_ids = set().union(*(indep_ident(flat(m)) for m in old))
        recent_ids = set().union(*(indep_ident(flat(m)) for m in recent))
        later = set().union(*(indep_ident(flat(m)) for m in after if m["role"] == "assistant"))
        need = (old_ids - recent_ids) & later & indep_ident(pending["text"])
        cand = (got.get("pruned") or {}).get("text") or pending["text"]
        lost = sorted(need - indep_ident(cand))
        final = next((e for e in reversed(events) if (e.get("meta") or {}).get("stage") == "final"), {})
        fm = final.get("meta") or {}
        row = {"session": session, "rules_tokens": pending["new_tokens"],
               "pruned_tokens": (got.get("pruned") or {}).get("new_tokens") or pending["new_tokens"],
               "applied_tokens": got["new_tokens"], "promoted": fm.get("promoted"), "verdict": fm.get("verdict"),
               "gate": fm.get("gate"), "need_in_rules": len(need), "lost": lost,
               "seconds": round(time.time() - t0), "steps": steps}
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{session}: rules {row['rules_tokens']:,} -> pruned {row['pruned_tokens']:,} "
              f"({100 * (1 - row['pruned_tokens'] / row['rules_tokens']):.1f}% smaller); {row['verdict']}; "
              f"lost {len(lost)} of {len(need)} used later {lost[:6]}; {row['seconds']}s", flush=True)
    return report()


def report() -> int:
    rows = [json.loads(x) for x in OUT.read_text().splitlines()] if OUT.exists() else []
    print(f"\n{'session':9s} {'rules':>8s} {'pruned':>8s} {'smaller':>8s} {'lost':>9s} {'check':>32s} {'s':>6s}")
    for r in rows:
        g = r.get("gate") or {}
        score = (f"{g.get('original_right')}->{g.get('replacement_right')}/{g.get('questions')}"
                 if g.get("questions") else g.get("check", ""))
        print(f"{r['session']:9s} {r['rules_tokens']:8,d} {r['pruned_tokens']:8,d} "
              f"{100 * (1 - r['pruned_tokens'] / r['rules_tokens']):7.1f}% "
              f"{len(r['lost']):3d} of {r['need_in_rules']:<4d} {str(r['verdict'])[:22]:>22s} {score:>9s} "
              f"{r['seconds']:6d}")
    if rows:
        rt = sum(r["rules_tokens"] for r in rows)
        at = sum(r["applied_tokens"] for r in rows)
        print(f"applied: {100 * (1 - at / rt):.1f}% smaller than rules-only over {len(rows)} breaks; "
              f"promoted {sum(1 for r in rows if r['promoted'])} of {len(rows)}")
    return 0


if __name__ == "__main__":
    if "--report" in sys.argv:
        raise SystemExit(report())
    raise SystemExit(main())
