"""One table: TokenSaver vs Headroom vs plain, from the latest runs on this machine.

Reads the newest `tokensaver-cli benchmark` result (~/.tokensaver/benchmark/)
and the newest Headroom runs of experiments/replay_history.py. All of them use
the same replay, token counts and cost model, so the rows compare directly.
Local, no calls. Usage: python3 experiments/compare_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import names  # noqa: E402

BENCH = names.home() / "benchmark"
HR = names.home() / "experiments" / "replay_history.jsonl"


def totals(rows):
    ok = [r for r in rows if "skipped" not in r]
    return (len(ok), sum(r["plain_tokens"] for r in ok), sum(r["ours_tokens"] for r in ok),
            sum(r["plain_cost"] for r in ok), sum(r["ours_cost"] for r in ok),
            sum(r.get("markers", 0) for r in ok))


def main() -> int:
    runs = {}
    files = sorted(BENCH.glob("benchmark-*.json"))
    if files:
        runs["tokensaver"] = json.loads(files[-1].read_text())["sessions"]
    if HR.exists():
        for line in HR.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if str(rec.get("variant", "")).startswith("headroom:") and "sessions" in rec:
                runs[rec["variant"].replace(":", "/")] = rec["sessions"]      # newest wins
    if not runs:
        print("nothing yet: run `tokensaver-cli benchmark` first")
        return 1
    print(f"{'':18s} {'sessions':>8s} {'tokens sent':>14s} {'saved':>7s} {'cost':>11s} {'saved':>7s}  notes")
    base = None
    for name, rows in runs.items():
        n, pt, ot, pc, oc, markers = totals(rows)
        if base is None:
            base = (pt, pc)
            print(f"{'plain Claude Code':18s} {n:8d} {pt:14,} {'':>7s} ${pc:10,.2f} {'':>7s}")
        note = f"{markers} retrieve markers (each would be an extra request)" if markers else ""
        print(f"{name:18s} {n:8d} {ot:14,} {100 * (1 - ot / pt):6.1f}% ${oc:10,.2f} {100 * (1 - oc / pc):6.1f}%  {note}")
    print("\nSame sessions, same token counts and cost model for every row; defaults for tokensaver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
