"""Per-session cost, newest-first by spend.

The point of this view: cost is not spread evenly. If a handful of runaway
sessions carry most of the bill, a threshold that only binds on those is worth
more than one that interrupts every session.
"""

from __future__ import annotations

from .pricing import resolve
from .transcripts import Session


def render(sessions: list[Session], limit: int = 15) -> str:
    rows = []
    for s in sessions:
        p = resolve(s.primary_model).price
        if not p or p.read == 0:
            continue
        cost = sum(r.read * p.read + r.write * p.write_1h + r.out * p.out
                   for r in s.requests) / 1e6
        peak = max((r.context for r in s.requests), default=0)
        rows.append((cost, peak, len(s.requests), s))
    rows.sort(reverse=True, key=lambda r: r[0])
    total = sum(r[0] for r in rows) or 1.0

    L = [f"Per-session cost  ({len(rows)} Anthropic-billed sessions, ${total:,.2f})",
         "",
         f"  {'#':>3s} {'cost':>9s} {'cum':>5s} {'peak ctx':>9s} {'reqs':>5s} "
         f"{'started':11s} {'model':17s} project"]
    cum = 0.0
    for i, (cost, peak, n, s) in enumerate(rows[:limit], 1):
        cum += cost
        proj = (s.cwd or s.project).rsplit("/", 1)[-1][:28]
        L.append(f"  {i:3d} {cost:9,.2f} {100 * cum / total:4.0f}% {peak:9,} {n:5d} "
                 f"{s.started[:10]:11s} {s.primary_model[:17]:17s} {proj}")
    L.append("")
    for k in (3, 5, 10):
        if len(rows) > k:
            share = 100 * sum(r[0] for r in rows[:k]) / total
            L.append(f"  top {k:2d} sessions = {share:3.0f}% of cost")
    return "\n".join(L)
