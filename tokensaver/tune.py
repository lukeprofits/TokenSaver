"""Fit background compaction's idle threshold to this user's own habits.

Only `bg_after_min` is tuned: it is the one setting with a real effect
(simulated on one user: 61 min 16%, 30 min 24%, 15 min 28% of the bill). The
sizes (keep, index, minimum) measured flat within noise, and tuning noise only
adds variance, so they stay at their defaults.

"Measuring is hard" is the normal case, so every rule below falls back to the
default rather than guessing:

* Not before the simulation has been checked against real tokens: fewer than
  MIN_AUDITED audited requests means "waiting", nothing changes.
* Too little history: fewer than MIN_SESSIONS sessions, under MIN_DAYS days,
  fewer than MIN_MOMENTS idle moments, or fewer than MIN_SESSIONS_WITH idle
  sessions. Use the default and say which.
* Too small a win: the best candidate must beat the default by MIN_GAIN_PP
  points of bill, or it is noise.
* Unstable: the best candidate must also win (beat the default) in both
  halves of the history, split by time. A habit that holds in only one half
  is not a habit.
* Bounds: values are clamped to the setting's bounds (config), and a value
  the user set themselves is never overridden.
* Hysteresis: move at most one candidate step per run, only if the new value
  beats the current one by MIN_GAIN_PP, and hold if the last run picked a
  different best (two runs must agree before moving).

`bg_warm_payback_max` is not fitted (2026-11-10): it stays at its default, and a
value an earlier version fitted is removed. A per-user count of requests after a break
(p40, then the median) measured worse than the default: priced on what really followed
each warm trim, 30 days here, limit 33 +$22.8, 43 (the median) ~+$24, 49 (the p40 fit)
+$25.4, 70 +$28.2, 100 +$16.4, none -$41.6. A count is a poor stand-in for money, and it
left out sessions that went through TokenSaver. The benchmark's "best limit here" line
prices the limits on each machine; per-user fitting comes back only if those disagree.

Local only: replays transcripts on disk, no API calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import background, config, sim, vault

CANDIDATES = (10.0, 15.0, 20.0, 30.0, 45.0, 61.0)
MIN_AUDITED = 20
MIN_SESSIONS = 10
MIN_DAYS = 7
MIN_MOMENTS = 15
MIN_SESSIONS_WITH = 5
MIN_GAIN_PP = 1.0


@dataclass
class Decision:
    value: float
    source: str                     # default | tuned | held | user
    why: str
    table: dict = field(default_factory=dict)   # candidate -> net saving % of bill
    halves: tuple = ()
    payback: "Decision | None" = None


def decide(table: dict[float, float], halves: tuple[dict, dict], current: float, default: float,
           prev_best: float | None, counts: dict) -> Decision:
    """The choice, from per-candidate net savings (percent of bill) for all the
    history and for each half. Pure: every rule is tested on its own."""
    if counts.get("sessions", 0) < MIN_SESSIONS:
        return Decision(default, "default", f"only {counts.get('sessions', 0)} sessions (need {MIN_SESSIONS})", table)
    if counts.get("days", 0) < MIN_DAYS:
        return Decision(default, "default", f"only {counts.get('days', 0):.0f} days of history (need {MIN_DAYS})", table)
    if counts.get("moments", 0) < MIN_MOMENTS or counts.get("sessions_with", 0) < MIN_SESSIONS_WITH:
        return Decision(default, "default",
                        f"only {counts.get('moments', 0)} idle moments in {counts.get('sessions_with', 0)} sessions "
                        f"(need {MIN_MOMENTS} in {MIN_SESSIONS_WITH})", table)
    if not table:
        return Decision(default, "default", "nothing to compare", table)
    best = max(table, key=lambda c: (table[c], -abs(c - default)))
    gain = table[best] - table.get(default, 0.0)
    if best != default and gain < MIN_GAIN_PP:
        return Decision(default, "default", f"best ({best:g} min) beats the default by only {gain:.1f} pts", table, halves)
    if best != default:
        for h in halves:
            if not h or h.get(best, 0.0) <= h.get(default, 0.0):
                return Decision(default, "default",
                                f"{best:g} min does not win in both halves of the history (unstable)", table, halves)
    if prev_best is not None and prev_best != best:
        return Decision(current, "held", f"last run picked {prev_best:g} min, this one {best:g}: holding", table, halves)
    if best == current:
        return Decision(current, "held" if current != default else "default", "already at the best value", table, halves)
    if table[best] - table.get(current, 0.0) < MIN_GAIN_PP:
        return Decision(current, "held", f"{best:g} min beats the current {current:g} by under {MIN_GAIN_PP} pts",
                        table, halves)
    order = sorted(CANDIDATES)
    i, j = order.index(current) if current in order else order.index(default), order.index(best)
    step = order[i + (1 if j > i else -1)]       # one step toward the best
    return Decision(step, "tuned", f"{best:g} min is best (+{gain:.1f} pts vs default); moving one step to {step:g}",
                    table, halves)


def _audited() -> int:
    import sqlite3
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        n = conn.execute("SELECT count(*) FROM events WHERE kind='token_audit'").fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def _pass_rate() -> tuple[float, str]:
    # The rules-only compaction applies without a model check (2026-10-23); a
    # checked prune only makes an applied one smaller, which the sim leaves out.
    return 1.0, "rules-only compaction applies without a check"


def run(apply: bool = True, force: bool = False) -> Decision:
    payback = _payback(apply)
    default = config.SPECS["bg_after_min"].default
    current, source = config.resolve("bg_after_min")
    if source in ("user", "env"):
        return Decision(current, "user", f"you set bg_after_min yourself ({source}); not tuning it",
                        payback=payback)
    audited = _audited()
    if audited < MIN_AUDITED and not force:
        d = Decision(current, "held", f"waiting for real token data: {audited}/{MIN_AUDITED} audited requests",
                     payback=payback)
        _record(d, apply)
        return d
    data = sim.sessions()
    factor, factor_why = sim.token_factor()
    sim.TOKENS_PER_ESTIMATED_SAVED = factor
    min_tokens = None                        # the proxy's rule: enough old history to replace
    keep_chars = background.KEEP_TOKENS * 4
    pass_rate, pass_why = _pass_rate()
    starts = sorted(sim._ts(d.requests[0].timestamp) for d in data) if data else []
    counts = {"sessions": len(data), "days": (starts[-1] - starts[0]) / 86400 if len(starts) > 1 else 0,
              "moments": 0, "sessions_with": 0}
    first = min(CANDIDATES)
    for d in data:
        n = len(sim.moments(d, first * 60, min_tokens))
        counts["moments"] += n
        counts["sessions_with"] += n > 0
    shrink = {d.stem: sim.estimate_shrink(d, first * 60, min_tokens) for d in data}
    ordered = sorted(data, key=lambda d: sim._ts(d.requests[0].timestamp))
    halves_data = (ordered[:len(ordered) // 2], ordered[len(ordered) // 2:])
    table, halves = {}, ({}, {})
    for c in CANDIDATES:
        res = [sim.price_session(d, shrink[d.stem], c * 60, min_tokens, keep_chars, pass_rate) for d in data]
        table[c] = round(100 * sim.bill_saving(res)[1], 2)
        for h, part in zip(halves, halves_data):
            rs = [r for r in res if r.stem in {d.stem for d in part}]
            h[c] = round(100 * sim.bill_saving(rs)[1], 2)
    prev = (config.section("tune").get("best"))
    best_now = max(table, key=lambda c: table[c]) if table else None
    d = decide(table, halves, current, default, prev, counts)
    d.why += (f" [{pass_why}; token factor {factor_why}; {counts['sessions']} sessions, "
              f"{counts['moments']} idle moments]")
    if apply and d.source == "tuned":
        config.set_value("bg_after_min", d.value, "tuned", why=d.why,
                         evidence={"table": table, "halves": halves, "counts": counts})
        config.set_value("bg_prep_idle_min", max(5.0, min(10.0, d.value - 4)), "tuned",
                         why=f"derived from bg_after_min={d.value:g}")
    _record(d, apply, best=best_now)
    d.payback = payback
    return d


def _payback(apply: bool) -> Decision:
    """The warm payback limit: the user's own value, else the default; a value an earlier
    version fitted is removed (see the module docstring)."""
    value, source = config.resolve("bg_warm_payback_max")
    if source in ("user", "env"):
        return Decision(value, "user", "you set bg_warm_payback_max yourself")
    if apply and source == "tuned":
        config.unset("bg_warm_payback_max", "tuned")
    return Decision(config.SPECS["bg_warm_payback_max"].default, "default",
                    "not fitted per user; the default paid best on real trims")


def _record(d: Decision, apply: bool, best: float | None = None) -> None:
    if not apply:
        return
    st = config.section("tune")
    hist = (st.get("history") or [])[-9:]
    hist.append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "value": d.value,
                 "source": d.source, "why": d.why})
    st.update(last_run=hist[-1]["at"], status=d.source, why=d.why, history=hist)
    if best is not None:
        st["best"] = best
    config.set_section("tune", st)


def render(d: Decision) -> str:
    lines = [f"bg_after_min -> {d.value:g} min ({d.source}): {d.why}"]
    if d.table:
        lines.append("  net saving of bill by idle threshold (all history | first half | second half):")
        h1, h2 = d.halves if d.halves else ({}, {})
        for c in sorted(d.table):
            lines.append(f"    {c:4g} min  {d.table[c]:5.1f}%  | {h1.get(c, 0):5.1f}% | {h2.get(c, 0):5.1f}%")
    if d.payback:
        lines.append(f"bg_warm_payback_max -> {d.payback.value:g} requests ({d.payback.source}): {d.payback.why}")
    return "\n".join(lines)
