"""Out-of-sample test of keep-alive policies on this machine's history. Local, $0.

Every policy decides each break using only what was known before that day began (counts
from breaks that had ended, day ends from earlier days). The model's two knobs (stratum
strength, lower quantile) are picked on the first half of the days and scored on the
second half; today's rule (`keepalive.due`'s FREE_BUMPS/MAX_BUMPS/day end, big sessions)
is scored on the same breaks. A bump costs a read of the context, a kept-warm return saves
a rewrite at the 1-hour price minus a read (no compaction modelled: the same for all).
Usage: python3 experiments/keepalive_oos.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re  # noqa: E402
import time  # noqa: E402
from tokensaver import keepalive as ka, pricing, sim, transcripts  # noqa: E402
from tokensaver.keepalive import (BUMP_AFTER_S, _day, _entrypoint, _hour,  # noqa: E402
                                  interactive, stop_hour)

H = 3600
T = ka.BUMP_AFTER_S

# ---- the learned model (moved here from tokensaver/keepalive.py, 2026-11-04) ----
# Out of sample it lost to the shipped rule (+$39 vs +$59 on the second half of 28 days,
# and +$12.84 vs +$14.14 on the first), so it is not shipped; re-run as history grows.

#
# Each past break of an interactive session is one record: when it began (t0), when the
# user came back (elapsed seconds) or how long it was watched without a return (censored),
# whether the session had been resumed after a break before, and when another interactive
# session was first used during it. At bump k (sent at k*T after t0), the cache would
# otherwise die at the end of the previous coverage; the bump pays only for a return
# after that and before its own coverage ends:
#   q_k = P(back in (kT, 60 min + (k-1)T] | not back by kT)   already covered, no value
#   h_k = P(back in (60 min + (k-1)T, 60 min + kT] | not back by kT)   the bump's value
# estimated per stratum (daypart at the decision, another session active, returned before),
# shrunk to the user's pooled estimate, which starts from a uniform-ignorance prior. The
# decision takes a lower credible bound of h_k, so thin data bumps less, never more.

LEARN_WINDOWS = 12           # 12 bump windows (~12 h) is also HOLD_MAX_S
LEARNED_MIN_BREAKS = 30      # the user's own breaks reaching bump 1 before the model decides
LEARNED_MIN_DAYS = 10
STRATUM_STRENGTH = 10        # pseudo-breaks the stratum borrows from the user's pooled estimate
LOWER_Q = 0.2                # decide on the 20th percentile of h_k
# The two above are chosen by experiments/keepalive.py --oos (fit on the first half of the
# days, scored on the second); see its output for the values tried.


def break_records(sessions=None, now: float | None = None) -> list[dict]:
    """Every break over BUMP_AFTER_S in an interactive session, oldest first."""
    if sessions is None:
        forks = sim.forks()
        sessions = [s for s in transcripts.load() if s.requests and s.path.stem not in forks
                    and interactive(f"(external, {_entrypoint(s.path) or 'cli'})")]
    times = {s.path.stem: sorted(sim._ts(r.timestamp) for r in s.requests) for s in sessions}
    allreq = sorted((t, k) for k, ts in times.items() for t in ts)
    now = now or (allreq[-1][0] if allreq else time.time())
    out = []
    for s in sessions:
        rq = sorted(s.requests, key=lambda r: sim._ts(r.timestamp))
        returned = False
        for k, r in enumerate(rq):
            t = sim._ts(r.timestamp)
            g = sim._ts(rq[k + 1].timestamp) - t if k + 1 < len(rq) else None
            if g is not None and g <= BUMP_AFTER_S:
                continue
            e = g if g is not None and g <= 7 * 86400 else None
            other = next((t2 - t for t2, k2 in allreq if t2 > t and k2 != s.path.stem), None)
            out.append({"t0": t, "e": e, "obs": (g if e is not None else min(now - t, 7 * 86400)),
                        "rb": returned, "o": other, "size": r.context, "model": s.primary_model,
                        "sid": s.path.stem, "reqs": k + 1})
            if g is not None and g > H:
                returned = True
    out.sort(key=lambda b: b["t0"])
    return out


def _daypart(ends: dict, t0: float, at: float) -> str:
    stop = stop_hour(ends, t0)
    if stop is None:
        return "day"
    return "day" if _day(at) == _day(t0) and _hour(at) < stop else "late"


def _stratum(ends: dict, b: dict, k: int, T: int) -> tuple:
    d = k * T
    return (_daypart(ends, b["t0"], b["t0"] + d), b["o"] is not None and b["o"] < d, bool(b["rb"]))


def hazard_counts(records: list[dict], ends: dict, T: int = BUMP_AFTER_S) -> dict:
    """{(k, stratum): [at_risk, covered, hits]}, k = 1..LEARN_WINDOWS. One session gives at
    most max(3, 20%) of any cell's weight, so a single long-running session cannot decide."""
    cells: dict = {}
    per_session: dict = {}
    for b in records:
        for k in range(1, LEARN_WINDOWS + 1):
            d = k * T
            if b["e"] is not None and b["e"] <= d:
                break
            if b["e"] is None and b["obs"] < d:
                break                                  # censored: not watched this long
            prev_cov, cov = H + (k - 1) * T, H + k * T
            key = (k, _stratum(ends, b, k, T))
            per_session.setdefault(key, {}).setdefault(b["sid"], [0, 0, 0])
            c = per_session[key][b["sid"]]
            c[0] += 1
            if b["e"] is not None and b["e"] <= prev_cov:
                c[1] += 1
            elif b["e"] is not None and b["e"] <= cov:
                c[2] += 1
    for key, by_s in per_session.items():
        total = sum(v[0] for v in by_s.values())
        cap = max(3, 0.2 * total)
        agg = [0.0, 0.0, 0.0]
        for v in by_s.values():
            w = min(1.0, cap / v[0]) if v[0] else 0
            for i in range(3):
                agg[i] += w * v[i]
        cells[key] = agg
    return cells


def _betacf(a: float, b: float, x: float) -> float:
    m, eps = 200, 3e-12
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for i in range(1, m + 1):
        m2 = 2 * i
        aa = i * (b - i) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c if abs(c) > 1e-30 else 1e30
        h *= d * c
        aa = -(a + i) * (qab + i) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c if abs(c) > 1e-30 else 1e30
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    import math
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def beta_quantile(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def posteriors(cells: dict, k: int, stratum: tuple, strength: float = STRATUM_STRENGTH,
               lower_q: float = LOWER_Q) -> tuple[float, float]:
    """(lower bound of h_k, mean of q_k) for this stratum. The user's pooled estimate for
    window k starts from Beta(1, 1); the stratum borrows `strength` pseudo-breaks from it."""
    pooled = [0.0, 0.0, 0.0]
    for (kk, _), v in cells.items():
        if kk == k:
            for i in range(3):
                pooled[i] += v[i]
    n_u, cov_u, hit_u = pooled
    p_hit = (hit_u + 1) / (n_u + 2)
    p_cov = (cov_u + 1) / (n_u + 2) if n_u else 0.0
    n, cov, hit = cells.get((k, stratum), [0.0, 0.0, 0.0])
    a = hit + strength * p_hit
    b = (n - hit) + strength * (1 - p_hit)
    q_mean = (cov + strength * p_cov) / (n + strength)
    return beta_quantile(lower_q, a, b), q_mean


def plan_value(cells: dict, ends: dict, b: dict, k: int, value: float, cost: float, T: int = BUMP_AFTER_S,
               strength: float = STRATUM_STRENGTH, lower_q: float = LOWER_Q) -> float:
    """Expected net of bumping now (bump k) and acting well afterwards, by backward induction:
    V_j = max(0, h_j*B - C + (1 - h_j - q_j) * V_{j+1}); the decision to bump is V_k > 0.
    Signals for later windows are taken as they would be if nothing changes (daypart moves
    with the clock; 'another session' and 'returned before' stay)."""
    v_next = 0.0
    for j in range(LEARN_WINDOWS, k - 1, -1):
        h, q = posteriors(cells, j, _stratum(ends, b, j, T), strength, lower_q)
        v = h * value - cost + max(0.0, 1 - h - q) * v_next
        v_next = max(0.0, v)
    return v_next


def learned_ready(st: dict) -> bool:
    return (int(st.get("breaks_at_k1") or 0) >= LEARNED_MIN_BREAKS
            and len(st.get("ends") or {}) >= LEARNED_MIN_DAYS)




def counts_before(records, ends, day_start, cache={}):
    key = (id(records), day_start)
    if key not in cache:
        known = [dict(b, e=b["e"] if b["e"] is not None and b["t0"] + b["e"] < day_start else None,
                      obs=min(b["obs"], day_start - b["t0"]))
                 for b in records if b["t0"] < day_start]
        cache[key] = hazard_counts(known, ends)
    return cache[key]


def simulate(b, decide):
    """(spent, saved, bumps) for one break under a decision function decide(k) -> bool."""
    p = pricing.price_for(b["model"])
    B, C = b["size"] * (p.write_1h - p.read) / 1e6, b["size"] * p.read / 1e6
    spent, cov, k = 0.0, H, 0
    while True:
        k += 1
        d = k * T
        if b["e"] is not None and b["e"] <= d:
            break
        if b["e"] is None and b["obs"] < d:
            break
        if not decide(k):
            break
        spent += C
        cov = H + k * T
    kept = b["e"] is not None and H < b["e"] <= cov and k > 1
    return spent, (B if kept else 0.0), k - 1


def rule(b, ends):
    big = b["reqs"] >= ka.BIG_REQUESTS or b["rb"] or b["size"] >= ka.BIG_TOKENS

    def decide(k):
        if not big or k > ka.MAX_BUMPS or k * T > ka.HOLD_MAX_S:
            return False
        if k <= ka.FREE_BUMPS:
            return True
        stop = ka.stop_hour(ends, b["t0"])
        at = b["t0"] + k * T
        return stop is not None and ka._day(at) == ka._day(b["t0"]) and ka._hour(at) < stop
    return decide


def model(b, ends, records, strength, lower_q):
    p = pricing.price_for(b["model"])
    B, C = b["size"] * (p.write_1h - p.read) / 1e6, b["size"] * p.read / 1e6
    day_start = b["t0"] - (b["t0"] - ka.DAY_CUTOFF_H * H) % 86400        # roughly the day's start
    cells = counts_before(records, ends, day_start)

    def decide(k):
        if k * T > ka.HOLD_MAX_S:
            return False
        return plan_value(cells, ends, b, k, B, C, T, strength, lower_q) > 0
    return decide


def score(breaks, make):
    spent = saved = 0.0
    for b in breaks:
        s, v, _ = simulate(b, make(b))
        spent += s
        saved += v
    return saved - spent, saved, spent


def main() -> int:
    fk = sim.forks()
    ss = [s for s in transcripts.load() if s.requests and s.path.stem not in fk
          and ka.interactive(f"(external, {ka._entrypoint(s.path) or 'cli'})")]
    records = [b for b in break_records(ss) if pricing.price_for(b["model"])]
    ends = ka.day_ends([sim._ts(r.timestamp) for s in ss for r in s.requests])
    bill = sum((r.inp * p.inp + r.read * p.read + r.write_1h * p.write_1h + r.write_5m * p.write_5m
                + r.out * p.out) / 1e6 for s in ss for r in s.requests for p in [pricing.price_for(s.primary_model)] if p)
    days = sorted({ka._day(b["t0"]) for b in records})
    mid = days[len(days) // 2]
    first = [b for b in records if ka._day(b["t0"]) < mid]
    second = [b for b in records if ka._day(b["t0"]) >= mid]
    print(f"{len(ss)} interactive sessions, {len(records)} breaks over {T // 60} min, {len(days)} days, "
          f"bill ${bill:,.2f}; first half {len(first)} breaks, second half {len(second)}")
    grid = [(s_, q) for s_ in (3, 10, 30) for q in (0.1, 0.2, 0.35, 0.5)]
    print("\nchoosing the model's knobs on the first half:")
    best = None
    for s_, q in grid:
        net, _, _ = score(first, lambda b: model(b, ends, records, s_, q))
        print(f"  strength {s_:2d}, lower quantile {q:.2f}: net ${net:+.2f}")
        if best is None or net > best[0]:
            best = (net, s_, q)
    _, s_, q = best
    print(f"\nchosen: strength {s_}, lower quantile {q}")
    for name, part in (("first half (in sample for the model)", first), ("second half (out of sample)", second)):
        r = score(part, lambda b: rule(b, ends))
        m = score(part, lambda b: model(b, ends, records, s_, q))
        print(f"\n{name}:")
        print(f"  today's rule: net ${r[0]:+.2f} (saved ${r[1]:.2f}, spent ${r[2]:.2f})")
        print(f"  learned:      net ${m[0]:+.2f} (saved ${m[1]:.2f}, spent ${m[2]:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
