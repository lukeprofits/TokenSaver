"""Bigger levers than shrinking the old part, priced on real sessions (local, $0).

A. Split: of all context tokens sent, how much is the fixed part (system,
   tools), the compacted old part, and the recent part since the last free
   point, under today's rules.
B. Cache lifetime x compaction: what if the whole chain used the 5-minute
   cache, so every pause over 5 minutes becomes a free point where compaction
   applies at no extra rewrite? Modelled cost per request:
   - the cache expired (gap > TTL), or a compaction applied: the whole prompt
     is written at that TTL's price;
   - otherwise: new tokens written, the rest read.
   All variants, the baseline included, use this same model, so they compare
   like for like; a calibration line compares the modelled 1h baseline with
   the real bill. Re-reads are charged as in `sim.price_session`.
C. Fewer requests: tool-only follow-up requests where the previous response
   made a single read-only call and so did this one. If each such pair had
   been one response with two calls, one full-context request disappears.
   Dependency between the calls is unknown: an UPPER bound.
D. Exploration a subagent could absorb: tool results none of whose
   identifiers Claude names later, weighted by how many later requests
   resend them. An upper bound on what moving exploration out of the main
   context could save (the subagent's own cost is not subtracted).

Usage: python3 experiments/levers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import background, config, sim  # noqa: E402
from stack import indep_ident  # noqa: E402

MIN_TOKENS = 100_000
READ_ONLY_BASH = ("cat ", "ls", "grep ", "rg ", "sed -n", "head ", "tail ", "find ", "wc ",
                  "git status", "git log", "git diff", "git show", "pwd", "echo ")


def model_cost(data, shrink, ttl_s, write_price, compact_after_s, payback_max, split=None):
    """Modelled cost of one session (see module docstring), with re-reads."""
    p, rq = data.price, data.requests
    w = write_price(p)
    k = (shrink or 0.0) * sim.TOKENS_PER_ESTIMATED_SAVED
    floor, keep = data.session.floor, background.KEEP_TOKENS
    total, red, prev, app_at = 0.0, 0.0, 0, None
    extra = [0] * (len(rq) + 2)
    apps = 0
    for i, r in enumerate(rq):
        c = r.context
        if prev and c < 0.5 * prev:
            red, app_at = 0.0, None             # Claude Code's own /compact
        prev = c
        s = c - red
        gap = sim._ts(r.timestamp) - sim._ts(rq[i - 1].timestamp) if i else 1e9
        cold = gap > ttl_s
        rewrite = cold
        if k and compact_after_s is not None and gap > compact_after_s and c > MIN_TOKENS:
            new = c - max(c - floor - keep, 0) * k
            if new < s:
                ok = cold
                if not cold and p.read > 0:
                    ok = (new * w - s * p.read) / ((s - new) * p.read) <= payback_max
                if ok:
                    red, s, rewrite, app_at = c - new, new, True, (i, c)
                    apps += 1
                    sim._charge_rereads(data, i, keep * 4, extra, len(rq))
        carried = sum(extra[:i + 1])
        s += carried
        if rewrite:
            cost = r.inp * p.inp + max(s - r.inp, 0) * w
        else:
            cost = r.inp * p.inp + r.write * w + max(s - r.write - r.inp, 0) * p.read + extra[i] * (w - p.read)
        total += (cost + r.out * p.out) / 1e6
        if split is not None:
            recent = (c - app_at[1] + keep) if app_at else max(c - floor, 0)
            recent = min(recent, max(s - floor, 0))
            split["floor"] += min(floor, s)
            split["recent"] += recent
            split["old"] += max(s - floor - recent, 0)
    return total, apps


def batching(data):
    """(mergeable requests, their modelled read cost, all requests)."""
    by_req: dict[int, list] = {}
    user_text: set[int] = set()
    for b in data.blocks:
        if b.role == "assistant" and b.type == "tool_use":
            by_req.setdefault(b.request, []).append(b)
        if b.role == "user" and b.type == "text" and b.text.strip():
            user_text.add(b.request)

    def read_only(calls):
        if len(calls) != 1:
            return False
        c = calls[0]
        if c.tool in ("Read", "Grep", "Glob"):
            return True
        cmd = ((c.input or {}).get("command") or "").lstrip()
        return c.tool == "Bash" and cmd.startswith(READ_ONLY_BASH) and "&&" not in cmd and ">" not in cmd

    rq, p = data.requests, data.price
    n, cost = 0, 0.0
    for i in range(2, len(rq)):
        # request i follows a response with one read-only call, and request i-1 did too
        if i in user_text or (i - 1) in user_text:
            continue
        if read_only(by_req.get(i, [])) and read_only(by_req.get(i - 1, [])):
            n += 1
            cost += (rq[i].context * p.read + rq[i].out * p.out) / 1e6
    return n, cost, len(rq)


def exploration(data):
    """(uncited tool-result token-requests, all message token-requests)."""
    blocks, n = data.blocks, len(data.requests)
    later_ids: list[set] = [set() for _ in range(n + 2)]
    for b in blocks:
        if b.role == "assistant" and b.type in ("text", "tool_use"):
            txt = b.text if b.type == "text" else str(b.input)
            later_ids[min(b.request, n + 1)] |= {x for x in indep_ident(txt) if len(x) >= 6}
    suffix: list[set] = [set() for _ in range(n + 3)]
    for i in range(n + 1, -1, -1):
        suffix[i] = suffix[i + 1] | later_ids[i]
    uncited = total = 0
    for b in blocks:
        weight = len(b.text) / 4 * max(n - b.request, 0)
        total += weight
        if b.type == "tool_result" and len(b.text) > 300:
            own = {x for x in indep_ident(b.text) if len(x) >= 6} - set(indep_ident(str(b.input or "")))
            if own and not (own & suffix[b.request + 1]):
                uncited += weight
    return uncited, total


def main() -> int:
    data = sim.sessions()
    sim.TOKENS_PER_ESTIMATED_SAVED = sim.token_factor()[0]
    payback = config.get("bg_warm_payback_max")
    print(f"{len(data)} sessions; payback limit {payback}; estimating shrink...", flush=True)
    shrink = {d.stem: sim.estimate_shrink(d, 15 * 60, MIN_TOKENS) for d in data}
    real = sum((r.inp * d.price.inp + r.read * d.price.read + r.write * d.price.write_1h + r.out * d.price.out) / 1e6
               for d in data for r in d.requests)
    w1, w5 = (lambda p: p.write_1h), (lambda p: p.write_5m)
    variants = [
        ("1h, no compaction (model baseline)", 3600, w1, None),
        ("1h, compact after 15 min (today)", 3600, w1, 15 * 60),
        ("1h, compact after 5 min (payback)", 3600, w1, 5 * 60),
        ("5m, no compaction", 300, w5, None),
        ("5m, compact at every pause > 5 min", 300, w5, 5 * 60),
        ("5m, + warm compaction after 2 min (payback)", 300, w5, 2 * 60),
    ]
    split = {"floor": 0.0, "old": 0.0, "recent": 0.0}
    results = []
    for name, ttl, wp, after in variants:
        tot = apps = 0
        for d in data:
            c, a = model_cost(d, shrink[d.stem], ttl, wp, after, payback,
                              split if name.endswith("(today)") else None)
            tot += c
            apps += a
        results.append((name, tot, apps))
    base = results[0][1]
    print(f"\nreal bill ${real:,.2f}; modelled 1h baseline ${base:,.2f} (calibration {100 * (base / real - 1):+.1f}%)")
    print("\nB. cache lifetime x compaction (vs modelled baseline)")
    for name, tot, apps in results:
        print(f"  {name:45s} ${tot:9,.2f}  {100 * (1 - tot / base):+6.1f}%  compactions {apps}")
    s = sum(split.values())
    print("\nA. where the sent context tokens are, today's rules")
    for k in ("floor", "old", "recent"):
        print(f"  {k:7s} {100 * split[k] / s:5.1f}%")
    n = c = allr = 0
    u = t = 0.0
    for d in data:
        a, b, m = batching(d)
        n, c, allr = n + a, c + b, allr + m
        x, y = exploration(d)
        u, t = u + x, t + y
    print(f"\nC. mergeable read-only follow-ups: {n:,} of {allr:,} requests ({100 * n / allr:.1f}%), "
          f"~${c:,.2f} = {100 * c / real:.1f}% of the real bill (upper bound)")
    print(f"D. uncited tool results: {100 * u / t:.1f}% of resent message tokens (upper bound, "
          f"subagent cost not subtracted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
