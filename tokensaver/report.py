"""Measured baseline: what was actually sent and what it actually cost."""

from __future__ import annotations

from dataclasses import dataclass, field

from .pricing import VERIFIED, resolve
from .transcripts import Session


@dataclass
class Totals:
    inp: int = 0
    read: int = 0
    write: int = 0
    write_5m: int = 0
    write_1h: int = 0
    out: int = 0
    requests: int = 0
    context_sum: int = 0
    unpriced_requests: int = 0
    unpriced_models: set[str] = field(default_factory=set)
    assumed_models: set[str] = field(default_factory=set)
    cost_inp: float = 0.0
    cost_read: float = 0.0
    cost_write: float = 0.0
    cost_out: float = 0.0

    @property
    def cost(self) -> float:
        return self.cost_inp + self.cost_read + self.cost_write + self.cost_out

    @property
    def mean_context(self) -> float:
        return self.context_sum / self.requests if self.requests else 0.0


def accumulate(sessions: list[Session], reprice_as: str | None = None) -> Totals:
    """Sum usage and cost. `reprice_as` prices every request at one model's
    rates, for answering 'what would this same work cost on X'."""
    t = Totals()
    override = resolve(reprice_as).price if reprice_as else None
    if reprice_as and override is None:
        raise SystemExit(f"TokenSaver: no verified prices for model {reprice_as!r}")

    for sess in sessions:
        for r in sess.requests:
            t.requests += 1
            t.inp += r.inp
            t.read += r.read
            t.write += r.write
            t.write_5m += r.write_5m
            t.write_1h += r.write_1h
            t.out += r.out
            t.context_sum += r.context

            if override is not None:
                p = override
            else:
                res = resolve(r.model)
                p = res.price
                if res.assumed:
                    t.assumed_models.add(res.model_id)
            if p is None:
                t.unpriced_requests += 1
                t.unpriced_models.add(r.model or "<none>")
                continue
            # Fall back to the 1h rate only when the TTL split is unreported;
            # that is the Claude Code subscription default.
            if r.write_5m or r.write_1h:
                wcost = (r.write_5m * p.write_5m + r.write_1h * p.write_1h) / 1e6
            else:
                wcost = r.write * p.write_1h / 1e6
            t.cost_inp += r.inp * p.inp / 1e6
            t.cost_read += r.read * p.read / 1e6
            t.cost_write += wcost
            t.cost_out += r.out * p.out / 1e6
    return t


def _pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "    - "


def render(sessions: list[Session], reprice_as: str | None = None) -> str:
    t = accumulate(sessions, reprice_as)
    L: list[str] = []
    label = f"repriced as {reprice_as}" if reprice_as else "as billed"
    span_lo = min((s.started[:10] for s in sessions if s.started), default="?")
    span_hi = max((s.ended[:10] for s in sessions if s.ended), default="?")

    L.append(f"TokenSaver baseline  ({label}; prices verified {VERIFIED})")
    L.append(f"  {len(sessions)} sessions, {t.requests:,} billed requests, {span_lo} -> {span_hi}")
    L.append("")
    L.append(f"  {'component':14s} {'tokens':>16s} {'per request':>12s} {'cost':>10s}   share")
    rows = [
        ("uncached input", t.inp, t.cost_inp),
        ("cache read", t.read, t.cost_read),
        ("cache write", t.write, t.cost_write),
        ("output", t.out, t.cost_out),
    ]
    for name, tok, cost in rows:
        per = tok / t.requests if t.requests else 0
        L.append(f"  {name:14s} {tok:16,} {per:12,.0f} {cost:10,.2f}   {_pct(cost, t.cost)}")
    L.append(f"  {'TOTAL':14s} {t.inp + t.read + t.write + t.out:16,} "
             f"{t.mean_context:12,.0f} {t.cost:10,.2f}")
    L.append("")
    if t.write or t.write_5m or t.write_1h:
        ttl = f"5m {t.write_5m:,} / 1h {t.write_1h:,}" if (t.write_5m or t.write_1h) else "unreported (billed at 1h)"
        L.append(f"  cache write TTL split: {ttl}")

    ctxs = sorted(r.context for s in sessions for r in s.requests)
    if ctxs:
        n = len(ctxs)
        L.append(f"  context per request: median {ctxs[n // 2]:,}  "
                 f"p90 {ctxs[int(n * 0.9)]:,}  max {ctxs[-1]:,}")
    floors = sorted(s.floor for s in sessions if s.floor)
    if floors:
        L.append(f"  session floor (system+tools+project context): "
                 f"median {floors[len(floors) // 2]:,}  min {floors[0]:,}  max {floors[-1]:,}")

    comps = [c for s in sessions for c in s.compactions]
    L.append("")
    L.append(f"  compactions observed: {len(comps)}")
    for c in comps:
        L.append(f"    {c.timestamp[:19]}  {c.trigger}: {c.pre_tokens:,} -> {c.post_tokens:,} tok, "
                 f"{c.duration_ms / 1000:.0f}s, {c.preserved_count} messages preserved")
    over = sum(1 for s in sessions if any(r.context > 200_000 for r in s.requests))
    L.append(f"  sessions exceeding 200k context: {over}/{len(sessions)}")

    if t.assumed_models:
        L.append("")
        L.append(f"  ASSUMED PRICES for {', '.join(sorted(t.assumed_models))} "
                 f"- run `tokensaver doctor` for why")
    if t.unpriced_requests:
        L.append("")
        L.append(f"  NOT PRICED: {t.unpriced_requests:,} requests from "
                 f"{', '.join(sorted(t.unpriced_models))} (no verified rate; excluded from cost)")
    bad = sum(s.unparsed_lines for s in sessions)
    if bad:
        L.append(f"  unparsed transcript lines: {bad} (format drift; counts above may undercount)")
    return "\n".join(L)
