"""What proxied sessions really cost, against what plain Claude Code would have.

Token percentages hide the thing that decides the bill: the prompt cache. The
first real trial saved 15.5% of tokens and still cost 37% MORE, because six
side requests and two re-applied compactions rebuilt 0.25-0.56M-token caches.
So this prices every proxied request from its billed usage and compares it
with a plain-Claude counterfactual:

- sizes: each request's original size is its sent size times the audited
  original/sent token ratio for that request (count_tokens), or, unaudited,
  the bytes ratio scaled by BYTES_TO_TOKENS;
- caching: plain Claude reads the previous main request's prefix while the
  cache is warm (gap under the 1h TTL) and writes only what is new; after an
  expired gap it rebuilds; a side request reads the main cache.
- calibration: the same model run on the SENT sizes, keeping the real misses,
  must land near the real bill, or the counterfactual means nothing. Measured
  on the trial: $70.44 modelled vs $71.25 billed.

Every warm cache write over MISS_TOKENS is listed with a cause. Misses with no
known cause are charged to TokenSaver (plain Claude assumed to hit), which
can only understate the saving.

Local: vault reads only, no API calls.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

from . import vault
from .pricing import price_for

TTL_S = 3600
MISS_TOKENS = 50_000
BYTES_TO_TOKENS = 0.70      # measured: tokens saved per byte saved, 272 audited requests


@dataclass
class Miss:
    ts: str
    session: str
    request_class: str
    tokens: int
    dollars: float
    cause: str
    ours: bool


@dataclass
class Bill:
    requests: int = 0
    real: float = 0.0
    plain: float = 0.0
    calibration: float = 0.0
    audited: int = 0
    write_5m: int = 0          # tokens written at the 5-minute price (none since 2026-10-22)
    misses: list = field(default_factory=list)

    @property
    def saved(self) -> float:
        return self.plain - self.real


def _t(ts: str) -> float:
    return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone


def _cost(p, inp: float, read: float, write: float, out: float, write_5m: float = 0.0) -> float:
    """`write` at the 1h price, `write_5m` of it at the 5-minute price."""
    w5 = min(write_5m, write)
    return (inp * p.inp + read * p.read + (write - w5) * p.write_1h + w5 * p.write_5m
            + out * p.out) / 1e6


def compute(session_prefix: str = "", since: str = "") -> Bill:
    conn = sqlite3.connect(vault.DB_PATH)
    rows = conn.execute("SELECT id, ts, session_id FROM events WHERE kind='api_request' AND "
                        "session_id LIKE ? AND ts >= ? ORDER BY id",
                        (session_prefix + "%", since)).fetchall()
    audits = conn.execute("SELECT id, session_id FROM events WHERE kind='token_audit' AND "
                          "session_id LIKE ? AND ts >= ? ORDER BY id",
                          (session_prefix + "%", since)).fetchall()
    conn.close()
    ratio_by: dict = {}
    for rid, session in audits:
        m = vault.read_meta(rid)
        # Only thinking-fair counts (audit.py, 2026-10-22); older ones priced
        # plain Claude too low, and the bytes estimate below is closer.
        if m.get("tokens_sent") and m.get("method") == "no_thinking":
            ratio_by.setdefault((session, m.get("bytes_in"), m.get("bytes_out")),
                                m["tokens_original"] / m["tokens_sent"])
    reqs = []
    for rid, ts, session in rows:
        m = vault.read_meta(rid)
        u = m.get("usage") or {}
        if m.get("status") != 200 or not u or (m.get("path") or "").split("?")[0] != "/v1/messages":
            continue
        p = price_for(m.get("model") or "")
        if p is None:
            continue
        rw = m.get("rewrite") if isinstance(m.get("rewrite"), dict) else {}
        reqs.append((ts, session, m.get("request_class") or "main", u, rw, p))

    b = Bill()
    last: dict = {}            # session -> (time, plain size, sent size) of its last main request
    for ts, session, cls, u, rw, p in reqs:
        inp = int(u.get("input_tokens") or 0)
        rd = int(u.get("cache_read_input_tokens") or 0)
        w = int(u.get("cache_creation_input_tokens") or 0)
        out = int(u.get("output_tokens") or 0)
        sent = inp + rd + w
        key = (session, rw.get("bytes_in"), rw.get("bytes_out"))
        if key in ratio_by:
            ratio = ratio_by[key]
            b.audited += 1
        elif rw.get("bytes_in") and rw.get("bytes_out"):
            ratio = 1 + (rw["bytes_in"] / rw["bytes_out"] - 1) * BYTES_TO_TOKENS
        else:
            ratio = 1.0
        plain_size = sent * ratio
        w5 = int((u.get("cache_creation") or {}).get("ephemeral_5m_input_tokens") or 0)
        b.write_5m += w5
        real = _cost(p, inp, rd, w, out, w5)
        b.requests += 1
        b.real += real
        t = _t(ts)
        prev = last.get(session)
        warm = prev is not None and t - prev[0] < TTL_S
        cause = None
        if warm and w > MISS_TOKENS:
            if cls != "main":
                cause = ("side request sent unrewritten (did not match the main history)"
                         if not rw else "side request rewritten but missed")
            elif rw.get("bg_applied"):
                cause = "compaction applied on a warm cache"
            elif prev and sent < 0.5 * prev[2]:
                cause = "Claude Code replaced the history (/compact)"
            else:
                cause = "unknown (possibly Anthropic-side)"
        ours = cause is not None and not cause.startswith("Claude Code")
        if cause:
            b.misses.append(Miss(ts, session or "", cls, w, w * (p.write_1h - p.read) / 1e6, cause, ours))
        # plain Claude, same request
        if cls == "main":
            if warm and not (cause and not ours):
                p_rd = min(prev[1], plain_size)
            elif warm:
                p_rd = 0.0                   # a rebuild plain Claude would have paid too
            else:
                p_rd = 0.0
            # Calibration: the same caching model on the sent sizes. It keeps
            # the real misses (there it uses the real read) and otherwise
            # assumes the previous request's prefix is read, as for plain.
            c_rd = rd if (cause or not warm) else min(prev[2], sent)
            last[session] = (t, plain_size, sent)
        else:
            p_rd = plain_size - inp - min(w, MISS_TOKENS // 25)
            c_rd = rd if cause else sent - inp - min(w, MISS_TOKENS // 25)
        b.plain += _cost(p, inp, max(p_rd, 0), max(plain_size - p_rd - inp, 0), out)
        b.calibration += _cost(p, inp, c_rd, max(sent - c_rd - inp, 0), out, w5)
    return b


def lines(session_prefix: str = "", since: str = "", top: int = 8) -> list[str]:
    try:
        b = compute(session_prefix, since)
    except Exception as exc:
        return [f"Bill: could not compute ({type(exc).__name__}: {exc})", ""]
    if not b.requests:
        return []
    pct = 100 * b.saved / b.plain if b.plain else 0.0
    out = [f"Bill: ${b.real:,.2f} real vs ~${b.plain:,.2f} for plain Claude "
           f"({'saved' if b.saved >= 0 else 'COST'} ${abs(b.saved):,.2f}, {pct:+.1f}%) over {b.requests} requests "
           f"({b.audited} with audited token ratios)"]
    if b.write_5m:
        out.append(f"  written at the 5-minute price by cache_tail_5m (removed 2026-10-22; the next request "
                   f"wrote it again at 1h): {b.write_5m:,} tokens")
    ours = [m for m in b.misses if m.ours]
    if ours:
        by: dict = {}
        for m in ours:
            n, d = by.get(m.cause, (0, 0.0))
            by[m.cause] = (n + 1, d + m.dollars)
        out.append(f"  warm cache rebuilds charged to TokenSaver: {len(ours)}, ${sum(m.dollars for m in ours):,.2f}")
        for cause, (n, d) in sorted(by.items(), key=lambda kv: -kv[1][1])[:top]:
            out.append(f"    {n:3d}  ${d:6.2f}  {cause}")
    out.append(f"  model check: the same cache model on the sent sizes gives ${b.calibration:,.2f} "
               f"(billed ${b.real:,.2f}); plain Claude priced from audited original sizes, and misses "
               f"with no known cause charged to tokensaver")
    out.append("")
    return out
