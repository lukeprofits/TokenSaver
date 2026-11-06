"""Claude Code's idle compaction vs leaving breaks to TokenSaver (local, $0).

Claude Code compacts a session by itself when it sits idle with a big context (setting
`idleCompaction`, default on; `CLAUDE_CODE_IDLE_COMPACT_MIN_TOKENS`, default 200K): it asks
Claude for a summary (a paid request) and the summary replaces the history. Seen here about
55 minutes into a break, just before the prompt cache would expire. TokenSaver trims at the
same moments without a model, keeping everyone's words verbatim.

For every idle compaction in this machine's transcripts (one with no `/compact` command
next to it; the transcript labels both kinds "manual"), the session is replayed twice with
the benchmark's own replay and one local size estimate on both sides:
- as it ran: the summary history afterwards, plus the summary request's own bill
  (from the vault when it went through TokenSaver, else estimated from its size);
- without it: the history continues, and TokenSaver's break-time trim decides.
Both are TokenSaver's cost from that point on. Not modelled: answer quality (the summary is
model-written and drops details; the trim keeps every word of yours and Claude's).

    python3 experiments/idle_compaction.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import background as bg, benchmark, pricing, sim, transcripts, vault  # noqa: E402

IDLE_MIN_S = 45 * 60          # a compaction this long after the last reply, with no /compact typed


def _t(ts: str) -> float:
    return sim._ts(ts[:19] + "Z")


def count(model, msgs, thinking=False, **_):
    """Local estimate, the same on both sides (thinking left out, as the benchmark counts)."""
    plain = [bg.strip_thinking(m) if m.get("role") == "assistant" else m for m in msgs]
    return bg.size_tokens(plain)


def idle_boundaries(lines: list) -> list[int]:
    out, last_reply = [], None
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("type") == "assistant" and r.get("timestamp") and not r.get("isSidechain"):
            last_reply = _t(r["timestamp"])
        if r.get("subtype") != "compact_boundary" or not r.get("timestamp") or last_reply is None:
            continue
        near = "".join(lines[max(0, i - 8): i + 8])
        if "<command-name>/compact" in near:
            continue                                     # typed by the user
        if _t(r["timestamp"]) - last_reply >= IDLE_MIN_S:
            out.append(i)
    return out


def _is_boundary(ln: str) -> bool:
    try:
        return json.loads(ln).get("subtype") == "compact_boundary"
    except ValueError:
        return False


def without(lines: list, idx: int) -> list:
    """The transcript as if that compaction had not run: no boundary, no summary."""
    keep = []
    for i, ln in enumerate(lines):
        if i == idx:
            continue
        if i > idx:
            try:
                if json.loads(ln).get("isCompactSummary"):
                    continue
            except ValueError:
                pass
        keep.append(ln)
    return keep


def summary_cost(session: str, at: float, pre_tokens: int, model: str) -> tuple[float, str]:
    """The summary request's bill: the vault's when it went through TokenSaver."""
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        rows = conn.execute("SELECT id, ts FROM events WHERE kind='api_request' AND session_id = ?",
                            (session,)).fetchall()
        conn.close()
        for rid, ts in rows:
            if abs(_t(ts) - at) > 300:
                continue
            m = vault.read_meta(rid)
            if m.get("request_class") != "compaction" or "count_tokens" in (m.get("path") or ""):
                continue
            u, p = m.get("usage") or {}, pricing.price_for(m.get("model") or model)
            if u and p:
                return ((int(u.get("input_tokens") or 0) * p.inp + int(u.get("cache_read_input_tokens") or 0) * p.read
                         + int(u.get("cache_creation_input_tokens") or 0) * p.write_1h
                         + int(u.get("output_tokens") or 0) * p.out) / 1e6, "billed")
    except Exception:
        pass
    p = pricing.price_for(model)
    return (pre_tokens * p.read + 8_000 * p.out) / 1e6, "estimated (warm read + 8K summary)"


def main() -> int:
    rows = []
    for d in sim.sessions(include_rewritten=True):
        path = d.session.path
        lines = path.read_text(errors="replace").splitlines(keepends=True)
        for idx in idle_boundaries(lines):
            b = json.loads(lines[idx])
            at, pre = _t(b["timestamp"]), int((b.get("compactMetadata") or {}).get("preTokens") or 0)
            # Only this compaction's stretch: from the start of its history to the next one
            # (earlier and later parts are the same either way).
            bounds = [i for i, ln in enumerate(lines) if '"compact_boundary"' in ln and _is_boundary(ln)]
            lo = max([i for i in bounds if i < idx], default=-1) + 1
            hi = min([i for i in bounds if i > idx], default=len(lines))
            part = lines[lo:hi]
            tmp_real = Path(tempfile.mkdtemp()) / path.name
            tmp_real.write_text("".join(part))
            tmp = Path(tempfile.mkdtemp()) / path.name
            tmp.write_text("".join(without(part, idx - lo)))
            s_real = SimpleNamespace(path=tmp_real, primary_model=d.session.primary_model)
            s_off = SimpleNamespace(path=tmp, primary_model=d.session.primary_model)
            mk = lambda s: SimpleNamespace(session=s, price=d.price, stem=path.stem,
                                           through_ts=getattr(d, "through_ts", False))
            real = benchmark.replay(mk(s_real), count, 8, 900, 70, bg.TOKENS_PER_SIGNATURE_CHAR)
            off = benchmark.replay(mk(s_off), count, 8, 900, 70, bg.TOKENS_PER_SIGNATURE_CHAR)
            sc, how = summary_cost(path.stem, at, pre, d.session.primary_model)
            rows.append((path.stem[:8], time.strftime("%m-%d %H:%M", time.localtime(at)), pre,
                         real["ours_cost"] + sc, off["ours_cost"], sc, how,
                         real.get("compactions", 0), off.get("compactions", 0)))
            print("  done:", rows[-1][:2], flush=True)
    if not rows:
        print("no idle compactions found in this machine's transcripts")
        return 0
    print(f"{'session':9} {'when':12} {'context':>8} {'with idle compaction':>21} {'without':>9}  summary request")
    for s, when, pre, a, b, sc, how, ca, cb in rows:
        print(f"{s:9} {when:12} {pre:>8,} {a:>20,.2f}$ {b:>8,.2f}$  ${sc:.2f} {how}; TokenSaver trims {ca} vs {cb}")
    a = sum(r[3] for r in rows)
    b = sum(r[4] for r in rows)
    print(f"\n{len(rows)} idle compactions; from each one's history start to the next, as TokenSaver runs: with ${a:,.2f}, without ${b:,.2f} "
          f"({'idle compaction saved' if a < b else 'idle compaction cost'} ${abs(a - b):,.2f})")
    print("local size estimate on both sides (no API calls); answer quality not modelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
