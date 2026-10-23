"""Replay a user's own sessions to price background-compaction settings.

The importable form of experiments/compaction_net.py, for `tune`. For each
real session (test forks excluded) it replays the billed requests and, at
every idle gap longer than `after_s` with more than `min_tokens` of context,
applies that session's compaction shrink. The shrink comes from
`background.build` on the session's own history, which is deterministic and
needs no model. It then charges the re-reads compaction would cause: a file
view cut by the compaction that Claude later edits or names, without having
re-read it in the real session, is re-read in full and carried to the end.

All local; no API calls. The shrink is measured in characters and scaled to
tokens by a factor the token audit measured (`TOKENS_PER_ESTIMATED_SAVED`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import background, passes
from .pricing import resolve
from .transcripts import DEFAULT_ROOT, Session, load


def _ts(x: str) -> float:
    return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


def _first_uuid(path: Path) -> str | None:
    try:
        with path.open(errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("uuid"):
                    return rec["uuid"]
    except OSError:
        pass
    return None


def forks(root: Path = DEFAULT_ROOT) -> set[str]:
    """Session files that copy another one's history (forks, test copies): the
    first record's uuid matches an older file's."""
    files = sorted(root.glob("*/*.jsonl"), key=lambda p: os.stat(p).st_birthtime
                   if hasattr(os.stat(p), "st_birthtime") else os.stat(p).st_mtime)
    owner: dict[str, str] = {}
    out: set[str] = set()
    for p in files:
        u = _first_uuid(p)
        if u is None:
            continue
        if u in owner:
            out.add(p.stem)
        else:
            owner[u] = p.stem
    return out


@dataclass
class SessionData:
    session: Session
    requests: list
    price: object
    _blocks: list | None = field(default=None, repr=False)

    @property
    def blocks(self):
        if self._blocks is None:
            self._blocks = passes.from_transcript(self.session.path)
        return self._blocks

    @property
    def stem(self) -> str:
        return self.session.path.stem


def rewritten() -> set[str]:
    """Sessions whose requests the proxy rewrote. Their transcripts record
    usage of the REWRITTEN requests (compactions included), so they are not a
    plain baseline: replaying compaction over them would count it twice."""
    import sqlite3
    from . import vault
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        rows = conn.execute("SELECT DISTINCT session_id FROM events WHERE kind='api_request' "
                            "AND json_extract(meta, '$.rewrite.bytes_out') IS NOT NULL").fetchall()
        conn.close()
        return {r[0] for r in rows if r[0]}
    except Exception:
        return set()


def sessions(root: Path = DEFAULT_ROOT) -> list[SessionData]:
    fk = forks(root) | rewritten()
    out = []
    for s in load(root):
        if not s.requests or s.path.stem in fk:
            continue
        price = resolve(s.primary_model).price
        rq = [r for r in s.requests if r.timestamp]
        if price and rq:
            out.append(SessionData(s, rq, price))
    return out


def threshold(data: SessionData, min_tokens: int | None) -> int:
    """Context above which a compaction qualifies. None: the proxy's rule, at
    least `background.MIN_REMOVABLE` of old history beyond this session's
    system/tools floor and the recent tail kept verbatim."""
    if min_tokens is not None:
        return min_tokens
    return data.session.floor + background.KEEP_TOKENS + background.MIN_REMOVABLE


def moments(data: SessionData, after_s: float, min_tokens: int | None) -> list[int]:
    """Indexes of requests that follow an idle gap longer than `after_s` with
    more than `min_tokens` of context (before any compaction)."""
    min_tokens = threshold(data, min_tokens)
    rq = data.requests
    return [i for i in range(1, len(rq))
            if _ts(rq[i].timestamp) - _ts(rq[i - 1].timestamp) > after_s and rq[i].context > min_tokens]


def _conversation_until(path: Path, request_index: int) -> list[dict]:
    """API-shaped messages sent before the `request_index`-th assistant reply."""
    messages: list[dict] = []
    seen, n = set(), 0
    with path.open(errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict) or rec.get("isSidechain"):
                continue
            role, content = msg.get("role"), msg.get("content")
            if role not in ("user", "assistant") or content is None:
                continue
            if role == "assistant" and msg.get("id") and msg["id"] not in seen:
                seen.add(msg["id"])
                n += 1
                if n > request_index:
                    break
            items = content if isinstance(content, list) else [{"type": "text", "text": content}]
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = messages[-1]["content"] + items
            else:
                messages.append({"role": role, "content": list(items)})
    return messages


def estimate_shrink(data: SessionData, after_s: float, min_tokens: int | None) -> float | None:
    """How much of the compactable OLD part this session's compaction removes,
    (old - new) / old, from `background.build` at its first qualifying moment.
    None if it has none or nothing to cut. `price_session` applies it to the
    old part at each moment, which grows as the session does (measured: the
    trial's compactions removed 48% of the request, then 56% later on)."""
    # The first moment where a compaction can be built: like the proxy, which
    # simply tries again at the next break. (Measured: with request indexes
    # counted correctly, the first break often falls just under the history
    # minimum, and stopping there gave a $197 session no compaction at all.)
    for i in moments(data, after_s, min_tokens)[:4]:
        res = background.build(_conversation_until(data.session.path, i))
        if res and res["old_tokens"]:
            return max(0.0, 1 - res["new_tokens"] / res["old_tokens"])
    return None


@dataclass
class Result:
    stem: str
    base: float
    gross: float
    net: float
    applications: int
    rereads: int


# Measured on the first real trial (count_tokens on original vs sent): the two
# distinct compactions removed 208,648 and 265,338 real tokens where the local
# estimate (`background.size_tokens`) said 229,127 and 283,957: 0.91 and 0.93.
TOKENS_PER_ESTIMATED_SAVED = 0.92
FACTOR_BOUNDS = (0.7, 1.1)
FACTOR_MIN_COMPACTIONS = 3


def token_factor() -> tuple[float, str]:
    """The factor refitted from this user's own compactions: every request
    that applied one was counted both ways (count_tokens), so real tokens
    saved over the local estimate of tokens saved is measured, not assumed.
    Median, clamped to FACTOR_BOUNDS; the default until there are
    FACTOR_MIN_COMPACTIONS distinct ones."""
    import sqlite3
    from . import vault
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        reqs = conn.execute("SELECT id, session_id FROM events WHERE kind='api_request' "
                            "AND json_extract(meta, '$.rewrite.bg_applied') = 1").fetchall()
        auds = conn.execute("SELECT id, session_id FROM events WHERE kind='token_audit' "
                            "AND json_extract(meta, '$.reason') = 'compaction_applied'").fetchall()
        conn.close()
    except Exception:
        return TOKENS_PER_ESTIMATED_SAVED, "default (vault unreadable)"
    est = {}
    for rid, session in reqs:
        rw = vault.read_meta(rid).get("rewrite") or {}
        try:
            old, new = (int(x) for x in str(rw.get("bg_tokens")).split("->"))
        except ValueError:
            continue
        est[(session, rw.get("bytes_in"), rw.get("bytes_out"))] = (old - new, rw.get("bg_tokens"))
    ratios, seen = [], set()
    for rid, session in auds:
        m = vault.read_meta(rid)
        e = est.get((session, m.get("bytes_in"), m.get("bytes_out")))
        if not e or e[0] <= 0 or (session, e[1]) in seen:
            continue                      # one per distinct compaction
        seen.add((session, e[1]))
        ratios.append((m["tokens_original"] - m["tokens_sent"]) / e[0])
    if len(ratios) < FACTOR_MIN_COMPACTIONS:
        return TOKENS_PER_ESTIMATED_SAVED, (f"default {TOKENS_PER_ESTIMATED_SAVED} "
                                            f"({len(ratios)}/{FACTOR_MIN_COMPACTIONS} measured compactions)")
    ratios.sort()
    med = ratios[len(ratios) // 2]
    f = min(max(med, FACTOR_BOUNDS[0]), FACTOR_BOUNDS[1])
    return f, f"measured {f:.2f} (median of {len(ratios)} compactions)"


def price_session(data: SessionData, shrink: float | None, after_s: float, min_tokens: int | None,
                  keep_chars: int, pass_rate: float = 1.0,
                  payback_max: float | None = None) -> Result:
    """Cost of one session's real requests with compaction at `after_s`, net
    of the re-reads it would cause. `pass_rate` scales the shrink by how often
    the quality gate lets a compaction through.

    The rules are the proxy's:
    - a compaction is built from the ORIGINAL history, so its reduction is
      `shrink` of the original old part (context above the system/tools floor,
      minus the recent part kept verbatim); a later one only adds what goes beyond
      the reduction already in place (measured: a second compaction saved 68K
      tokens per request, not the full shrink of an already-shrunk request);
    - on a warm cache (gap up to the 61 min TTL) it applies only if its payback
      (requests needed to repay the fresh write) is within `payback_max`
      (default: config `bg_warm_payback_max`); on a cold cache always."""
    from . import config
    if payback_max is None:
        payback_max = config.get("bg_warm_payback_max")
    p, rq = data.price, data.requests
    min_tokens = threshold(data, min_tokens)
    k = (shrink or 0.0) * pass_rate * TOKENS_PER_ESTIMATED_SAVED
    floor = data.session.floor              # system + tools: never compacted
    keep = background.KEEP_TOKENS            # the recent part stays verbatim
    base = gross = net = 0.0
    red, prev, apps, rereads = 0.0, 0, 0, 0
    extra = [0] * (len(rq) + 2)
    for i, r in enumerate(rq):
        c = r.context
        if prev and c < 0.5 * prev:
            red = 0.0                          # Claude Code compacted: a new history
        prev = c
        sim = c - red
        gap = _ts(r.timestamp) - _ts(rq[i - 1].timestamp) if i else 0
        write_extra = 0.0
        if k and gap > after_s and c > min_tokens:
            new = c - max(c - floor - keep, 0) * k   # built from the original history
            if new < sim:
                warm = gap <= 61 * 60
                if not warm:
                    n = 0.0
                elif p.read > 0:
                    n = (new * p.write_1h - sim * p.read) / ((sim - new) * p.read)
                else:
                    n = float("inf")           # reads free: a fresh write never repays
                if not warm or n <= payback_max:
                    if warm:                   # the new prefix is a fresh write
                        write_extra = new * (p.write_1h - p.read) / 1e6
                    red = c - new
                    sim = new
                    apps += 1
                    rereads += _charge_rereads(data, i, keep_chars, extra, len(rq))
        cost = (r.inp * p.inp + r.read * p.read + r.write * p.write_1h + r.out * p.out) / 1e6
        base += cost
        # What is written can be no more than the (reduced) prompt: a rebuild
        # after a compaction writes the compacted prefix, not the original.
        write = min(r.write, max(sim - r.inp, 0))
        read = max(sim - write - r.inp, 0)
        g = (r.inp * p.inp + read * p.read + write * p.write_1h + r.out * p.out) / 1e6 + write_extra
        gross += g
        carried = sum(extra[:i + 1])
        net += g + carried * p.read / 1e6 + extra[i] * (p.write_1h - p.read) / 1e6
    return Result(data.stem, base, gross, net, apps, rereads)


def _charge_rereads(data: SessionData, i: int, keep_chars: int, extra: list, n_req: int) -> int:
    """Charge re-reads for file views a compaction at request i cuts (see
    module docstring). Returns how many."""
    blocks = data.blocks
    before = [b for b in blocks if b.request < i + 1]
    after = [b for b in blocks if b.request >= i + 1]
    acc, cut = 0, len(before)
    for j in range(len(before) - 1, -1, -1):
        acc += len(before[j].text)
        if acc > keep_chars:
            cut = j
            break
    old_views, recent = {}, set()
    for j, b in enumerate(before):
        v = passes.view_of(b) if b.type == "tool_result" else None
        if not v:
            continue
        if j < cut and len(b.original) > background.EXTRACT_RESULT_CHARS:
            old_views[v[0]] = len(b.original)
        elif j >= cut:
            recent.add(v[0])
    n = 0
    for path, size in old_views.items():
        if path in recent:
            continue
        name = os.path.basename(path)
        for b in after:
            touched = None
            if b.type == "tool_use" and isinstance(b.input, dict):
                fp = b.input.get("file_path") or ""
                if b.tool == "Read" and fp.endswith(name):
                    touched = "read"
                elif b.tool in ("Edit", "Write", "MultiEdit") and fp.endswith(name):
                    touched = "edit"
            elif b.type == "tool_result" and passes.view_of(b) and passes.view_of(b)[0] == path:
                touched = "read"
            elif b.role == "assistant" and b.type == "text" and name in b.text:
                touched = "mention"
            if touched:
                if touched != "read":
                    at = min(max(b.request - 1, i + 1), n_req)
                    extra[at] += size // 4
                    n += 1
                break
    return n


def bill_saving(results: list[Result]) -> tuple[float, float]:
    """(gross, net) saving as a fraction of the bill."""
    base = sum(r.base for r in results)
    if not base:
        return 0.0, 0.0
    return 1 - sum(r.gross for r in results) / base, 1 - sum(r.net for r in results) / base
