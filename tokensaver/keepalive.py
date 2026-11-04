"""Keep a session's cache warm through daytime breaks ("bumps").

Anthropic drops a cached prompt an hour after the START of the last request
that used it, and the next request then writes the whole context again at the
1-hour price (Opus 5.5: $8/M) instead of reading it ($0.20/M): 40x. A cache
read refreshes the entry, and Anthropic documents pre-warming with
`max_tokens: 0` (nothing generated, nothing added to the conversation), "on a
scheduled interval" too. So shortly before the hour is up the proxy re-sends
the session's last request with `max_tokens: 0`: one read, 2.5% of a rewrite,
and the cache lives another hour.

It pays only if the user comes back before the cache would have died anyway.
The rule (FREE_BUMPS, MAX_BUMPS): in a big or returned-to session, two bumps
after any break, more only before this user's usual end of day (`stop_hour`,
learned from their own history), never through the night. Replayed on 29 days
and 115 breaks: +6.9% of the bill (experiments/keepalive.py).

A kept-warm return is warm for the proxy too (`last_warm`), so break-time
compaction still happens only at its own swap points, as on any warm request.

Safety (each from a review of the first version):
- the login stays in this process's memory only, for a session that may be
  bumped, at most HOLD_MAX_S, and is dropped on a newer login, 401/403, the
  day-end stop, or when the session closes;
- only interactive Claude Code sessions (`cli` / IDE entrypoints), only
  api.anthropic.com, only 1-hour caches, only known models;
- each send re-checks the clock and stops HARD_STOP_S before the hour; no
  retry after a timeout (it may have landed); 429/5xx stop the break; no bump
  when the last real response shows little rate-limit headroom;
- every bump is checked (a read about the cached size is a hit); a miss sent
  on time moves this user's timing earlier once, a second turns it off; any
  output token turns it off (an upstream ignored `max_tokens: 0`).
Results go to the vault (`cache_bump`) and `keepalive.json`; a rolling net
below zero turns it off.

Setting `cache_keepalive`: auto | on | off. Never raises into the proxy.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import config, pricing, vault

H = 3600
# When: the later the better (a return before the bump needs none, and each bump covers an
# hour from its own start). Replayed: 45 min $64.97, 50 $66.51, 55 $68.28, 58 $68.82,
# 59.5 $69.05 net over 29 days. 58 min keeps 2 min of margin against ~6 s of known delay
# (5 s scheduling + the send); it pays unless late bumps exceed ~1 in 300.
BUMP_AFTER_S = 58 * 60
BACKOFF_AFTER_S = 55 * 60    # after one on-time miss for this user
HARD_STOP_S = 60             # never send within this of the hour (or 3x the upload time, if longer)
UPLOAD_BPS = 1_000_000       # assumed upload speed for that estimate, bytes/s (conservative)
# How many (experiments/keepalive.py, 29 days, 115 breaks; same accounting for all):
#   2 bumps after any break, plus up to 4 more while before the usual day end,
#   only in a big or returned-to session:                               +6.9% of the bill
#   day end only, max 8, any session:                                  +5.3%
#   bumping all night (20-24 per break):                         +0.1% / -0.7%
# Most returns come 1-2.6 h later, which 2 bumps cover; a next-morning return needs 16-30
# bumps (40-75% of the rewrite) and even the biggest sessions are picked up the next
# morning only about half the time, so nights lose every way it was sliced.
FREE_BUMPS = 2
MAX_BUMPS = 6
BIG_REQUESTS = 50            # "big": this many requests, returned to after a break before,
BIG_TOKENS = 150_000         # or this much context (a session first seen after a restart)
MIN_SAVING_USD = 0.01        # a warm return worth less than this is not worth a request
WARMUP_DAYS = 5              # past days of use before the day end is trusted
WEEKDAY_MIN = 4              # same-weekday samples before that weekday's own day end is used
DAY_END_Q = 0.75             # p75 of past day ends, minus MARGIN_H
MARGIN_H = 1.0
DAY_CUTOFF_H = 4             # activity before 4 am counts toward the previous day
HOLD_MAX_S = 12 * H          # a login is never held longer than this
OPEN_POLL_S = 120            # the status line polls an open session every 5-10 s
RATE_HEADROOM = 0.2          # no bump with less than this share of a rate limit left
DAILY_CAP_USD = 2.0          # a rail against bugs: at least this, or 2% of the recent daily bill
DAILY_CAP_SHARE = 0.02
HIT_SHARE = 0.9              # a bump read at least this share of the cached size and wrote
WRITE_SHARE = 0.1            # under this share: the cache was alive
LEARN_EVERY_S = 6 * H
AUTO_OFF_MIN = 10            # bumped breaks before the rolling net can turn it off
AUTO_OFF_DAYS = 14
UPSTREAM_OK = ("https://api.anthropic.com",)
STATE = vault.VAULT_DIR / "keepalive.json"
# `auto` turns on only once a real bump has been seen to work (accepted with Claude Code's own
# settings, read the cache, and the next request found it warm). Until then auto = off.
VERIFIED = False
TICK_S = 5
MAX_SESSIONS = 20            # held at once (each holds its last request, a few MB)
SEND_WORKERS = 4

_LOCK = threading.Lock()
_SESSIONS: dict[str, dict] = {}     # in memory only (holds auth)
_COUNTS: dict[str, dict] = {}       # per session: requests seen, returned after a break before
_WARM: dict[str, float] = {}        # per session: start of the last bump that kept the cache alive
_POLLED: dict[str, float] = {}      # per session: last status-line poll (the session is open)
_CRED = [None]                      # the newest login seen on a main request
_STARTED = [False]
_POOL = ThreadPoolExecutor(max_workers=SEND_WORKERS, thread_name_prefix="keepalive")
UPSTREAM = [config.get("upstream")]
_TTL_1H = re.compile(rb'"ttl"\s*:\s*"1h"')
_CC = re.compile(rb'"cache_control"')


# ---- the learned day end ---------------------------------------------------------

def _day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(t - DAY_CUTOFF_H * H))


def _hour(t: float) -> float:
    lt = time.localtime(t - DAY_CUTOFF_H * H)
    return lt.tm_hour + lt.tm_min / 60 + DAY_CUTOFF_H      # 4 am = 4.0, 1 am the next night = 25.0


def _weekday(t: float) -> int:
    return time.localtime(t - DAY_CUTOFF_H * H).tm_wday


def day_ends(times) -> dict[str, tuple[int, float]]:
    """{day: (weekday, hour of the last activity)} from request times."""
    out: dict[str, tuple[int, float]] = {}
    for t in times:
        d = _day(t)
        if d not in out or _hour(t) > out[d][1]:
            out[d] = (_weekday(t), _hour(t))
    return out


def stop_hour(ends: dict, t: float) -> float | None:
    """When extra bumps stop on the day of `t`, from earlier days only; None before the warm-up."""
    today = _day(t)
    past = {d: v for d, v in ends.items() if d < today}
    if len(past) < WARMUP_DAYS:
        return None
    same = sorted(h for w, h in past.values() if w == _weekday(t))
    hours = same if len(same) >= WEEKDAY_MIN else sorted(h for _, h in past.values())
    return hours[int(DAY_END_Q * (len(hours) - 1))] - MARGIN_H


def interactive(user_agent: str | None) -> bool:
    """Claude Code in a terminal or an IDE; not `claude -p`, the SDKs or CI ("sdk-*")."""
    m = re.search(r"\(external,\s*([^)]+)\)", user_agent or "")
    entry = (m.group(1).strip().lower() if m else "")
    return entry == "cli" or any(k in entry for k in ("vscode", "jetbrains", "ide", "desktop"))


def learn() -> dict:
    """Day ends and the recent daily bill from Claude Code's transcripts (every
    interactive session, any route). Small: kept in STATE."""
    from . import sim, transcripts
    times, spend = [], {}
    forks = sim.forks()
    for s in transcripts.load():
        if s.path.stem in forks:
            continue
        p = pricing.price_for(s.primary_model)
        for r in s.requests:
            t = sim._ts(r.timestamp)
            times.append(t)
            if p:
                spend[_day(t)] = spend.get(_day(t), 0.0) + (
                    r.inp * p.inp + r.read * p.read + r.write_1h * p.write_1h + r.write_5m * p.write_5m
                    + r.out * p.out) / 1e6
    ends = day_ends(times)
    recent = sorted(spend)[-7:]
    with _LOCK:
        st = load_state()
        st.update(ends={d: list(v) for d, v in sorted(ends.items())[-90:]},
                  daily_bill=round(sum(spend[d] for d in recent) / len(recent), 2) if recent else 0.0,
                  learned_at=time.time())
        _save_state(st)
    return st


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(st: dict) -> None:
    try:
        STATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(STATE)
    except OSError:
        pass
    _CACHED.update(at=0.0)


def _update_state(**kw) -> dict:
    with _LOCK:
        st = load_state()
        st.update(kw)
        _save_state(st)
        return st


# ---- on or off ------------------------------------------------------------------

_CACHED = {"at": 0.0, "st": {}}


def _state_cached() -> dict:
    """The state, re-read at most every few seconds (enabled() runs on every main request)."""
    if time.time() - _CACHED["at"] > 30:
        _CACHED.update(at=time.time(), st=load_state())
    return _CACHED["st"]


def enabled(st: dict | None = None) -> tuple[bool, str]:
    mode = config.get("cache_keepalive")
    if mode == "off":
        return False, "off (you turned it off)"
    st = _state_cached() if st is None else st
    if st.get("auto_off") and mode != "on":
        return False, f"off by itself: {st['auto_off']}"
    if mode == "on":
        return True, "on"
    if not VERIFIED:
        return False, "off until it is verified on a real break (`tokensaver keepalive on` to try it)"
    days = len(st.get("ends") or {})
    if days < WARMUP_DAYS:
        return True, f"on (auto; {FREE_BUMPS} bumps per break while it learns your hours: {days} of {WARMUP_DAYS} days)"
    return True, "on (auto)"


def bump_after(st: dict | None = None) -> int:
    return BACKOFF_AFTER_S if (st or _state_cached()).get("backoff") else BUMP_AFTER_S


def describe() -> str:
    st = load_state()
    on, why = enabled(st)
    r = results(st)
    tail = (f"; kept warm {r['kept']} breaks, saved ${r['saved']:.2f}, spent ${r['spent']:.2f} "
            f"(${r['wasted']:.2f} on breaks with no return)" if r["bumps"] else "")
    stop = stop_hour({d: tuple(v) for d, v in (st.get("ends") or {}).items()}, time.time())
    when = (f", extra bumps stop at {int(stop) % 24:02d}:{int(stop % 1 * 60):02d} today"
            if on and stop is not None else "")
    return why + when + tail


def results(st: dict | None = None, now: float | None = None) -> dict:
    """Rolling AUTO_OFF_DAYS totals from the state's ledger of finished breaks."""
    st = load_state() if st is None else st
    now = now or time.time()
    rows = [r for r in st.get("breaks") or [] if now - r["t"] < AUTO_OFF_DAYS * 86400]
    return {"bumps": sum(r["bumps"] for r in rows), "kept": sum(1 for r in rows if r["kept"]),
            "saved": sum(r["saved"] for r in rows), "spent": sum(r["spent"] for r in rows),
            "wasted": sum(r["spent"] for r in rows if not r["kept"]), "breaks": len(rows)}


def _settle(st: dict, rec: dict) -> None:
    """Add one finished break and apply the auto-off rule."""
    st.setdefault("breaks", []).append(rec)
    st["breaks"] = [r for r in st["breaks"] if time.time() - r["t"] < 60 * 86400][-500:]
    r = results(st)
    if r["breaks"] >= AUTO_OFF_MIN and r["saved"] < r["spent"]:
        st["auto_off"] = (f"spent ${r['spent']:.2f} to save ${r['saved']:.2f} over "
                          f"{AUTO_OFF_DAYS} days ({r['bumps']} bumps)")


# ---- what the proxy tells this module -----------------------------------------------

def last_warm(session: str | None) -> float | None:
    """Start of the last bump that kept this session's cache alive (the proxy's
    cold/warm test uses it: a kept-warm return is warm)."""
    return _WARM.get(session or "")


def polled(session: str) -> None:
    """The status line asked about this session: it is open in a terminal or IDE."""
    _POLLED[session] = time.time()


def _cred(headers: dict) -> str | None:
    for k, v in headers.items():
        if k.lower() in ("authorization", "x-api-key"):
            return v
    return None


def _headroom_ok(rl: dict) -> bool:
    """False when any `anthropic-ratelimit-*-remaining` is under RATE_HEADROOM of its limit."""
    for k, v in (rl or {}).items():
        k = k.lower()
        if k.endswith("-remaining"):
            limit = rl.get(k[:-len("remaining")] + "limit") or rl.get(k.replace("-remaining", "-limit"))
            try:
                if limit and float(v) < RATE_HEADROOM * float(limit):
                    return False
            except (TypeError, ValueError):
                continue
    return True


def eligible(path: str, sent: bytes, headers: dict, usage: dict, model: str | None,
             rate_limits: dict | None = None) -> str | None:
    """None if this request's session may be bumped, else why not."""
    if not interactive(next((v for k, v in headers.items() if k.lower() == "user-agent"), "")):
        return "not an interactive Claude Code session"
    if not any(UPSTREAM[0].rstrip("/").startswith(u) for u in UPSTREAM_OK):
        return "upstream is not api.anthropic.com"
    r = pricing.resolve(model or "")
    if r.price is None or getattr(r, "assumed", False) or not r.price.read:
        return "unknown model (its prices decide whether a bump pays)"
    n_cc = len(_CC.findall(sent))
    if not n_cc or len(_TTL_1H.findall(sent)) != n_cc:
        return "not a 1-hour cache"
    cc = usage.get("cache_creation") or {}
    if int(usage.get("cache_read_input_tokens") or 0) == 0 and not int(cc.get("ephemeral_1h_input_tokens") or 0):
        return "no 1-hour cache in use"
    if not _headroom_ok(rate_limits or {}):
        return "rate limit nearly used"
    return None


def remember(session: str | None, path: str, sent: bytes | None, headers: dict, started: float,
             usage: dict, model: str | None, rate_limits: dict | None = None) -> None:
    """After a main request went through: settle the previous break, and keep what
    a bump needs (memory only). Never raises."""
    try:
        if not session or not sent:
            return
        size = sum(int(usage.get(k) or 0) for k in
                   ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        cred = _cred(headers)
        with _LOCK:
            if cred and _CRED[0] not in (None, cred):
                # A different login (another account or key): nothing held may be used again.
                for k in list(_SESSIONS):
                    _SESSIONS[k]["headers"], _SESSIONS[k]["dropped"] = {}, "a different login is in use"
            _CRED[0] = cred or _CRED[0]
            prev = _SESSIONS.get(session)
            c = _COUNTS.setdefault(session, {"reqs": 0, "returned": False, "last": None})
            if c["last"] is not None and started - c["last"] > H:
                c["returned"] = True
            c["reqs"] += 1
            c["last"] = started
            if len(_COUNTS) > 500:
                for k in sorted(_COUNTS, key=lambda k: _COUNTS[k]["last"] or 0)[:100]:
                    _COUNTS.pop(k, None)
            big = c["reqs"] >= BIG_REQUESTS or c["returned"] or size >= BIG_TOKENS
            _WARM.pop(session, None)          # a real request: the proxy's own clock takes over
        if prev and prev.get("bumps"):
            read = int(usage.get("cache_read_input_tokens") or 0)
            _record_break(session, prev, started, kept=read >= HIT_SHARE * prev["size"])
        on, _ = enabled()
        p = pricing.price_for(model or "")
        why = (None if on else "off") or (None if big else "not big") or (
            None if p and size * (p.write_1h - p.read) / 1e6 >= MIN_SAVING_USD else "too small") or \
            eligible(path, sent, headers, usage, model, rate_limits)
        if why:
            with _LOCK:
                _SESSIONS.pop(session, None)
            return
        with _LOCK:
            if len(_SESSIONS) >= MAX_SESSIONS and session not in _SESSIONS:
                oldest = min(_SESSIONS, key=lambda k: _SESSIONS[k]["started"])
                _SESSIONS.pop(oldest)
            _SESSIONS[session] = {"path": path, "sent": sent, "headers": dict(headers), "started": started,
                                  "size": size, "model": model or "", "bumps": 0, "spent": 0.0,
                                  "last_bump": None, "misses": 0, "inflight": False}
        if not _STARTED[0]:
            _STARTED[0] = True
            threading.Thread(target=_loop, daemon=True, name="keepalive").start()
    except Exception:
        pass


def _record_break(session: str, prev: dict, t: float, kept: bool) -> None:
    p = pricing.price_for(prev["model"])
    saved = prev["size"] * (p.write_1h - p.read) / 1e6 if kept and p else 0.0
    rec = {"t": t, "session": session[:8], "bumps": prev["bumps"], "spent": round(prev["spent"], 4),
           "saved": round(saved, 4), "kept": kept, "misses": prev["misses"], "size": prev["size"]}
    with _LOCK:
        st = load_state()
        _settle(st, rec)
        _save_state(st)
    try:
        vault.record("cache_bump", tool="proxy", session_id=session, index=False,
                     summary=f"break settled: {'kept warm' if kept else 'not kept'}; {prev['bumps']} bumps, "
                             f"spent ${prev['spent']:.3f}, saved ${saved:.2f}", meta=rec)
    except Exception:
        pass


def _open(session: str) -> bool:
    """Is the session still open? The status line's polls say so precisely; without
    them (status line off), a running `claude` process at least; cannot tell = no."""
    seen = _POLLED.get(session)
    if seen is not None:
        return time.time() - seen < OPEN_POLL_S
    try:
        r = subprocess.run(["pgrep", "-x", "claude"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def due(s: dict, now: float, ends: dict, spent_today: float, cap: float, after_s: int | None = None) -> str | None:
    """None if this session should be bumped now, else why not."""
    if s.get("dropped"):
        return s["dropped"]
    if s.get("inflight"):
        return "not yet"
    if s["bumps"] >= MAX_BUMPS:
        return f"{MAX_BUMPS} bumps this break"
    if now - s["started"] > HOLD_MAX_S:
        return "held too long"
    base = s["last_bump"] or s["started"]
    if now - base < (after_s or bump_after()):
        return "not yet"
    if now - base > H - _stop_margin(s):
        return "too late: the cache may be gone"
    if s["bumps"] >= FREE_BUMPS:
        # Past the first two: only while before this user's usual end of the day.
        stop = stop_hour(ends, s["started"])
        if stop is None:
            return "still learning this user's hours"
        if _day(now) != _day(s["started"]) or _hour(now) >= stop:
            return "past the usual end of the day"
    if spent_today >= cap:
        return f"daily cap ${cap:.2f} reached"
    return None


def _stop_margin(s: dict) -> float:
    """Never send this close to the hour: at least HARD_STOP_S, or 3x the time to upload the body."""
    return max(HARD_STOP_S, 3 * len(s.get("sent") or b"") / UPLOAD_BPS)


def bump(session: str, s: dict) -> dict:
    """Send the session's last request again with max_tokens 0. Returns what happened."""
    body = json.loads(s["sent"])
    body["max_tokens"] = 0
    body["stream"] = False
    data = json.dumps(body, ensure_ascii=False).encode()
    headers = {k: v for k, v in s["headers"].items() if k.lower() not in ("content-length", "host")}
    headers["Content-Length"] = str(len(data))
    req = urllib.request.Request(UPSTREAM[0].rstrip("/") + s["path"], data=data, headers=headers, method="POST")
    t = time.time()
    base = s["last_bump"] or s["started"]
    if t - base > H - _stop_margin(s):                 # re-checked at the moment of sending
        return {"status": -1, "error": "too late at send time", "t": t}
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            usage = json.loads(r.read()).get("usage") or {}
    except urllib.error.HTTPError as exc:
        try:
            err = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            err = ""
        return {"status": exc.code, "error": err, "t": t, "retry_after": exc.headers.get("retry-after")}
    except Exception as exc:
        return {"status": 0, "error": f"{type(exc).__name__}: {exc}"[:200], "t": t}
    read = int(usage.get("cache_read_input_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    p = pricing.price_for(s["model"])
    cost = (read * p.read + write * p.write_1h + int(usage.get("input_tokens") or 0) * p.inp + out * p.out) / 1e6
    hit = read >= HIT_SHARE * s["size"] and write < WRITE_SHARE * s["size"]
    return {"status": 200, "t": t, "read": read, "write": write, "output": out, "cost": cost, "hit": hit,
            "on_time": t - base < bump_after() + 30}


def _apply(session: str, s: dict, res: dict) -> None:
    """Fold one bump's result into the session and the user's state."""
    status = res.get("status")
    stop = None
    if status == 200:
        s["bumps"] += 1
        s["last_bump"] = res["t"]
        s["spent"] += res["cost"]
        _add_today(res["cost"])
        if res.get("output"):
            _update_state(auto_off="the upstream generated output for a max_tokens 0 request")
            stop = "output generated"
        elif res["hit"]:
            _WARM[session] = res["t"]
        else:
            s["misses"] += 1
            stop = "the cache was already gone"
            st = load_state()
            if res.get("on_time"):
                if st.get("backoff"):
                    _update_state(auto_off="a bump sent on time found the cache gone, twice")
                else:
                    _update_state(backoff=time.time())       # bump earlier from now on
    elif status in (401, 403):
        stop = f"login refused ({status})"
    elif status == 400:
        stop = "bump refused (400)"
        st = load_state()
        refused = [t for t in st.get("refused") or [] if time.time() - t < AUTO_OFF_DAYS * 86400] + [time.time()]
        _update_state(refused=refused, **({"auto_off": f"Anthropic refused bumps twice: {res.get('error', '')[:120]}"}
                                         if len(refused) >= 2 else {}))
    elif status in (0, -1) or status == 429 or (isinstance(status, int) and status >= 500):
        stop = f"stopped for this break ({status}: {str(res.get('error'))[:60]})"   # never retried
    if stop:
        s["dropped"], s["headers"] = stop, {}
    s["inflight"] = False
    with _LOCK:
        replaced = _SESSIONS.get(session) is not s
    if replaced and status == 200:
        _add_late_cost(session, res["cost"])
    try:
        vault.record("cache_bump", tool="proxy", session_id=session, index=False,
                     summary=f"bump {status}: " + (
                         f"read {res.get('read', 0):,}, wrote {res.get('write', 0):,}, ${res.get('cost', 0):.3f}"
                         + ("" if res.get("hit") else " (MISS: the cache was gone)")
                         if status == 200 else str(res.get("error"))[:160]),
                     meta={k: v for k, v in res.items() if k != "error"})
    except Exception:
        pass


def _add_today(cost: float) -> None:
    with _LOCK:
        st = load_state()
        today = st.get("today") or {}
        if today.get("day") != _day(time.time()):
            today = {"day": _day(time.time()), "spent": 0.0}
        today["spent"] = round(float(today["spent"]) + cost, 4)
        st["today"] = today
        _save_state(st)


def _add_late_cost(session: str, cost: float) -> None:
    """A bump that landed after the session's next request: its cost joins that break."""
    with _LOCK:
        st = load_state()
        for r in reversed(st.get("breaks") or []):
            if r["session"] == session[:8]:
                r["spent"] = round(r["spent"] + cost, 4)
                break
        _save_state(st)


def tick(now: float | None = None, send=bump, is_open=None, wait: bool = False) -> list[str]:
    """One pass over the remembered sessions. Returns what it did (tests, logs)."""
    is_open = is_open or _open
    now = now or time.time()
    st = load_state()
    on, why = enabled(st)
    try:
        from . import routing
        rs = routing.state()
        if on and rs.get("managed") and not rs.get("on"):
            on, why = False, "TokenSaver is off or paused"
    except Exception:
        pass
    with _LOCK:
        items = list(_SESSIONS.items())
    if not on:
        with _LOCK:
            _SESSIONS.clear()             # nothing held while off
        return [f"off: {why}"] if items else []
    ends = {d: tuple(v) for d, v in (st.get("ends") or {}).items()}
    today = st.get("today") or {}
    spent_today = float(today.get("spent") or 0) if today.get("day") == _day(now) else 0.0
    cap = max(DAILY_CAP_USD, DAILY_CAP_SHARE * float(st.get("daily_bill") or 0))
    after_s = bump_after(st)
    done, futures = [], []
    for session, s in items:
        if s["bumps"] and not s.get("inflight") and now > s["last_bump"] + H:
            _forget(session)                     # nobody came back while it was warm: wasted
            _record_break(session, s, s["last_bump"], kept=False)
            continue
        if not s.get("inflight") and now - (s["last_bump"] or s["started"]) > H:
            _forget(session)                     # the cache is gone either way: hold nothing
            continue
        why_not = due(s, now, ends, spent_today, cap, after_s)
        if why_not == "not yet":
            continue
        if why_not is None and not is_open(session):
            why_not = "the session is closed"
        if why_not:
            s["headers"], s["dropped"] = {}, s.get("dropped") or why_not
            continue
        s["inflight"] = True
        futures.append(_POOL.submit(lambda k=session, v=s: _apply(k, v, send(k, v))))
        done.append(session[:8])
    if wait:
        for f in futures:
            f.result()
    return done


def _forget(session: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session, None)


forget = _forget


def held() -> list[dict]:
    """For the health check: which sessions may be bumped and when next. No login data."""
    with _LOCK:
        items = list(_SESSIONS.items())
    return [{"session": k[:8], "size": s["size"], "bumps": s["bumps"], "dropped": s.get("dropped"),
             "next_bump": round((s["last_bump"] or s["started"]) + bump_after()),
             "open": (time.time() - _POLLED[k] < OPEN_POLL_S) if k in _POLLED else None} for k, s in items]


def _loop() -> None:
    while True:
        time.sleep(TICK_S)
        try:
            tick()
        except Exception:
            pass
