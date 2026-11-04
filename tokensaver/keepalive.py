"""Keep a session's cache warm through daytime breaks ("bumps").

Anthropic drops a cached prompt an hour after it was last used, and the next
request then writes the whole context again at the 1-hour price (Opus 5.5:
$8/M) instead of reading it ($0.20/M): 40x. A cache read refreshes the entry,
and Anthropic documents pre-warming with `max_tokens: 0` (nothing generated,
nothing added to the conversation), "on a scheduled interval" too. So, about
50 minutes after a session's last request, the proxy can re-send that exact
request with `max_tokens: 0`: one read, 2.5% of a rewrite, and the cache
lives another hour.

It pays only if the user comes back to that session before the cache would
have died anyway. The rule (see FREE_BUMPS): two bumps after any break in a
big or returned-to session, a few more only before this user's usual end of
day, learned from their own history (`stop_hour`), never through the night.
Replayed on 29 days, 115 breaks: +6.9% of the bill (experiments/keepalive.py).

A warm return changes nothing else: break-time compaction still happens at
its own swap points (the payback rule decides, as on any warm request).

Credentials: the last main request's headers (with its login) stay in this
process's memory only, never on disk, only for a session that may be bumped,
and are dropped at the day-end stop, after MAX_BUMPS, on any 401/403, or
when the session is evicted. Every bump is checked: a read about the cached
size is a hit; a large write means the cache was already gone (we paid a
write), and that session is not bumped again. Results go to the vault
(`cache_bump`), and a rolling net below zero turns it off by itself.

Setting `cache_keepalive`: auto (default: after a few days of history to
learn from) | on | off. Never raises into the proxy.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import config, pricing, vault

H = 3600
BUMP_AFTER_S = 50 * 60       # the hour runs from the START of the last request; long replies eat into it
# The rule (experiments/keepalive.py, 29 days, 115 breaks; same accounting for all):
#   2 bumps after any break, at any hour, plus up to 4 more while before the usual day end,
#   only for a session that is big or was returned to before:          +6.9% of the bill
#   the earlier rule (day end only, max 8, any session):                 +5.3%
#   bumping all night (20-24 per break):                         +0.1% / -0.7%
# Most returns come 1-2.6 h later, which 2 bumps cover; a next-morning return needs 16-30
# bumps (40-75% of the rewrite) and only about half of even the biggest sessions are picked
# up the next morning, so nights lose every way it was sliced.
FREE_BUMPS = 2               # per break, whatever the hour
MAX_BUMPS = 6                # per break, the rest only before the usual day end
BIG_REQUESTS = 50            # "big": this many requests, or returned to after a break before,
BIG_TOKENS = 150_000         # or this much context (a session first seen after a proxy restart)
MIN_TOKENS = 20_000          # below this a rewrite is cents
WARMUP_DAYS = 5              # past days of use before auto bumps at all
WEEKDAY_MIN = 4              # same-weekday samples before the weekday's own day end is used
DAY_END_Q = 0.75             # the measured best variant: p75 of past day ends, minus MARGIN_H
MARGIN_H = 1.0
DAY_CUTOFF_H = 4             # activity before 4 am counts toward the previous day
DAILY_CAP_USD = 2.0          # spent on bumps per day, at least; or 2% of the recent daily bill
DAILY_CAP_SHARE = 0.02
HIT_SHARE = 0.9              # a bump read at least this share of the cached size: kept warm
LEARN_EVERY_S = 6 * H
AUTO_OFF_MIN = 10            # bumps before the rolling net can turn it off
AUTO_OFF_DAYS = 14
STATE = vault.VAULT_DIR / "keepalive.json"
# `auto` turns on only once a real bump has been seen to work (accepted with Claude Code's own
# settings, read the cache, and the next request found it warm). Until then auto = off, and `on`
# is the way to try it.
VERIFIED = False
TICK_S = 30
MAX_SESSIONS = 20            # held in memory at once (each holds its last request, a few MB)

_LOCK = threading.Lock()
_SESSIONS: dict[str, dict] = {}     # in memory only (holds auth)
_COUNTS: dict[str, dict] = {}       # per session: requests seen, returned after a break before
_STARTED = [False]
UPSTREAM = [config.get("upstream")]


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
    """When bumping stops on the day of `t`, from earlier days only; None before the warm-up."""
    today = _day(t)
    past = {d: v for d, v in ends.items() if d < today}
    if len(past) < WARMUP_DAYS:
        return None
    same = sorted(h for w, h in past.values() if w == _weekday(t))
    hours = same if len(same) >= WEEKDAY_MIN else sorted(h for _, h in past.values())
    return hours[int(DAY_END_Q * (len(hours) - 1))] - MARGIN_H


def learn() -> dict:
    """Day ends and the recent daily bill from Claude Code's transcripts (every
    session, any route). Small: kept in STATE."""
    from . import transcripts, sim
    times, spend = [], {}
    for s in transcripts.load():
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


# ---- on or off ------------------------------------------------------------------

_CACHED = {"at": 0.0, "st": {}}


def _state_cached() -> dict:
    """The state, re-read at most every TICK_S (enabled() runs on every main request)."""
    if time.time() - _CACHED["at"] > TICK_S:
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


def describe() -> str:
    st = load_state()
    on, why = enabled(st)
    r = results(st)
    tail = (f"; kept warm {r['kept']} breaks, saved ${r['saved']:.2f}, spent ${r['spent']:.2f} "
            f"(${r['wasted']:.2f} on breaks with no return)" if r["bumps"] else "")
    stop = stop_hour({d: tuple(v) for d, v in (st.get("ends") or {}).items()}, time.time())
    when = f", stops at {int(stop) % 24:02d}:{int(stop % 1 * 60):02d} today" if on and stop is not None else ""
    return why + when + tail


def results(st: dict | None = None, now: float | None = None) -> dict:
    """Rolling AUTO_OFF_DAYS totals from the state's ledger of finished breaks."""
    st = load_state() if st is None else st
    now = now or time.time()
    rows = [r for r in st.get("breaks") or [] if now - r["t"] < AUTO_OFF_DAYS * 86400]
    return {"bumps": sum(r["bumps"] for r in rows), "kept": sum(1 for r in rows if r["kept"]),
            "saved": sum(r["saved"] for r in rows), "spent": sum(r["spent"] for r in rows),
            "wasted": sum(r["spent"] for r in rows if not r["kept"]),
            "misses": sum(r.get("misses", 0) for r in rows)}


def _settle(st: dict, rec: dict) -> None:
    """Add one finished break and apply the auto-off rule."""
    st.setdefault("breaks", []).append(rec)
    st["breaks"] = [r for r in st["breaks"] if time.time() - r["t"] < 60 * 86400][-500:]
    r = results(st)
    if r["misses"] >= 2:
        st["auto_off"] = f"{r['misses']} bumps found the cache already gone in {AUTO_OFF_DAYS} days"
    elif r["bumps"] >= AUTO_OFF_MIN and r["saved"] < r["spent"]:
        st["auto_off"] = (f"spent ${r['spent']:.2f} to save ${r['saved']:.2f} over "
                          f"{AUTO_OFF_DAYS} days ({r['bumps']} bumps)")


# ---- the proxy's side --------------------------------------------------------------

def remember(session: str | None, path: str, sent: bytes | None, headers: dict, started: float,
             usage: dict, model: str | None) -> None:
    """After a main request went through: settle the previous break's bumps, and
    keep what a bump needs. Never raises."""
    try:
        if not session or not sent:
            return
        size = sum(int(usage.get(k) or 0) for k in
                   ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        with _LOCK:
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
        if prev and prev.get("bumps"):
            _settle_return(session, prev, usage, started)
        on, _ = enabled()
        if not on or size < MIN_TOKENS or not big:
            with _LOCK:
                _SESSIONS.pop(session, None)
            return
        with _LOCK:
            if len(_SESSIONS) >= MAX_SESSIONS and session not in _SESSIONS:
                oldest = min(_SESSIONS, key=lambda k: _SESSIONS[k]["started"])
                _SESSIONS.pop(oldest)
            _SESSIONS[session] = {"path": path, "sent": sent, "headers": dict(headers), "started": started,
                                  "size": size, "model": model or "", "bumps": 0, "spent": 0.0,
                                  "last_bump": None, "misses": 0}
        if not _STARTED[0]:
            _STARTED[0] = True
            threading.Thread(target=_loop, daemon=True, name="keepalive").start()
    except Exception:
        pass


def _price(model: str):
    return pricing.price_for(model) or pricing.price_for("claude-opus-5-5")


def _settle_return(session: str, prev: dict, usage: dict, started: float) -> None:
    # Kept warm: the return read what the bumps kept alive. A compaction applied on
    # the warm return (it paid) reads less and counts as not kept: conservative.
    read = int(usage.get("cache_read_input_tokens") or 0)
    _record_break(session, prev, started, kept=read >= HIT_SHARE * prev["size"])


def _record_break(session: str, prev: dict, t: float, kept: bool) -> None:
    p = _price(prev["model"])
    saved = prev["size"] * (p.write_1h - p.read) / 1e6 if kept else 0.0
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


def _claude_running() -> bool:
    try:
        r = subprocess.run(["pgrep", "-x", "claude"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return True            # cannot tell: the day-end rule still applies


def due(s: dict, now: float, ends: dict, spent_today: float, cap: float) -> str | None:
    """None if this session should be bumped now, else why not."""
    if s.get("dropped"):
        return s["dropped"]
    if s["bumps"] >= MAX_BUMPS:
        return f"{MAX_BUMPS} bumps this break"
    base = s["last_bump"] or s["started"]
    if now - base < BUMP_AFTER_S:
        return "not yet"
    if now - base > H - 120:
        return "too late: the cache may be gone"     # a bump then risks a full write
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
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            usage = json.loads(r.read()).get("usage") or {}
        status = 200
    except urllib.error.HTTPError as exc:
        status, usage = exc.code, {}
        try:
            err = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            err = ""
        return {"status": status, "error": err, "t": t}
    except Exception as exc:
        return {"status": 0, "error": f"{type(exc).__name__}: {exc}"[:200], "t": t}
    read = int(usage.get("cache_read_input_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)
    p = _price(s["model"])
    cost = (read * p.read + write * p.write_1h + int(usage.get("input_tokens") or 0) * p.inp) / 1e6
    hit = read >= HIT_SHARE * s["size"] and write < 0.1 * s["size"]
    return {"status": status, "t": t, "read": read, "write": write, "cost": cost, "hit": hit}


def tick(now: float | None = None, send=bump, running=_claude_running) -> list[str]:
    """One pass over the remembered sessions. Returns what it did (tests, logs)."""
    now = now or time.time()
    st = load_state()
    if now - float(st.get("learned_at") or 0) > LEARN_EVERY_S:
        try:
            st = learn()
        except Exception:
            pass
    on, why = enabled(st)
    try:
        from . import routing
        rs = routing.state()
        if on and rs.get("managed") and not rs.get("on"):
            on, why = False, "TokenSaver is off or paused"
    except Exception:
        pass
    done = []
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
    for session, s in items:
        if s["bumps"] and now > s["last_bump"] + H:
            # The last bump's hour is over and nobody came back: wasted. Settled here,
            # so the next request of this session (if any) starts afresh.
            forget(session)
            _record_break(session, s, s["last_bump"], kept=False)
            continue
        if now - (s["last_bump"] or s["started"]) > H:
            forget(session)                  # the cache is gone either way: hold nothing
            continue
        why_not = due(s, now, ends, spent_today, cap)
        if why_not == "not yet":
            continue
        if why_not is None and not running():
            why_not = "Claude Code is not running"
        if why_not:
            if why_not != "not yet" and s.get("headers"):
                s["headers"] = {}            # the login is not kept past this session's last chance
                s["dropped"] = why_not
            continue
        res = send(session, s)
        if res.get("status") in (401, 403):
            s["dropped"], s["headers"] = f"login refused ({res['status']})", {}
        elif res.get("status") == 200:
            s["bumps"] += 1
            s["last_bump"] = res["t"]
            s["spent"] += res["cost"]
            spent_today += res["cost"]
            if not res["hit"]:
                s["misses"] += 1
                s["dropped"], s["headers"] = "the cache was already gone", {}
        elif res.get("status") == 400:
            s["dropped"], s["headers"] = "bump refused (400)", {}
            with _LOCK:
                st2 = load_state()
                st2["auto_off"] = f"Anthropic refused the bump: {res.get('error', '')[:120]}"
                _save_state(st2)
        try:
            vault.record("cache_bump", tool="proxy", session_id=session, index=False,
                         summary=f"bump {res.get('status')}: " + (
                             f"read {res.get('read', 0):,}, wrote {res.get('write', 0):,}, ${res.get('cost', 0):.3f}"
                             + ("" if res.get("hit") else " (MISS: the cache was gone)")
                             if res.get("status") == 200 else str(res.get("error"))[:160]),
                         meta={k: v for k, v in res.items() if k != "error"})
        except Exception:
            pass
        done.append(f"{session[:8]}: {res.get('status')}")
    if done:
        with _LOCK:
            st3 = load_state()
            st3["today"] = {"day": _day(now), "spent": round(spent_today, 4)}
            _save_state(st3)
    return done


def held() -> list[dict]:
    """For the health check: which sessions may be bumped and when next. No login data."""
    with _LOCK:
        items = list(_SESSIONS.items())
    return [{"session": k[:8], "size": s["size"], "bumps": s["bumps"], "dropped": s.get("dropped"),
             "next_bump": round((s["last_bump"] or s["started"]) + BUMP_AFTER_S)} for k, s in items]


def forget(session: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session, None)


def _loop() -> None:
    while True:
        time.sleep(TICK_S)
        try:
            tick()
        except Exception:
            pass
