"""How long a request's prompt cache lives: 5 minutes or 1 hour.

Claude Code picks it per request (read from its 2.1.287 source):
1. `FORCE_PROMPT_CACHING_5M` → 5m;
2. `CLAUDE_CODE_PROMPT_CACHE_TTL` (main thread) / `CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL`;
3. the settings `promptCacheTtl` / `subagentPromptCacheTtl`;
4. a subagent's own frontmatter TTL;
5. `ENABLE_PROMPT_CACHING_1H` (`_BEDROCK` on Bedrock) → 1h;
6. otherwise: no Claude subscription login (API key, Console, gateway, Bedrock, Vertex) → 5m;
   a subscription in extra usage (overage) → 5m; a subscription otherwise → 1h for the main
   thread only, 5m for subagents and side requests.

So it differs per user, per request class and even mid-session (a subscriber going into
overage). Nothing here guesses which rule applied: the request body says it exactly (the
`cache_control` markers' `ttl`; no ttl means 5m), and so does the usage of any request that
wrote to the cache (`ephemeral_5m_input_tokens` vs `ephemeral_1h_input_tokens`).
"""

from __future__ import annotations

SECONDS = {"5m": 300, "1h": 3600}
DEFAULT = "1h"        # unknown: TokenSaver's behaviour before the TTL was read


def _marker(block) -> str | None:
    cc = block.get("cache_control") if isinstance(block, dict) else None
    if not isinstance(cc, dict):
        return None
    return "1h" if cc.get("ttl") == "1h" else "5m"


def of_request(body: dict) -> str | None:
    """The TTL of the last cache marker in `messages` (the entry the next request reads),
    else of the system markers; None if the request caches nothing."""
    if not isinstance(body, dict):
        return None
    for m in reversed(body.get("messages") or []):
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for b in reversed(content):
                t = _marker(b)
                if t:
                    return t
    system = body.get("system")
    if isinstance(system, list):
        for b in reversed(system):
            t = _marker(b)
            if t:
                return t
    return None


def of_usage(r) -> str | None:
    """From a transcripts.Request: the TTL most of its cache write went to; None if it
    wrote nothing or the split is not reported."""
    w5, w1 = int(getattr(r, "write_5m", 0) or 0), int(getattr(r, "write_1h", 0) or 0)
    if not (w5 or w1):
        return None
    return "1h" if w1 >= w5 else "5m"


def per_request(reqs) -> list[str]:
    """Each request's TTL from its usage: carried forward over requests that wrote nothing,
    the session's first ones take the first TTL seen; none reported at all = DEFAULT."""
    seen = [of_usage(r) for r in reqs]
    first = next((t for t in seen if t), DEFAULT)
    out, cur = [], first
    for t in seen:
        cur = t or cur
        out.append(cur)
    return out


def seconds(ttl: str | None) -> int:
    return SECONDS.get(ttl or DEFAULT, SECONDS[DEFAULT])


def write_price(p, ttl: str | None) -> float:
    return p.write_5m if (ttl or DEFAULT) == "5m" else p.write_1h


# ---- choosing the main conversation's lifetime from this user's own history ----------
#
# A 1h write costs 2x the input price and a 5m one 1.25x, so 1h pays only where pauses of
# 5-60 minutes land on big contexts. Measured on this project's history (2026-11-10): 95% of
# gaps were under 5 minutes, 4.3% 5-60 minutes, and still 1h was 29% cheaper ($897 vs
# $1,260), because those pauses fall on the biggest contexts; 5m won only in short sessions.
# So each user's history decides, priced both ways with the same model, and TokenSaver only
# ever writes "1h" (where 5m is cheaper, Claude Code's own default for an API key already is
# 5m, and on a subscription within its plan we leave its choice alone).
#
# Claude Code reads the setting on every request, so a change reaches open sessions too and
# each pays one rewrite of its cache: decided at setup/update, re-checked weekly, changed
# only well outside a band and two checks running.

WINDOW_DAYS = 30
MIN_SESSIONS = 5            # with MIN_REQUESTS or more; fewer: the user's default, 1h
MIN_REQUESTS = 10
SET_MARGIN = 0.02           # setup sets 1h when it is at least this much cheaper
FLIP_BAND = 0.10            # the weekly re-check changes ours only past this, twice running
RECHECK_S = 7 * 86400
MIN_VERSION = (2, 1, 242)   # Claude Code that knows `promptCacheTtl`
KEY = "promptCacheTtl"


def price_history(reqs, p, ttl: str) -> float:
    """One session's main requests with every cache entry at `ttl`: a request reads the
    previous one's context if the gap is within the lifetime, else writes it all again."""
    from . import sim
    cost, prev = 0.0, None
    for k, r in enumerate(reqs):
        gap = sim._ts(r.timestamp) - sim._ts(reqs[k - 1].timestamp) if k else 0
        hit = 0 if prev is None or gap > seconds(ttl) else min(prev, r.context)
        cost += (hit * p.read + (r.context - hit) * write_price(p, ttl) + r.out * p.out) / 1e6
        prev = r.context
    return cost


def hit_accuracy(reqs) -> tuple[int, int]:
    """(right, of) for the model on the lifetime the user actually ran: did it predict a cache
    read where the usage shows one? Times are response ends, the lifetime runs from request
    start, so this is the check on how far the gaps can be trusted."""
    from . import sim
    ttls, right, n = per_request(reqs), 0, 0
    for k in range(1, len(reqs)):
        gap = sim._ts(reqs[k].timestamp) - sim._ts(reqs[k - 1].timestamp)
        predicted = gap <= seconds(ttls[k - 1])
        seen = reqs[k].read >= 0.5 * reqs[k - 1].context
        right += predicted == seen
        n += 1
    return right, n


def choose(root=None, now: float | None = None) -> dict:
    """{"set": bool, "cost_1h", "cost_5m", "sessions", "accuracy", "why"}: local, free."""
    import time
    from . import sim
    now = time.time() if now is None else now
    kw = {"root": root} if root is not None else {}
    c1 = c5 = 0.0
    used = right = of = 0
    for d in sim.sessions(include_rewritten=True, **kw):
        rq = d.requests
        if not rq or now - sim._ts(rq[-1].timestamp) > WINDOW_DAYS * 86400:
            continue
        c1 += price_history(rq, d.price, "1h")
        c5 += price_history(rq, d.price, "5m")
        used += len(rq) >= MIN_REQUESTS
        a, b = hit_accuracy(rq)
        right, of = right + a, of + b
    out = {"cost_1h": round(c1, 2), "cost_5m": round(c5, 2), "sessions": used,
           "accuracy": round(right / of, 3) if of else None}
    if used < MIN_SESSIONS:
        return dict(out, set=True, why=f"only {used} sessions of {MIN_REQUESTS}+ requests in "
                                         f"{WINDOW_DAYS} days: the default, 1 hour")
    if c1 <= (1 - SET_MARGIN) * c5:
        return dict(out, set=True, why=f"1 hour ${c1:,.2f} vs 5 minutes ${c5:,.2f} on your last {WINDOW_DAYS} days")
    return dict(out, set=False, why=f"5 minutes is as cheap or cheaper (${c5:,.2f} vs ${c1:,.2f} with "
                                    f"1 hour): Claude Code's own choice stands")


# ---- the setting ------------------------------------------------------------------------

def _state_path():
    from . import vault
    return vault.VAULT_DIR / "cachettl.json"


def load_state() -> dict:
    import json
    try:
        return json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    import json
    p = _state_path()
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(p)


def current() -> tuple[str | None, bool]:
    """(the `promptCacheTtl` in ~/.claude/settings.json, whether it is ours)."""
    from . import claude_settings
    data = claude_settings.read_json(claude_settings.SETTINGS) or {}
    v = data.get(KEY)
    st = load_state()
    return v, v is not None and st.get("added") == v


def apply(decision: dict, backup: bool = True) -> str:
    """Write "1h" unless the user has their own value. Returns what happened."""
    import time
    from . import claude_settings
    data = claude_settings.read_json(claude_settings.SETTINGS)
    if data is None:
        raise RuntimeError(f"{claude_settings.SETTINGS} does not parse; not touching it")
    st = load_state()
    cur = data.get(KEY)
    if cur is not None and st.get("added") != cur:
        return f"{KEY} is already set to {cur!r} by you: left alone"
    st.update(decision=decision, checked_at=time.time())
    if cur == "1h":
        save_state(dict(st, added="1h"))
        return "1-hour cache already set"
    data[KEY] = "1h"
    claude_settings.write_json(claude_settings.SETTINGS, data, backup=backup)
    save_state(dict(st, added="1h", at=time.time()))
    return f"set {KEY} to \"1h\" in {claude_settings.SETTINGS}"


def remove(backup: bool = True) -> str | None:
    """Take ours out, only while it still holds our value."""
    from . import claude_settings
    st = load_state()
    data = claude_settings.read_json(claude_settings.SETTINGS)
    if data is None or not st.get("added"):
        return None
    if data.get(KEY) == st["added"]:
        del data[KEY]
        claude_settings.write_json(claude_settings.SETTINGS, data, backup=backup)
        msg = f"removed {KEY} from {claude_settings.SETTINGS}"
    else:
        msg = None
    st.pop("added", None)
    save_state(st)
    return msg


def claude_version() -> tuple[int, ...] | None:
    import re
    import subprocess
    try:
        r = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=30)
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", r.stdout or "")
        return tuple(int(x) for x in m.groups()) if m else None
    except Exception:
        return None


def recheck(now: float | None = None) -> str | None:
    """Weekly, from the watchdog: change our own value only when the other choice has been
    FLIP_BAND cheaper on two checks running. Never touches a user's value or runs while the
    user turned it off; the first check only records. Returns what it did."""
    import time
    now = time.time() if now is None else now
    st = load_state()
    if st.get("mode") == "off" or now - float(st.get("checked_at") or 0) < RECHECK_S:
        return None
    v, ours = current()
    if v is not None and not ours:
        return None
    save_state(dict(st, checked_at=now))   # a failing read is not retried every 15 s
    d = choose(now=now)
    st = load_state()
    st.update(checked_at=now, decision=d)
    if ours:
        want_off = st.get("mode") != "on" and d["cost_5m"] <= (1 - FLIP_BAND) * d["cost_1h"] \
            and d["sessions"] >= MIN_SESSIONS
        st["streak"] = int(st.get("streak") or 0) + 1 if want_off else 0
        save_state(st)
        if st["streak"] >= 2:
            msg = remove(backup=False)
            st = dict(load_state(), streak=0, decision=d, auto_removed=now)
            save_state(st)
            return msg
        return None
    want_on = d["cost_1h"] <= (1 - FLIP_BAND) * d["cost_5m"]
    st["streak"] = int(st.get("streak") or 0) + 1 if want_on else 0
    save_state(st)
    if st["streak"] >= 2:
        msg = apply(d, backup=False)
        save_state(dict(load_state(), streak=0))
        return msg
    return None


def set_mode(mode: str) -> str:
    """`tokensaver cache-ttl auto|on|off`: on = always 1 hour, off = remove ours and never set
    it, auto = follow this user's history (the default)."""
    import time
    st = dict(load_state(), mode=mode, streak=0)
    save_state(st)
    if mode == "off":
        return remove() or "nothing of ours to remove"
    d = choose()
    if mode == "on" or d["set"]:
        return apply(d)
    save_state(dict(load_state(), decision=d, checked_at=time.time()))
    return d["why"]


def describe() -> str:
    v, ours = current()
    st = load_state()
    d = st.get("decision") or {}
    nums = (f"; your last {WINDOW_DAYS} days: 1 hour ${d['cost_1h']:,.2f} vs 5 minutes ${d['cost_5m']:,.2f}"
            if "cost_1h" in d else "")
    if v is not None and not ours:
        return f"{v} (your own setting){nums}"
    if st.get("mode") == "off":
        return f"Claude Code's default (you turned TokenSaver's choice off){nums}"
    if ours:
        old = claude_version()
        warn = "; needs Claude Code 2.1.242+" if old is not None and old < MIN_VERSION else ""
        return f"1 hour for the main conversation (set by TokenSaver{', always' if st.get('mode') == 'on' else ''}){nums}{warn}"
    return f"Claude Code's default (5 minutes is as cheap here){nums}" if d else "not decided yet (setup or update decides)"
