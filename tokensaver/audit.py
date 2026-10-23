"""Real token counts for what the rewrite saved, on real traffic.

Every saving this project reports was measured in characters on old
transcripts. Tokens are what is billed, and the two do not move together:
paths, JSON and code tokenize differently from prose. So for a sample of real
requests the proxy asks Anthropic's count_tokens endpoint for both the original
body and the one it actually sent, and records the pair.

Why this is safe to leave on (Anthropic's docs, platform.claude.com/docs/en/
build-with-claude/token-counting): "Token counting is free to use", and "Token
counting and message creation have separate and independent rate limits. Usage
of one does not count against the limits of the other." The count is an
estimate, but both bodies go through the same estimator, so their difference
is what we want.

Rules, each because the alternative could make a real turn worse:

* Nothing happens in the request path. `offer` runs after the response has
  been flushed to the client and only queues a job. One worker thread, a small
  bounded queue, and a rate limit of one pair per minute and a daily cap.
* The pair is always counted together: two calls, no offset arithmetic
  between count_tokens and billed usage, which are different meters.
* Only allowlisted body fields and headers are sent. Server tools are dropped
  (count_tokens rejects them); they are the same in both bodies, so the
  difference is unaffected. No `x-claude-code-*` header goes out, so the call
  is never mistaken for a turn.
* Auth headers live only in memory, only for the job's life (at most 10 min).
  The vault row holds counts, never credentials.
* Thinking is removed from every body counted (2026-10-22). Opus 5.5 binds
  each thinking block to the exact bytes before it, and our edits were made
  before Claude thought, so the ORIGINAL body is the mismatched history: its
  thinking is dropped by the count, which made plain Claude look ~19K tokens
  cheaper than it was. With thinking removed from both bodies (valid per the
  docs), the difference is the content saving alone; plain = what was really
  billed + that saving.
* On 429/5xx it backs off; on 400/401/403 it turns itself off for the life of
  the process and records why. It never raises.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request

from . import config, vault

BODY_FIELDS = ("model", "messages", "system", "tools", "tool_choice", "thinking")
HEADER_FIELDS = ("authorization", "x-api-key", "anthropic-version", "anthropic-beta", "user-agent")
MIN_GAP_S = None                 # None: config `audit_min_gap_s` (default 1 s); tests set a number
JOB_MAX_AGE_S = 600              # drop jobs (and their auth headers) after this
SETTLE_S = 2                     # let the turn that just ended finish first
BACKOFF = (300, 6 * 3600)
QUEUE_MAX = 32                   # absorbs a burst of tool-loop requests

_q: queue.PriorityQueue = queue.PriorityQueue(maxsize=QUEUE_MAX)
_lock = threading.Lock()
# The newest real count per session, for the status line (`proxy.estimate`):
# {"orig": plain tokens, "sent": real tokens, "bytes_in", "bytes_out", "t"}.
LATEST: dict[str, dict] = {}


def _remember(session: str | None, orig: int, sent: int, bytes_in, bytes_out) -> None:
    if not session or not orig:
        return
    LATEST[session] = {"orig": int(orig), "sent": int(sent), "bytes_in": bytes_in, "bytes_out": bytes_out,
                       "t": time.time()}
    if len(LATEST) > 50:
        LATEST.pop(next(iter(LATEST)))
_state = {"started": False, "disabled": None, "pause_until": 0.0, "backoff": 0.0,
          "last_call": 0.0, "day": "", "calls_today": 0, "dropped": 0, "seq": 0,
          "limit": None, "remaining": None, "reset_at": None}
_per_session: dict[str, int] = {}
UPSTREAM = None                  # set by the proxy; None means config's upstream


def enabled() -> bool:
    return config.get("audit") == "on" and not _state["disabled"]


def status() -> dict:
    return {"on": config.get("audit") == "on", "disabled": _state["disabled"],
            "dropped": _state["dropped"], "calls_today": _state["calls_today"],
            "limit": _state["limit"], "remaining": _state["remaining"]}


def _reason(session: str | None, rstats: dict, changed: bool) -> tuple[int, str] | None:
    """Sampling priority (lower first) and why, or None to skip.

    Every Nth request is sampled whether or not anything was rewritten, so
    the overall saving is not biased upward by looking only where the rewrite
    acted. The first *changed* request per session is sampled too: that is
    what shows the rewrite working (a session's first request is usually
    empty, with nothing to save)."""
    key = session or ""
    st = _per_session.setdefault(key, {"n": 0, "changed_seen": False})
    st["n"] += 1
    if rstats.get("bg_applied"):
        return 0, "compaction_applied"
    if changed and not st["changed_seen"]:
        st["changed_seen"] = True
        return 1, "first_changed"
    if st["n"] % max(int(config.get("audit_every")), 1) == 0:
        return 2, "every_n"
    return None


def offer(*, path: str, request_class: str | None, session: str | None, headers: dict,
          body: bytes | None, sent: bytes | None, rstats: dict, status: int,
          usage: dict | None = None, base: tuple | None = None) -> None:
    """Queue a count for this request if it is sampled. O(1); never raises."""
    try:
        if not enabled() or not body or not sent:
            return
        if path.split("?")[0] != "/v1/messages" or (request_class or "main") != "main" or status != 200:
            return
        changed = sent is not body and sent != body
        picked = _reason(session, rstats or {}, changed)
        if picked is None:
            return
        prio, why = picked
        if not changed:
            # Nothing was rewritten: the billed usage is the exact count of
            # both, so no API call. Recorded so totals include 0% requests.
            u = usage or {}
            n = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens",
                                                  "cache_creation_input_tokens"))
            if n:
                _remember(session, n, n, len(body), len(sent))
                vault.record("token_audit", tool="proxy", session_id=session, index=False,
                             summary=f"tokens {n:,} -> {n:,} (unchanged), {why}",
                             meta={"reason": why, "tokens_original": n, "tokens_sent": n,
                                   "unchanged": True, "source": "usage",
                                   "bytes_in": len(body), "bytes_out": len(sent)})
            return
        hdrs = {k.lower(): v for k, v in headers.items() if k.lower() in HEADER_FIELDS}
        with _lock:
            _state["seq"] += 1
            job = (prio, _state["seq"], {"t": time.time(), "path": path, "session": session,
                                         "headers": hdrs, "body": body, "sent": sent,
                                         "usage": dict(usage or {}),
                                         "why": why, "rstats": dict(rstats or {}), "base": base})
            if _q.full():
                # A higher-priority job evicts the lowest one; otherwise drop this one.
                items = []
                while not _q.empty():
                    items.append(_q.get_nowait())
                items.append(job)
                items.sort(key=lambda j: (j[0], j[1]))
                _state["dropped"] += len(items) - QUEUE_MAX
                for it in items[:QUEUE_MAX]:
                    _q.put_nowait(it)
            else:
                _q.put_nowait(job)
            if not _state["started"]:
                _state["started"] = True
                threading.Thread(target=_worker, daemon=True).start()
    except Exception:
        pass


def payload(raw: bytes, no_thinking: bool = False) -> dict | None:
    """The count_tokens body for one request: allowlisted fields only, and no
    server tools (the endpoint rejects them). `no_thinking`: thinking blocks
    removed from the messages (see the module notes)."""
    try:
        req = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(req, dict) or not isinstance(req.get("messages"), list):
        return None
    out = {k: req[k] for k in BODY_FIELDS if k in req}
    if no_thinking:
        from .background import strip_thinking
        out["messages"] = [strip_thinking(m) if isinstance(m, dict) else m for m in out["messages"]]
    if isinstance(out.get("tools"), list):
        out["tools"] = [t for t in out["tools"]
                        if isinstance(t, dict) and t.get("type") in (None, "custom")]
        if not out["tools"]:
            out.pop("tools")
    return out


def _parse_reset(value: str | None) -> float | None:
    """`anthropic-ratelimit-requests-reset` is an RFC 3339 time."""
    if not value:
        return None
    try:
        import calendar
        v = value.strip().replace("Z", "+00:00")
        main, _, tz = v.partition("+")
        main = main.split(".")[0]
        return float(calendar.timegm(time.strptime(main, "%Y-%m-%dT%H:%M:%S")))
    except (ValueError, TypeError):
        return None


def _note_limits(headers: dict) -> None:
    """Remember the request limit Anthropic reports on every response, so the
    pace below follows the real limit instead of a guess."""
    h = {k.lower(): v for k, v in (headers or {}).items()}
    try:
        if "anthropic-ratelimit-requests-remaining" in h:
            _state["remaining"] = int(h["anthropic-ratelimit-requests-remaining"])
            _state["limit"] = int(h.get("anthropic-ratelimit-requests-limit") or 0) or None
            _state["reset_at"] = _parse_reset(h.get("anthropic-ratelimit-requests-reset"))
    except ValueError:
        pass


def pace_gap(now: float | None = None) -> float:
    """Seconds to wait between calls: the configured floor, or longer when
    Anthropic's own headers say the window would run out first. The requests
    left are spread evenly until the window resets, with a job's worth
    spare, so the audit stays under the limit rather than finding it with a
    429. A job is up to CALLS_PER_JOB calls made back to back (original, sent,
    alias attribution), so the budget is in jobs."""
    now = time.time() if now is None else now
    floor = MIN_GAP_S if MIN_GAP_S is not None else float(config.get("audit_min_gap_s"))
    rem, reset = _state.get("remaining"), _state.get("reset_at")
    if rem is None or reset is None or reset <= now:
        return floor
    jobs_left = rem // CALLS_PER_JOB - 1
    if jobs_left < 1:
        return max(floor, reset - now + 1)        # out of room: wait for the window
    return max(floor, (reset - now) / jobs_left)


def _count(path: str, headers: dict, body: dict) -> tuple[int | None, int, dict]:
    """(input_tokens or None, HTTP status, response headers)."""
    base = (UPSTREAM or config.get("upstream")).rstrip("/")
    query = ("?" + path.split("?", 1)[1]) if "?" in path else ""
    req = urllib.request.Request(base + "/v1/messages/count_tokens" + query,
                                 data=json.dumps(body).encode(), method="POST",
                                 headers=dict(headers, **{"content-type": "application/json",
                                                         "accept-encoding": "identity"}))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _note_limits(dict(resp.headers))
            return int(json.loads(resp.read()).get("input_tokens")), resp.status, dict(resp.headers)
    except urllib.error.HTTPError as exc:
        _note_limits(dict(exc.headers or {}))
        return None, exc.code, dict(exc.headers or {})
    except Exception:
        return None, 0, {}


# Every forward pass is priced on every counted request, by counting the
# request once more with that one pass toggled (off if it was on, on if it was
# off). A pass that changed nothing in a request costs no call. The proxy
# switches a pass off for new free points when it measures as a net cost.
ATTRIBUTE = ("heredocs", "repeat_runs", "format", "json_compact", "path_aliases")
CALLS_PER_JOB = 2 + len(ATTRIBUTE)


def _toggled(orig: dict, name: str, base: tuple) -> tuple[dict | None, bool]:
    """(the request as built with pass `name` toggled, whether it was on).

    `base` is (messages, aliases, exclude) as the proxy really built the
    request: the history after any compaction, before the forward passes,
    the alias plan in force and the passes switched off for this session."""
    try:
        from . import passes
        messages, aliases, exclude = base
        exclude = tuple(exclude or ())
        if name == "path_aliases":
            was_on = isinstance(aliases, dict)
            alt = "off" if was_on else (passes.hindsight_plan(messages) or None)
            if alt is None:
                return None, was_on
            new = passes.rewrite_messages(messages, exclude=exclude, aliases=alt)
        else:
            was_on = name not in exclude
            ex = tuple(x for x in exclude if x != name) if not was_on else exclude + (name,)
            new = passes.rewrite_messages(messages, exclude=ex, aliases=aliases)
        out = dict(orig)
        out["messages"] = new
        return out, was_on
    except Exception:
        return None, False


def _day() -> str:
    return time.strftime("%Y-%m-%d")


def _worker() -> None:
    while True:
        try:
            _, _, job = _q.get()
            if _state["disabled"]:
                continue
            now = time.time()
            if now - job["t"] > JOB_MAX_AGE_S:
                _state["dropped"] += 1
                continue
            if _state["day"] != _day():
                _state.update(day=_day(), calls_today=0)
            if _state["calls_today"] + 2 > int(config.get("audit_max_per_day")):
                _state["dropped"] += 1
                continue
            # Sleep in short steps and re-check: a pause can be lifted, or the
            # audit switched off, while this job waits.
            while True:
                if _state["disabled"]:
                    break
                gap = pace_gap()
                wait = (max(_state["pause_until"], _state["last_call"] + gap, job["t"] + SETTLE_S)
                        - time.time())
                if wait <= 0:
                    break
                time.sleep(min(wait, 1.0))
            if not _state["disabled"]:
                _run(job)
        except Exception:
            pass


def _run(job: dict) -> None:
    orig, sent = payload(job["body"], no_thinking=True), payload(job["sent"], no_thinking=True)
    if orig is None or sent is None:
        return
    started = time.time()
    counts, statuses = [], []
    for body in (orig, sent):
        n, code, rh = _count(job["path"], job["headers"], body)
        _state["last_call"] = time.time()
        _state["calls_today"] += 1
        statuses.append(code)
        if n is None:
            _on_error(code, rh, job)
            return
        counts.append(n)
    _state["backoff"] = 0.0
    u = job.get("usage") or {}
    real = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens",
                                            "cache_creation_input_tokens"))
    if not real:
        n, code, rh = _count(job["path"], job["headers"], payload(job["sent"]))
        _state["last_call"] = time.time()
        _state["calls_today"] += 1
        if n is None:
            _on_error(code, rh, job)
            return
        real = n
    content_nt = list(counts)          # (original, sent), thinking removed from both
    counts = [real + (content_nt[0] - content_nt[1]), real]
    rs = job["rstats"]
    # Per-pass attribution, so each pass's token effect is measured, not
    # guessed (measured first: bytes fell 0.7% while tokens rose 1.5%; path
    # aliases were the cause). Variants start from what the proxy really
    # started from (`base`), so they price the current configuration,
    # compacted requests included. `passes[name]["saving"]`: tokens the pass saves when on (negative = it
    # costs), from one extra count with only that pass toggled. 0 with no call
    # when toggling it changes nothing in this request.
    per_pass, without = {}, {}
    base = job.get("base")
    if base:
        sent_msgs = json.dumps(sent.get("messages"), sort_keys=True)
        for name in ATTRIBUTE:
            variant, was_on = _toggled(orig, name, base)
            if variant is not None:
                variant = payload(json.dumps(variant).encode(), no_thinking=True)
            if variant is None:
                continue
            variant_msgs = json.dumps(variant["messages"], sort_keys=True)
            if variant_msgs == sent_msgs:
                per_pass[name] = {"on": was_on, "saving": 0, "changed": False}
                continue
            n, code, rh = _count(job["path"], job["headers"], variant)
            _state["last_call"] = time.time()
            _state["calls_today"] += 1
            if n is None:
                _on_error(code, rh, job)
                break
            # chars: how much toggling changed the request. A token is at least
            # one character, so a token effect beyond it is a broken count.
            per_pass[name] = {"on": was_on, "saving": (n - content_nt[1]) if was_on else (content_nt[1] - n),
                              "changed": True, "chars": abs(len(variant_msgs) - len(sent_msgs))}
            if name == "path_aliases" and was_on:
                without[name] = n
    _remember(job["session"], counts[0], counts[1], rs.get("bytes_in"), rs.get("bytes_out"))
    vault.record("token_audit", tool="proxy", session_id=job["session"],
                 summary=f"tokens {counts[0]:,} -> {counts[1]:,} "
                         f"({100 * (1 - counts[1] / max(counts[0], 1)):.1f}% saved), {job['why']}",
                 meta={"reason": job["why"], "tokens_original": counts[0], "tokens_sent": counts[1],
                       "bytes_in": rs.get("bytes_in"), "bytes_out": rs.get("bytes_out"),
                       "bg_applied": bool(rs.get("bg_applied")), "bg_cut": rs.get("bg_cut"),
                       "model": orig.get("model"), "seconds": round(time.time() - started, 2),
                       "without": without, "passes": per_pass, "aliases": rs.get("aliases"),
                       "attribution_base": "as_built" if base else "original",
                       "method": "no_thinking", "tokens_original_nt": content_nt[0],
                       "tokens_sent_nt": content_nt[1]},
                 index=False)


def _on_error(code: int, headers: dict, job: dict) -> None:
    why = f"count_tokens returned {code or 'no response'}"
    if code in (400, 401, 403):
        _state["disabled"] = why
    else:
        retry = 0.0
        try:
            retry = float({k.lower(): v for k, v in headers.items()}.get("retry-after") or 0)
        except ValueError:
            pass
        _state["backoff"] = min(max(_state["backoff"] * 2, BACKOFF[0]), BACKOFF[1])
        _state["pause_until"] = time.time() + max(retry, _state["backoff"])
    vault.record("token_audit_error", tool="proxy", session_id=job["session"],
                 summary=why + ("; audit off for this process" if _state["disabled"] else
                                f"; pausing {(_state['pause_until'] - time.time()) / 60:.0f} min"),
                 meta={"status": code, "disabled": bool(_state["disabled"])}, index=False)
