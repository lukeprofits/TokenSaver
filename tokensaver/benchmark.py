"""What TokenSaver WOULD have saved on this machine's own Claude Code history.

Runs on a plain clone: it reads Claude Code's transcripts
(~/.claude/projects), changes nothing, and needs no proxy, setup or vault.
Every real session (test forks, sessions the proxy already rewrote and
non-Claude sessions left out) is replayed request by request through today's
code, the way the proxy would have handled it:

- the forward passes (`passes.rewrite_messages`, aliases off);
- background compaction (`background.build`, deterministic) at a quiet gap
  over `bg_after_min`, decided like `proxy._decide`: on counted tokens; cold
  cache, apply if smaller; warm, only within `bg_warm_payback_max` requests.
  The kept tail's thinking is stripped from then on, as the proxy does.

Defaults only: the user's own tuned settings are not used, so two people's
numbers are comparable.

Sizes. Plain is the real billed prompt of each request (input + cache read +
cache write, from the transcript: system and tools included). TokenSaver is
plain minus what it removes, counted on every STEP-th request and every
decision with Anthropic's count_tokens endpoint (free; separate rate limit):

    count(original messages, thinking removed) - count(ours, thinking removed)
    + F x (thinking signature characters removed)

F (thinking tokens per signature character) is measured on this machine's
own requests where possible (counted with and without thinking), per model:
first on exact sent requests, then on rebuilt ones; otherwise the fallback
`DEFAULT_SIG_FACTOR` is used, the report says so, and it shows how far a 30%
error in it would move the result. Between counted requests the removal is interpolated within
the same compaction epoch.

Cost, the same model for both: a request reads the previous one's size from
cache unless the gap was over 1h or a compaction was just applied; the rest is
written at 1h; output tokens are the real ones; each request at its own
model's price. The plain side is printed next to the real bill as a check.

Cache keep-alive is replayed with the proxy's own rule (`keepalive.replay_bumps`),
once proven on the machine: kept-warm breaks are priced as reads, every bump is
paid for (also after a session's last request), and trims follow the warm-return
rule. Off when the user turned keep-alive off.

Not modelled (each would move the result a little): re-reads of cut content
(0 measured in a real trial). Fast mode (the default) also leaves out local AI
(about 0.5-1% more where a machine runs it); `full` runs it on each compaction
the replay applies, through the proxy's own code (`proxy.prune_offline`), and
counts every request: hours instead of minutes.

The count calls send the transcripts' messages to Anthropic (who processed
them originally), with the user's own Claude Code login or ANTHROPIC_API_KEY.
Credentials are read in this process and never printed or stored. The result
file holds numbers only: no paths, prompts or content.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import names
from . import __version__, audit, background as bg, cachettl, config, passes, pricing, sim, transcripts

# Tokens per signature char when this machine can't measure (see thinking_factors):
# the one default, kept with its history in background.TOKENS_PER_SIGNATURE_CHAR.
DEFAULT_SIG_FACTOR = bg.TOKENS_PER_SIGNATURE_CHAR
COUNT_URL = "https://api.anthropic.com/v1/messages/count_tokens"
OUT_DIR = names.home() / "benchmark"


# ---- credentials and counting -----------------------------------------------

def credentials() -> tuple[dict, str] | None:
    """(auth headers, where they came from), or None. Never printed."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return {"x-api-key": key}, "ANTHROPIC_API_KEY"
    raw = None
    if platform.system() == "Darwin":
        try:
            raw = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                                 capture_output=True, text=True, timeout=20).stdout
        except Exception:
            raw = None
    if not raw:
        p = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / ".credentials.json"
        try:
            raw = p.read_text()
        except OSError:
            raw = None
    try:
        token = json.loads(raw)["claudeAiOauth"]["accessToken"] if raw else None
    except (ValueError, KeyError, TypeError):
        token = None
    if not token:
        return None
    return {"authorization": f"Bearer {token}", "anthropic-beta": "oauth-2025-04-20"}, "Claude Code login"


# ---- making a rebuilt history countable ------------------------------------------
#
# count_tokens checks a request the way the API does, but the benchmark sends no tool
# list (the transcripts do not keep it). Two things in real histories then fail, measured
# on a coworker's run (1,102 of 1,544 counts, 2026-11-04), and one failure repeats for
# every later request of that session, since each one carries the same message:
# - server-tool blocks (web search/fetch, MCP connector, code execution, ...: a
#   `server_tool_use` and its `*_tool_result`): rejected without their tool and beta;
# - a tool call without its result at the start of the next message: when Claude calls a
#   server tool and a client tool in one turn, the server result arrives in the next
#   assistant message, after the client result, and an Esc-interrupted call has none.
# So every count goes through `countable`, which applies the same rules to both sides
# (plain and TokenSaver), keeping the measured difference exact: unsupported blocks become
# text holding their JSON, and every tool call gets its result right after it (moved from
# later in the history, or a one-line placeholder when none was recorded).

COUNTABLE_TYPES = {"text", "image", "document", "search_result", "tool_use", "tool_result",
                   "thinking", "redacted_thinking"}
PLACEHOLDER = "[no result recorded]"


def _as_text(block: dict) -> dict:
    body = {k: v for k, v in block.items() if k != "cache_control"}
    return {"type": "text", "text": f"[{block.get('type')}] "
            + json.dumps(body, ensure_ascii=False, separators=(",", ":"))}


def _convert(block, rep: dict, strict: bool = False):
    """One block made countable; nested tool_result content too."""
    if not isinstance(block, dict):
        return block
    t = block.get("type")
    if t == "tool_reference":
        rep["converted"] = rep.get("converted", 0) + 1
        return {"type": "text", "text": f"[tool {block.get('tool_name') or block.get('name') or ''}]"}
    if t not in COUNTABLE_TYPES or (strict and t not in ("text", "tool_use", "tool_result")):
        rep["converted"] = rep.get("converted", 0) + 1
        return _as_text(block)
    if strict:
        # Only the basic fields: whatever extra field the counter refused goes into the text.
        keep = {"text": ("type", "text"), "tool_use": ("type", "id", "name", "input"),
                "tool_result": ("type", "tool_use_id", "content", "is_error")}[t]
        extra = {k: v for k, v in block.items() if k not in keep and k != "cache_control"}
        base = {k: block[k] for k in keep if k in block}
        if t == "tool_result":
            c = base.get("content")
            if isinstance(c, list):
                base["content"] = [x if isinstance(x, dict) and x.get("type") == "text" and set(x) <= {"type", "text"}
                                   else _as_text(x) for x in c]
        if extra:
            rep["converted"] = rep.get("converted", 0) + 1
            if t == "text":
                base["text"] = base.get("text", "") + " " + json.dumps(extra, ensure_ascii=False, separators=(",", ":"))
        return base
    if t == "tool_result" and isinstance(block.get("content"), list):
        return dict(block, content=[_convert(b, rep, strict) for b in block["content"]])
    return dict(block)


def _items(m: dict) -> list:
    c = m.get("content")
    return list(c) if isinstance(c, list) else [{"type": "text", "text": c if isinstance(c, str) else ""}]


def countable(messages: list, strict: dict | set | None = None) -> tuple[list, dict]:
    """(a copy count_tokens accepts, {"converted", "moved", "placeholder", "stray", "duplicate",
    "empty", "merged"}). Every rule the counter has refused a history for is checked here
    first, so a known problem never costs a refused count.
    `strict`: canonical-JSON hash of a message a 400 named -> level. 1: only basic block types
    and fields; 2: every block of it as text (its tool calls too, so their results become
    text as well). A set means level 1 for each."""
    rep: dict = {}
    levels = dict.fromkeys(strict, 1) if isinstance(strict, set) else (strict or {})
    out = []
    seen_calls: set = set()
    for n, m in enumerate(messages):
        lvl = levels.get(_msg_hash(m), 0) if levels else 0
        role = m.get("role")
        content = []
        for b in _items(m):
            if lvl >= 2 and isinstance(b, dict) and b.get("type") != "text":
                rep["converted"] = rep.get("converted", 0) + 1
                b = _as_text(b)
            b = _convert(b, rep, lvl >= 1)
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and not str(b.get("text") or "").strip():
                rep["empty"] = rep.get("empty", 0) + 1          # "text content blocks must be non-empty"
                continue
            if t in ("thinking", "redacted_thinking") and role != "assistant":
                rep["converted"] = rep.get("converted", 0) + 1
                b = _as_text(b)
            if t == "tool_use":
                if b.get("id") in seen_calls:                  # "`tool_use` ids must be unique"
                    rep["duplicate"] = rep.get("duplicate", 0) + 1
                    continue
                seen_calls.add(b.get("id"))
            content.append(b)
        if out and out[-1]["role"] == role:                     # roles must alternate
            rep["merged"] = rep.get("merged", 0) + 1
            out[-1]["content"] += content
        else:
            out.append({"role": role, "content": content, "_src": n})
    # Pairing: the results for an assistant message's tool calls lead the next user message.
    i = 0
    while i < len(out):
        m = out[i]
        if m["role"] != "assistant":
            if m["role"] == "user":
                prev_ids = ({b.get("id") for b in out[i - 1]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"}
                            if i and out[i - 1]["role"] == "assistant" else set())
                fixed = []
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") not in prev_ids:
                        rep["stray"] = rep.get("stray", 0) + 1
                        b = _as_text(b)
                    fixed.append(b)
                m["content"] = fixed
            i += 1
            continue
        ids = [b.get("id") for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not ids:
            i += 1
            continue
        if i + 1 >= len(out) or out[i + 1]["role"] != "user":
            out.insert(i + 1, {"role": "user", "content": [], "_src": None})
        nxt = out[i + 1]
        have = {b.get("tool_use_id"): b for b in nxt["content"]
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in ids}
        for tid in ids:
            if tid in have:
                continue
            found = None
            for j in range(i + 2, len(out)):
                for k, b in enumerate(out[j]["content"]):
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tid:
                        found = out[j]["content"].pop(k)
                        break
                if found:
                    break
            if found:
                rep["moved"] = rep.get("moved", 0) + 1
                have[tid] = found
            else:
                rep["placeholder"] = rep.get("placeholder", 0) + 1
                have[tid] = {"type": "tool_result", "tool_use_id": tid, "content": PLACEHOLDER}
        rest = [b for b in nxt["content"] if not (isinstance(b, dict) and b.get("type") == "tool_result"
                                                  and b.get("tool_use_id") in have)]
        nxt["content"] = [have[t] for t in ids] + rest
        i += 1
    # A message left empty by a move is not sent empty; `_src` (where each came from) is
    # kept aside for the caller.
    kept: list = []
    for m in out:
        if not m["content"]:
            continue
        if kept and kept[-1]["role"] == m["role"]:            # an empty one between them is gone
            rep["merged"] = rep.get("merged", 0) + 1
            kept[-1]["content"] += m["content"]
        else:
            kept.append(m)
    out = kept
    rep["_origin"] = [m.pop("_src") for m in out]
    return out, rep


def _msg_hash(m: dict) -> str:
    return hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()


def _countable(x):
    """Back-compat: the countable copy only."""
    if isinstance(x, list) and all(isinstance(m, dict) and "role" in m for m in x):
        return countable(x)[0]
    return _convert(x, {}) if isinstance(x, dict) else x


BUILTIN_TOOLS = {"Bash", "BashOutput", "KillShell", "Read", "Edit", "MultiEdit", "Write", "Glob", "Grep",
                 "WebFetch", "WebSearch", "Task", "Agent", "TodoWrite", "NotebookEdit", "ExitPlanMode",
                 "EnterPlanMode", "ToolSearch", "Skill", "AskUserQuestion", "Monitor", "LSP"}


def shape(messages: list, index: int | None) -> list:
    """Roles and block types around `index`, safe to share: no text, inputs or results;
    tool names only for Claude Code's own tools ("mcp" / "other" otherwise)."""
    if index is None:
        return []
    out = []
    for j in range(max(0, index - 2), min(len(messages), index + 2)):
        m = messages[j]
        prev = messages[j - 1] if j else None
        nxt = messages[j + 1] if j + 1 < len(messages) else None
        blocks = []
        for b in _items(m):
            if not isinstance(b, dict):
                continue
            d = {"type": b.get("type")}
            name = b.get("name")
            if name:
                d["tool"] = name if name in BUILTIN_TOOLS else ("mcp" if str(name).startswith("mcp__") else "other")
            if b.get("type") == "tool_use":
                d["result_next"] = bool(nxt) and any(isinstance(x, dict) and x.get("tool_use_id") == b.get("id")
                                                     for x in _items(nxt))
            if b.get("type") == "tool_result":
                d["call_before"] = bool(prev) and any(isinstance(x, dict) and x.get("id") == b.get("tool_use_id")
                                                      for x in _items(prev))
            blocks.append(d)
        out.append({"index": j, "role": m.get("role"), "blocks": blocks})
    return out


# ---- counting -----------------------------------------------------------------

# How long one count waits for a rate limit before it is given up. A sample (one request of
# the session, the rest interpolated) waits a minute: on this machine counting every 8th request
# instead of every one moved the result by ~0.1 point, so a few skipped samples cost little,
# while waiting hours would make the benchmark useless. A count that DECIDES something (a
# compaction) waits longer, since without it the replay could not decide at all.
SAMPLE_BUDGET_S = 60
RATE_BUDGET_S = 15 * 60
BUSY_S = 120                 # after a sample gives up, skip optional samples this long
_MSG_INDEX = re.compile(r"messages\.(\d+)")


class Counter:
    """count_tokens, paced and patient. Rate limits (429), overload (5xx) and network errors
    are waited out: Anthropic's `retry-after` is honoured, the pace follows its
    `anthropic-ratelimit-requests-*` headers (the limit is shared by everyone on one
    account), and backoff has random jitter so many people do not retry in lockstep. Only
    a count still blocked after RATE_BUDGET_S is given up, and it is reported as rate
    limited, never mixed with a rejected request."""

    def __init__(self, auth: dict, rate_per_s: float = 5.0, sleep=time.sleep):
        self.headers = dict(auth, **{"anthropic-version": "2023-06-01", "content-type": "application/json"})
        self.gap, self.next_at, self.lock, self.sleep = 1.0 / rate_per_s, 0.0, threading.Lock(), sleep
        self.calls = self.failed = self.rate_limited = self.attempts = self.skipped_busy = 0
        self.busy_until = 0.0
        self.errors: dict[str, int] = {}
        self.repairs: dict[str, int] = {}
        self.repaired_sessions = 0
        self.rejections: list[dict] = []
        self.strict: dict[str, int] = {}
        self.escalated: dict[int, int] = {}      # strict level -> messages raised to it

    def __call__(self, model: str, messages: list, thinking: bool = False, patient: bool = False) -> int | None:
        self.calls += 1
        budget = RATE_BUDGET_S if patient else SAMPLE_BUDGET_S
        first = None
        for attempt in range(5):
            fixed, rep = countable(messages, self.strict)
            body = audit.payload(json.dumps({"model": model, "messages": fixed}).encode(),
                                 no_thinking=not thinking)
            tokens, err = self.count_body(body, budget=budget)
            if tokens is not None:
                for k, v in rep.items():
                    if not k.startswith("_"):
                        self.repairs[k] = self.repairs.get(k, 0) + v
                if first:
                    self._reject(*first, recovered=True)   # counted, but worth fixing at the source
                return tokens
            m = _MSG_INDEX.search(err)
            if err.startswith("400") and first is None:
                # The shape as first refused (later tries are stricter and would hide the cause).
                first = (err, shape(body["messages"], int(m.group(1)) if m else None))
            if err.startswith("400") and m:
                # Make the named message stricter, for this and every later count: first only
                # basic blocks and fields, then all of it as text. Text always counts, so a
                # history shape not seen before costs a retry, not the count.
                idx = int(m.group(1))
                origin = rep.get("_origin") or []
                src = origin[idx] if idx < len(origin) else None
                h = _msg_hash(messages[src]) if src is not None else None
                if h is not None and self.strict.get(h, 0) < 2:
                    self.strict[h] = self.strict.get(h, 0) + 1
                    self.escalated[self.strict[h]] = self.escalated.get(self.strict[h], 0) + 1
                    continue
            break
        if err.startswith("rate limited"):
            self.rate_limited += 1
            self.busy_until = time.time() + BUSY_S       # ease off optional samples for a while
        else:
            self.failed += 1
            self.errors[err] = self.errors.get(err, 0) + 1
            m = _MSG_INDEX.search(err)
            self._reject(*(first or (err, shape(body["messages"], int(m.group(1)) if m else None))))
        return None

    def _reject(self, err: str, where: list, recovered: bool = False) -> None:
        """Keep up to 5 refusals of each kind (failed / recovered by a retry), one per error."""
        kind = [r for r in self.rejections if bool(r.get("recovered")) == recovered]
        key = re.sub(r"\d+", "N", err)
        if len(kind) < 5 and not any(re.sub(r"\d+", "N", r["error"]) == key for r in kind):
            self.rejections.append(dict({"error": err, "shape": where}, **({"recovered": True} if recovered else {})))

    def _pace(self, headers) -> None:
        h = {k.lower(): v for k, v in (headers or {}).items()}
        try:
            rem = int(h.get("anthropic-ratelimit-requests-remaining"))
        except (TypeError, ValueError):
            return
        reset = audit._parse_reset(h.get("anthropic-ratelimit-requests-reset"))
        if reset is None:
            return
        left = max(reset - time.time(), 0.0)
        # Spread what is left of the window evenly; with a few left, wait for the window.
        spacing = left / rem if rem > 2 else left + 1
        with self.lock:
            self.next_at = max(self.next_at, time.time() + min(spacing, left + 1))

    def busy(self) -> bool:
        """Anthropic's rate limit is saturated right now: optional samples are skipped."""
        return time.time() < self.busy_until

    def count_body(self, body: dict, beta: str | None = None,
                   budget: float | None = None) -> tuple[int | None, str]:
        """(input tokens, "") or (None, "<status> <message>" | "rate limited ...")."""
        headers = dict(self.headers)
        if beta:
            headers["anthropic-beta"] = ",".join(x for x in (headers.get("anthropic-beta"), beta) if x)
        start, attempt, err = time.time(), 0, "no response"
        while True:
            with self.lock:
                wait = max(self.next_at, 0) - time.time()
                self.next_at = max(self.next_at, time.time()) + self.gap
            if wait > 0:
                self.sleep(wait)
            self.attempts += 1
            req = urllib.request.Request(COUNT_URL, data=json.dumps(body).encode(), headers=headers,
                                         method="POST")
            retry_after = None
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    self._pace(r.headers)
                    return json.loads(r.read())["input_tokens"], ""
            except urllib.error.HTTPError as e:
                self._pace(e.headers)
                if e.code not in (429, 500, 502, 503, 504, 529):
                    try:
                        msg = (json.loads(e.read()).get("error") or {}).get("message", "")[:200]
                    except Exception:
                        msg = ""
                    return None, f"{e.code} {msg}"
                err = f"rate limited ({e.code})" if e.code == 429 else f"rate limited (overloaded {e.code})"
                try:
                    retry_after = float(e.headers.get("retry-after"))
                except (TypeError, ValueError):
                    retry_after = None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = f"rate limited (network: {type(e).__name__})"
            attempt += 1
            backoff = retry_after if retry_after is not None else min(60.0, 2.0 ** attempt)
            backoff *= 1 + random.random() * 0.5               # jitter: not everyone at once
            if time.time() - start + backoff > (budget if budget is not None else RATE_BUDGET_S):
                return None, err
            self.sleep(backoff)


def _pair(fn, a, b, patient: bool = False):
    """fn on both, in parallel. `patient`: the counter may wait longer for a rate limit
    (the count decides something, e.g. a compaction, rather than being one sample of many)."""
    if patient and _takes_patient(fn):
        inner = fn
        fn = lambda m: inner(m, patient=True)
    out = [None, None]
    t = threading.Thread(target=lambda: out.__setitem__(1, fn(b)))
    t.start()
    out[0] = fn(a)
    t.join()
    return out


def _takes_patient(fn) -> bool:
    import inspect
    try:
        return "patient" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# ---- thinking factor ----------------------------------------------------------

# What the factor is: the tokens Anthropic bills for Claude's earlier reasoning
# ("thinking"), per character of the signature it travels with. The benchmark
# needs it for the thinking TokenSaver removes (the tail of a break-time trim);
# the plain side is the transcript's real usage and never uses it.
#
# Measured by counting a request with and without its thinking. On Opus 5.5+
# thinking counts only on the exact bytes first sent (system, tools and every
# earlier message), so requests rebuilt from transcripts read ~0. Source A uses
# the exact sent requests TokenSaver keeps (the stored request with the sent-
# bytes ledger's messages: `_rewrite` changes only `messages`), and proves each
# count included the thinking with the binding check set to "error".
BINDING_BETA = "thinking-binding-controls-2026-08-01"
MIN_SIG_CHARS = 20_000
FACTOR_BOUNDS = (0.02, 0.5)


def _sent_requests(per_model: int = 4) -> dict[str, list[tuple[dict, int]]]:
    """{model: [(count_tokens body as sent, signature chars)]} from this machine's
    TokenSaver history: first the thinking-rate samples (`thinking`, kept 30
    days), then each session's latest main request whose messages the ledger
    holds (the 7-day debug copies). Read-only; nothing leaves the machine here."""
    import sqlite3
    from . import thinking, vault
    out: dict[str, list] = {}
    seen: set = set()
    for session, e in sorted(thinking._index()["samples"].items(), key=lambda kv: -kv[1].get("stored_at", 0)):
        sample = thinking.load(session)
        model = (sample or {}).get("model") or ""
        if sample and len(out.setdefault(model, [])) < per_model:
            out[model].append((sample, e.get("sig_chars") or bg._signature_chars(sample["messages"])))
            seen.add(thinking._safe(session))
    state = vault.VAULT_DIR / "proxy_state"
    if not state.is_dir() or not vault.DB_PATH.exists():
        return out
    try:
        conn = sqlite3.connect(f"file:{vault.DB_PATH}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return out
    try:
        for led_path in sorted(state.glob("*.bg.sent.json"), key=lambda p: -p.stat().st_mtime):
            try:
                sent = json.loads(led_path.read_text()).get("sent")
            except (OSError, ValueError):
                continue
            if not isinstance(sent, list) or bg._signature_chars(sent) < MIN_SIG_CHARS:
                continue
            session = led_path.name[:-len(".bg.sent.json")]
            if session in seen:
                continue                        # already measured from its sample
            for (eid,) in conn.execute("SELECT id FROM events WHERE kind='api_request' AND session_id=? "
                                       "ORDER BY id DESC LIMIT 5", (session,)):
                got = vault.read(eid)
                meta = vault.read_meta(eid) or {}
                if not got or meta.get("request_class") not in (None, "main") or meta.get("status") != 200:
                    continue
                rw = meta.get("rewrite") or {}
                if rw.get("fallback") or rw.get("error") or meta.get("thinking_dropped"):
                    break                       # this session's thinking is not known good
                try:
                    req = json.loads(got[1])
                except ValueError:
                    continue
                if len(req.get("messages") or []) != len(sent):
                    continue                    # not the request the ledger holds
                body = audit.payload(json.dumps(dict(req, messages=sent)).encode())
                model = req.get("model") or ""
                if body and len(out.setdefault(model, [])) < per_model:
                    out[model].append((body, bg._signature_chars(sent)))
                break
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return out


def _measure_sent(body: dict, sig: int, count: Counter) -> tuple[float | None, str]:
    """(tokens per signature char, "") on one exact sent request, or (None, why):
    the same measurement the proxy makes continuously (`audit.measure_thinking`)."""
    got = audit.measure_thinking(body, count.count_body)
    return got["factor"], got["why"]


def thinking_factors(sessions: list, count: Counter, per_model: int = 4) -> dict[str, dict]:
    """{model: {"value", "source": "sent" | "rebuilt" | "fallback", "n", "how"}}.
    First the proxy's own continuous measurements on this machine (enough of
    them for that model), then source A (exact sent requests, measured now),
    then source B (requests rebuilt from transcripts: fine for models without
    byte binding), then the fallback."""
    out: dict[str, dict] = {}
    tried: dict[str, list[str]] = {}            # why sent requests did not measure, per model
    models = {d.session.primary_model for d in sessions} | set(_sent_requests(per_model))
    for model in models:
        try:
            v, why, info = audit.signature_factor(model)
        except Exception:
            continue
        if info["source"] == "measured":
            out[model] = {"value": v, "source": "weekly", "n": info["n"],
                          "how": f"TokenSaver's weekly measurements on this machine: {info['n']} sent requests, "
                                 f"{info['sessions']} sessions"}
    for model, items in _sent_requests(per_model).items():
        if model in out:
            continue
        results = [_measure_sent(b, s, count) for b, s in items]
        good = [r for r, _ in results if r is not None]
        tried[model] = [why for r, why in results if r is None]
        if good:
            out[model] = {"value": sorted(good)[len(good) // 2], "source": "sent", "n": len(good),
                          "how": f"measured on this machine: {len(good)} sent requests, thinking verified valid"}
    picks: dict[str, list] = {}
    for d in sessions:
        messages, reqs = transcripts.api_history(d.session.path)
        if not reqs:
            continue
        entry = reqs[len(reqs) * 2 // 3]
        hist = transcripts.request_messages(messages, entry)
        sig = bg._signature_chars(hist)
        model = entry[1].model
        if model not in out and sig > MIN_SIG_CHARS and len(picks.setdefault(model, [])) < per_model:
            picks[model].append((hist, sig))
    for model, items in picks.items():
        ratios = []
        for hist, sig in items:
            w, wo = _pair(lambda t: count(model, hist, thinking=t), True, False)
            if w and wo and w > wo:
                ratios.append((w - wo) / sig)
        good = [r for r in ratios if FACTOR_BOUNDS[0] <= r <= FACTOR_BOUNDS[1]]
        if good:
            out[model] = {"value": sorted(good)[len(good) // 2], "source": "rebuilt", "n": len(good),
                          "how": f"measured on this machine: {len(good)} rebuilt requests"}
        else:
            out[model] = {"value": DEFAULT_SIG_FACTOR, "source": "fallback", "n": 0,
                          "how": "fallback: TokenSaver's reference value; not measurable here"}
    for model, whys in tried.items():
        if whys and out.get(model, {}).get("source") != "sent":
            fa = out.setdefault(model, {"value": DEFAULT_SIG_FACTOR, "source": "fallback", "n": 0,
                                        "how": "fallback: TokenSaver's reference value"})
            fa["how"] += f"; {len(whys)} sent request(s) tried: " + "; ".join(sorted(set(whys)))
    return out


# ---- one session ----------------------------------------------------------------

def replay(d, count: Counter, step: int, after_s: float, payback: float, factor: float,
           transform=None, prune=None, keepalive_ends: dict | None = None,
           main_ttl: str | None = None) -> dict:
    """One session. `transform(k, hist) -> messages` replaces tokensaver's
    rewrite with another tool's (experiments use it for Headroom); then no
    compaction is emulated. `prune(cand, hist) -> cand` (`--benchmark full`)
    runs the local AI stages on each compaction the replay applies; the
    decision to apply is made on the rules-only compaction, as the proxy does.
    `keepalive_ends` (the user's day ends): cache keep-alive replayed with the
    proxy's own rule (`keepalive.replay_bumps`): a break kept warm is priced as a
    cache read and its trim decided as on a warm return; every bump is paid for,
    including after a session's last request, where nobody came back.
    `main_ttl` ("1h"): the cache lifetime TokenSaver sets for the main conversation
    (`cachettl.choose`); the TokenSaver side is priced and decided with it, plain Claude Code
    keeps the lifetime each request really had."""
    from . import keepalive as ka
    kept_warm: set[int] = set()
    bump_events: list[tuple[int, int]] = []     # (request the bumps followed, how many)
    returned = False
    tinfo: dict = {}
    messages, entries = transcripts.api_history(d.session.path, tinfo)
    if not entries:
        return {"skipped": "no billed requests"}
    # When Claude Code compacted the history during a break (it does so by itself about 55
    # minutes in): live, keep-alive stops holding the session there (the old cache is dead).
    compacted_at = {start: sim._ts(ts) for start, ts in tinfo.get("boundaries") or [] if ts}
    reqs = [e[1] for e in entries]
    model = d.session.primary_model
    prices = [pricing.price_for(r.model) or d.price for r in reqs]
    # Each request's cache lifetime, from what its usage says it wrote (`cachettl`): a request
    # reads the previous one's entry while the gap is within that entry's lifetime.
    ttls = cachettl.per_request(reqs)
    ours_ttls = [main_ttl] * len(reqs) if main_ttl else ttls
    expired = lambda k, gap, t=ttls: k > 0 and gap > cachettl.seconds(t[k - 1])
    removed_at: dict[int, int] = {}
    sig_at: dict[int, int] = {}    # thinking signature chars TokenSaver removed (priced by `factor`)
    epochs: list[int] = []
    applied, epoch, decisions, applied_at = None, 0, 0, []
    failed = 0                     # counts Anthropic refused for this session

    # A session that went through TokenSaver: its usage numbers are what TokenSaver sent, so
    # plain Claude Code's size is rebuilt from the original messages (Claude Code's transcript
    # keeps them as written): Anthropic's count of them plus the session's fixed part (system
    # prompt and tools), measured where TokenSaver could not have trimmed anything but the
    # small every-message passes: before the session's first break (no compaction yet), as
    # its real size minus the count of the same messages rewritten that way.
    through = bool(getattr(d, "through_ts", False))
    plain_at: dict[int, int] = {}
    floor_samples: list[int] = []
    first_break = next((k for k in range(1, len(reqs))
                        if sim._ts(reqs[k].timestamp) - sim._ts(reqs[k - 1].timestamp) > after_s), len(reqs))

    def removed(hist, ours, k=None) -> tuple[int, int] | None:
        a, b = _pair(lambda m: count(model, m), hist, ours)
        if a is None or b is None:
            return None
        sig = bg._signature_chars(hist) - bg._signature_chars(ours)
        if through and k is not None:
            plain_at[k] = a + round(factor * bg._signature_chars(hist))
            if k < first_break:
                floor_samples.append(reqs[k].context - (b + round(factor * bg._signature_chars(ours))))
        return (a - b) + round(factor * sig), sig

    def floor_now() -> int | None:
        if not floor_samples:
            return None
        return max(sorted(floor_samples)[len(floor_samples) // 2], 0)

    last_epoch_start = 0
    for k, entry in enumerate(entries):
        r = entry[1]
        hist = transcripts.request_messages(messages, entry)
        gap = sim._ts(r.timestamp) - sim._ts(reqs[k - 1].timestamp) if k else 0
        if entry[2] != last_epoch_start:                       # Claude Code compacted: a new history
            applied, epoch, last_epoch_start = None, epoch + 1, entry[2]
        if transform is not None:
            ours = None
            if k % step == 0 or k == len(entries) - 1:
                ours = transform(k, hist)
            else:
                transform(k, hist)                             # stateful tools must see every request
        else:
            if applied is not None and bg.apply(hist, applied) is None:
                applied, epoch = None, epoch + 1
            decided = None                     # (cand, s_now, s_new, reps) when a compaction was priced
            if k and gap > after_s and bg.worth_compacting(hist):
                cand = bg.build(hist)
                if cand and (applied is None or cand["cut"] > applied["cut"]):
                    cand = dict(cand, tail_end=len(hist))
                    now = passes.rewrite_messages(bg.apply(hist, applied) if applied else hist, aliases="off")
                    new = passes.rewrite_messages(bg.apply(hist, cand), aliases="off")
                    n_now, n_new = _pair(lambda m: count(model, m), now, new, patient=True)
                    decisions += 1
                    if not (n_now and n_new):
                        failed += 1
                    if n_now and n_new:
                        n_now += round(factor * bg._signature_chars(now))
                        n_new += round(factor * bg._signature_chars(new))
                        # system + tools: not in the count (for a TokenSaver session, the floor
                        # measured before its first break: its usage is already trimmed)
                        floor = (floor_now() or 0) if through else max(r.context - n_now, 0)
                        s_now, s_new = n_now + floor, n_new + floor
                        p = prices[k]
                        w = cachettl.write_price(p, ours_ttls[k])
                        reps = (s_new * w - s_now * p.read) / max((s_now - s_new) * p.read, 1)
                        decided = (cand, s_now, s_new, reps)
            cold = expired(k, gap, ours_ttls)
            if keepalive_ends is not None and k and ours_ttls[k - 1] == "1h":   # live: 1-hour caches only
                # The break before this request: would keep-alive have kept the cache, and at what cost?
                big = k >= ka.BIG_REQUESTS or returned or reqs[k - 1].context >= ka.BIG_TOKENS
                warm_trim = decided is not None and decided[2] < decided[1] and decided[3] <= payback
                t_prev = sim._ts(reqs[k - 1].timestamp)
                ka_gap, new_history = gap, entry[2] != entries[k - 1][2]
                if new_history and entry[2] in compacted_at:
                    ka_gap = max(compacted_at[entry[2]] - t_prev, 0)
                n, warm = ka.replay_bumps(t_prev, ka_gap, big, keepalive_ends, worth_it=not warm_trim,
                                          size=reqs[k - 1].context)
                warm = warm and not new_history     # a new history is written whatever the cache holds
                if n:
                    bump_events.append((k - 1, n))
                if warm:
                    kept_warm.add(k)
                    cold = False
            if gap > 3600:
                returned = True
            if decided is not None:
                cand, s_now, s_new, reps = decided
                if s_new < s_now and (cold or reps <= payback):
                    if prune is not None:
                        cand = prune(cand, hist)
                    applied, epoch = cand, epoch + 1
                    applied_at.append(k)
            ours = None
            needed = (applied_at and applied_at[-1] == k) or k == len(entries) - 1 or k == 0
            sample = k % step == 0
            if sample and not needed and getattr(count, "busy", lambda: False)():
                count.skipped_busy += 1           # the rate limit is saturated: interpolate this one
                sample = False
            if sample or needed:
                ours = passes.rewrite_messages(bg.apply(hist, applied) if applied else hist, aliases="off")
        epochs.append(epoch)
        if ours is not None:
            g = removed(hist, ours, k)
            if g is not None:
                removed_at[k], sig_at[k] = g
            else:
                failed += 1

    def fill(at: dict[int, int]) -> list[int]:
        """One value per request: counted ones as is, the rest interpolated within the epoch."""
        vals = []
        for k, ep in enumerate(epochs):
            if k in at:
                vals.append(at[k])
                continue
            before = [j for j in at if j < k and epochs[j] == ep]
            after = [j for j in at if j > k and epochs[j] == ep]
            if before and after:
                a, b = max(before), min(after)
                vals.append(round(at[a] + (at[b] - at[a]) * (k - a) / (b - a)))
            else:
                vals.append(at[max(before)] if before else at[min(after)] if after else 0)
        return vals

    rem, sig_rem = fill(removed_at), fill(sig_at)

    if keepalive_ends is not None and transform is None and ours_ttls[-1] == "1h" \
            and not ka.closed_by_user(d.session.path):
        # After the session's last request: bumps for someone who never came back (as it runs
        # live while the session stays open; a /clear or /exit closed it, so none then).
        last = len(reqs) - 1
        big = len(reqs) >= ka.BIG_REQUESTS or returned or reqs[last].context >= ka.BIG_TOKENS
        n, _ = ka.replay_bumps(sim._ts(reqs[last].timestamp), None, big, keepalive_ends,
                              size=reqs[last].context)
        if n:
            bump_events.append((last, n))

    def price(sizes, rebuilt, warm=frozenset(), bumps=(), t=ttls) -> float:
        cost, prev = 0.0, None
        for k, (r, size, p) in enumerate(zip(reqs, sizes, prices)):
            gap = sim._ts(r.timestamp) - sim._ts(reqs[k - 1].timestamp) if k else 0
            hit = 0 if prev is None or (expired(k, gap, t) and k not in warm) or k in rebuilt else min(prev, size)
            cost += (hit * p.read + (size - hit) * cachettl.write_price(p, t[k]) + r.out * p.out) / 1e6
            prev = size
        # a bump reads the context its session had then
        return cost + sum(n * sizes[i] * prices[i].read / 1e6 for i, n in bumps)

    if through:
        if not plain_at or floor_now() is None:
            return {"skipped": "went through TokenSaver and could not be rebuilt (no counts)",
                    "through_tokensaver": True}
        fl = floor_now()
        plain = [max(v + fl, 1) for v in fill(plain_at)]
    else:
        plain = [r.context for r in reqs]
    ours = [max(s - x, 1) for s, x in zip(plain, rem)]
    # The same result with the thinking factor 30% lower / higher (compaction
    # decisions held fixed): how much the headline leans on that one number.
    ours_lo = [max(o + round(0.3 * factor * s), 1) for o, s in zip(ours, sig_rem)]
    ours_hi = [max(o - round(0.3 * factor * s), 1) for o, s in zip(ours, sig_rem)]
    real = sum((r.inp * p.inp + r.read * p.read + (r.write - r.write_5m) * p.write_1h
                + r.write_5m * p.write_5m + r.out * p.out) / 1e6 for r, p in zip(reqs, prices))
    breaks = sum(1 for k in range(1, len(reqs)) if sim._ts(reqs[k].timestamp) - sim._ts(reqs[k - 1].timestamp) > after_s)
    rebuilt = set(applied_at)
    bump_cost = sum(n * ours[i] * prices[i].read / 1e6 for i, n in bump_events)
    # What the kept-warm returns would otherwise have paid: the cached part written again.
    kept_saved = sum(min(ours[k - 1], ours[k]) * (prices[k].write_1h - prices[k].read) / 1e6
                     for k in kept_warm if k not in rebuilt)
    return {"requests": len(reqs), "model": model, "breaks": breaks, "counted": len(removed_at),
            "decisions": decisions, "compactions": len(applied_at), "count_failures": failed,
            "transcript_duplicates": tinfo.get("duplicates", 0),
            "plain_tokens": sum(plain), "ours_tokens": sum(ours),
            "plain_cost": round(price(plain, set()), 4),
            "ours_cost": round(price(ours, rebuilt, kept_warm, bump_events, ours_ttls), 4),
            "ours_cost_factor_lo": round(price(ours_lo, rebuilt, kept_warm, bump_events, ours_ttls), 4),
            "ours_cost_factor_hi": round(price(ours_hi, rebuilt, kept_warm, bump_events, ours_ttls), 4),
            # plain Claude Code with only the cache lifetime changed: that part of the saving
            "plain_cost_ours_ttl": round(price(plain, set(), t=ours_ttls), 4),
            "keepalive_bumps": sum(n for _, n in bump_events), "keepalive_kept": len(kept_warm),
            "keepalive_cost": round(bump_cost, 4), "keepalive_saved": round(kept_saved, 4),
            "thinking_chars_removed": sum(sig_rem), "real_cost": round(real, 4),
            "write_5m_tokens": sum(r.write_5m for r in reqs), "write_1h_tokens": sum(r.write_1h for r in reqs),
            "through_tokensaver": through,
            **({"live": live_facts(d.session.path.stem)} if through else {}),
            # what was billed, priced by the same cache model as plain (so the two compare fairly)
            **({"floor_tokens": floor_now(),
                "billed_model_cost": round(price([r.context for r in reqs], set()), 4)} if through else {})}


# ---- the whole machine ------------------------------------------------------------

def _pct(a: float, b: float) -> float:
    return 100 * (1 - a / b) if b else 0.0


def _family(model: str) -> str:
    for f in ("opus", "sonnet", "haiku", "fable", "mythos"):
        if f in model:
            return f
    return model or "?"


def _version(model: str) -> str:
    """'claude-opus-5-5' -> 'opus 5.5', 'claude-haiku-4-5-20251001' -> 'haiku 4.5': each
    version is its own model, with its own behaviour and prices."""
    parts = [p for p in (model or "?").split("-") if p != "claude" and not (p.isdigit() and len(p) == 8)]
    name = [p for p in parts if not p.isdigit()]
    nums = [p for p in parts if p.isdigit()]
    return " ".join(name) + (" " + ".".join(nums) if nums else "") if name else (model or "?")


def _bucket(n: int) -> str:
    return "under 50 requests" if n < 50 else "50-199 requests" if n < 200 else "200+ requests"


UNRELIABLE_SHARE = 0.05      # more failed counts than this: say so loudly


def reliability(rows: list[dict], count) -> list[str]:
    """A plain warning when too many token counts failed, with why it matters."""
    if not count.calls or count.failed / count.calls <= UNRELIABLE_SHARE:
        return []
    hit = sorted((r for r in rows if r.get("count_failures")), key=lambda r: -r["count_failures"])
    out = ["",
           f"*** WARNING: {count.failed:,} of {count.calls:,} token counts failed "
           f"({100 * count.failed / count.calls:.0f}%). The savings above are probably TOO LOW.",
           "    Why: the benchmark rebuilds each past conversation from Claude Code's transcript files",
           "    and asks Anthropic's token counter to measure it. If a rebuilt conversation does not",
           "    match what Claude Code really sent, Anthropic refuses to count it, and usually every",
           "    later request in that session too. Those requests are then estimated from earlier",
           "    ones, and no break-time trimming can be decided there, so those sessions show little",
           "    or no saving. It says nothing about TokenSaver itself, which sees the real requests.",
           "    Usual cause: a Claude Code update changed how it writes transcripts."]
    if hit:
        out.append("    Most affected sessions:")
        for r in hit[:3]:
            out.append(f"      {r['requests']:5d} requests, {r['model'].replace('claude-', '')}: {r['count_failures']} counts failed, "
                       f"shown {_pct(r['ours_cost'], r['plain_cost']):.1f}% cheaper (likely more)")
    if count.errors:
        k, v = max(count.errors.items(), key=lambda kv: kv[1])
        out.append(f"    Most common error ({v}x): {k[:150]}")
    out += ["    What to do: `tokensaver update`, then run the benchmark again. If this warning stays,",
            "    please send the saved result file: it lists the shape of each refused request (block",
            "    types only, no text from your sessions), which is what is needed to fix it."]
    return out


def commit() -> str | None:
    """The checkout's commit (read from .git, no git command), so a result file says which
    TokenSaver made it."""
    try:
        git = Path(__file__).resolve().parent.parent / ".git"
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = git / head[5:]
            if ref.exists():
                return ref.read_text().strip()[:12]
            for line in (git / "packed-refs").read_text().splitlines():
                if line.endswith(" " + head[5:]):
                    return line.split()[0][:12]
            return None
        return head[:12]
    except OSError:
        return None


def live_facts(session_id: str) -> dict:
    """What the live proxy recorded for one session (this machine's vault; counts and reasons
    only), so an "as it ran" figure that differs from the replay explains itself: which code
    ran, whether it rewrote, which compactions it prepared, applied or dropped and why, whether
    it fell back to plain relay, and the free token counts it made of the original against
    what it sent."""
    import collections
    import sqlite3
    from . import vault
    out: dict = {"requests": 0, "rewritten": 0, "errors": 0, "fallback": 0, "applied": 0,
                 "skipped": {}, "prepared": 0, "dropped_pending": 0, "fallback_events": 0,
                 "bumps": 0, "codes": [], "first": None, "last": None,
                 "audit_original": 0, "audit_sent": 0, "audited": 0}
    if not session_id or not vault.DB_PATH.exists():
        return out
    skipped: collections.Counter = collections.Counter()
    codes: dict = {}
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        rows = conn.execute("SELECT id, kind, ts FROM events WHERE session_id = ? AND kind IN "
                            "('api_request', 'token_audit', 'bg_compaction_prepared', 'bg_dropped_pending', "
                            "'rewrite_fallback', 'cache_bump') ORDER BY id", (session_id,)).fetchall()
        conn.close()
    except Exception:
        return out
    counted = {"bg_compaction_prepared": "prepared", "bg_dropped_pending": "dropped_pending",
               "rewrite_fallback": "fallback_events", "cache_bump": "bumps"}
    for rid, kind, ts in rows:
        if kind in counted:
            out[counted[kind]] += 1
            continue
        try:
            m = vault.read_meta(rid)
        except Exception:
            continue
        if kind == "token_audit":
            if m.get("tokens_original") and m.get("tokens_sent"):
                out["audit_original"] += int(m["tokens_original"])
                out["audit_sent"] += int(m["tokens_sent"])
                out["audited"] += 1
            continue
        if kind != "api_request" or m.get("request_class") not in ("main", None) \
                or str(m.get("path") or "").split("?")[0] != "/v1/messages":
            continue
        rw = m.get("rewrite") or {}
        out["requests"] += 1
        out["first"] = out["first"] or ts
        out["last"] = ts
        code = m.get("code") or "unrecorded"
        codes[code] = codes.get(code, 0) + 1
        if rw.get("bytes_out") and rw.get("bytes_in") and rw["bytes_out"] < rw["bytes_in"]:
            out["rewritten"] += 1
        out["errors"] += bool(rw.get("error"))
        out["fallback"] += bool(rw.get("fallback"))
        out["applied"] += bool(rw.get("bg_applied"))
        if rw.get("bg_skipped"):
            skipped[str(rw["bg_skipped"])[:80]] += 1
    out["skipped"] = dict(skipped.most_common(5))
    out["codes"] = sorted(codes.items(), key=lambda kv: -kv[1])
    return out


def live_line(rows: list[dict]) -> list[str]:
    """Why "as it ran" differs from the replay, from what the live proxy recorded."""
    ts = [r for r in rows if r.get("through_tokensaver") and r.get("live")]
    if not ts:
        return []
    L = [r["live"] for r in ts]
    tot = lambda k: sum(x.get(k) or 0 for x in L)
    seen = tot("requests")
    if not seen:
        return [" as it ran: this machine's vault has no record of those sessions' requests "
                "(older than its retention, or recorded under another session id)"]
    replay_applied = sum(r.get("compactions") or 0 for r in ts)
    codes: dict = {}
    for x in L:
        for c, n in x.get("codes") or []:
            codes[c] = codes.get(c, 0) + n
    skipped: dict = {}
    for x in L:
        for why, n in (x.get("skipped") or {}).items():
            skipped[why] = skipped.get(why, 0) + n
    first = min(x["first"] for x in L if x.get("first"))
    last = max(x["last"] for x in L if x.get("last"))
    out = [f" as it ran ({first[:10]} to {last[:10]}, {seen:,} requests the proxy saw): "
           f"{tot('rewritten'):,} rewritten, {tot('applied')} compactions applied live vs {replay_applied} "
           f"in today's replay; {tot('prepared')} prepared, {tot('dropped_pending')} dropped as stale; "
           f"{tot('fallback')} sent plain after a fallback ({tot('fallback_events')} fallback events), "
           f"{tot('errors')} rewrite errors; {tot('bumps')} keep-alive bumps"]
    if tot("audited"):
        out.append(f"   live token counts on {tot('audited'):,} requests: {tot('audit_original'):,} original -> "
                   f"{tot('audit_sent'):,} sent ({_pct(tot('audit_sent'), tot('audit_original')):.1f}% fewer)")
    if skipped:
        out.append("   compactions not applied live: "
                   + "; ".join(f"{why} x{n}" for why, n in sorted(skipped.items(), key=lambda kv: -kv[1])[:3]))
    out.append("   proxy code: " + ", ".join(f"{c} x{n}" for c, n in
                                             sorted(codes.items(), key=lambda kv: -kv[1])[:4]))
    return out


def measured_line(rows: list[dict]) -> str:
    """For sessions that really went through TokenSaver: what you actually paid, against the
    rebuilt plain cost and against the replay of today's TokenSaver on the same sessions."""
    ts = [r for r in rows if r.get("through_tokensaver")]
    if not ts:
        return " sessions already through TokenSaver: none"
    plain = sum(r["plain_cost"] for r in ts)
    real = sum(r["real_cost"] for r in ts)
    ours = sum(r["ours_cost"] for r in ts)
    # Both sides through the same cache model: what was billed (sizes TokenSaver sent) against the
    # plain sizes rebuilt from the original messages. The real bill is shown beside it: where it is
    # above the billed model, there were cache misses the model does not expect.
    billed = sum(r.get("billed_model_cost") or r["real_cost"] for r in ts)
    return (f" {len(ts)} sessions already went through TokenSaver: {_pct(billed, plain):.1f}% cheaper as it ran "
            f"(${billed:,.2f} for what was sent vs ${plain:,.2f} plain, rebuilt from the original messages, "
            f"the same cost model; you were billed ${real:,.2f}). Today's TokenSaver on them, replayed: "
            f"{_pct(ours, plain):.1f}% cheaper")


def cache_lifetime_line(rows: list[dict], decision: dict | None = None, main_ttl: str | None = None) -> str:
    """Which cache lifetime the sessions used (plain is priced by each request's own), and what
    TokenSaver's choice for the main conversation is worth."""
    short = sum(1 for r in rows if r.get("write_5m_tokens", 0) > r.get("write_1h_tokens", 0))
    ran = (f"5 minutes on {short} of {len(rows)} sessions (Claude Code uses 1 hour only for a Claude "
           "subscription within its plan's usage)") if short else "1 hour"
    acc = f"; the cache model predicted {100 * decision['accuracy']:.0f}% of real cache reads" \
        if decision and decision.get("accuracy") is not None else ""
    if main_ttl != "1h" or not short:
        why = f"; TokenSaver leaves it ({decision['why']})" if decision and short else ""
        return f" cache lifetime: {ran}{why}{acc}"
    plain = sum(r.get("plain_cost", 0) for r in rows)
    with_1h = sum(r.get("plain_cost_ours_ttl", r.get("plain_cost", 0)) for r in rows)
    return (f" cache lifetime: {ran}; TokenSaver sets 1 hour for the main conversation (included above): "
            f"${plain:,.2f} -> ${with_1h:,.2f} on its own, and keep-alive then applies"
            f"{acc}; chosen on this same history")


def keepalive_line(rows: list[dict], ends) -> str:
    if ends is None:
        return " cache keep-alive: off (not included)"
    bumps = sum(r.get("keepalive_bumps", 0) for r in rows)
    kept = sum(r.get("keepalive_kept", 0) for r in rows)
    cost = sum(r.get("keepalive_cost", 0) for r in rows)
    saved = sum(r.get("keepalive_saved", 0) for r in rows)
    return (f" cache keep-alive (included above): {bumps} bumps kept {kept} breaks warm; "
            f"spent ${cost:,.2f}, saved ~${saved:,.2f} in rewrites")


def rate_limit_warning(count) -> list[str]:
    """Counts estimated because Anthropic's rate limit was saturated: given up after waiting,
    or skipped while it was busy. Interpolated from the counts around them (the same as the
    requests a normal run does not count), so a few cost little; many are worth a rerun."""
    est = count.rate_limited + getattr(count, "skipped_busy", 0)
    if not est:
        return []
    share = est / max(count.calls + getattr(count, "skipped_busy", 0), 1)
    out = ["", f" {est:,} token counts were estimated because Anthropic's rate limit was busy "
               f"({count.rate_limited} gave up after waiting, {getattr(count, 'skipped_busy', 0)} skipped while busy)."]
    if share > UNRELIABLE_SHARE:
        out += [f"*** That is {100 * share:.0f}% of the counts: run the benchmark again later for a precise result.",
                "    The limit is shared by everyone on your account (for example many people running it at once)."]
    return out


def summarize(rows: list[dict]) -> list[str]:
    ok = [r for r in rows if "skipped" not in r]
    lines = []

    def line(label, group):
        pt, ot = sum(r["plain_tokens"] for r in group), sum(r["ours_tokens"] for r in group)
        pc, oc = sum(r["plain_cost"] for r in group), sum(r["ours_cost"] for r in group)
        lines.append(f"  {label:26s} {len(group):4d} sessions  {_pct(ot, pt):5.1f}% fewer tokens  "
                     f"{_pct(oc, pc):5.1f}% cheaper  (${pc:,.2f} -> ${oc:,.2f})")
    line("ALL", ok)
    for title, key in (("by model", lambda r: _version(r["model"])), ("by length", lambda r: _bucket(r["requests"])),
                       ("by compactions", lambda r: "no compaction" if not r["compactions"]
                        else "1 compaction" if r["compactions"] == 1 else "2+ compactions")):
        lines.append(f" {title}:")
        groups: dict[str, list] = {}
        for r in ok:
            groups.setdefault(key(r), []).append(r)
        for g in sorted(groups):
            line(g, groups[g])
    return lines


def factor_sensitivity(rows: list) -> list[str]:
    """One line, only when some session priced its removed thinking with the fallback:
    how far the headline would move if that value were 30% off either way."""
    fb = [r for r in rows if r.get("factor_source") == "fallback" and r.get("thinking_chars_removed")]
    if not fb:
        return []
    plain = sum(r["plain_cost"] for r in rows)
    base = sum(r["ours_cost"] for r in rows)
    lo = base - sum(r["ours_cost"] - r["ours_cost_factor_lo"] for r in fb)
    hi = base - sum(r["ours_cost"] - r["ours_cost_factor_hi"] for r in fb)
    spread = max(abs(_pct(lo, plain) - _pct(base, plain)), abs(_pct(hi, plain) - _pct(base, plain)))
    return [f" thinking fallback used for {len(fb)} session(s): if {DEFAULT_SIG_FACTOR:.3f} were 30% off, "
            f"\"cheaper\" would move by at most {spread:.1f} points"]


# Measured on an M3 Max (64 GB), experiments/final_check.py, 8 real breaks, 2026-10-23: the
# local AI stages took 291-1214 s per break (median ~650 s) with the 35B; Nimble alone is
# quicker (fast tier, no deep stages or check), here estimated at a third of that.
SECONDS_PER_BREAK = {"deep": 650, "fast": 220}
FULL_MIN_CONTEXT = 60_000      # a break below this rarely has old history worth compacting


def full_estimate(data, after_s: float, tier: str) -> tuple[int, float]:
    """(breaks that may get a compaction, seconds of local AI) for `--benchmark full`:
    an upper bound, since only breaks whose compaction pays are pruned."""
    n = 0
    for d in data:
        reqs = d.requests
        for k in range(1, len(reqs)):
            if sim._ts(reqs[k].timestamp) - sim._ts(reqs[k - 1].timestamp) > after_s \
                    and reqs[k].context >= FULL_MIN_CONTEXT:
                n += 1
    return n, n * SECONDS_PER_BREAK.get(tier, SECONDS_PER_BREAK["deep"])


def _hours(seconds: float) -> str:
    h = seconds / 3600
    return f"about {h:.0f} hours" if h >= 1.5 else f"about {max(1, round(seconds / 60))} minutes"


def run(step: int = 8, since: str | None = None, yes: bool = False, out: Path | None = None,
        mode: str = "fast") -> int:
    # Sessions that already went through TokenSaver are included: their plain sizes are rebuilt
    # from the original messages (see replay), so before, after, or only-with TokenSaver all count.
    data = [d for d in sim.sessions(include_rewritten=True) if not since or d.session.ended[:10] >= since]
    data = [d for d in data if all(r.model.startswith("claude") for r in d.requests)]
    n_req = sum(len(d.requests) for d in data)
    after_s, payback = config.SPECS["bg_after_min"].default * 60, config.SPECS["bg_warm_payback_max"].default
    tier = None
    if mode == "full":
        from . import localai, prune as pr
        tier, why = pr.tier()
        if tier == "off":
            print(f"`--benchmark full` runs the local AI models, and this machine has none ({why}; "
                  f"local AI: {localai.describe()}).")
            print("Run `tokensaver --benchmark` (fast) instead: it covers everything except local AI.")
            return 2
        step = 1
    calls = n_req * 2 // step + 50
    print(f"TokenSaver benchmark {__version__} ({mode}): {len(data)} Claude Code sessions, "
          f"{n_req:,} requests on this machine.")
    print("It replays each one through today's TokenSaver and compares with what you actually sent.")
    n_ts = sum(1 for d in data if getattr(d, "through_ts", False))
    if n_ts:
        print(f"{n_ts} of them already went through TokenSaver: their plain Claude Code size is rebuilt from "
              "the original messages, and what you paid is shown as a measured saving.")
    print(f"About {calls:,} token counts (Anthropic's count_tokens endpoint: free, no messages are created),")
    print(f"roughly {max(1, calls // 250)}-{max(2, calls // 120)} minutes. They send your sessions' messages to Anthropic,")
    print("which processed them originally. The result file holds numbers only.")
    if mode == "full":
        breaks, secs = full_estimate(data, after_s, tier)
        print(f"Full mode also runs local AI ({tier} tier) on each break that gets a compaction: up to "
              f"{breaks} breaks, {_hours(secs + calls / 180 * 60)} in all on this machine. It can run "
              "overnight; the models pause while you use the computer hard.")
    else:
        print("Fast mode leaves out local AI (about 0.5-1% more where a machine runs it): "
              "`tokensaver --benchmark full` includes it and takes hours.")
    cred = credentials()
    if cred is None:
        print("\nNo credentials: log in to Claude Code, or set ANTHROPIC_API_KEY.")
        return 2
    auth, source = cred
    print(f"Credentials: your {source} (read here, never printed or stored).")
    if not yes:
        if not sys.stdin.isatty():
            print("\nRun with --yes to start.")
            return 3
        if input("\nStart? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    count = Counter(auth)
    # Cache keep-alive is part of TokenSaver unless switched off: replayed with the proxy's
    # own rule, on this user's day ends from their interactive sessions.
    from . import keepalive as ka
    ka_ends = None
    if config.get("cache_keepalive") != "off":
        ka_ends = ka.day_ends([sim._ts(r.timestamp) for d in data
                               if ka.interactive(f"(external, {ka._entrypoint(d.session.path) or 'cli'})")
                               for r in d.requests])
    # The main conversation's cache lifetime as TokenSaver would set it here (`cachettl`):
    # 1 hour where this user's history shows it is cheaper, unless they set it themselves.
    ttl_now, ttl_ours = cachettl.current()
    ttl_decision = cachettl.choose()
    main_ttl = None
    if (ttl_now is None or ttl_ours) and cachettl.load_state().get("mode") != "off" \
            and (ttl_decision["set"] or cachettl.load_state().get("mode") == "on"):
        main_ttl = "1h"
    pruner = None
    if mode == "full":
        import tempfile
        from . import proxy
        state = Path(tempfile.mkdtemp(prefix="tokensaver-bench-"))
        seq = [0]

        def pruner(cand, hist):
            seq[0] += 1
            t = time.time()
            got, _ = proxy.prune_offline(f"bench-{seq[0]}", hist, dict(cand), state)
            print(f"      local AI on a break: {cand['new_tokens']:,} -> {got.get('new_tokens', cand['new_tokens']):,} "
                  f"tokens ({time.time() - t:.0f}s)", flush=True)
            return dict(cand, **{k: got[k] for k in ("text", "new_tokens", "drop_refs", "stage") if k in got})
    print("\nmeasuring thinking tokens per model...", flush=True)
    factors = thinking_factors(data, count)
    for m, fa in factors.items():
        print(f"  {_version(m)}: {fa['value']:.3f} tokens per signature char ({fa['how']})")
    fallback = {"value": DEFAULT_SIG_FACTOR, "source": "fallback", "n": 0, "how": "fallback"}
    rows, t0 = [], time.time()
    # The replay's own size estimates (worth_compacting, choose_cut) use the same
    # per-model rate as its arithmetic, not the default.
    bg.SIG_FACTOR = lambda model=None: factors.get(model, fallback)["value"]
    for i, d in enumerate(sorted(data, key=lambda d: len(d.requests)), 1):
        fa = factors.get(d.session.primary_model, fallback)
        f = fa["value"]
        bg.MODEL.set(d.session.primary_model)
        t = time.time()
        repaired_before = sum(count.repairs.values())
        try:
            r = replay(d, count, step, after_s, payback, f, prune=pruner, keepalive_ends=ka_ends,
                       main_ttl=main_ttl)
        except Exception as exc:
            r = {"skipped": f"{type(exc).__name__}"}
        if sum(count.repairs.values()) > repaired_before:
            count.repaired_sessions += 1
        r["session"] = hashlib.sha256(d.stem.encode()).hexdigest()[:8]     # not the real id
        r["factor_source"] = fa["source"]
        rows.append(r)
        if "skipped" in r:
            print(f"  [{i}/{len(data)}] skipped: {r['skipped']}", flush=True)
            continue
        print(f"  [{i}/{len(data)}] {r['requests']:5d} requests, {_version(r['model']):10s} "
              f"{_pct(r['ours_tokens'], r['plain_tokens']):5.1f}% tokens  {_pct(r['ours_cost'], r['plain_cost']):5.1f}% cost  "
              f"compactions {r['compactions']}  ({time.time() - t:.0f}s)", flush=True)
    ok = [r for r in rows if "skipped" not in r]
    plainside = [r for r in ok if not r.get("through_tokensaver")]
    pc, rc = sum(r["plain_cost"] for r in plainside), sum(r["real_cost"] for r in plainside)
    lines = ["", "What TokenSaver would have saved on this machine (defaults, vs plain Claude Code, "
                 + ("with local AI, every request counted):" if mode == "full" else
                    "without local AI: `--benchmark full` includes it):")]
    lines += summarize(rows)
    lines += [f" cost model check (sessions without TokenSaver): plain modelled ${pc:,.2f} vs your real bill ${rc:,.2f}",
              cache_lifetime_line(ok, ttl_decision, main_ttl),
              measured_line(ok),
              *live_line(ok),
              f" token counts: {count.calls:,} calls, {count.failed} failed"
              + (f" ({'; '.join(f'{k} x{v}' for k, v in sorted(count.errors.items(), key=lambda kv: -kv[1])[:3])})"
                 if count.errors else "")
              + (f", {count.rate_limited} gave up on a rate limit after waiting" if count.rate_limited else ""),
              (" history repairs so Anthropic's counter accepts it (the same on both sides): "
               + ", ".join(f"{v} {k}" for k, v in sorted(count.repairs.items()))
               + f" in {count.repaired_sessions} sessions") if count.repairs else " history repairs: none needed",
              *([f" transcripts: {dups:,} repeated records or tool calls left out (the API saw each once)"]
                if (dups := sum(r.get("transcript_duplicates") or 0 for r in rows)) else []),
              *([" refused counts retried with the named message made stricter: "
                 + ", ".join(f"{v} at level {k}" for k, v in sorted(count.escalated.items()))]
                if count.escalated else []),
              keepalive_line(ok, ka_ends),
              # no same-quote nesting inside f-strings: Python before 3.12 (macOS's own 3.9) rejects it
              " thinking: " + ", ".join(f"{_version(m)} {fa['value']:.3f} ({fa['source']})"
                                        for m, fa in factors.items()),
              f" took {time.time() - t0:.0f}s"]
    lines += factor_sensitivity(ok)
    lines += reliability(ok, count)
    lines += rate_limit_warning(count)
    print("\n".join(lines))
    report = {"tokensaver": __version__, "commit": commit(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "step": step, "mode": mode,
              "settings": {"bg_after_min": after_s / 60, "bg_warm_payback_max": payback},
              "cache_ttl": {"main_ttl": main_ttl, **ttl_decision},
              "thinking_factors": factors,
              "counts": {"calls": count.calls, "failed": count.failed, "rate_limited": count.rate_limited,
                         "attempts": count.attempts, "repairs": count.repairs,
                         "repaired_sessions": count.repaired_sessions,
                         "escalated": {str(k): v for k, v in sorted(count.escalated.items())},
                         "transcript_duplicates": sum(r.get("transcript_duplicates") or 0 for r in rows)},
              "rejections": count.rejections,
              "summary": lines[1:], "sessions": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = out or OUT_DIR / f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"\nSaved {path} (numbers only; safe to share).")
    return 0
