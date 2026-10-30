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

Not modelled (each would move the result a little): re-reads of cut content
(0 measured in a real trial) and the optional local-model prune stages
(about 0.5-1% more). The compaction itself is rules only, as replayed here.

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
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import names
from . import __version__, audit, background as bg, config, passes, pricing, sim, transcripts

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


def _countable(x):
    """A copy count_tokens accepts without the session's tool list: a tool-search
    reference becomes a text block naming the tool (the same on both sides)."""
    if isinstance(x, list):
        return [_countable(v) for v in x]
    if isinstance(x, dict):
        if x.get("type") == "tool_reference":
            return {"type": "text", "text": f"[tool {x.get('tool_name') or x.get('name') or ''}]"}
        return {k: _countable(v) for k, v in x.items()}
    return x


class Counter:
    def __init__(self, auth: dict, rate_per_s: float = 5.0):
        self.headers = dict(auth, **{"anthropic-version": "2023-06-01", "content-type": "application/json"})
        self.gap, self.last, self.lock = 1.0 / rate_per_s, 0.0, threading.Lock()
        self.calls = self.failed = 0
        self.errors: dict[str, int] = {}

    def __call__(self, model: str, messages: list, thinking: bool = False) -> int | None:
        body = audit.payload(json.dumps({"model": model, "messages": _countable(messages)}).encode(),
                             no_thinking=not thinking)
        tokens, err = self.count_body(body)
        if tokens is None:
            self.errors[err] = self.errors.get(err, 0) + 1
            self.failed += 1
        return tokens

    def count_body(self, body: dict, beta: str | None = None) -> tuple[int | None, str]:
        """(input tokens, "") or (None, "<status> <message>") for a whole count_tokens
        body. Errors are not tallied here: a caller probing validity expects some."""
        headers = dict(self.headers)
        if beta:
            headers["anthropic-beta"] = ",".join(x for x in (headers.get("anthropic-beta"), beta) if x)
        err = "no response"
        for attempt in range(5):
            with self.lock:
                wait = self.last + self.gap - time.time()
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
                self.calls += 1
            req = urllib.request.Request(COUNT_URL, data=json.dumps(body).encode(), headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    return json.loads(r.read())["input_tokens"], ""
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503, 529):
                    time.sleep(5 * (attempt + 1))
                    continue
                try:
                    msg = (json.loads(e.read()).get("error") or {}).get("message", "")[:80]
                except Exception:
                    msg = ""
                return None, f"{e.code} {msg}"
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(5 * (attempt + 1))
        return None, err


def _pair(fn, a, b):
    out = [None, None]
    t = threading.Thread(target=lambda: out.__setitem__(1, fn(b)))
    t.start()
    out[0] = fn(a)
    t.join()
    return out


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
    TokenSaver history: each session's latest main request whose messages the
    ledger holds. Read-only; nothing leaves the machine here."""
    import sqlite3
    from . import vault
    out: dict[str, list] = {}
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
           transform=None) -> dict:
    """One session. `transform(k, hist) -> messages` replaces tokensaver's
    rewrite with another tool's (experiments use it for Headroom); then no
    compaction is emulated."""
    messages, entries = transcripts.api_history(d.session.path)
    if not entries:
        return {"skipped": "no billed requests"}
    reqs = [e[1] for e in entries]
    model = d.session.primary_model
    prices = [pricing.price_for(r.model) or d.price for r in reqs]
    removed_at: dict[int, int] = {}
    sig_at: dict[int, int] = {}    # thinking signature chars TokenSaver removed (priced by `factor`)
    epochs: list[int] = []
    applied, epoch, decisions, applied_at = None, 0, 0, []
    failed = 0                     # counts Anthropic refused for this session

    def removed(hist, ours) -> tuple[int, int] | None:
        a, b = _pair(lambda m: count(model, m), hist, ours)
        if a is None or b is None:
            return None
        sig = bg._signature_chars(hist) - bg._signature_chars(ours)
        return (a - b) + round(factor * sig), sig

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
            if k and gap > after_s and bg.worth_compacting(hist):
                cand = bg.build(hist)
                if cand and (applied is None or cand["cut"] > applied["cut"]):
                    cand = dict(cand, tail_end=len(hist))
                    now = passes.rewrite_messages(bg.apply(hist, applied) if applied else hist, aliases="off")
                    new = passes.rewrite_messages(bg.apply(hist, cand), aliases="off")
                    n_now, n_new = _pair(lambda m: count(model, m), now, new)
                    decisions += 1
                    if not (n_now and n_new):
                        failed += 1
                    if n_now and n_new:
                        n_now += round(factor * bg._signature_chars(now))
                        n_new += round(factor * bg._signature_chars(new))
                        floor = max(r.context - n_now, 0)       # system + tools: not in the count
                        s_now, s_new = n_now + floor, n_new + floor
                        p = prices[k]
                        reps = (s_new * p.write_1h - s_now * p.read) / max((s_now - s_new) * p.read, 1)
                        if s_new < s_now and (gap > 3600 or reps <= payback):
                            applied, epoch = cand, epoch + 1
                            applied_at.append(k)
            ours = None
            if k % step == 0 or (applied_at and applied_at[-1] == k) or k == len(entries) - 1:
                ours = passes.rewrite_messages(bg.apply(hist, applied) if applied else hist, aliases="off")
        epochs.append(epoch)
        if ours is not None:
            g = removed(hist, ours)
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

    def price(sizes, rebuilt) -> float:
        cost, prev = 0.0, None
        for k, (r, size, p) in enumerate(zip(reqs, sizes, prices)):
            gap = sim._ts(r.timestamp) - sim._ts(reqs[k - 1].timestamp) if k else 0
            hit = 0 if prev is None or gap > 3600 or k in rebuilt else min(prev, size)
            cost += (hit * p.read + (size - hit) * p.write_1h + r.out * p.out) / 1e6
            prev = size
        return cost

    plain = [r.context for r in reqs]
    ours = [max(s - x, 1) for s, x in zip(plain, rem)]
    # The same result with the thinking factor 30% lower / higher (compaction
    # decisions held fixed): how much the headline leans on that one number.
    ours_lo = [max(o + round(0.3 * factor * s), 1) for o, s in zip(ours, sig_rem)]
    ours_hi = [max(o - round(0.3 * factor * s), 1) for o, s in zip(ours, sig_rem)]
    real = sum((r.inp * p.inp + r.read * p.read + (r.write - r.write_5m) * p.write_1h
                + r.write_5m * p.write_5m + r.out * p.out) / 1e6 for r, p in zip(reqs, prices))
    breaks = sum(1 for k in range(1, len(reqs)) if sim._ts(reqs[k].timestamp) - sim._ts(reqs[k - 1].timestamp) > after_s)
    return {"requests": len(reqs), "model": model, "breaks": breaks, "counted": len(removed_at),
            "decisions": decisions, "compactions": len(applied_at), "count_failures": failed,
            "plain_tokens": sum(plain), "ours_tokens": sum(ours),
            "plain_cost": round(price(plain, set()), 4), "ours_cost": round(price(ours, set(applied_at)), 4),
            "ours_cost_factor_lo": round(price(ours_lo, set(applied_at)), 4),
            "ours_cost_factor_hi": round(price(ours_hi, set(applied_at)), 4),
            "thinking_chars_removed": sum(sig_rem), "real_cost": round(real, 4)}


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
            "    please open an issue with the saved result file (numbers only, safe to share)."]
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


def run(step: int = 8, since: str | None = None, yes: bool = False, out: Path | None = None) -> int:
    data = [d for d in sim.sessions() if not since or d.session.ended[:10] >= since]
    data = [d for d in data if all(r.model.startswith("claude") for r in d.requests)]
    n_req = sum(len(d.requests) for d in data)
    calls = n_req * 2 // step + 50
    print(f"TokenSaver benchmark {__version__}: {len(data)} Claude Code sessions, {n_req:,} requests on this machine.")
    print("It replays each one through today's TokenSaver and compares with what you actually sent.")
    print(f"About {calls:,} token counts (Anthropic's count_tokens endpoint: free, no messages are created),")
    print(f"roughly {max(1, calls // 250)}-{max(2, calls // 120)} minutes. They send your sessions' messages to Anthropic,")
    print("which processed them originally. The result file holds numbers only.")
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
    after_s, payback = config.SPECS["bg_after_min"].default * 60, config.SPECS["bg_warm_payback_max"].default
    print("\nmeasuring thinking tokens per model...", flush=True)
    factors = thinking_factors(data, count)
    for m, fa in factors.items():
        print(f"  {_version(m)}: {fa['value']:.3f} tokens per signature char ({fa['how']})")
    fallback = {"value": DEFAULT_SIG_FACTOR, "source": "fallback", "n": 0, "how": "fallback"}
    rows, t0 = [], time.time()
    for i, d in enumerate(sorted(data, key=lambda d: len(d.requests)), 1):
        fa = factors.get(d.session.primary_model, fallback)
        f = fa["value"]
        t = time.time()
        try:
            r = replay(d, count, step, after_s, payback, f)
        except Exception as exc:
            r = {"skipped": f"{type(exc).__name__}"}
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
    pc, rc = sum(r["plain_cost"] for r in ok), sum(r["real_cost"] for r in ok)
    lines = ["", "What TokenSaver would have saved on this machine (defaults, vs plain Claude Code):"]
    lines += summarize(rows)
    lines += [f" cost model check: plain modelled ${pc:,.2f} vs your real bill ${rc:,.2f}",
              f" token counts: {count.calls:,} calls, {count.failed} failed"
              + (f" ({'; '.join(f'{k} x{v}' for k, v in sorted(count.errors.items(), key=lambda kv: -kv[1])[:3])})"
                 if count.errors else ""),
              # no same-quote nesting inside f-strings: Python before 3.12 (macOS's own 3.9) rejects it
              " thinking: " + ", ".join(f"{_version(m)} {fa['value']:.3f} ({fa['source']})"
                                        for m, fa in factors.items()),
              f" took {time.time() - t0:.0f}s"]
    lines += factor_sensitivity(ok)
    lines += reliability(ok, count)
    print("\n".join(lines))
    report = {"tokensaver": __version__, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "step": step,
              "settings": {"bg_after_min": after_s / 60, "bg_warm_payback_max": payback},
              "thinking_factors": factors,
              "counts": {"calls": count.calls, "failed": count.failed},
              "summary": lines[1:], "sessions": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = out or OUT_DIR / f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"\nSaved {path} (numbers only; safe to share).")
    return 0
