"""Headroom vs TokenSaver on the same real request bodies (local, $0).

Replays one proxied session's stored request bodies (the vault keeps what
Claude Code sent, before any rewrite) in order, three ways:

- headroom/<profile>: like its Anthropic proxy. Each request reuses the bytes
  it returned for the prefix it already sent (append-only), so only new
  messages are compressed, once, at first send (`frozen_message_count`).
  Profiles: "coding" (its proxy default) and "agent-90" (its aggressive one,
  ML text compression).
- TokenSaver: forward passes (aliases off) plus background compaction built
  deterministically (`background.build`, no QA gate) and applied with the
  proxy's rules: after a quiet gap over `bg_after_min`, cold always, warm only
  within the payback limit; the same bytes kept until Claude Code rebuilds
  the history.
- plain: the bodies as sent.

Pricing (the same for all): the real billed size of each original request
(input + cache read + cache write, from the vault) is scaled by each
variant's characters; the cache is emulated by the longest common prefix with
the previous request's bytes (read), the rest written at 1h; a gap over 1h
writes everything. Output tokens are the real ones. Headroom's CCR markers
(content removed, fetchable) are counted and reported, since a fetch is an
extra full-context request this replay does not charge.

Must run with a Python that has headroom-ai installed:
  <venv>/bin/python experiments/headroom_compare.py [session_prefix] [profile ...]
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import names  # noqa: E402
from tokensaver import background, config, passes, pricing, vault  # noqa: E402

OUT = names.home() / "experiments" / "headroom_compare.jsonl"


def bodies(prefix: str):
    conn = sqlite3.connect(vault.DB_PATH)
    rows = conn.execute("SELECT ts, blob_sha, meta FROM events WHERE kind='api_request' AND session_id LIKE ? "
                        "AND blob_sha IS NOT NULL ORDER BY id", (prefix + "%",)).fetchall()
    conn.close()
    for ts, sha, meta in rows:
        m = json.loads(meta or "{}")
        u = m.get("usage") or {}
        if m.get("status") != 200 or not u:
            continue
        p = vault.BLOB_DIR / sha[:2] / sha[2:4] / sha
        try:
            body = json.loads(p.read_bytes())
        except Exception:
            continue
        if not isinstance(body.get("messages"), list) or not body.get("tools"):
            continue                       # side requests without tools: not main traffic
        t = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        real = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + \
               (u.get("cache_creation_input_tokens") or 0)
        yield t, body, real, u.get("output_tokens") or 0


def dump(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:                         # binary search on prefix equality
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class Ledger:
    def __init__(self, price):
        self.p, self.prev, self.cost, self.tokens, self.writes = price, None, 0.0, 0, 0

    def add(self, text: str, fixed_chars: int, orig_chars: int, real: int, out: int, gap: float):
        tok_per_char = real / max(orig_chars + fixed_chars, 1)
        size = round((len(text) + fixed_chars) * tok_per_char)
        hit = 0 if self.prev is None or gap > 3600 else round((common_prefix(self.prev, text) + fixed_chars)
                                                              * tok_per_char)
        hit = min(hit, size)
        self.cost += (hit * self.p.read + (size - hit) * self.p.write_1h + out * self.p.out) / 1e6
        self.tokens += size
        self.writes += size - hit
        self.prev = text


def warm_up(compress, wait_s: int = 600) -> bool:
    """Kompress loads in the background and passes text through until ready;
    poll with a large prose block until it compresses."""
    text = " ".join(f"Step {i}: the build ran the integration tests against the staging database and "
                    f"reported that every check passed without warnings." for i in range(200))
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": [
        {"type": "tool_use", "id": "w1", "name": "Bash", "input": {"command": "make test"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "w1", "content": text}]},
        {"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]
    t0 = time.time()
    while time.time() - t0 < wait_s:
        r = compress(msgs, model="claude-opus-4-20250514", savings_profile="agent-90")
        if r.tokens_after < 0.9 * r.tokens_before:
            return True
        time.sleep(10)
    return False


def headroom_run(rows, profile):
    from headroom import compress
    # "<profile>-token": Headroom's token mode, approximated as compressing the
    # whole history on every request (no frozen prefix); the cache emulation
    # then charges whatever prefix the recompression changes.
    token_mode = profile.endswith("-token")
    profile = profile[:-len("-token")] if token_mode else profile
    if profile == "agent-90":
        # Measured: Kompress starts loading only on a default-profile call, and
        # passes text through until loaded (a cold replay silently ran without it).
        compress(rows[-1][1]["messages"], model="claude-opus-4-20250514", model_limit=1_000_000)
        time.sleep(150)
        print(f"  kompress ready: {warm_up(compress)}", flush=True)
    prev_raw, prev_out, outs, markers, ml = None, None, [], 0, 0
    for t, body, real, out in rows:
        raw = body["messages"]
        frozen = 0
        if prev_raw is not None and not token_mode:
            while frozen < min(len(raw), len(prev_raw)) and dump(raw[frozen]) == dump(prev_raw[frozen]):
                frozen += 1
        start = (prev_out[:frozen] if prev_out is not None else []) + raw[frozen:]
        try:
            r = compress(start, model="claude-opus-4-20250514", model_limit=1_000_000,
                         savings_profile=profile, frozen_message_count=frozen)
            res = r.messages
            ml += any(x.startswith(("router:text:", "router:mixed:")) for x in r.transforms_applied)
        except Exception as exc:
            print("  headroom error:", type(exc).__name__, str(exc)[:200], flush=True)
            res = start
        if prev_out is not None and frozen and dump(res[:frozen]) != dump(prev_out[:frozen]):
            print(f"  headroom rewrote the frozen prefix at a request ({frozen} messages)", flush=True)
        markers += sum(dump(m).count("headroom_retrieve") for m in res[frozen:])
        prev_raw, prev_out = raw, res
        outs.append(res)
    print(f"  requests where Headroom's ML text compression ran: {ml}", flush=True)
    return outs, markers


def tokensaver_run(rows, price):
    after_s = config.get("bg_after_min") * 60
    payback = config.get("bg_warm_payback_max")
    applied, outs, last_t, prev_size = None, [], None, None
    for t, body, real, out in rows:
        msgs = body["messages"]
        gap = t - last_t if last_t else 0
        last_t = t
        if applied is not None and background.apply(msgs, applied) is None:
            applied = None                  # Claude Code rebuilt the history
        if gap > after_s and background.worth_compacting(msgs):
            cand = background.build(msgs)
            ok_cut = cand and (applied is None or cand["cut"] > applied["cut"])
            if ok_cut:
                cur = background.apply(msgs, applied) if applied else msgs
                new = background.apply(msgs, cand)
                if new is not None:
                    s_now, s_new = len(dump(cur)), len(dump(new))
                    warm = gap <= 3600
                    n = (s_new * price.write_1h - s_now * price.read) / max((s_now - s_new) * price.read, 1)
                    if s_new < s_now and (not warm or n <= payback):
                        applied = dict(cand, tail_end=len(msgs))   # the proxy strips the tail's thinking
        m = background.apply(msgs, applied) if applied else msgs
        outs.append(passes.rewrite_messages(m, aliases="off"))
    return outs


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: headroom_compare.py <session id prefix> [profile ...]")
        return 2
    prefix = sys.argv[1]
    profiles = sys.argv[2:] or ["coding", "agent-90"]
    rows = list(bodies(prefix))
    if not rows:
        print("no stored bodies for", prefix)
        return 1
    price = pricing.price_for(rows[0][1].get("model") or "") or pricing.price_for("claude-opus-5-5")
    print(f"{prefix}: {len(rows)} main requests with stored bodies", flush=True)
    variants = {"plain": [r[1]["messages"] for r in rows]}
    t0 = time.time()
    variants["tokensaver"] = tokensaver_run(rows, price)
    print(f"  TokenSaver replayed in {time.time() - t0:.0f}s", flush=True)
    extra = {}
    for prof in profiles:
        t0 = time.time()
        variants[f"headroom/{prof}"], extra[prof] = headroom_run(rows, prof)
        print(f"  headroom/{prof} replayed in {time.time() - t0:.0f}s; retrieve markers in new content: "
              f"{extra[prof]}", flush=True)
    results = {}
    for name, outs in variants.items():
        led, last = Ledger(price), None
        for (t, body, real, out), msgs in zip(rows, outs):
            fixed = len(dump({k: body[k] for k in ("system", "tools") if k in body}))
            orig = len(dump(body["messages"]))
            led.add(dump(msgs), fixed, orig, real, out, t - last if last else 0)
            last = t
        results[name] = led
    base = results["plain"]
    print(f"\n{'variant':22s} {'tokens sent':>14s} {'saved':>7s} {'cost':>9s} {'saved':>7s} {'cache writes':>13s}")
    for name, led in results.items():
        print(f"{name:22s} {led.tokens:14,} {100 * (1 - led.tokens / base.tokens):6.1f}% "
              f"${led.cost:8.2f} {100 * (1 - led.cost / base.cost):6.1f}% {led.writes:13,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps({"session": prefix, "at": time.time(), "markers": extra,
                            **{n: {"tokens": l.tokens, "cost": round(l.cost, 4), "writes": l.writes}
                               for n, l in results.items()}}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
