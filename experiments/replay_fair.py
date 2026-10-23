"""One real session replayed under today's code, priced on fair real counts.

For every stored main request of a session (the vault keeps what Claude Code
sent), two variants: plain (the bodies as sent) and TokenSaver today
(`headroom_compare.tokensaver_run`: forward passes, compaction at free points
with the kept tail's thinking stripped). Each request's size is

    count_tokens(messages with thinking removed)    (free; real tokenizer)
  + system and tools (in both counts)
  + 0.14 x signature characters of the thinking it still carries

Thinking is priced apart because Opus 5.5 binds each thinking block to the
exact bytes before it: rebuilt bodies would have their thinking dropped by the
count, which made plain Claude look ~19K tokens cheaper per request before
(2026-10-22). 0.14 tokens per signature character was measured on this
machine with thinking valid.

Cache model, the same for both: a request reads the cached part of the
previous one, i.e. the previous request's size times the share of its
messages (cache_control ignored, which only says where to cache) that this
request repeats exactly; the rest is written at 1h, and a gap over 1h writes
all. Measuring the read against the previous request's real size, not this
one's characters, matters: thinking signatures are many characters but few
tokens, so a character share made every new reply look like a large write.
Output tokens are the real ones.

The OAuth token is read from the Keychain inside this process and never
printed or stored. Usage: python3 experiments/replay_fair.py [session_prefix]
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import names  # noqa: E402
from tokensaver import audit, background as bg, pricing, proxy  # noqa: E402
from headroom_compare import bodies, common_prefix, dump, tokensaver_run  # noqa: E402

OUT = names.home() / "experiments" / "replay_fair.jsonl"


def counter(headers: dict):
    lock, state = threading.Lock(), {"last": 0.0}

    def count(body: dict) -> int | None:
        for attempt in range(4):
            with lock:                       # ~5 calls/s at most
                wait = state["last"] + 0.2 - time.time()
                if wait > 0:
                    time.sleep(wait)
                state["last"] = time.time()
            req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens?beta=true",
                                         data=json.dumps(body).encode(), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read())["input_tokens"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 529, 500, 503):
                    time.sleep(5 * (attempt + 1))
                    continue
                print(f"  count failed: {e.code} {e.read()[:200]!r}", flush=True)
                return None
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(5 * (attempt + 1))      # network hiccup: retry
        return None
    return count


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: replay_fair.py <session id prefix>")
        return 2
    prefix = sys.argv[1]
    rows = list(bodies(prefix))
    if not rows:
        print("no stored bodies for", prefix)
        return 1
    price = pricing.price_for(rows[0][1].get("model") or "") or pricing.price_for("claude-opus-5-5")
    variants = {"plain": [r[1]["messages"] for r in rows], "tokensaver": tokensaver_run(rows, price)}
    print(f"{prefix}: {len(rows)} main requests; {2 * len(rows)} free count calls", flush=True)

    cred = json.loads(subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                                     capture_output=True, text=True, check=True).stdout)
    import sqlite3
    from tokensaver import vault
    conn = sqlite3.connect(vault.DB_PATH)
    rid = conn.execute("SELECT id FROM events WHERE kind='api_request' AND session_id LIKE ? "
                       "AND blob_sha IS NOT NULL ORDER BY id DESC LIMIT 1", (prefix + "%",)).fetchone()[0]
    conn.close()
    h = {k.lower(): v for k, v in (vault.read_meta(rid).get("headers") or {}).items()}
    headers = {"authorization": f"Bearer {cred['claudeAiOauth']['accessToken']}",
               "anthropic-version": h.get("anthropic-version", "2023-06-01"),
               "anthropic-beta": h.get("anthropic-beta", ""), "content-type": "application/json"}
    del cred
    count = counter(headers)

    sizes = {name: [None] * len(rows) for name in variants}
    # Counts from an earlier run of the same session and code are reused, so
    # a changed cost model needs no new calls; only missing ones are fetched.
    try:
        prior = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        prior = [x for x in prior if x.get("session") == prefix and x.get("rows") == len(rows)
                 and x.get("code") == proxy.code_id()]
        if prior:
            for name in variants:
                old = (prior[-1].get("per_request") or {}).get(name) or []
                if len(old) == len(rows):
                    sizes[name] = list(old)
    except (OSError, ValueError):
        pass
    jobs = [(name, i) for name in variants for i in range(len(rows)) if sizes[name][i] is None]
    print(f"  {len(jobs)} counts to fetch", flush=True)

    def work():
        while True:
            try:
                name, i = jobs.pop()
            except IndexError:
                return
            body = rows[i][1]
            msgs = variants[name][i]
            payload = audit.payload(json.dumps(dict(body, messages=msgs)).encode(), no_thinking=True)
            n = count(payload)
            if n is not None:
                sizes[name][i] = n + round(bg.TOKENS_PER_SIGNATURE_CHAR * bg._signature_chars(msgs))
    threads = [threading.Thread(target=work) for _ in range(4)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"  counted in {time.time() - t0:.0f}s", flush=True)

    res = {}
    for name, outs in variants.items():
        cost = tokens = writes = 0.0
        prev, prev_size, last = None, 0, None
        for (t, body, real, out), msgs, size in zip(rows, outs, sizes[name]):
            if size is None:
                size = real                  # a failed count: the billed size, never a guess smaller
            text = dump(proxy._canon(msgs))
            gap = t - last if last else 0
            hit = 0
            if prev is not None and gap <= 3600:
                hit = min(size, round(prev_size * common_prefix(prev, text) / max(len(prev), 1)))
            cost += (hit * price.read + (size - hit) * price.write_1h + out * price.out) / 1e6
            tokens += size
            writes += size - hit
            prev, prev_size, last = text, size, t
        res[name] = {"tokens": int(tokens), "cost": round(cost, 4), "writes": int(writes),
                     "failed": sum(1 for x in sizes[name] if x is None)}
    base = res["plain"]
    print(f"\n{'variant':12s} {'tokens sent':>13s} {'saved':>7s} {'cost':>9s} {'saved':>7s} {'cache writes':>13s}")
    for name, r in res.items():
        print(f"{name:12s} {r['tokens']:13,} {100 * (1 - r['tokens'] / base['tokens']):6.1f}% ${r['cost']:8.2f} "
              f"{100 * (1 - r['cost'] / base['cost']):6.1f}% {r['writes']:13,}"
              + (f"  ({r['failed']} counts failed)" if r["failed"] else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps({"session": prefix, "at": time.time(), "rows": len(rows), "code": proxy.code_id(), **res,
                            "per_request": {n: s for n, s in sizes.items()}}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
