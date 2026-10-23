"""Anthropic's on-demand compaction vs ours, on a real break. COSTS MONEY.

On-demand compaction (`compact-2026-09-04`, docs build-with-claude/
compaction-on-demand) has the API write a signed summary of the older turns;
the kept turns' thinking stays valid under preserved thinking, where ours (a
summary we write) must strip it. This measures, on one stored request:

1. the request as sent is valid (count_tokens, prefix check "error": free);
2. ours: the extractive compaction, kept tail without thinking (free count);
3. the API's: one summarization call on the same older part (PAID, cold:
   about $4/M input + $20/M output on Opus 5.5; ~$0.6 for a 175-message request). Live it
   would run while the cache is warm and read it at $0.20/M instead;
4. the API's result counted with the prefix check on (free): 200 means the
   kept thinking held.
Optional `--gate`: the local QA gate on both texts (local model, slow, $0).

The OAuth token is read from the Keychain in this process, never printed or
stored. The signed block is saved to ~/.tokensaver/experiments/api_compact.jsonl.
Usage: python3 experiments/api_compact.py [event_id] --yes [--gate]
       python3 experiments/api_compact.py [event_id] --reuse --legacy-user-text   (free recount)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import names  # noqa: E402
from tokensaver import audit, background as bg, passes, pricing, vault  # noqa: E402

OUT = names.home() / "experiments" / "api_compact.jsonl"
BINDING, COMPACT = "thinking-binding-controls-2026-08-01", "compact-2026-09-04"
INSTRUCTIONS = (
    "Summarize the earlier part of this coding session for the assistant that will continue it. "
    "Keep every concrete detail later work may need: file paths, identifiers, commands, error "
    "messages, numbers, configuration values, decisions and the reasons for them, what was tried "
    "and failed, and the user's requests and constraints in their own words where it matters. "
    "Say what is finished and what is still open. Do not call tools; respond with the summary text only.")


def call(url: str, body: dict, headers: dict, timeout: int = 600) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("usage: api_compact.py <vault event id of a stored request> [--yes]")
        return 2
    rid = int(args[0])
    meta = vault.read_meta(rid)
    orig = json.loads(vault.read(rid)[1])
    rw = meta.get("rewrite") or {}
    if rw.get("aliases") == "plan" or rw.get("bg_cut"):
        print("this request ran with an alias plan or a compaction; its sent bytes cannot be rebuilt")
        return 2
    ms = orig["messages"]
    if "--legacy-user-text" in sys.argv:
        # Rebuild the bytes as the code of the time sent them: before
        # 2026-10-22 the passes could edit the user's own text blocks.
        real_from = passes.from_messages

        def from_messages(*a, **k):
            blocks = real_from(*a, **k)
            for b in blocks:
                if b.role == "user" and b.type == "text":
                    b.fixed = False
            return blocks
        passes.from_messages = from_messages
    sent = passes.rewrite_messages(ms, aliases="off", exclude=tuple(rw.get("passes_off") or ()))
    prep = bg.build(ms)
    if not prep:
        print("no compaction is possible on this request")
        return 2
    cut = prep["cut"]
    ours = passes.rewrite_messages(bg.apply(ms, dict(prep, tail_end=len(ms))), aliases="off",
                                   exclude=tuple(rw.get("passes_off") or ()))
    p = pricing.price_for(orig.get("model") or "")
    est_in = bg.size_tokens(sent[:cut]) + len(json.dumps([orig.get("system"), orig.get("tools")])) // 4
    print(f"event {rid}: {len(ms)} messages, compaction cut at {cut}; summary call input ~{est_in:,} tokens (older part + system + tools)")
    print(f"the summary call costs about ${est_in * p.inp / 1e6 + 4000 * p.out / 1e6:.2f} (cold)")
    if "--yes" not in sys.argv and "--reuse" not in sys.argv:
        print("add --yes to run it")
        return 0

    cred = json.loads(subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                                     capture_output=True, text=True, check=True).stdout)
    h = {k.lower(): v for k, v in meta["headers"].items()}
    betas = ",".join(x for x in [h.get("anthropic-beta", ""), BINDING, COMPACT] if x)
    headers = {"authorization": f"Bearer {cred['claudeAiOauth']['accessToken']}",
               "anthropic-version": h.get("anthropic-version", "2023-06-01"),
               "anthropic-beta": betas, "content-type": "application/json"}
    del cred
    query = ("?" + meta["path"].split("?", 1)[1]) if "?" in meta.get("path", "") else ""
    base = "https://api.anthropic.com/v1/messages"
    strict = dict(orig.get("thinking") or {"type": "adaptive"}, block_binding={"prefix_mismatch_behavior": "error"})

    def count(messages: list) -> tuple[int, str]:
        body = audit.payload(json.dumps(dict(orig, messages=messages)).encode())
        body["thinking"] = strict
        st, r = call(base + "/count_tokens" + query, body, headers, 60)
        return (r.get("input_tokens", 0), "ok") if st == 200 else (0, f"{st} {(r.get('error') or {}).get('message', '')[:160]}")

    now_n, now_s = count(sent)
    ours_n, ours_s = count(ours)
    print(f"  as sent             {now_n:9,} tokens  check: {now_s}")
    print(f"  ours (tail no thinking) {ours_n:9,} tokens  check: {ours_s}")

    if "--reuse" in sys.argv:
        # Free: the signed block from an earlier paid run of this event.
        prior = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        prior = [x for x in prior if x.get("event") == rid and x.get("cut") == cut]
        if not prior:
            print("no saved block for this event and cut")
            return 2
        block = prior[-1]["block"]
        api_n, api_s = count([{"role": "assistant", "content": [block]}] + sent[cut:])
        print(f"  API compaction      {api_n:9,} tokens  check: {api_s}  (saved block, no paid call)")
        return 0
    req = {k: orig[k] for k in ("model", "system", "tools", "thinking") if k in orig}
    req.update(max_tokens=16000, messages=sent[:cut],
               compaction={"type": "summarize", "instructions": INSTRUCTIONS})
    t0 = time.time()
    st, r = call(base + query, req, headers)
    secs = time.time() - t0
    if st != 200 or r.get("stop_reason") != "compaction":
        print(f"  API compaction failed: {st} {r.get('stop_reason')} {json.dumps(r.get('error') or {})[:300]}")
        return 1
    block = r["content"][0]
    it = next((x for x in (r.get("usage") or {}).get("iterations", []) if x.get("type") == "compaction"), {})
    cost = (it.get("input_tokens", 0) * p.inp + it.get("cache_read_input_tokens", 0) * p.read
            + it.get("cache_creation_input_tokens", 0) * p.write_1h + it.get("output_tokens", 0) * p.out) / 1e6
    live = (it.get("input_tokens", 0) * p.read + it.get("output_tokens", 0) * p.out) / 1e6
    after = [{"role": "assistant", "content": [block]}] + sent[cut:]
    api_n, api_s = count(after)
    print(f"  API compaction      {api_n:9,} tokens  check: {api_s}")
    print(f"    summary: {len(block.get('content', '')):,} chars in {secs:.0f}s; call cost ${cost:.2f} "
          f"(live, warm cache: ~${live:.2f}); usage {json.dumps(it)}")
    row = {"event": rid, "cut": cut, "now": now_n, "ours": ours_n, "api": api_n, "api_check": api_s,
           "ours_check": ours_s, "cost": round(cost, 4), "live_cost": round(live, 4), "seconds": round(secs),
           "usage": it, "block": block, "ours_text": prep["text"]}
    if "--gate" in sys.argv:
        import episodes_lib as episodes
        old = ms[:cut]
        current = episodes.current_digest(ms, cut)
        for name, text in (("ours", prep["text"]), ("api", block.get("content", ""))):
            g = bg.gate(old, lambda i, j, t=text: t)
            gc = bg.gate(old, lambda i, j, t=text: t, current=current)
            keep = lambda d: {k: d.get(k) for k in ("passed", "original_right", "replacement_right")}
            row[f"gate_{name}"], row[f"gate_current_{name}"] = keep(g), keep(gc)
            print(f"  gate {name:4s}: standard {keep(g)}  current work {keep(gc)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
