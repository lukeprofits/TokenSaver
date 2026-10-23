"""Do our request bodies pass Anthropic's preserved-thinking check?

On Opus 5.5 a thinking block from an earlier turn stays valid only while
`system`, `tools` and every message before it are byte-identical to what was
sent when the thinking was produced (docs: build-with-claude/preserved-thinking).
Old accounts silently drop invalid blocks; accounts created on or after
2026-08-31 get a 400. This counts one stored request several ways with
`thinking.block_binding.prefix_mismatch_behavior` set, which opts the call into
the check on any account:

- "error": 200 means every thinking block is valid, 400 names the first that is not;
- "drop_block": the count without the invalid blocks, plus `input_transformations`
  when the endpoint reports them.

count_tokens is free. The OAuth token is read from the Keychain inside this
process and never printed or stored; only counts and error messages are shown.
Usage: python3 experiments/thinking_binding.py [event_id] [--build-only]
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import audit, background as bg, passes, vault  # noqa: E402

BETA = "thinking-binding-controls-2026-08-01"
DEFAULT_EVENT = None            # pass the vault event id of a stored request to test


def no_thinking(messages: list, where=lambda i: True) -> list:
    out = copy.deepcopy(messages)
    for i, m in enumerate(out):
        if where(i) and m.get("role") == "assistant" and isinstance(m.get("content"), list):
            m["content"] = [b for b in m["content"]
                            if b.get("type") not in ("thinking", "redacted_thinking")] or [{"type": "text", "text": "."}]
    return out


def variants(orig: dict, meta: dict) -> dict:
    ms = orig["messages"]
    off = tuple((meta.get("rewrite") or {}).get("passes_off") or ())
    sent = passes.rewrite_messages(ms, aliases="off", exclude=off)
    prep = bg.build(ms)
    out = {"original": ms, "sent": sent,
           "original, no thinking": no_thinking(ms), "sent, no thinking": no_thinking(sent)}
    if prep:
        comp = passes.rewrite_messages(bg.apply(ms, prep), aliases="off", exclude=off)
        out["compacted (today)"] = comp
        out["compacted, kept tail without thinking"] = no_thinking(comp)
        new = passes.rewrite_messages(bg.apply(ms, dict(prep, tail_end=len(ms))), aliases="off", exclude=off)
        out["compacted (new code: tail_end strip)"] = new
    return out


def count(body: dict, headers: dict, query: str) -> tuple[int, dict]:
    req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens" + query,
                                 data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("usage: thinking_binding.py <vault event id of a stored request>")
        return 2
    rid = int(args[0])
    meta = vault.read_meta(rid)
    orig = json.loads(vault.read(rid)[1])
    vs = variants(orig, meta)
    print(f"event {rid}: {len(orig['messages'])} messages")
    if "--build-only" in sys.argv:
        for name, ms in vs.items():
            print(f"  {name:40s} {len(ms):4d} messages {len(json.dumps(ms)):9,} bytes")
        return 0
    cred = json.loads(subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                                     capture_output=True, text=True, check=True).stdout)
    h = {k.lower(): v for k, v in meta["headers"].items()}
    betas = ",".join(x for x in [h.get("anthropic-beta", ""), BETA] if x)
    headers = {"authorization": f"Bearer {cred['claudeAiOauth']['accessToken']}",
               "anthropic-version": h.get("anthropic-version", "2023-06-01"),
               "anthropic-beta": betas, "content-type": "application/json"}
    del cred
    query = ("?" + meta["path"].split("?", 1)[1]) if "?" in meta.get("path", "") else ""
    for name, ms in vs.items():
        for mode in ("error", "drop_block"):
            body = audit.payload(json.dumps(dict(orig, messages=ms)).encode())
            body["thinking"] = dict(body.get("thinking") or {"type": "adaptive"},
                                    block_binding={"prefix_mismatch_behavior": mode})
            status, resp = count(body, headers, query)
            if status == 200:
                extra = {k: v for k, v in resp.items() if k != "input_tokens"}
                tr = extra.pop("input_transformations", None)
                note = f"; dropped {len(tr)} (first {tr[0].get('path')})" if tr else ("; none dropped" if tr == [] else "")
                print(f"  {name:40s} {mode:10s} 200 {resp.get('input_tokens', 0):9,} tokens{note}"
                      + (f" {json.dumps(extra)[:200]}" if extra else ""))
            else:
                msg = (resp.get("error") or {}).get("message", "")
                print(f"  {name:40s} {mode:10s} {status} {msg[:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
