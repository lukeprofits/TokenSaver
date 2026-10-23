"""The opt-in lossy layer on real breaks: Headroom's Kompress vs our 35B line picker.

Both run only inside a compaction of the OLD part (a free point: the prefix
is rebuilt anyway) and only on old document views and tool-result excerpts
longer than `background.SHRINK_MIN_CHARS` (see `background.render`). For each
real break (from bg_compact.moments) and method it reports:
- extra shrink of the compacted text over the extractive one;
- used-later loss: identifiers unique to the old part that Claude used after
  the break, present in the extractive text, gone from the lossy one;
- both QA gates (standard, and the current-work one), unless
  TOKENSAVER_LOSSY_GATE=0;
- seconds.

Methods:
- kompress: Headroom's ML compressor (Apache-2.0), via its library, on each
  piece as a tool result (needs headroom-ai in this Python);
- lines35b: the local 35B lists the line numbers to keep, given a digest of
  the current work; survivors stay byte-exact, lines carrying a detail Claude
  itself wrote are always kept, and a run of left-out lines becomes one
  "[... N lines left out]" marker.

Local only, $0. Usage: <python> experiments/lossy_compare.py [method ...]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import names  # noqa: E402
from tokensaver import background as bg, compress as cmp  # noqa: E402
import episodes_lib as episodes  # noqa: E402
from bg_compact import moments  # noqa: E402
from stack import indep_ident  # noqa: E402

OUT = names.home() / "experiments" / "lossy_compare.jsonl"
# Which sessions to draw breaks from: TOKENSAVER_SESSIONS=<id prefix>,<id prefix>; default all.
SESSIONS = tuple(filter(None, os.environ.get("TOKENSAVER_SESSIONS", "").split(",")))


def body_of(text: str) -> str:
    return text.split("\n\n## Identifiers from the earlier part\n\n")[0]


def kompress_fn():
    from headroom import compress
    # Measured: Kompress loads only after a default call, then a pause; until
    # then it passes text through silently.
    probe = [{"role": "user", "content": "x " * 2000}]
    compress(probe, model="claude-opus-4-20250514")
    time.sleep(150)

    def fn(kind, text):
        msgs = [{"role": "assistant", "content": [{"type": "tool_use", "id": "k1", "name": "Bash", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "k1", "content": text}]}]
        r = compress(msgs, model="claude-opus-4-20250514", savings_profile="agent-90", protect_recent=0)
        c = r.messages[-1]["content"][0].get("content")
        out = c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
        return out if out and len(out) < len(text) else text
    return fn


LINES_PROMPT = """You help shorten an OLD part of a coding session that an AI assistant will reread before continuing the work below.

CURRENT WORK (most recent part of the session):
{current}

OLD {kind}, numbered lines:
{numbered}

List the line numbers to KEEP: anything the assistant could need to continue the current work or answer a question about this session: names, paths, commands, numbers, errors, decisions, results, configuration values. Leave out boilerplate, repetition, progress output and details that only mattered at the time.
Answer with JSON only: {{"keep": [line numbers]}}"""


def lines35b_fn(current: str, keep_details: set[str], model: str = bg.LOCAL_MODEL):
    def fn(kind, text):
        lines = text.split("\n")
        if len(lines) < 4:
            return text
        numbered = "\n".join(f"{i + 1}: {x[:400]}" for i, x in enumerate(lines))
        body = json.dumps({"model": model, "stream": False, "think": False, "format": "json",
                           "prompt": LINES_PROMPT.format(current=current[-3000:], kind=kind,
                                                         numbered=numbered[:24_000]),
                           "options": {"temperature": 0, "num_predict": 800, "num_ctx": 16384}}).encode()
        try:
            req = urllib.request.Request(bg.OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                keep = set(int(x) - 1 for x in json.loads(json.loads(resp.read())["response"]).get("keep", []))
        except Exception:
            return text
        out, gap = [], 0
        for i, x in enumerate(lines):
            if i in keep or i == 0 or (keep_details and cmp.details(x) & keep_details):
                if gap:
                    out.append(f"[... {gap} lines left out]")
                    gap = 0
                out.append(x)
            else:
                gap += 1
        if gap:
            out.append(f"[... {gap} lines left out]")
        res = "\n".join(out)
        return res if len(res) < len(text) else text
    return fn


def main() -> int:
    methods = sys.argv[1:] or ["kompress", "lines35b"]
    with_gate = os.environ.get("TOKENSAVER_LOSSY_GATE", "1") == "1"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kfn = kompress_fn() if "kompress" in methods else None
    seen = set()
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: p.stat().st_size):
        if SESSIONS and path.stem[:8] not in SESSIONS:
            continue
        for before, after in moments(path):
            fp = bg.fingerprint(before[:50])
            if fp in seen:
                continue
            seen.add(fp)
            cut = bg.choose_cut(before)
            if not cut:
                continue
            old, recent = before[:cut], before[cut:]
            base, _ = bg._extract(before, cut)
            old_ids = set().union(*(indep_ident(json.dumps(m)) for m in old))
            recent_ids = set().union(*(indep_ident(json.dumps(m)) for m in recent))
            later = set().union(*(indep_ident(json.dumps(m)) for m in after if m["role"] == "assistant"))
            need = (old_ids - recent_ids) & later & indep_ident(body_of(base["text"]))
            current = episodes.current_digest(before, cut)
            print(f"\n{path.stem[:8]}: today {base['new_tokens']:,} tokens; identifiers used later: {len(need)}",
                  flush=True)
            for method in methods:
                t = time.time()
                fn = kfn if method == "kompress" else lines35b_fn(current, bg.mentioned(before))
                cache: dict = {}
                shrink = lambda kind, text: cache.setdefault(text, fn(kind, text))
                res, cand_for = bg._extract(before, cut, shrink=shrink)
                lost = need - indep_ident(body_of(res["text"]))
                extra = 1 - res["new_tokens"] / max(base["new_tokens"], 1)
                g = bg.gate(old, cand_for) if with_gate else {}
                gc = bg.gate(old, cand_for, current=current) if with_gate else {}
                strip = lambda d: {k: v for k, v in d.items() if k in ("passed", "original_right", "replacement_right")}
                row = {"session": path.stem[:8], "method": method, "base_tokens": base["new_tokens"],
                       "tokens": res["new_tokens"], "extra_shrink": round(extra, 4), "need": len(need),
                       "lost": sorted(lost), "gate": strip(g), "gate_current": strip(gc),
                       "seconds": round(time.time() - t)}
                with OUT.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {method:9s} {base['new_tokens']:,} -> {res['new_tokens']:,} ({100 * extra:.1f}% smaller); "
                      f"lost {len(lost)} of {len(need)} used later; gate {strip(g)} current {strip(gc)}; "
                      f"{row['seconds']}s", flush=True)
                for ident in sorted(lost)[:6]:
                    print(f"      LOST {ident!r}")
            break                           # one moment per session
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
