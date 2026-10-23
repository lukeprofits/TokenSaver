"""Background compaction on real idle breaks: time, size, and the QA gate.

Finds moments in local transcripts where the developer was away for over an
hour with more than `background.MIN_TOKENS` of context, prepares the
replacement exactly as the proxy would, and reports how long it took, how much
smaller the request gets, and whether the gate passed. A second, stricter check
uses this file's own extractor (not the guard): of the identifiers unique to
the old part that Claude used again after the break, how many does the
replacement still carry?

Local only: ollama, zero API calls. Slow (minutes per moment). The proxy no
longer runs this check on the rules-only compaction; it is kept to measure it.
Usage: python3 experiments/bg_compact.py [max_moments]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import names  # noqa: E402
from tokensaver import background as bg  # noqa: E402
from stack import indep_ident  # noqa: E402

OUT = Path(os.environ.get("TOKENSAVER_BG_OUT") or names.home() / "experiments" / "bg_compact.jsonl")


# Moments are selected by total context size, as when these were measured
# (the package dropped this setting for `bg_min_removable`, 2026-10-22).
MIN_TOKENS = 100_000

def moments(path: Path):
    """(messages before the request that followed a >61 min gap, messages after)."""
    messages, last_ts, found = [], None, []
    for line in path.open(errors="replace"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict) or rec.get("isSidechain"):
            continue
        role, content = msg.get("role"), msg.get("content")
        if role not in ("user", "assistant") or content is None:
            continue
        ts = rec.get("timestamp")
        if role == "assistant" and ts:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            if last_ts and t - last_ts > 61 * 60:
                found.append(len(messages))
            last_ts = t
        items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = messages[-1]["content"] + items
        else:
            messages.append({"role": role, "content": list(items)})
    for k in found:
        if bg.size_tokens(messages[:k]) >= MIN_TOKENS:
            yield messages[:k], messages[k:]


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    OUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    done = 0
    seen = set()          # test forks repeat another session's history
    only = set(filter(None, os.environ.get("TOKENSAVER_BG_SESSIONS", "").split(",")))
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: p.stat().st_size):
        if only and path.stem[:8] not in only:
            continue
        for before, after in moments(path):
            if done >= limit:
                return 0
            fp = bg.fingerprint(before[:50])
            if fp in seen:
                continue
            seen.add(fp)
            res = bg.prepare(before)
            if not res:
                continue
            # prepare no longer runs the check (2026-10-23); this experiment still measures it
            res["gate"] = bg.gate(before[:res["cut"]], bg._extract(before, res["cut"])[1])
            done += 1
            cut = res["cut"]
            old_ids = set().union(*(indep_ident(json.dumps(m)) for m in before[:cut]))
            recent_ids = set().union(*(indep_ident(json.dumps(m)) for m in before[cut:]))
            later = set().union(*(indep_ident(json.dumps(m)) for m in after if m["role"] == "assistant"))
            need = (old_ids - recent_ids) & later
            kept = need & indep_ident(res["text"])
            row = {"session": path.stem[:8], "old_tokens": res["old_tokens"], "new_tokens": res["new_tokens"],
                   "total_tokens": bg.size_tokens(before), "seconds": res["seconds"],
                   "summary_seconds": res["summary_seconds"], "gate": res["gate"],
                   "needed_later": len(need), "kept": len(kept)}
            with OUT.open("a") as f:
                f.write(json.dumps(dict(row, text=res["text"])) + "\n")
            shrink = 1 - (bg.size_tokens(before) - res["old_tokens"] + res["new_tokens"]) / bg.size_tokens(before)
            print(f"{row['session']}: {row['total_tokens']:,} tokens -> {100 * shrink:.0f}% smaller; "
                  f"{row['seconds']}s ({row['summary_seconds']}s summary); gate { {k: v for k, v in row['gate'].items() if k != 'items'} }; "
                  f"identifiers used after the break kept {len(kept)}/{len(need)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
