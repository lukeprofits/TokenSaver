"""Measure what local compression actually achieves on real transcript content.

Two numbers matter, not one:
  ratio    - how much shorter the compressed text is
  retention - what fraction of the ORIGINAL's hard details survive verbatim

A high ratio with poor retention is the failure mode this project exists to
avoid, so both are reported and neither is averaged away.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# The ollama HTTP API, not `ollama run`: the CLI writes ANSI spinner escapes
# even with no TTY attached, and on a thinking model it streams the chain of
# thought to stdout - an earlier version of this script captured that
# reasoning and scored it as the "compressed" text.
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODELS = ("qwen3.6:35b-mlx", "qwen3:4b")

PROMPT = """You compress text for an AI coding assistant's memory. The compressed version replaces the original in the assistant's context, so it must stay usable for future decisions.

RULES - these are absolute:
- Preserve VERBATIM every: file path, function/class/variable name, error code, exception type, test name, number, version, command, flag, URL, and quoted string.
- Preserve every decision made, and WHY.
- Preserve every failure, what was tried, and what the result was.
- Remove: pleasantries, restatement, hedging, narration of intent, repeated information, filler.
- Do not add interpretation. Do not speculate. Do not summarise away specifics.
- If something cannot be shortened without losing a specific, keep it as-is.

Output ONLY the compressed text. No preamble, no explanation, no markdown fences.

TEXT TO COMPRESS:
"""

# "Hard details": the things that must survive. Deliberately broad.
HARD = re.compile(r"""
    (?:[A-Za-z0-9_.\-/]+\.(?:py|java|kt|ts|tsx|js|rb|go|rs|dart|json|yaml|yml|xml|sql|sh|md))
  | (?:[A-Z][A-Za-z0-9]*(?:Exception|Error|Warning))
  | (?:\b[a-z_]+_[a-z_]+\b)
  | (?:\b[A-Z_]{3,}\b)
  | (?:\b\d+\.\d+(?:\.\d+)?\b)
  | (?:\b\d{3,}\b)
  | (?:https?://\S+)
""", re.VERBOSE)


def hard_details(text: str) -> set[str]:
    return {m.group(0) for m in HARD.finditer(text)}


def compress(text: str, model: str, timeout: int = 900) -> tuple[str, float]:
    body = json.dumps({
        "model": model,
        "prompt": PROMPT + text,
        "stream": False,
        "think": False,          # the qwen3 family reasons by default
        "options": {"temperature": 0, "num_predict": 4096},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    out = (payload.get("response") or "").strip()
    # Belt and braces: drop any leaked reasoning block or code fence.
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    out = re.sub(r"^```[a-z]*\n|\n```$", "", out).strip()
    return out, elapsed


def samples(limit_per_kind: int = 2, min_chars: int = 1500, max_chars: int = 6000):
    """Real content from real sessions, biggest-first within the size window."""
    found: dict[str, list[tuple[int, str]]] = {"assistant_text": [], "tool_result": []}
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl")):
        for line in path.open(errors="replace"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            content = (rec.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    kind, text = "assistant_text", block.get("text") or ""
                elif block.get("type") == "tool_result":
                    kind, text = "tool_result", json.dumps(block.get("content"))
                else:
                    continue
                if min_chars <= len(text) <= max_chars:
                    found[kind].append((len(text), text))
    out = []
    for kind, items in found.items():
        items.sort(reverse=True)
        seen = set()
        for _, text in items:
            key = text[:200]
            if key in seen:
                continue
            seen.add(key)
            out.append((kind, text))
            if sum(1 for k, _ in out if k == kind) >= limit_per_kind:
                break
    return out


def main() -> int:
    rows = samples()
    if not rows:
        print("no samples in the size window", file=sys.stderr)
        return 1
    model = sys.argv[1] if len(sys.argv) > 1 else MODELS[0]
    print(f"model: {model}\n")
    print(f"{'kind':16s} {'orig':>7s} {'comp':>7s} {'ratio':>7s} "
          f"{'details':>9s} {'kept':>6s} {'secs':>6s}")
    total_o = total_c = 0
    for kind, text in rows:
        try:
            out, secs = compress(text, model)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"{kind:16s} {len(text):7,}   ERROR {exc}")
            continue
        if not out:
            print(f"{kind:16s} {len(text):7,} {'EMPTY':>7s}")
            continue
        orig_d = hard_details(text)
        kept = orig_d & hard_details(out)
        ratio = 1 - len(out) / len(text)
        retention = len(kept) / len(orig_d) if orig_d else 1.0
        total_o += len(text); total_c += len(out)
        print(f"{kind:16s} {len(text):7,} {len(out):7,} {ratio*100:6.1f}% "
              f"{len(orig_d):9d} {retention*100:5.0f}% {secs:6.1f}")
        missing = sorted(orig_d - kept)[:6]
        if missing:
            print(f"                 LOST: {', '.join(missing)}")
    if total_o:
        print(f"\noverall char reduction: {100*(1-total_c/total_o):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
