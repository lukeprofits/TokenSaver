"""Compression via a schema-constrained local model, measured on real content.

Why schema-constrained: free-form prompting of qwen3:4b returned its entire
deliberation instead of the compressed text - 263 characters in, 17,079 out,
49 seconds. Forcing a JSON schema fixed that and ran in 0.9s, but the model
then echoed the input verbatim: 0% compression. Both failures are silent, and
both would poison a context if shipped. So every run here reports ratio AND
detail retention, and a run that changes nothing is a failure, not a pass.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:11434/api/generate"

SCHEMA = {
    "type": "object",
    "properties": {"compressed": {"type": "string"}},
    "required": ["compressed"],
    "additionalProperties": False,
}

PROMPT = """Rewrite the TEXT below as compactly as possible for an AI coding assistant's memory.

Keep verbatim: file paths, function/class/variable names, exception types, error codes, test names, numbers, versions, commands, flags, URLs.
Keep: every decision and its reason, every failure and its result.
Delete: hedging, pleasantries, restatement, narration, filler, repetition.

Your output MUST be shorter than the input. Do not echo the input. Use terse fragments, not sentences.

TEXT:
"""

HARD = re.compile(r"""
    (?:[A-Za-z0-9_.\-/]+\.(?:py|java|kt|ts|tsx|js|rb|go|rs|dart|json|yaml|yml|xml|sql|sh|md))
  | (?:[A-Z][A-Za-z0-9]*(?:Exception|Error|Warning))
  | (?:\b[a-z]+_[a-z_]+\b)
  | (?:\b[A-Z_]{3,}\b)
  | (?:\b\d+\.\d+(?:\.\d+)?\b)
  | (?:\b\d{2,}\b)
  | (?:https?://\S+)
""", re.VERBOSE)


def details(text: str) -> set[str]:
    return {m.group(0) for m in HARD.finditer(text)}


def _result_text(content) -> str:
    """Pull the human-readable text out of a tool_result's content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if isinstance(p, str))
    if isinstance(content, dict):
        for key in ("text", "stdout", "content", "output"):
            if isinstance(content.get(key), str):
                return content[key]
    return str(content)


def compress(text: str, model: str, timeout: int = 600) -> tuple[str, float]:
    body = json.dumps({
        "model": model, "prompt": PROMPT + text, "stream": False,
        "think": False, "format": SCHEMA,
        "options": {"temperature": 0, "num_predict": 8192},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    raw = (payload.get("response") or "").strip()
    try:
        return json.loads(raw)["compressed"], elapsed
    except (ValueError, KeyError):
        return "", elapsed


def samples(per_kind: int = 2, lo: int = 1500, hi: int = 5000):
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
                    # json.dumps would escape the real newlines into "\n"
                    # literals, which makes line-based analysis see one giant
                    # line and measure 0% compression. Pull the text out.
                    kind, text = "tool_result", _result_text(block.get("content"))
                else:
                    continue
                if lo <= len(text) <= hi:
                    found[kind].append((len(text), text))
    out = []
    for kind, items in found.items():
        items.sort(reverse=True)
        seen: set[str] = set()
        for _, text in items:
            if text[:200] in seen:
                continue
            seen.add(text[:200])
            out.append((kind, text))
            if sum(1 for k, _ in out if k == kind) >= per_kind:
                break
    return out


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6:35b-mlx"
    rows = samples()
    print(f"model: {model}   schema-constrained\n")
    print(f"{'kind':15s} {'orig':>6s} {'comp':>6s} {'ratio':>7s} {'kept':>6s} "
          f"{'secs':>6s} {'ch/s':>6s}  verdict")
    for kind, text in rows:
        try:
            out, secs = compress(text, model)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"{kind:15s} {len(text):6,}   ERROR {exc}")
            continue
        if not out:
            print(f"{kind:15s} {len(text):6,} {'--':>6s} {'schema broke':>7s}")
            continue
        ratio = 1 - len(out) / len(text)
        orig_d = details(text)
        kept = len(orig_d & details(out)) / len(orig_d) if orig_d else 1.0
        # A run that grew the text, or echoed it, is a failure.
        verdict = ("GREW" if ratio < 0 else
                   "ECHOED" if ratio < 0.05 else
                   "LOST DETAIL" if kept < 0.95 else "ok")
        print(f"{kind:15s} {len(text):6,} {len(out):6,} {ratio*100:6.1f}% "
              f"{kept*100:5.0f}% {secs:6.1f} {len(text)/secs:6.0f}  {verdict}")
        lost = sorted(orig_d - details(out))[:8]
        if lost:
            print(f"                LOST: {', '.join(lost)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
