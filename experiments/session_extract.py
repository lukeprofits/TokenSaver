"""Extractive compression across a whole session, with state carried forward.

Compressing one message in isolation finds almost nothing: the redundancy in
these transcripts is cross-message (a file read twice, a schema restated, the
same progress log emitted every build), and it only becomes visible when the
set of already-seen lines persists across the session. That is also how it
would run in production - the vault holds the session's history.

Reports the two numbers that matter together: total reduction, and whether any
hard detail present in the original session is absent from the compressed one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extractive import compress  # noqa: E402
from schema_compress import _result_text, details  # noqa: E402


def session_blocks(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
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
                out.append(("text", block.get("text") or ""))
            elif block.get("type") == "tool_result":
                out.append(("tool_result", _result_text(block.get("content"))))
            elif block.get("type") == "tool_use":
                out.append(("tool_use", json.dumps(block.get("input") or {})))
    return out


def run(path: Path) -> dict | None:
    blocks = session_blocks(path)
    if not blocks:
        return None
    seen: set[str] = set()
    known: set[str] = set()
    orig_total = comp_total = 0
    orig_details: set[str] = set()
    comp_details: set[str] = set()
    per_kind: dict[str, list[int]] = {}

    for kind, text in blocks:
        out, _ = compress(text, seen, known)
        orig_total += len(text)
        comp_total += len(out)
        orig_details |= details(text)
        comp_details |= details(out)
        row = per_kind.setdefault(kind, [0, 0])
        row[0] += len(text)
        row[1] += len(out)

    lost = orig_details - comp_details
    return {
        "name": path.name[:8],
        "orig": orig_total,
        "comp": comp_total,
        "ratio": 1 - comp_total / orig_total if orig_total else 0,
        "details": len(orig_details),
        "lost": lost,
        "per_kind": per_kind,
    }


def main() -> int:
    root = Path.home() / ".claude/projects"
    paths = sorted(root.glob("*/*.jsonl"), key=lambda p: -p.stat().st_size)[:6]
    print(f"{'session':9s} {'orig':>10s} {'comp':>10s} {'ratio':>7s} "
          f"{'details':>8s} {'lost':>5s}")
    agg_o = agg_c = 0
    for path in paths:
        r = run(path)
        if not r:
            continue
        agg_o += r["orig"]
        agg_c += r["comp"]
        print(f"{r['name']:9s} {r['orig']:10,} {r['comp']:10,} "
              f"{r['ratio']*100:6.1f}% {r['details']:8,} {len(r['lost']):5d}")
        if r["lost"]:
            print(f"          LOST: {', '.join(sorted(r['lost'])[:6])}")
        for kind, (o, c) in sorted(r["per_kind"].items()):
            if o:
                print(f"          {kind:12s} {o:9,} -> {c:9,}  {100*(1-c/o):5.1f}%")
    if agg_o:
        print(f"\noverall: {agg_o:,} -> {agg_c:,} chars  "
              f"{100*(1-agg_c/agg_o):.1f}% reduction, zero detail loss required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
