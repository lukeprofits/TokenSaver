"""Extractive compression: delete whole lines, keep survivors byte-exact.

No model. It cannot hallucinate, cannot drop a path from the middle of a line,
cannot invert a negation, and runs in microseconds. Its ceiling is lower than a
rewrite's - it can only delete - but every character it keeps is original.

Measured against the same samples and the same detail metric as the
model-based attempt in schema_compress.py, so the two are comparable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema_compress import details, samples  # noqa: E402

# Lines worth keeping even when they look repetitive.
KEEP = re.compile(r"""
    error|fail|failed|exception|traceback|assert|fatal|panic|abort
  | warn|denied|refused|timeout|timed\ out
  | \bE\d{3,}\b | \b[45]\d\d\b
  | ^\s*at\s | ^\s*File\s"
""", re.IGNORECASE | re.VERBOSE)

# Lines that carry no information once you have seen one of them.
NOISE = re.compile(r"""
    ^\s*$
  | ^[\s\-=_*#~.]+$
  | ^\s*(?:ok|pass(?:ed|ing)?|done|skipped|\.+)\s*$
  | ^\s*\[?\d+%\]?\s*$
  | ^\s*(?:Downloading|Downloaded|Resolving|Fetching|Progress|Collecting)\b
""", re.IGNORECASE | re.VERBOSE)

HEAD_LINES, TAIL_LINES = 25, 25


def compress(text: str, seen: set[str] | None = None,
             known: set[str] | None = None) -> tuple[str, dict]:
    """Return compressed text plus a breadcrumb describing what was removed.

    `seen` carries lines already present earlier in the session and `known`
    the hard details already present. Both must persist across the session or
    nothing dedups: with a per-message detail set, a repeated line's details
    look new in every new message and the line is always kept - measured as
    0.4% reduction before this was fixed.
    """
    seen = seen if seen is not None else set()
    lines = text.split("\n")
    kept: list[str] = []
    dropped_dup = dropped_noise = 0
    # Hard details already present in what we are keeping. A line may only be
    # dropped if it introduces none that are new - measured: dropping lines
    # that merely "looked unremarkable" lost 39% of identifiers.
    kept_details = known if known is not None else set()

    for i, line in enumerate(lines):
        key = re.sub(r"\s+", " ", line).strip()
        in_window = i < HEAD_LINES or i >= len(lines) - TAIL_LINES
        line_details = details(line)
        introduces_new = bool(line_details - kept_details)

        # Order matters. A line carrying an unseen identifier is kept no matter
        # what - that is the no-loss guarantee. Everything after this point
        # introduces nothing new, so removing it cannot lose information.
        if introduces_new or KEEP.search(line):
            kept.append(line)
            kept_details |= line_details
            if len(key) >= 12:
                seen.add(key)
            continue
        # A repeat is a repeat wherever it sits, so this precedes the head/tail
        # window. With the window first, any message shorter than
        # HEAD+TAIL lines was kept whole and nothing ever dropped (0.7%).
        if len(key) >= 12 and key in seen:
            dropped_dup += 1
            continue
        if NOISE.match(line):
            dropped_noise += 1
            continue
        if in_window:
            kept.append(line)
            if len(key) >= 12:
                seen.add(key)
            continue
        kept.append(line)
        if len(key) >= 12:
            seen.add(key)

    removed = dropped_dup + dropped_noise
    if removed:
        kept.insert(min(HEAD_LINES, len(kept)),
                    f"[TokenSaver: {removed} lines removed "
                    f"({dropped_dup} duplicate, {dropped_noise} noise) "
                    f"- no unseen identifier dropped; full output in vault]")
    return "\n".join(kept), {"dup": dropped_dup, "noise": dropped_noise}


def main() -> int:
    rows = samples()
    print("extractive (no model)\n")
    print(f"{'kind':15s} {'orig':>6s} {'comp':>6s} {'ratio':>7s} {'kept':>6s} "
          f"{'secs':>7s}  removed")
    import time
    for kind, text in rows:
        t0 = time.perf_counter()
        out, stats = compress(text)
        secs = time.perf_counter() - t0
        ratio = 1 - len(out) / len(text)
        orig_d = details(text)
        kept = len(orig_d & details(out)) / len(orig_d) if orig_d else 1.0
        print(f"{kind:15s} {len(text):6,} {len(out):6,} {ratio*100:6.1f}% "
              f"{kept*100:5.0f}% {secs:7.4f}  {stats}")
        lost = sorted(orig_d - details(out))[:8]
        if lost:
            print(f"                LOST: {', '.join(lost)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
