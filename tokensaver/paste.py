"""Giant machine-made pastes in old history: found, and shortened the way old tool output is.

A developer sometimes pastes a whole log, stack trace or data dump into a message. When a
break-time trim replaces the old part of the session, that paste is resent in full on every
later request, where an old tool result would be cut to its head. Measured on one coworker's
history (2026-11-17): 42% of what his trims covered was user text, and his trims kept 63% of
it (20-25% elsewhere), so most of them did not pay back.

The rule for what the developer wrote stays: their own words are never edited. This module
only touches a *region* of a message that is clearly machine output and very large:

* at least MIN_LINES lines and MIN_CHARS characters (a 10- or 100-line paste is left alone);
* at least MACHINE_SHARE of its non-blank lines look machine-made (timestamps, log levels,
  stack frames, JSON, CSV, build/test output, logfmt, ids) and almost none look like code
  (CODE_SHARE: a pasted source file may be needed exactly);
* a sentence or question around the paste is outside the region and is kept as written.

What a cut region keeps: its first and last HEAD lines, one line of each distinct error or
warning shape (with how often it repeated), and every line carrying a detail mentioned
elsewhere in the session (by Claude, or by the developer after the paste). A one-line note
says how many lines were left out. Nothing to fetch: a pointer is refused by the model
(see the plan, R2), and the developer can paste again.

Only ever applied inside a trim's old part (`background.render`), never to the recent
history kept verbatim, and only when `bg_trim_pastes` is on. Deterministic: the same text
and the same `keep` give the same bytes.
"""

from __future__ import annotations

import re

from . import compress

MIN_LINES = 200
MIN_CHARS = 10_000
MACHINE_SHARE = 0.8
CODE_SHARE = 0.05
MAX_PROSE_RUN = 3            # prose lines allowed in a row inside a region (log headers, notes)
HEAD = 40                    # lines kept at each end of a cut region
SHAPES_MAX = 60              # distinct error/warning shapes kept
CITED_MAX_CHARS = 4_000      # lines kept for a detail used elsewhere in the session
LINE_MAX_CHARS = 400         # a kept line beyond the head and tail
DISTINCT_MAX = 3             # a detail on more lines than this does not single a line out

_MACHINE = re.compile(
    r"""\b\d{4}-\d{2}-\d{2}[T\s]\d{1,2}:\d{2}            # 2026-11-17 10:18 / ISO time
      | \b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b              # 10:18:31.123
      | ^\s*\[\s*\d+(?:\.\d+)?\]                         # [ 12.345678] (dmesg)
      | \b(?:TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|SEVERE|FATAL|CRITICAL)\b
      | \blevel=\w+
      | ^\s+at\s+\S+\s*\(                                # Java / JS stack frame
      | ^\s+at\s+\S+:\d+:\d+\)?\s*$
      | ^\s*File\ ".+",\ line\ \d+                        # Python
      | ^Traceback\ \(most\ recent
      | ^\s*Caused\ by:
      | ^\s+\.\.\.\ \d+\ (?:more|common\ frames)
      | ^\s*\#\d+\s+0x[0-9a-fA-F]+                       # native frame
      | ^\s*[\[\]{}],?\s*$                               # JSON structure
      | ^\s*"[^"\n]{1,120}"\s*:                          # JSON key
      | ^\s*(?:PASS|FAIL|FAILED|ok|not\ ok|SKIP|✓|✗|×|√)\b
      | ^\s*\[(?:INFO|ERROR|WARN|WARNING|DEBUG)\]
      | ^\s*(?:npm|yarn|pnpm)\ (?:ERR|WARN|notice)
      | ^\s*(?:warning|error)(?:\[\w+\])?:\s
      | ^\s*>\s*Task\ :
      | ^\s*[ewi]:\ \S
      | ^\s*\d+\ (?:passing|failing|pending|tests?\ (?:passed|failed))
      | (?:\b[\w.-]+=\S+.*){2,}                          # logfmt: two key=value pairs
      | \b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b
      | \b0x[0-9a-fA-F]{6,}\b
      | \b[0-9a-f]{12,}\b
    """, re.X)
_CODE = re.compile(
    r"""^\s*(?:def\ |class\ |import\ |from\ \S+\ import\ |function\b|const\ |let\ |var\ |
               public\ |private\ |protected\ |return\b|if\s*\(|for\s*\(|while\s*\(|\#include\b|
               package\ |func\ |fn\ |impl\b|@\w+\s*$|@\w+\()""", re.X)
_SIGNAL = re.compile(r"\b(?:ERROR|FATAL|CRITICAL|SEVERE|WARN|WARNING|Exception|Error|FAIL|FAILED|"
                     r"panic|Traceback|Caused by)\b|\b(?:error|warning):")
_SHAPE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s][\d:.,]+Z?|\b0x[0-9a-fA-F]+\b|\b[0-9a-f]{8,}\b|\d+")
MARK = "lines of this pasted output left out"
NOTE = ("[TokenSaver: {n:,} " + MARK + " to keep the history short; "
        "its first and last lines, each kind of error or warning, and every line referred to "
        "later are kept]")


def _machine(line: str) -> bool:
    if _MACHINE.search(line):
        return True
    s = line.strip()
    if len(s) >= 20:
        if s.count(",") >= 3 and ". " not in s:                       # CSV
            return True
        if s.count("\t") >= 2:                                        # TSV
            return True
        letters = sum(c.isalpha() for c in s)
        if letters < 0.4 * len(s):                                    # tables of numbers, hex
            return True
    return False


def regions(text: str) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Line spans [i, j) of `text` that qualify, and why big runs that did not were left
    alone ({"code": n, "mostly prose": n, ...}: diagnostics, counts only)."""
    lines = text.split("\n")
    out: list[tuple[int, int]] = []
    left: dict[str, int] = {}
    if len(lines) < MIN_LINES and len(text) < MIN_CHARS:
        return out, left
    kind = ["blank" if not ln.strip() else "machine" if _machine(ln) else "prose" for ln in lines]
    i, n = 0, len(lines)
    while i < n:
        if kind[i] != "machine":
            i += 1
            continue
        j, last, run = i, i, 0
        while j < n:
            if kind[j] == "machine":
                last, run = j, 0
            elif kind[j] == "prose":
                run += 1
                if run > MAX_PROSE_RUN:
                    break
            j += 1
        a, b = i, last + 1
        i = b
        span = lines[a:b]
        if len(span) < MIN_LINES or sum(len(x) + 1 for x in span) < MIN_CHARS:
            if len(span) >= MIN_LINES // 2:
                left["too small"] = left.get("too small", 0) + 1
            continue
        nonblank = [k for k in kind[a:b] if k != "blank"]
        if sum(k == "machine" for k in nonblank) < MACHINE_SHARE * len(nonblank):
            left["mostly prose"] = left.get("mostly prose", 0) + 1
            continue
        if sum(bool(_CODE.match(x)) for x in span) >= CODE_SHARE * len(span):
            left["reads like code"] = left.get("reads like code", 0) + 1
            continue
        out.append((a, b))
    if not out and not left and len(text) >= MIN_CHARS and len(lines) < MIN_LINES:
        left["under 200 lines"] = 1
    return out, left


def _shape(line: str) -> str:
    return _SHAPE.sub("#", line.strip())[:200]


def cut(text: str, keep: set[str] | None = None) -> tuple[str, int]:
    """(`text` with each qualifying region shortened, lines left out). Unchanged when no
    region qualifies. Lines outside the regions are untouched."""
    spans, _ = regions(text)
    if not spans:
        return text, 0
    lines = text.split("\n")
    out, pos, dropped = [], 0, 0
    for a, b in spans:
        out += lines[pos:a]
        body = lines[a:b]
        kept: dict[int, str] = {}
        for k in range(min(HEAD, len(body))):
            kept[k] = body[k]
        for k in range(max(len(body) - HEAD, 0), len(body)):
            kept[k] = body[k]
        shapes: dict[str, list] = {}
        cited = 0
        # A detail marks a line as referred to only if it is distinctive in the paste: "10" or
        # "INFO" on every line of a log would otherwise spend the budget on its first lines.
        per_line = [compress.details(ln) for ln in body] if keep else []
        seen: dict[str, int] = {}
        for ds in per_line:
            for x in ds:
                seen[x] = seen.get(x, 0) + 1
        for k, ln in enumerate(body):
            if k in kept:
                continue
            if _SIGNAL.search(ln):
                s = _shape(ln)
                if s in shapes:
                    shapes[s][1] += 1
                elif len(shapes) < SHAPES_MAX:
                    shapes[s] = [k, 1]
                    kept[k] = ln[:LINE_MAX_CHARS]
                    continue
            if keep and cited < CITED_MAX_CHARS and any(
                    seen[x] <= DISTINCT_MAX for x in per_line[k] & keep):
                kept[k] = ln[:LINE_MAX_CHARS]
                cited += len(kept[k])
        for s, (k, times) in shapes.items():
            if times > 1:
                kept[k] += f"   [this kind of line {times:,} times]"
        gone = len(body) - len(kept)
        if gone <= 0:
            out += body
        else:
            dropped += gone
            out.append(NOTE.format(n=gone))
            prev = -1
            for k in sorted(kept):
                if prev >= 0 and k > prev + 1:
                    out.append("[...]")
                out.append(kept[k])
                prev = k
        pos = b
    out += lines[pos:]
    return "\n".join(out), dropped
