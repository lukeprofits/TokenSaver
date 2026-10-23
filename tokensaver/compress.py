"""The guard's "hard details": file names, identifiers, error classes, numbers,
hashes, URLs. Compaction uses it to keep every tool-result line carrying a
detail Claude later mentioned, and to build the identifier index.

`HARD` had no camelCase rule once, so `isFeatureEnabled`, `setUserName` and
`tearDown` were unprotected, and so were acronym-led names (`UIButton`,
`HTTPClient`). Retention still scored 1.0, because it was scored with this
same regex; only an independent extractor (experiments/stack.py) showed the
loss. Guard and judge must differ.

The line-deleting pass that lived here (`compress`, used by `dedup_noise`)
was removed 2026-10-22: it deleted repeated lines from code views, diffs and
records, and blank lines and `---` from pasted text. experiments/extractive.py
keeps the version that first measured it.
"""

from __future__ import annotations

import re

# "Hard details": the things that must survive, deliberately over-inclusive.
# A false positive here only costs compression; a false negative loses an
# identifier, which is the failure this whole project exists to avoid.
HARD = re.compile(r"""
    (?:[A-Za-z0-9_.\-/]+\.(?:py|java|kt|ts|tsx|js|jsx|rb|go|rs|dart|swift
                            |json|yaml|yml|xml|toml|sql|sh|md|txt|csv))
  | (?:[A-Z][A-Za-z0-9]*(?:Exception|Error|Warning|Failure))
  | (?:\b[a-z][a-z0-9]*_[a-z0-9_]+\b)
  | (?:\b[a-z]+[A-Z][A-Za-z0-9]*\b)
  | (?:\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b)
  | (?:\b[A-Z]{2,}[a-z][A-Za-z0-9]*\b)
  | (?:\b[A-Z][A-Z0-9_]{2,}\b)
  | (?:\b\d+\.\d+(?:\.\d+)?\b)
  | (?:\b\d{2,}\b)
  | (?:https?://\S+)
  | (?:\b[0-9a-f]{7,40}\b)
""", re.VERBOSE)


def details(text: str) -> set[str]:
    """Every hard detail in `text`. Used both to protect lines and to score."""
    return {m.group(0) for m in HARD.finditer(text)}
