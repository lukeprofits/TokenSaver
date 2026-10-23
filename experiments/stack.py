"""Measure the forward passes (`passes.FORWARD`) on every transcript on disk.

For each pass, in pipeline order: the **marginal** saving it adds over the
passes before it, and two retention numbers, both conversation-wide.

  guard  - `compress.details`, the regex the passes use to protect lines
  indep  - a second extractor that no pass uses: identifiers, quoted strings,
           long words

Guard retention alone is circular. It scored 1.0 while camelCase identifiers
were being dropped, because the guard could not see them and neither could
the judge. Only `indep` caught it.

Retention is reported twice: what is still *in context*, and what is
*recoverable* (in context, or in a block whose bytes moved to the vault, like
a heredoc script). Every pass must score 1.0 recoverable. (The pointer passes
this once also measured were removed 2026-10-22; see passes.py.)

Local only: reads ~/.claude/projects/*/*.jsonl. Zero API calls. Note the
`heredocs` pass writes vaulted scripts to the TokenSaver vault.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import passes  # noqa: E402
from tokensaver.compress import details  # noqa: E402
from tokensaver.transcripts import read_session  # noqa: E402

# Quotes must not touch a letter, or "I'm ... don't" reads as a quoted string.
IDENT = re.compile(r"""\b[A-Za-z_][A-Za-z0-9_]*[A-Z0-9_][A-Za-z0-9_]*\b
                     | (?<![A-Za-z])"[^"\n]{3,80}"(?![A-Za-z])
                     | (?<![A-Za-z])'[^'\n]{3,80}'(?![A-Za-z])""", re.X)
WORDS = re.compile(r"\b[a-z]{8,}\b")


def indep_ident(text: str) -> set[str]:
    return {m.group(0) for m in IDENT.finditer(text)}


def indep_words(text: str) -> set[str]:
    return {m.group(0) for m in WORDS.finditer(text)}


def indep(text: str) -> set[str]:
    return indep_ident(text) | indep_words(text)


def context_sets(blocks, fn):
    out: set[str] = set()
    for b in blocks:
        out |= fn(b.text)
    return out


def ratio_at(blocks, request: int) -> tuple[int, int]:
    """(original chars, rewritten chars) of what request `request` resends.
    Forward passes only look backwards, so this is causal as it stands."""
    orig = comp = 0
    for b in blocks:
        if b.request >= request:
            break
        orig += len(b.original)
        comp += len(b.text)
    return orig, comp


def main() -> int:
    root = Path.home() / ".claude/projects"
    paths = sorted(root.glob("*/*.jsonl"))
    names = [name for name, _ in passes.FORWARD]
    removed = {n: 0 for n in names}
    total = 0
    # conversation-wide, keyed by session so equal strings in two sessions
    # never cover for each other
    orig = {"guard": set(), "indep": set()}
    final = {"guard": set(), "indep": set()}
    recoverable = {"guard": set(), "indep": set()}
    lost_by = {n: {"guard": set(), "indep": set()} for n in names}
    causal_o = causal_c = 0           # message chars at each request, causal
    ctx_tokens = msg_tokens_est = 0   # provider-reported context, and the floor-excluded share

    for path in paths:
        blocks = passes.from_transcript(path)
        if not blocks:
            continue
        key = path.name
        fns = {"guard": details, "indep": indep}
        before = {k: context_sets(blocks, f) for k, f in fns.items()}
        for k in fns:
            orig[k] |= {(key, w) for w in before[k]}
        total += sum(len(b.original) for b in blocks)

        state = passes.State()
        for name, fn in passes.FORWARD:
            blocks = fn(blocks, state)
            after = {k: context_sets(blocks, f) for k, f in fns.items()}
            for k in fns:
                lost_by[name][k] |= {(key, w) for w in before[k] - after[k]}
            before = after
        for b in blocks:
            for k, f in fns.items():
                if b.vaulted:
                    recoverable[k] |= {(key, w) for w in f(b.original)}
            for n, saved in b.removed_by.items():
                removed[n] += saved
        for k in fns:
            final[k] |= {(key, w) for w in before[k]}
            recoverable[k] |= {(key, w) for w in before[k]}

        # Causal share of what each billed request resends, weighted by that
        # request's provider-reported context, with the session floor
        # (system + tools + project context) counted as incompressible.
        session = read_session(path)
        floor = session.floor
        for i, req in enumerate(session.requests, start=1):
            o, c = ratio_at(blocks, i)
            if not o:
                continue
            msg = max(req.context - floor, 0)
            ctx_tokens += req.context
            msg_tokens_est += msg
            causal_o += msg
            causal_c += msg * c / o

    print(f"{len(paths)} transcripts, {total:,} message chars\n")
    print(f"{'pass':18s} {'marginal':>9s} {'cum':>7s}   lost in context (guard / indep)")
    cum = 0
    for n in names:
        cum += removed[n]
        print(f"{n:18s} {100 * removed[n] / total:8.1f}% {100 * cum / total:6.1f}%   "
              f"{len(lost_by[n]['guard']):5d} / {len(lost_by[n]['indep']):5d}")
    print()
    for k in ("guard", "indep"):
        o = orig[k]
        print(f"retention {k:5s}: in context {len(o & final[k]) / len(o):.4f}   "
              f"recoverable {len(o & recoverable[k]) / len(o):.4f}")
    print()
    print("Causal, weighted by each request's real context, with the session floor")
    print("(system + tools) treated as incompressible:")
    if causal_o:
        share = 1 - causal_c / causal_o
        print(f"  message tokens resent, causal: {100 * share:.1f}% smaller")
        print(f"  whole context resent (floor included): "
              f"{100 * (causal_o - causal_c) / ctx_tokens:.1f}% smaller "
              f"(messages are {100 * msg_tokens_est / ctx_tokens:.0f}% of context)")

    rng = random.Random(4)
    for n in names:
        lost = sorted(w for _, w in lost_by[n]["indep"] - recoverable["indep"])
        if lost:
            print(f"\nunrecoverable indep losses from {n} ({len(lost)}), sample:")
            print("  " + ", ".join(rng.sample(lost, min(15, len(lost)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
