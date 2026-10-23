"""Summarize old assistant prose with a local model, then repair what it lost.

Assistant prose is 11.8% of message content and the largest target left once
the lossless and pointer passes are done. Compressing it means rewriting, which
the passes avoid, so this measures whether a rewrite can be made safe.

Earlier measurement (one prompt, qwen3.6:35b): full detail retention only at
22% shrink; at 50-58% it lost 25-67% of identifiers. Two changes here:

1. Several prompt styles, since one prompt is one data point.
2. **Deterministic repair.** After the model answers, every hard detail
   (`compress.details`) missing from its output gets the original sentence
   that carried it appended verbatim. Guard retention is then 1.0 by
   construction, and the model's job shrinks to "make the rest short". The
   repair costs length, so ratio is reported before and after it.

Retention is judged by both extractors from `stack.py`. The guard is circular
for the repair - it repairs exactly what it can see - so the independent
number is the one that counts. Measured per block, which is stricter than
conversation-wide: prose rarely gets deduplicated against anyway.

Local only: ollama on 127.0.0.1, zero API calls. Slow: about 10-30s per block
on the 35B. Usage: python3 experiments/prose.py [model] [n_blocks]
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver.compress import details  # noqa: E402
from stack import indep_ident, indep_words  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

RULES = """Preserve VERBATIM every file path, identifier (function, class, variable, flag, env var), number, version, error text, command, URL, test name and quoted string.
Keep every decision and its reason, every failure and what was tried, every open question and every promise of future work. Never invert or soften a negation ("not", "never", "no", "only")."""

PROMPTS = {
    # The earlier single prompt's intent, restated.
    "plain": f"""Rewrite the assistant message below to be as short as possible without losing information.
{RULES}
Remove pleasantries, restatement, hedging, narration of intent and filler.
Output ONLY the rewritten message.""",
    # Dense note style: articles and connective prose go, structure stays.
    "telegraphic": f"""Rewrite the assistant message below as terse telegraphic notes: drop articles, filler and full sentences; keep fragments, symbols (->, =, :) and bullet structure.
{RULES}
Output ONLY the notes.""",
    # Length-targeted: the 22% ceiling above came from prompts with no target.
    "short40": f"""Rewrite the assistant message below in AT MOST 40% of its length. Cut explanation, examples, transitions and anything a reader could re-derive; keep conclusions, decisions with their reasons, results, and next steps.
{RULES}
Output ONLY the rewritten message.""",
    "short25": f"""Rewrite the assistant message below in AT MOST 25% of its length, as dense notes. Keep only conclusions, decisions with a one-clause reason, results, failures, open questions and next steps.
{RULES}
Output ONLY the notes.""",
    # Facts list: forces the model to enumerate rather than paraphrase.
    "facts": f"""Extract every fact, decision, result, number and open item from the assistant message below as a flat list of short bullets, one fact per bullet, in original order. Nothing else.
{RULES}
Output ONLY the bullets.""",
}

# QA run 2's real losses were reasons, negations and lost state, not
# identifiers. Sentences carrying these markers are never given to the model.
PROTECT = re.compile(r"""\b(?:because|since|so\ that|due\ to|instead|rather\ than
    |not|never|no|none|nothing|without|only|unless|except|until|but
    |\w+n't|cannot|won't|can't)\b""", re.IGNORECASE | re.VERBOSE)

PROTECT_PROMPT = f"""Rewrite the assistant message below to be as short as possible without losing information.
Markers like [[P1]] stand for sentences that must stay exactly where they are: copy every marker unchanged and do not rewrite around them.
{RULES}
Remove pleasantries, restatement, hedging, narration of intent and filler.
Output ONLY the rewritten message."""

SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [s for s in SENT.split(text) if s.strip()]


def repair(original: str, summary: str) -> str:
    """Append, verbatim, each original sentence carrying a detail the summary lost."""
    missing = details(original) - details(summary)
    if not missing:
        return summary
    add = []
    for s in sentences(original):
        if details(s) & missing:
            add.append(s.strip())
            missing -= details(s)
        if not missing:
            break
    return summary.rstrip() + "\n[verbatim]\n" + "\n".join(add)


def protect(text: str, model: str) -> str:
    """Model compresses only sentences without a reason/negation marker.

    Protected sentences go to the model as [[Pn]] markers and come back
    verbatim. A marker the model dropped is appended, so no protected sentence
    can be lost; ordering may then differ, which the QA test sees."""
    keep: list[str] = []

    def mark(m: re.Match) -> str:
        s = m.group(0)
        if not PROTECT.search(s):
            return s
        keep.append(s)
        return f"[[P{len(keep)}]]"

    # Sentences as spans, so the separators (and line structure) survive.
    # Punctuation ends a sentence only before whitespace, as in SENT, so
    # `prose.py` and `3.6` stay whole.
    masked = re.sub(r".+?(?:[.!?]+(?=\s)|$)", mark, text, flags=re.M)
    if not keep:
        out, _ = ollama(PROMPTS["plain"], text, model)
        return out
    out, _ = ollama(PROTECT_PROMPT, masked, model)
    missing = []
    for i, s in enumerate(keep, 1):
        tag = f"[[P{i}]]"
        if tag in out:
            out = out.replace(tag, s, 1)
        else:
            missing.append(s)
    out = re.sub(r"\[\[P\d+\]\]", "", out)   # duplicated or invented markers
    return out + ("\n" + " ".join(missing) if missing else "")


def summarize(variant: str, text: str, model: str) -> str:
    if variant == "protect":
        return protect(text, model)
    out, _ = ollama(PROMPTS[variant], text, model)
    return out


def extractive(text: str) -> str:
    """No-model baseline: drop sentences that carry no hard detail."""
    return "\n".join(s.strip() for s in sentences(text) if details(s))


def ollama(prompt: str, text: str, model: str, timeout: int = 900) -> tuple[str, float]:
    body = json.dumps({
        "model": model,
        "prompt": f"{prompt}\n\nMESSAGE:\n{text}",
        "stream": False,
        "think": False,          # the qwen3 family reasons by default
        "options": {"temperature": 0, "num_predict": 4096},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = (json.loads(resp.read()).get("response") or "").strip()
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    out = re.sub(r"^```[a-z]*\n|\n```$", "", out).strip()
    return out, time.perf_counter() - t0


def sample(n: int, min_chars: int = 400, max_chars: int = 6000, seed: int = 7) -> list[str]:
    """Assistant prose blocks from real sessions, seeded so runs compare."""
    pool = []
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl")):
        for line in path.open(errors="replace"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") or {}
            if rec.get("type") != "assistant" or not isinstance(msg.get("content"), list):
                continue
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text") or ""
                    if min_chars <= len(t) <= max_chars:
                        pool.append(t)
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def _kept(fn, original: str, out: str) -> float:
    want = fn(original)
    return len(want & fn(out)) / len(want) if want else 1.0


def score(original: str, out: str) -> tuple[float, float, float]:
    """(guard, independent identifiers, long words) retention.

    Identifiers must survive a rewrite. Long words need not - a paraphrase
    changes them by design - so they are a loose signal only, never a gate."""
    return (_kept(details, original, out), _kept(indep_ident, original, out),
            _kept(indep_words, original, out))


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6:35b-mlx"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    blocks = sample(n)
    print(f"model {model}, {len(blocks)} assistant blocks, "
          f"{sum(map(len, blocks)):,} chars\n")
    rows: dict[str, list[tuple]] = {"extractive": []}
    for k in PROMPTS:
        rows[k], rows[k + "+repair"] = [], []
    lost: dict[str, set[str]] = {k: set() for k in rows}
    secs = 0.0
    for idx, text in enumerate(blocks):
        out = extractive(text)
        rows["extractive"].append((len(text), len(out), *score(text, out)))
        lost["extractive"] |= indep_ident(text) - indep_ident(out)
        for k, prompt in PROMPTS.items():
            try:
                out, s = ollama(prompt, text, model)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"  block {idx} {k}: {exc}")
                continue
            secs += s
            # A model that echoes or balloons is not compressing; count it as-is.
            for key, got in ((k, out), (k + "+repair", repair(text, out))):
                rows[key].append((len(text), len(got), *score(text, got)))
                lost[key] |= indep_ident(text) - indep_ident(got)
        print(f"  {idx + 1}/{len(blocks)} done", file=sys.stderr)

    print(f"{'variant':20s} {'n':>3s} {'shrink':>7s} {'guard':>7s} {'ident':>7s} "
          f"{'worst':>7s} {'words':>7s}")
    for k, r in rows.items():
        if not r:
            continue
        o = sum(x[0] for x in r)
        c = sum(x[1] for x in r)
        print(f"{k:20s} {len(r):3d} {100 * (1 - c / o):6.1f}% "
              f"{sum(x[2] for x in r) / len(r):7.3f} {sum(x[3] for x in r) / len(r):7.3f} "
              f"{min(x[3] for x in r):7.3f} {sum(x[4] for x in r) / len(r):7.3f}")
    print(f"\nmodel time {secs:,.0f}s")
    rng = random.Random(1)
    for k, words in lost.items():
        if k.endswith("+repair") and words:
            w = sorted(words)
            print(f"\n{k} identifier losses ({len(w)}), sample: {', '.join(rng.sample(w, min(15, len(w))))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
