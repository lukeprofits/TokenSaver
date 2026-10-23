"""Does a summary keep the *meaning*, not just the identifiers?

`prose.py` proves identifiers survive. It cannot see a dropped "not", a lost
reason, or a decision flipped. This asks questions instead:

1. From the original, the local model writes QUESTIONS short factual
   questions with reference answers - deliberately including decisions,
   reasons, negations and numbers.
2. Each candidate text (original, summaries, a crude cut-off) answers them
   with nothing else in view. "UNKNOWN" is allowed and counted as a miss.
3. The model grades each answer against the reference.

The same local model writes, answers and grades, which can flatter it. Two
controls keep that honest: the original must score near 1.0 (the grader is not
just harsh) and the first-third cut-off must score clearly lower (it is not
just lenient). If those two do not separate, the numbers mean nothing.

Local only: ollama, zero API calls. Every output is written to OUT for
reading by eye. Usage:
  python3 experiments/prose_qa.py [model] [n_blocks] [prompt,prompt,...]
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import names  # noqa: E402
from prose import PROMPTS, ollama, repair, sample, summarize  # noqa: E402

QUESTIONS = 6
# Real transcript text: kept in the private vault directory, never in the repo.
OUT = Path(os.environ.get("TOKENSAVER_QA_OUT")
           or names.home() / "experiments" / "prose_qa.jsonl")

ASK = f"""Read the assistant message below. Write exactly {QUESTIONS} short questions that a developer continuing this work might need answered from it, each with a short reference answer taken from the message.
Cover, where present: a decision and its reason, something explicitly NOT done or ruled out, a number or name, a result or failure, and a next step.
Output ONLY JSON: [{{"q": "...", "a": "..."}}, ...]"""

ANSWER = """Answer each question using ONLY the text below. If the text does not contain the answer, reply exactly UNKNOWN. Keep each answer short.
Output ONLY JSON: ["answer 1", "answer 2", ...]"""

# Run 2's grader marked "15" wrong against "15 tests pass" and "four" against
# "four commits": about half its misses on read-through were strictness, not
# loss. This one accepts terser answers but still rejects the real failures
# seen there: a different value, a flipped negation, a missing reason.
GRADE = """For each item, decide if the candidate answer gives the same information as the reference answer, for a developer who asked the question.
Count it as matching when it is shorter, terser or worded differently but names the same thing, value, reason or outcome.
Count it as NOT matching when it gives a different value or name, drops or flips a negation, omits the reason when the question asks why, or describes a different action. UNKNOWN never matches.
Output ONLY JSON: [true, false, ...]"""


def _json(text: str):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def questions(text: str, model: str) -> list[dict] | None:
    out, _ = ollama(ASK, text, model)
    qa = _json(out)
    if not isinstance(qa, list):
        return None
    qa = [x for x in qa if isinstance(x, dict) and x.get("q") and x.get("a")]
    return qa[:QUESTIONS] or None


def answer(candidate: str, qa: list[dict], model: str) -> list[str]:
    qs = "\n".join(f"{i + 1}. {x['q']}" for i, x in enumerate(qa))
    out, _ = ollama(ANSWER, f"{candidate}\n\nQUESTIONS:\n{qs}", model)
    got = _json(out)
    got = [str(a) for a in got] if isinstance(got, list) else []
    return (got + ["UNKNOWN"] * len(qa))[:len(qa)]


def grade(qa: list[dict], answers: list[str], model: str) -> list[bool]:
    items = "\n".join(f"{i + 1}. reference: {x['a']} | candidate: {a}"
                      for i, (x, a) in enumerate(zip(qa, answers)))
    out, _ = ollama(GRADE, items, model)
    got = _json(out)
    marks = [bool(g) for g in got] if isinstance(got, list) else []
    marks = (marks + [False] * len(qa))[:len(qa)]
    # UNKNOWN never matches, whatever the grader said.
    return [m and a.strip().upper() != "UNKNOWN" for m, a in zip(marks, answers)]


def regrade(model: str) -> int:
    """Re-grade the answers already in OUT with the current GRADE; no regeneration."""
    rows = [json.loads(line) for line in OUT.open() if line.strip()]
    for n, row in enumerate(rows):
        for got in row["variants"].values():
            try:
                got["correct"] = grade(row["qa"], got["answers"], model)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"  block {row['block']}: {exc}", file=sys.stderr)
        print(f"  {n + 1}/{len(rows)} regraded", file=sys.stderr)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report(rows, list(rows[0]["variants"]) if rows else [], model)
    return 0


def report(rows_out: list[dict], variants: list[str], model: str) -> None:
    score = {v: [0, 0] for v in variants}
    size = {v: [0, 0] for v in variants}
    for r in rows_out:
        for v in variants:
            got = r["variants"][v]
            score[v][0] += sum(got["correct"])
            score[v][1] += len(got["correct"])
            size[v][0] += len(r["original"])
            size[v][1] += len(got["text"])
    print(f"model {model}; questions answered correctly, by variant\n")
    print(f"{'variant':22s} {'shrink':>7s} {'correct':>9s}")
    for v in variants:
        ok, tot = score[v]
        o, c = size[v]
        if tot:
            print(f"{v:22s} {100 * (1 - c / o):6.1f}% {100 * ok / tot:7.1f}%  ({ok}/{tot})")
    print(f"\ncontrols: original should be near 100%, cutoff clearly lower. Details: {OUT}")

    # Paired against the original on the same questions: cancels most of the
    # grader's noise, which shows up as misses on the original itself.
    print(f"\n{'variant':22s} {'lost':>5s} {'gained':>7s}   (vs original, same questions)")
    for v in variants[1:]:
        pairs = [(o, c) for r in rows_out for o, c in
                 zip(r["variants"]["original"]["correct"], r["variants"][v]["correct"])]
        print(f"{v:22s} {sum(o and not c for o, c in pairs):5d} {sum(c and not o for o, c in pairs):7d}")
    for v in variants[2:]:
        print(f"\n--- {v}: right from the original, wrong from the summary")
        for r in rows_out:
            got = r["variants"][v]
            for i, (o, c) in enumerate(zip(r["variants"]["original"]["correct"], got["correct"])):
                if o and not c:
                    print(f"Q: {r['qa'][i]['q']}\n  ref: {r['qa'][i]['a']}\n  got: {got['answers'][i]}")


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6:35b-mlx"
    if len(sys.argv) > 2 and sys.argv[2] == "--regrade":
        return regrade(model)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    chosen = sys.argv[3].split(",") if len(sys.argv) > 3 else list(PROMPTS)
    prompts = {k: k for k in chosen}   # "protect" is not a prompt, see prose.protect
    variants = ["original", "cutoff"] + [k + "+repair" for k in prompts]
    rows_out: list[dict] = []
    OUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # TOKENSAVER_QA_SLICE="10:20" runs part of the same seeded sample and
    # appends, so a long run can be split into batches that each finish.
    lo, _, hi = (os.environ.get("TOKENSAVER_QA_SLICE") or f"0:{n}").partition(":")
    lo, hi = int(lo or 0), int(hi or n)
    with OUT.open("a" if lo else "w") as log:
        for idx, text in list(enumerate(sample(n)))[lo:hi]:
            try:
                qa = questions(text, model)
                if not qa:
                    print(f"  block {idx}: no usable questions", file=sys.stderr)
                    continue
                cands = {"original": text, "cutoff": text[:len(text) // 3]}
                for k, prompt in prompts.items():
                    out = summarize(k, text, model)
                    cands[k + "+repair"] = repair(text, out)
                row = {"block": idx, "original": text, "qa": qa, "variants": {}}
                for v, cand in cands.items():
                    answers = answer(cand, qa, model)
                    marks = grade(qa, answers, model)
                    row["variants"][v] = {"text": cand, "answers": answers, "correct": marks}
                rows_out.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"  block {idx}: {exc}", file=sys.stderr)
            print(f"  {idx + 1}/{n} done", file=sys.stderr)

    report(rows_out, variants, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
