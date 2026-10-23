"""Episode compaction (experimental): finished sub-tasks collapse to what was
asked and what was done.

A session is a run of sub-tasks: fix a failing build, write a test, rename
things. Once one is finished and the work has moved on, its process (the
investigation, plans, every tool call and result, the narration in between)
rarely matters; what was asked and what was done does. So, for the old part:

1. the local 35B splits it into episodes (message ranges), each marked
   finished or ongoing, with a one-line outcome;
2. Nimble scores each finished episode: is its detail still needed for the
   current work (the recent part)? With its probability;
3. a finished episode judged not needed keeps
   - E1: the developer's messages and the assistant's own final message of the
     episode, verbatim (extractive; nothing written by a model), or
   - E2: the developer's messages plus the 35B's one-line outcome, accepted only
     if every identifier in it appears in that episode (else E1).

Measured by experiments/episodes.py against what Claude actually used after
real breaks; nothing here is wired into the proxy.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import background as bg  # noqa: E402
from tokensaver import prune  # noqa: E402

SEG_CHUNK_CHARS = 60_000
SEG_LINE_CHARS = 600
NEEDED_MAX_P = 0.2              # collapse only when p(needed) is below this

SEG_PROMPT = """Below is part of a coding session between a developer and an AI assistant, one numbered message per line (#n ROLE: text; tool output is cut short).

Split it into EPISODES: consecutive messages working on one sub-task (for example: fixing one failing build, writing one test, one round of renames, answering one question). For each episode give the first and last message numbers, whether the sub-task was FINISHED within these messages (done, confirmed working, and the work moved on), and a one-line outcome in the form "<what> changed to/now <how>" using only names that appear in those messages.

Reply with a JSON list only, e.g.
[{"start": 0, "end": 14, "finished": true, "outcome": "build fixed: registry 401 solved by logging in again"}, {"start": 15, "end": 40, "finished": false, "outcome": "outbox test in progress"}]"""

_IDENT = re.compile(r"[A-Za-z_][\w.\-/]*[A-Z_.\-/0-9][\w.\-/]*|\b\w+\.\w{1,5}\b")


def _lines(messages: list, lo: int, hi: int) -> list[str]:
    out = []
    for i in range(lo, hi):
        r = bg.render([messages[i]], 300, 200)
        if r:
            out.append(f"#{i} " + r[0][:SEG_LINE_CHARS].replace("\n", " "))
    return out


def segment(messages: list, cut: int, model: str | None = None, log: list | None = None,
            progress=None) -> list[dict]:
    """Episodes over messages[:cut], chunk by chunk."""
    model = model or bg.LOCAL_MODEL
    eps, i = [], 0
    while i < cut:
        lines, size, j = [], 0, i
        while j < cut and (not lines or size < SEG_CHUNK_CHARS):
            for ln in _lines(messages, j, j + 1):
                lines.append(ln)
                size += len(ln)
            j += 1
        t = time.time()
        try:
            got = bg._json_list(bg.ollama(SEG_PROMPT, "\n".join(lines), model, timeout=prune.SPOT_TIMEOUT_S)) or []
        except Exception as exc:
            got = []
            if log is not None:
                log.append({"seg_error": f"{type(exc).__name__}: {exc}"[:200]})
        n = 0
        for e in got:
            try:
                s, en = int(e["start"]), int(e["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if i <= s <= en < j:
                # an episode reaching the end of a chunk may continue past it: not finished
                fin = bool(e.get("finished")) and en < j - 1
                eps.append({"start": s, "end": en, "finished": fin, "outcome": str(e.get("outcome") or "")[:300]})
                n += 1
        if progress:
            progress(f"segment #{i}-#{j - 1}: {n} episodes in {time.time() - t:.0f}s")
        i = j
    return eps


_NEED_Q = {"needed": {
    "type": "choice",
    "instructions": "Is the DETAIL of the EARLIER EPISODE (its tool calls, outputs and intermediate notes) still "
                    "needed to continue the CURRENT WORK, beyond knowing its outcome?",
    "criteria": {
        "yes": "The current work builds directly on details of that episode (its exact code, values, errors or "
               "steps), or it is unclear.",
        "no": "The episode is finished and the current work needs at most its outcome."}}}


def needed(episode_text: str, current: str) -> float:
    """p(detail still needed), from Nimble."""
    url = bg.OLLAMA_URL.split("/api/")[0] + "/v1/systemone"
    state = f"CURRENT WORK (latest messages):\n{current}\n\nEARLIER EPISODE:\n{episode_text}"
    body = json.dumps({"model": "nimble", "state": state, "questions": _NEED_Q}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        ans = json.loads(resp.read())["answers"]["needed"]
    return float((ans.get("probabilities") or {}).get("yes") or 0.0)


def current_digest(messages: list, cut: int, chars: int = 6_000) -> str:
    lines = bg.render(messages[cut:], 300, 200)
    return "\n".join(lines)[-chars:]


def _episode_text(messages: list, e: dict, chars: int = 5_000) -> str:
    text = "\n".join(_lines(messages, e["start"], e["end"] + 1))
    head, tail = text[:chars // 2], text[-chars // 2:]
    return f"OUTCOME: {e['outcome']}\n{head}\n...\n{tail}" if len(text) > chars else f"OUTCOME: {e['outcome']}\n{text}"


def outcome_ok(messages: list, e: dict) -> bool:
    """E2's one-line outcome may name only things that appear in the episode."""
    src = "\n".join(json.dumps(messages[i]) for i in range(e["start"], e["end"] + 1))
    names = {m.group(0).strip(".-/") for m in _IDENT.finditer(e["outcome"])}
    return bool(e["outcome"]) and all(n in src for n in names if len(n) > 3)


def collapse(messages: list, episodes: list[dict], variant: str) -> tuple[set[int], list[str]]:
    """(item ids to skip, outcome lines to add) for the episodes to collapse."""
    skip, notes = set(), []
    for e in episodes:
        last_text = None
        for i in range(e["start"], e["end"] + 1):
            m = messages[i]
            if m.get("role") == "assistant":
                for x in bg._items(m):
                    if isinstance(x, dict) and x.get("type") == "text" and x.get("text"):
                        last_text = x
        use_e2 = variant == "E2" and outcome_ok(messages, e)
        for i in range(e["start"], e["end"] + 1):
            m = messages[i]
            for x in bg._items(m):
                if not isinstance(x, dict):
                    continue
                if m.get("role") == "user" and x.get("type") == "text":
                    continue                    # the developer's words (and instructions in reminders) stay
                if m.get("role") == "system":
                    continue                    # instructions and notices: handled elsewhere
                if x is last_text and not use_e2:
                    continue                    # E1: the assistant's own conclusion stays
                skip.add(id(x))
        if use_e2:
            notes.append(f"- messages {e['start']}-{e['end']}: {e['outcome']}")
    return skip, notes


DROP_MIN_P = 0.8     # E4: Nimble must say a piece can go at this probability (E3, removed, used it too)


def recent_names(messages: list, cut: int) -> set[str]:
    names = set()
    for m in messages[cut:]:
        names |= {x.group(0) for x in _IDENT.finditer(json.dumps(m)) if len(x.group(0)) > 4}
    return names


# ---- E4: the 35B labels, Nimble verifies the drops ---------------------------
#
# Nimble alone never chose "learning" (measured: 0 of 63 pieces), so lasting
# facts were dropped as "process". The 35B reads the pieces properly and labels
# them; Nimble then answers an independent yes/no, "can this piece go without
# losing anything the current work needs?". A piece goes only if the 35B says
# plan/process/obsolete AND Nimble says yes at >= DROP_MIN_P AND it names
# nothing the recent part names.

LABEL_PROMPT = """Below are numbered pieces from one finished part of a coding session, and what the assistant is working on NOW.

Label each piece with exactly one word:
- result: the final state or outcome of that work, confirmed (what is true now);
- learning: a lasting fact about the code, tools or environment that is still true (where something lives, how an API behaves, why something is done a certain way);
- plan: a plan, proposal or intention;
- process: an intermediate step, attempt, command output or narration with no lasting fact beyond the result;
- obsolete: something stated earlier that is no longer true.
When unsure between learning and process, choose learning.

Reply with JSON only, e.g. {"P12": "process", "P13": "result"}"""

_VERIFY_Q = {"removable": {
    "type": "choice",
    "instructions": "Can the PIECE be removed from the history without losing anything the CURRENT WORK needs?",
    "criteria": {"yes": "It holds only plans, intermediate steps or outdated statements; nothing the current "
                        "work relies on.",
                 "no": "It holds a lasting fact, a result, or anything the current work may rely on, or it is unclear."}}}


def label_35b(pieces_: list, current: str, model: str | None = None) -> dict[int, str]:
    model = model or bg.LOCAL_MODEL
    out: dict[int, str] = {}
    chunk, size = [], 0
    for p in pieces_ + [None]:
        line = None if p is None else f"[P{p.n}] {p.kind}: " + " ".join(p.text.split())[:900]
        if p is None or (chunk and size + len(line) > 40_000):
            text = f"CURRENT WORK:\n{current[-4000:]}\n\nPIECES:\n" + "\n\n".join(chunk)
            try:
                raw = bg.ollama(LABEL_PROMPT, text, model, timeout=prune.SPOT_TIMEOUT_S)
                m = re.search(r"\{.*\}", raw, re.S)
                got = json.loads(m.group(0)) if m else {}
            except Exception:
                got = {}
            for k, v in (got or {}).items():
                if isinstance(k, str) and k.startswith("P") and k[1:].isdigit() and isinstance(v, str):
                    out[int(k[1:])] = v.strip().lower()
            chunk, size = [], 0
        if p is not None:
            chunk.append(line)
            size += len(line)
    return out


def verify_removable(p, current: str) -> float:
    url = bg.OLLAMA_URL.split("/api/")[0] + "/v1/systemone"
    state = f"CURRENT WORK:\n{current[-3000:]}\n\nPIECE:\n{p.text[:2500]}"
    body = json.dumps({"model": "nimble", "state": state, "questions": _VERIFY_Q}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        ans = json.loads(resp.read())["answers"]["removable"]
    return float((ans.get("probabilities") or {}).get("yes") or 0.0)
