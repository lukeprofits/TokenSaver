"""Background compaction: replace old history at free points, extractively.

Constraints, from the user: no lost context the developer would notice, no
delay, and no spend. Each one is met by a rule below.

* **No spend, no model.** The replacement is built by rules alone, on any
  machine. Local models (ollama) only add the optional prune stages, which a
  final check on the 35B guards (`proxy._prune_step`); never the API.
* **No delay.** It is prepared while the session is idle and applied on the
  request after a long enough quiet spell, at a free point the proxy decides
  on real counts (`proxy._decide`). If it is not ready, nothing changes.
* **Little lost context.** Only the old part is replaced, and extractively:
  every user and assistant word kept as written, each tool call as one line,
  each tool result as its head plus any line carrying a detail Claude later
  used, the latest view of each document, and a deterministic identifier
  index. The most recent ~`KEEP_TOKENS` stay verbatim. No model check: on 10
  real breaks the 35B's check did not track what was lost (2026-10-23); the
  after-the-fact signals in `proxy._quality` turn compaction off if Claude
  re-reads, re-runs or fails edits more after compactions.
* **Claude's thinking stays valid.** Opus 5.5 binds thinking to the exact
  bytes before it, so the kept tail's thinking is removed when the summary
  goes in (`apply`); thinking made after that is bound to the new prefix.

The index is built with the guard extractor (`compress.details`), and the
gate's scoring uses the model's answers. The two differ on purpose: a judge
that shares the guard's regex scores its own blind spots as perfect.

Nothing here hides content behind a retrieval instruction: vault pointers made
the Opus 5.5 safeguards refuse 4 of 4 turns that asked about hidden content.
Removed 2026-10-22: the abstractive (local-model summary) mode, 0 of 4 through
the gate, and opt-in "retrieve notes", where cutting deeper because a fetch
was possible lost money on 7 real breaks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter

from . import compress, config

# Settings come from `config` (file, env for manual runs). These module values
# are the current ones; `refresh()` re-reads them, and every function looks
# them up when called, so a tuned value reaches a running proxy.
OLLAMA_URL = config.get("ollama_url")
LOCAL_MODEL = config.get("local_model")

# The check model per machine (deep prune + the final check on a pruned
# compaction), picked by setup (`gate_model_for`), largest first. A tier is
# listed only if, on real breaks (experiments/gate_models.py), it failed every
# deliberately broken compaction the 35B failed and agreed with the 35B on the
# real ones. Below the smallest: no deep prune; compaction itself needs no model.
GATE_MODELS: list[tuple[int, str | None, str]] = [   # (min RAM in GB, "macos" | "linux" | None, ollama tag)
    (48, "macos", "qwen3.6:35b-mlx"),                 # the reference (MLX: Apple silicon)
]
# No smaller tier: experiments/gate_models.py, 2026-10-23, 8 real breaks. Each smaller model let
# through a broken compaction the 35B blocked: qwen3:14b 1 of 16 (all of Claude's text removed),
# qwen3:8b 4 of 16, gemma3:4b 1 of 4 (run stopped there); qwen3:4b gave no usable answers.
# Machines without the 35B get no deep prune (rules-only compaction still runs).


def gate_model_for(ram_bytes: int, platform: str) -> str | None:
    gb = ram_bytes / 2 ** 30
    for min_gb, plat, tag in GATE_MODELS:
        if gb >= min_gb and plat in (None, platform):
            return tag
    return None

KEEP_TOKENS = config.get("bg_keep_tokens")      # recent history kept verbatim
# What decides: how much old history a compaction would replace (the system,
# tools and the recent tail are never compacted). Simulated on 50 sessions:
# total size > 100K 20.39% of bill net, old part >= 15K 20.49% (4 more
# compactions on smaller sessions); flat from 5K to 15K, lower at 25K+.
MIN_REMOVABLE = config.get("bg_min_removable")
INDEX_SIZE = config.get("bg_index")             # 2,000: 95.6% recall, 1,000: 85.7%; gate passed at both, 1,000 ~5 points smaller
RESULT_CHARS = 1_500        # of each tool result shown to the QA gate
EXTRACT_RESULT_CHARS = 300  # of each old tool result kept by the extractive mode
QA_CHUNKS = 4               # 24 questions: 18 left pass/fail to noise (measured)
QA_QUESTIONS = 6
QA_MIN_RATIO = 0.8          # replacement must answer >= 80% of what the original did
CHARS_PER_TOKEN = 4

ASK = f"""Read the transcript excerpt below. Write exactly {QA_QUESTIONS} short questions that the assistant, continuing this work later, might need answered from it, each with a short reference answer taken from the excerpt.
Ask about facts that stay true as the work continues: the goal, a decision and its reason, the cause of a problem, an instruction from the developer, something explicitly NOT done or ruled out, which file or component holds something. Do not ask about counts or results that later work would change.
Output ONLY JSON: [{{"q": "...", "a": "..."}}, ...]"""

ANSWER = """Answer each question using ONLY the text below. If the text does not contain the answer, reply exactly UNKNOWN. Keep each answer short.
Output ONLY JSON: ["answer 1", "answer 2", ...]"""

GRADE = """For each item, decide if the candidate answer gives the same information as the reference answer, for a developer who asked the question.
Count it as matching when it is shorter, terser or worded differently but names the same thing, value, reason or outcome. The same file named by a relative or absolute path, or with or without its extension, is the same thing.
Count it as NOT matching when it gives a different value or name, drops or flips a negation, omits the reason when the question asks why, or describes a different action. UNKNOWN never matches.
Output ONLY JSON: [true, false, ...]"""


RECHECK = """For each item, answer only this: does the candidate state the reference's key fact (the same file, value, reason, cause or action), even if it says more, says it differently, or is cut off? A different fact, a flipped negation, or a missing reason when the question asks why is NO. UNKNOWN is NO.
Output ONLY JSON: [true, false, ...]"""


def refresh() -> None:
    """Re-read this module's settings from `config`."""
    global OLLAMA_URL, LOCAL_MODEL, KEEP_TOKENS, MIN_REMOVABLE, INDEX_SIZE
    OLLAMA_URL, LOCAL_MODEL = config.get("ollama_url"), config.get("local_model")
    KEEP_TOKENS, MIN_REMOVABLE = config.get("bg_keep_tokens"), config.get("bg_min_removable")
    INDEX_SIZE = config.get("bg_index")


# ---- local model -------------------------------------------------------------

LOCAL_CTX = 32_768
PROMPT_MAX_CHARS = int(LOCAL_CTX * 3.2) - 4096 * 4   # leave room for the answer


def ollama(prompt: str, text: str, model: str | None = None, timeout: int = 1800) -> str:
    model = model or LOCAL_MODEL
    # Ollama cuts an over-long prompt from the FRONT, silently: the
    # instructions go first. Measured: the gate was answering from the tail
    # of a 48K-token replacement. Refuse instead.
    if len(prompt) + len(text) > PROMPT_MAX_CHARS:
        raise ValueError(f"prompt of {len(prompt) + len(text):,} chars exceeds the local window")
    body = json.dumps({
        "model": model, "prompt": f"{prompt}\n\nTEXT:\n{text}", "stream": False,
        "think": False, "options": {"temperature": 0, "num_predict": 4096, "num_ctx": LOCAL_CTX},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = (json.loads(resp.read()).get("response") or "").strip()
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    return re.sub(r"^```[a-z]*\n|\n```$", "", out).strip()


def _json_list(text: str):
    m = re.search(r"\[.*\]", text, re.S)
    try:
        got = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    return got if isinstance(got, list) else None


# ---- history -----------------------------------------------------------------

# Thinking signatures are long base64 but bill far less than chars/4: valid
# thinking measured 0.14 tokens per signature character (a real session, 2026-10-22:
# 23,803 tokens for 169,800 chars; count_tokens with and without thinking).
TOKENS_PER_SIGNATURE_CHAR = 0.14


def _signature_chars(x) -> int:
    if isinstance(x, dict):
        n = len(x["signature"]) if x.get("type") in ("thinking", "redacted_thinking") and isinstance(
            x.get("signature"), str) else 0
        if x.get("type") == "redacted_thinking" and isinstance(x.get("data"), str):
            n += len(x["data"])
        return n + sum(_signature_chars(v) for k, v in x.items() if k in ("content",) and isinstance(v, list))
    if isinstance(x, list):
        return sum(_signature_chars(v) for v in x)
    return 0


def size_tokens(messages: list) -> int:
    """Estimated tokens: content at CHARS_PER_TOKEN, thinking signatures at
    their measured rate (they were counted as text before 2026-10-22, which
    made old parts with much thinking look far bigger than they bill)."""
    sig = _signature_chars(messages)
    total = len(json.dumps(messages, ensure_ascii=False))
    return round((total - sig) / CHARS_PER_TOKEN + sig * TOKENS_PER_SIGNATURE_CHAR)


def fingerprint(messages: list) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()[:24]


def _items(message: dict) -> list:
    c = message.get("content")
    return c if isinstance(c, list) else [{"type": "text", "text": c}] if isinstance(c, str) else []


def loaded_tools(messages: list) -> list[str]:
    """Names of deferred tools these messages loaded: Claude Code's tool search
    answers with `tool_reference` blocks inside a tool_result, and the API
    expands each into that tool's definition. A compaction that summarizes
    those messages away would unload the tools (Claude Code's own compaction
    keeps them for the same reason), so the proxy sends them undeferred while
    the compaction is in force. Sorted: the same bytes every request."""
    names: set[str] = set()
    for m in messages:
        for x in _items(m):
            if isinstance(x, dict) and x.get("type") == "tool_result" and isinstance(x.get("content"), list):
                for y in x["content"]:
                    if isinstance(y, dict) and y.get("type") == "tool_reference" and y.get("tool_name"):
                        names.add(str(y["tool_name"]))
    return sorted(names)


def undefer(tools, names) -> list | None:
    """`tools` with `defer_loading` dropped for `names`, or None if none of them
    is deferred there (nothing to change)."""
    if not isinstance(tools, list) or not names:
        return None
    want, out, changed = set(names), [], False
    for t in tools:
        if isinstance(t, dict) and t.get("defer_loading") and t.get("name") in want:
            t = {k: v for k, v in t.items() if k != "defer_loading"}
            changed = True
        out.append(t)
    return out if changed else None


def _is_turn_start(message: dict) -> bool:
    """A developer's own message: role user and no tool_result. Cutting
    anywhere else would leave a tool_result whose tool_use was summarized
    away, which the API rejects."""
    if message.get("role") != "user":
        return False
    items = _items(message)
    return bool(items) and not any(isinstance(x, dict) and x.get("type") == "tool_result" for x in items)


def worth_compacting(messages: list) -> bool:
    """Enough old history to replace (see MIN_REMOVABLE)."""
    cut = choose_cut(messages)
    return bool(cut) and size_tokens(messages[:cut]) >= MIN_REMOVABLE


def choose_cut(messages: list, keep_tokens: int | None = None) -> int | None:
    """Index of the first message kept verbatim, or None. It is the latest
    turn start that still leaves at least `keep_tokens` after it, so the
    recent work stays whole."""
    keep_tokens = KEEP_TOKENS if keep_tokens is None else keep_tokens
    tail = 0
    for i in range(len(messages) - 1, 0, -1):
        tail += size_tokens([messages[i]])
        if tail >= keep_tokens and _is_turn_start(messages[i]):
            return i
    return None


MENTIONED_LINE_CHARS = 240   # a kept line of an old tool result
MENTIONED_CAP = 2_000        # per tool result, beyond its head


def mentioned(messages: list) -> set[str]:
    """Hard details the assistant itself wrote in its own text. What Claude
    talked about is what mattered to it."""
    out: set[str] = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for x in _items(m):
            if isinstance(x, dict) and x.get("type") == "text":
                out |= compress.details(x.get("text") or "")
    return out


def _result_excerpt(text: str, head: int, keep: set[str] | None) -> str:
    if len(text) <= head:
        return text
    out = text[:head]
    if keep:
        extra, n = [], 0
        for line in text[head:].split("\n"):
            if n >= MENTIONED_CAP:
                break
            if compress.details(line) & keep:
                extra.append(line[:MENTIONED_LINE_CHARS])
                n += len(extra[-1])
        if extra:
            out += "\n[...]\n" + "\n".join(extra)
    return out + " [...]"


DOC_SUFFIXES = (".md", ".txt", ".rst")
DOC_CHARS = 20_000           # the latest view of a document is kept up to this
SHRINK_MIN_CHARS = 800       # the lossy profile only touches pieces longer than this


def latest_doc_views(messages: list) -> set[str]:
    """tool_use ids of the last Read of each document file. Plans and notes
    carry the goals and reasons; measured, cutting them to a head lost
    exactly those answers. Code is left cut: it is on disk, and Claude
    re-reads a file before editing it anyway."""
    last: dict[str, str] = {}
    for m in messages:
        for x in _items(m):
            if not (isinstance(x, dict) and x.get("type") == "tool_use"):
                continue
            inp = x.get("input") or {}
            if x.get("name") == "Read":
                paths = [str(inp.get("file_path") or "")]
            elif x.get("name") == "Bash" and _DOC_VIEW.search(str(inp.get("command") or "")):
                paths = _DOC_PATH.findall(str(inp.get("command")))
            else:
                continue
            for path in paths:
                if path.lower().endswith(DOC_SUFFIXES):
                    last[path] = x.get("id") or ""
    return set(last.values())


# A Bash view of a document: cat/sed/head/nl/tail on a .md/.txt/.rst path.
_DOC_VIEW = re.compile(r"(?:^|[;&|(]\s*)(?:cat|sed|head|tail|nl)\b")
_DOC_PATH = re.compile(r"[\w./~\-]+\.(?:md|txt|rst)\b", re.I)


# Results that say only "it worked": the call line above already implies them.
_ROUTINE = re.compile(r"^(?:The file \S+ has been (?:updated|created) successfully\.?|File created "
                      r"successfully at: \S+)\s*$", re.S)
_RULE = re.compile(r"^[\s─━═\-=_*#~.]{20,}$", re.M)


# Claude Code's own notices inside the history. Measured on a real compaction
# (659 messages): "file changed on disk" notices with their full diffs were
# 3.3% of the compacted text, and plan-mode instructions from a plan mode that
# had since ended 1.6%.
_NOTICE = re.compile(r"^Note: \S.* (?:changed on disk|was modified)", re.S)
_PLAN_ON = "Plan mode is active."
_PLAN_OFF = "## Exited Plan Mode"


def _system_text(m: dict) -> str:
    if m.get("role") != "system":
        return ""
    return "\n".join(x.get("text", "") for x in _items(m) if isinstance(x, dict))


def stale_plan_mode(messages: list) -> set[int]:
    """ids of plan-mode instruction messages followed later by an exit from
    plan mode: instructions for a mode that is over. The approved plan itself
    is kept elsewhere (the ExitPlanMode result, the plan file view)."""
    stale, pending = set(), []
    for m in messages:
        t = _system_text(m)
        if t.startswith(_PLAN_ON):
            pending.append(id(m))
        elif t.startswith(_PLAN_OFF):
            stale.update(pending)
            pending = []
    return stale


def render(messages: list, result_chars: int = RESULT_CHARS, call_chars: int = 800,
           keep: set[str] | None = None, full_ids: set[str] | None = None,
           drop_routine: bool = False, skip: set[int] | None = None, shrink=None) -> list[str]:
    """One line of text per message, for the compaction text and the QA gate.
    Thinking is left out; tool results are cut to their head, plus any line
    carrying a detail in `keep`. With `drop_routine`, Claude Code's
    file-changed notices are cut the same way (they are stale diffs, like an
    old file view). `skip` holds ids of messages, or of single content items,
    to leave out; a run of left-out items (not whole notices) is marked with
    one line, so Claude knows something was there. Nothing to fetch.

    `shrink(kind, text) -> text` (the "lines" prune stage, `prune.line_picker`)
    may shorten old document views ("doc"), tool-result excerpts ("result"),
    long system text ("system") and system reminders ("reminder"); never what
    the developer or Claude wrote."""
    out = []
    removed = 0
    for m in messages:
        if skip and id(m) in skip:
            continue
        role = (m.get("role") or "?").upper()
        parts = []
        for x in _items(m):
            if not isinstance(x, dict):
                continue
            if skip and id(x) in skip:
                removed += 1
                continue
            t = x.get("type")
            if t == "text" and x.get("text") and drop_routine and role == "SYSTEM" \
                    and _NOTICE.match(x["text"]):
                parts.append(_result_excerpt(x["text"], result_chars, keep))
            elif t == "text" and x.get("text"):
                text = x["text"]
                if shrink and len(text) > SHRINK_MIN_CHARS:
                    if role == "SYSTEM":
                        text = shrink("system", text)
                    elif role == "USER" and text.lstrip().startswith("<system-reminder>"):
                        text = shrink("reminder", text)
                parts.append(text)
            elif t == "tool_use":
                inp = x.get("input") or {}
                shown = inp.get("command") or inp.get("file_path") or json.dumps(inp)[:500]
                parts.append(f"[tool call {x.get('name')}] {str(shown)[:call_chars]}")
            elif t == "tool_result":
                c = x.get("content")
                text = c if isinstance(c, str) else "\n".join(
                    y.get("text", "") for y in (c or []) if isinstance(y, dict))
                if drop_routine and not x.get("is_error"):
                    if _ROUTINE.match(text.strip()):
                        continue
                    text = _RULE.sub("", text)
                if full_ids and x.get("tool_use_id") in full_ids:
                    head = text[:DOC_CHARS] + (" [...]" if len(text) > DOC_CHARS else "")
                    kind = "doc"
                else:
                    head = _result_excerpt(text, result_chars, keep)
                    kind = "result"
                if shrink and len(head) > SHRINK_MIN_CHARS:
                    head = shrink(kind, head)
                parts.append(f"[tool result{' ERROR' if x.get('is_error') else ''}] {head}")
        if parts:
            if removed:
                out.append(f"[{removed} earlier piece{'s' if removed > 1 else ''} removed here: "
                           f"superseded by later work in this session]")
                removed = 0
            out.append(f"{role}: " + "\n".join(parts))
    return out


def index(old: list, recent: list, size: int | None = None, shown: str = "") -> str:
    """The identifiers of the old part that neither the recent part nor the
    compacted text (`shown`) already carries, ranked by how often the assistant
    itself used them, then by frequency. Deterministic: no model.

    Measured: without `shown`, the index was 21% of the compacted text, mostly
    repeating identifiers the verbatim assistant text already showed."""
    said, freq = Counter(), Counter()
    for m in old:
        for x in _items(m):
            text = json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
            for d in compress.details(text):
                freq[d] += 1
                if m.get("role") == "assistant":
                    said[d] += 1
    present = set()
    for m in recent:
        present |= compress.details(json.dumps(m, ensure_ascii=False))
    ranked = sorted((d for d in freq if d not in present), key=lambda d: (-said[d], -freq[d], d))
    # Pick the top `size` first (the set the 95.6% recall was measured on),
    # then drop what the compacted text already shows. Filtering first only
    # backfilled the index with lower-ranked identifiers: measured, it stayed
    # the same size.
    visible = compress.details(shown) if shown else set()
    size = INDEX_SIZE if size is None else size
    return ", ".join(d for d in ranked[:size] if d not in visible)


# ---- the gate ----------------------------------------------------------------

def _qa(chunk: str, candidate: str, model: str, log: list | None = None,
        ask: str | None = None, ask_text: str | None = None) -> tuple[int, int, int]:
    """(questions, right from the original chunk, right from the candidate).
    `ask`/`ask_text` replace the question-writing prompt and its input (the
    current-work check shows the current work alongside the excerpt)."""
    qa = _json_list(ollama(ask or ASK, ask_text or chunk, model)) or []
    qa = [x for x in qa if isinstance(x, dict) and x.get("q") and x.get("a")][:QA_QUESTIONS]
    if not qa:
        return 0, 0, 0
    qs = "\n".join(f"{i + 1}. {x['q']}" for i, x in enumerate(qa))
    scores, detail = [], []
    for source in (chunk, candidate):
        answers = [str(a) for a in (_json_list(ollama(ANSWER, f"{source}\n\nQUESTIONS:\n{qs}", model)) or [])]
        answers = (answers + ["UNKNOWN"] * len(qa))[:len(qa)]
        items = "\n".join(f"{i + 1}. reference: {x['a']} | candidate: {a}"
                          for i, (x, a) in enumerate(zip(qa, answers)))
        marks = [bool(g) for g in (_json_list(ollama(GRADE, items, model)) or [])]
        marks = (marks + [False] * len(qa))[:len(qa)]
        # Second opinion on every rejection that is not UNKNOWN: the first
        # grader marked supersets, cut-off answers and rewordings wrong
        # (measured, half its rejections). Applied to both sides alike.
        redo = [i for i, (m, a) in enumerate(zip(marks, answers))
                if not m and a.strip().upper() != "UNKNOWN"]
        if redo:
            items2 = "\n".join(f"{n + 1}. question: {qa[i]['q']} | reference: {qa[i]['a']} | "
                                f"candidate: {answers[i]}" for n, i in enumerate(redo))
            second = [bool(g) for g in (_json_list(ollama(RECHECK, items2, model)) or [])]
            for n, i in enumerate(redo):
                if n < len(second) and second[n]:
                    marks[i] = True
        marks = [m and a.strip().upper() != "UNKNOWN" for m, a in zip(marks, answers)]
        scores.append(sum(marks))
        detail.append((answers, marks))
    if log is not None:
        for i, x in enumerate(qa):
            log.append({"q": x["q"], "ref": x["a"], "orig": detail[0][0][i], "orig_ok": detail[0][1][i],
                        "cand": detail[1][0][i], "cand_ok": detail[1][1][i]})
    return len(qa), scores[0], scores[1]


def _spans(messages: list, size: int = 12_000) -> list[tuple[int, int]]:
    """Consecutive message ranges whose rendered original is about `size` chars."""
    spans, start, n = [], 0, 0
    for i, m in enumerate(messages):
        n += sum(map(len, render([m])))
        if n >= size:
            spans.append((start, i + 1))
            start, n = i + 1, 0
    if start < len(messages):
        spans.append((start, len(messages)))
    return spans


ASK_CURRENT = f"""Below is what the assistant is working on NOW (CURRENT WORK), followed by an EARLIER excerpt of the same session.
Write exactly {QA_QUESTIONS} short questions that the assistant, continuing the CURRENT work, might need answered from the EARLIER excerpt, each with a short reference answer taken from the EARLIER excerpt.
Prefer facts the current work depends on: decisions and their reasons, how something was fixed, where something lives, instructions from the developer, anything ruled out. Do not ask about details of finished work that the current work does not build on.
Output ONLY JSON: [{{"q": "...", "a": "..."}}, ...]"""


def gate(old: list, candidate_for, model: str | None = None, current: str | None = None,
         original_for=None, only=None) -> dict:
    """Questions from evenly spaced stretches of the old history, answered
    from the stretch itself (the control) and from `candidate_for(i, j)`, the
    compressed form of the same stretch. Like for like, and both fit the
    local window. With `current` (a digest of the recent part), the questions
    are the ones the current work might need: the check for compactions that
    drop finished work's details on purpose.

    `original_for(i, j)` replaces the raw stretch as the control (the pruned
    check uses the rules-only compaction: did pruning lose what it had?), and
    `only(i, j)` limits the picks to stretches where something changed."""
    spans = _spans(old)
    if only is not None:
        spans = [sp for sp in spans if only(*sp)] or spans
    if not spans:
        return {"passed": False, "why": "nothing to test"}
    picks = sorted({round(k * (len(spans) - 1) / max(QA_CHUNKS - 1, 1)) for k in range(QA_CHUNKS)})
    total = orig = cand = 0
    log: list = []
    for k in picks:
        i, j = spans[k]
        chunk = (original_for(i, j) if original_for else "\n\n".join(render(old[i:j])))[:PROMPT_MAX_CHARS - 2_000]
        candidate = candidate_for(i, j)[:PROMPT_MAX_CHARS - 2_000]
        if current:
            ask_text = f"CURRENT WORK:\n{current[-6000:]}\n\nEARLIER EXCERPT:\n{chunk[:PROMPT_MAX_CHARS - 10_000]}"
            n, o, c = _qa(chunk, candidate, model, log, ask=ASK_CURRENT, ask_text=ask_text)
        else:
            n, o, c = _qa(chunk, candidate, model, log)
        total, orig, cand = total + n, orig + o, cand + c
    passed = orig >= 3 and cand >= QA_MIN_RATIO * orig
    return {"passed": passed, "questions": total, "original_right": orig, "replacement_right": cand,
            "items": log}


# ---- building the replacement ------------------------------------------------

def header(n_old: int, old_tokens: int) -> str:
    return (f"[TokenSaver: background compaction. The earlier part of this session "
            f"({n_old} messages, about {old_tokens:,} tokens) was summarized while the "
            f"developer was away; the most recent messages follow verbatim.]")


def _extract(messages: list, cut: int, prune: set[int] | None = None, shrink=None):
    """The extractive replacement for messages[:cut], and a per-stretch view of
    it for the gate. No model: every user and assistant word kept as written,
    each tool call as one line, each tool result as its head. It invents
    nothing; what it drops is tool output, and the files are still on disk."""
    old, recent = messages[:cut], messages[cut:]
    keep = mentioned(messages)
    docs = latest_doc_views(old)
    skip = stale_plan_mode(messages) | set(prune or ())
    body = "\n\n".join(render(old, EXTRACT_RESULT_CHARS, 200, keep=keep, full_ids=docs,
                               drop_routine=True, skip=skip, shrink=shrink))
    idx = index(old, recent, shown=body)
    text = (f"{header(cut, size_tokens(old))} Tool results are cut to their first "
            f"lines.\n\n## The earlier part of the session\n\n{body}"
            f"\n\n## Identifiers from the earlier part\n\n{idx}")

    def candidate_for(i: int, j: int) -> str:
        part = "\n\n".join(render(old[i:j], EXTRACT_RESULT_CHARS, 200, keep=keep, full_ids=docs,
                                   drop_routine=True, skip=skip, shrink=shrink))
        return f"{part}\n\n## Identifiers from the earlier part\n\n{idx[:12_000]}"
    result = {"cut": cut, "prefix_fp": fingerprint(old), "text": text, "mode": "extract",
              "old_tokens": size_tokens(old), "new_tokens": len(text) // CHARS_PER_TOKEN,
              "made": time.time(), "summary_seconds": 0}
    return result, candidate_for


def build(messages: list) -> dict | None:
    """The extractive compaction, deterministic and model-free, so tuning can
    estimate what compaction would save at no cost (the same text `prepare`
    makes)."""
    cut = choose_cut(messages)
    if not cut or size_tokens(messages[:cut]) < MIN_REMOVABLE:
        return None
    return _extract(messages, cut)[0]


# The rules-only compaction is applied without a model check (2026-10-23). It
# keeps every user and assistant word and cites what Claude used; on 10 real
# breaks the 35B's check did not track what was actually lost (it passed the
# breaks that lost most, and flipped pass/fail on the same compaction). The
# check now guards only the model-driven prune stages, once, at the end
# (proxy._prune_step); after-the-fact signals (proxy._quality) guard this one.
RULES_ONLY = {"passed": True, "check": "none (rules only)"}


def prepare(messages: list) -> dict | None:
    """Everything the proxy needs to apply a compaction later, or None. No
    model: seconds, on any machine.

    The result is pure data (cut index, prefix fingerprint, replacement text),
    so applying it is deterministic and costs nothing."""
    cut = choose_cut(messages)
    if not cut or size_tokens(messages[:cut]) < MIN_REMOVABLE:
        return None
    started = time.time()
    result, _ = _extract(messages, cut)
    result["gate"] = dict(RULES_ONLY)
    result["seconds"] = round(time.time() - started)
    return result


def apply(messages: list, prepared: dict) -> list | None:
    """`messages` with the prepared summary in place of the old part, or None
    when the history no longer starts the way it did when it was prepared.

    The summary goes in as a text block at the head of the first kept user
    message, so roles still alternate and no tool_result loses its call."""
    cut = prepared.get("cut")
    if not isinstance(cut, int) or cut <= 0 or cut >= len(messages):
        return None
    if fingerprint(messages[:cut]) != prepared.get("prefix_fp"):
        return None
    first = dict(messages[cut])
    if not _is_turn_start(first):
        return None
    first["content"] = [{"type": "text", "text": prepared["text"]}] + list(_items(first))
    out = [first] + list(messages[cut + 1:])
    # Preserved thinking (Opus 5.5+; docs build-with-claude/preserved-thinking):
    # a thinking block is valid only while everything before it is unchanged,
    # so the kept tail's thinking is invalid once its prefix is our summary:
    # old accounts drop it silently, accounts from 2026-08-31 get a 400.
    # Removing ALL thinking up to a point is valid (checked 2026-10-22 with
    # prefix_mismatch_behavior "error": 400 with it, 200 without). Only the
    # messages that existed when the compaction was applied (`tail_end`):
    # thinking produced after that was made on this prefix and stays.
    end = prepared.get("tail_end")
    if isinstance(end, int):
        out[:end - cut] = [strip_thinking(m) for m in out[:end - cut]]
    return out


def strip_thinking(message: dict) -> dict:
    """`message` without thinking or redacted_thinking blocks (a copy if any)."""
    c = message.get("content")
    if message.get("role") != "assistant" or not isinstance(c, list):
        return message
    kept = [x for x in c if not (isinstance(x, dict) and x.get("type") in ("thinking", "redacted_thinking"))]
    if len(kept) == len(c):
        return message
    return dict(message, content=kept or [{"type": "text", "text": "(continued)"}])
