"""Smart prune: a local model chooses which old pieces are superseded.

A human reading a long session can tell "that plan was approved, implemented
and described later" or "that attempt failed and was fixed later" and would
drop the earlier copy. This does the same for background compaction, but only
by SELECTING whole pieces to delete: survivors stay byte for byte, so it can
invent or distort nothing. The one way it can fail is deleting something that
was needed later, which experiments/smart_prune.py measures on real breaks.

Pieces (the units that may go):
- an assistant text block;
- a tool call together with its result.
Never: user text, system/hook/skill text, reminders, kept doc views.

Tiers, cheapest first:
- T1 `candidates`: for each piece, the later piece (rest of the old part, or the
  recent verbatim part) it overlaps most with; word 5-gram containment, or the
  same file path. No model, instant.
- T2/T3 `judge`: a local model answers YES/NO, "does LATER make EARLIER
  unnecessary?", twice with the order of the two pieces swapped. Deleted only on
  YES both times. Small prompts, one-word answers, so reading dominates.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field

from . import background as bg

SHINGLE = 5
MIN_CONTAIN = 0.35          # share of the earlier piece's 5-grams found in the later one
MIN_PIECE_CHARS = 200       # smaller pieces are not worth a question
PIECE_CHARS = 2_500         # what the judge sees of each piece

_WORD = re.compile(r"[A-Za-z0-9_./%-]+")
_PATH = re.compile(r"(?:~|/)[\w.\-/]+\.\w+|\b[\w\-]+/[\w.\-/]+\.\w+")


@dataclass
class Piece:
    n: int                      # order in the conversation
    old: bool                   # in the part being compacted
    kind: str                   # assistant | tool
    items: list                 # the content item dicts (removed together)
    text: str                   # what the compacted text would show
    paths: set = field(default_factory=set)
    grams: set = field(default_factory=set)


def _grams(text: str) -> set:
    w = [x.lower() for x in _WORD.findall(text)]
    return {" ".join(w[i:i + SHINGLE]) for i in range(max(len(w) - SHINGLE + 1, 0))}


def pieces(messages: list, cut: int, protect: set[str] | None = None) -> list[Piece]:
    """Pieces of the whole history; `old` marks those before `cut`.
    `protect` holds tool_use ids whose results must stay (kept doc views)."""
    protect = protect or set()
    results = {}
    for m in messages:
        for x in bg._items(m):
            if isinstance(x, dict) and x.get("type") == "tool_result":
                results[x.get("tool_use_id")] = x
    out: list[Piece] = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        for x in bg._items(m):
            if not isinstance(x, dict):
                continue
            if x.get("type") == "text" and x.get("text"):
                text = x["text"]
                p = Piece(len(out), i < cut, "assistant", [x], text)
            elif x.get("type") == "tool_use":
                if x.get("id") in protect:
                    continue
                res = results.get(x.get("id"))
                inp = x.get("input") or {}
                call = inp.get("command") or inp.get("file_path") or json.dumps(inp)[:500]
                body = ""
                if res is not None:
                    c = res.get("content")
                    body = c if isinstance(c, str) else "\n".join(
                        y.get("text", "") for y in (c or []) if isinstance(y, dict))
                text = f"[tool call {x.get('name')}] {call}\n[tool result{' ERROR' if res and res.get('is_error') else ''}] {body}"
                p = Piece(len(out), i < cut, "tool", [x] + ([res] if res is not None else []), text)
                if inp.get("file_path"):
                    p.paths.add(inp["file_path"])
            else:
                continue
            p.paths |= set(_PATH.findall(p.text[:2000]))
            p.grams = _grams(p.text[:20_000])
            out.append(p)
    return out


def candidates(ps: list[Piece]) -> list[tuple[Piece, Piece, float]]:
    """(earlier old piece, the later piece most likely to supersede it, score)."""
    out = []
    later_by_path: dict[str, list[Piece]] = {}
    for p in ps:
        for path in p.paths:
            later_by_path.setdefault(path, []).append(p)
    for a in ps:
        if not a.old or len(a.text) < MIN_PIECE_CHARS or not a.grams:
            continue
        best, score = None, 0.0
        for b in ps[a.n + 1:]:
            if b.kind != a.kind and not (a.paths & b.paths):
                continue
            s = len(a.grams & b.grams) / len(a.grams) if b.grams else 0.0
            if a.paths & b.paths and a.kind == b.kind == "tool":
                s = max(s, 0.5)               # the same file, looked at or changed again later
            if s > score:
                best, score = b, s
        if best is not None and score >= MIN_CONTAIN:
            out.append((a, best, round(score, 3)))
    return out


QUESTION = """Two pieces from one coding session between a developer and an AI assistant. {first_label} came earlier in the session than {second_label}.

Answer YES only if, for continuing this work, the EARLIER piece is no longer needed because the LATER piece makes it redundant: every fact, decision, file, value or reason in EARLIER that still matters is also in LATER, or EARLIER was a draft, attempt or view that LATER replaced, finished or corrected. Answer NO if EARLIER holds anything that still matters and is not in LATER, or if you are unsure.

Reply with one word: YES or NO."""


_THINKS: dict[str, bool] = {}     # models that reason in plain text before answering
THINK_TOKENS = 600


def _generate(model: str, prompt: str, num_predict: int, timeout: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(bg.OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = (json.loads(resp.read()).get("response") or "")
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def _ask(model: str, prompt: str, timeout: int = 600) -> str:
    """YES or NO. A one-word answer is asked for; a model that reasons first
    (measured: this qwen3:4b build ignores think=false and starts "Hmm, ...")
    gets room to reason, and its LAST yes/no counts. No answer is NO."""
    if not _THINKS.get(model):
        out = _generate(model, prompt, 3, timeout).upper()
        if out.startswith(("YES", "NO")):
            return "YES" if out.startswith("YES") else "NO"
        _THINKS[model] = True
    out = _generate(model, prompt, THINK_TOKENS, timeout).upper()
    found = re.findall(r"\b(YES|NO)\b", out)
    return found[-1] if found else "NO"


# Nimble (Bespoke Labs, Apache-2.0, 9B, `ollama pull nimble`) is a decision
# model: ollama's /v1/systemone takes the text plus named questions and returns
# each answer WITH its probability, scoring the answer tokens directly (no
# reasoning step). Measured here: 0.3-0.5 s per decision on an M3 Max. A piece
# is deleted only when p(yes) reaches NIMBLE_MIN_P.
NIMBLE_MIN_P = 0.9
_NIMBLE_Q = {"superseded": {
    "type": "choice",
    "instructions": "Is the EARLIER piece no longer needed for continuing this work because the LATER piece makes it redundant?",
    "criteria": {
        "yes": "Every fact, decision, file, value or reason in EARLIER that still matters is also in LATER, "
               "or EARLIER was a draft, attempt or view that LATER replaced, finished or corrected.",
        "no": "EARLIER holds something that still matters and is not in LATER, or it is unclear."}}}


def _nimble(state: str, timeout: int = 300) -> float:
    url = bg.OLLAMA_URL.split("/api/")[0] + "/v1/systemone"
    body = json.dumps({"model": "nimble", "state": state, "questions": _NIMBLE_Q}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ans = json.loads(resp.read())["answers"]["superseded"]
    return float((ans.get("probabilities") or {}).get("yes") or 0.0)


def judge(a: Piece, b: Piece, model: str) -> tuple[bool, list[str]]:
    """Deleted only if the model says YES with the pieces in both orders
    (generative models), or with p(yes) >= NIMBLE_MIN_P (nimble)."""
    if model.startswith("nimble"):
        p = _nimble(f"EARLIER:\n{a.text[:PIECE_CHARS]}\n\nLATER:\n{b.text[:PIECE_CHARS]}")
        return p >= NIMBLE_MIN_P, [f"p={p:.2f}"]
    ea, lb = a.text[:PIECE_CHARS], b.text[:PIECE_CHARS]
    votes = []
    for first_is_earlier in (True, False):
        if first_is_earlier:
            head = QUESTION.format(first_label="EARLIER", second_label="LATER")
            prompt = f"{head}\n\nEARLIER:\n{ea}\n\nLATER:\n{lb}"
        else:
            head = QUESTION.format(first_label="EARLIER", second_label="LATER")
            prompt = f"{head}\n\nLATER:\n{lb}\n\nEARLIER:\n{ea}"
        votes.append(_ask(model, prompt))
        if votes[-1] == "NO":
            break
    return all(v == "YES" for v in votes) and len(votes) == 2, votes


# ---- deep pass: the 35B spots restatements, Nimble judges them -------------
#
# Word overlap only proposes near-copies, so a plan or explanation restated
# later in other words never reaches the judge. The larger local model reads a
# chunk of old pieces in full plus one-line headers of every later piece and
# names (old, later) pairs it thinks are superseded; it writes only numbers.
# Every pair it names is then judged like any other, so a wrong proposal costs
# a judge call, not a deletion.

SPOT_CHUNK_CHARS = 42_000       # old pieces shown in full per call
SPOT_PIECE_CHARS = 1_400
SPOT_HEADERS_CHARS = 30_000     # later pieces, one line each
SPOT_HEADER_LEN = 110
SPOT_TIMEOUT_S = 300
PROGRESS = None                 # a callable(str) to report progress (experiments/smart_prune.py sets it)

SPOT_PROMPT = """Below are numbered pieces from one coding session between a developer and an AI assistant: OLD pieces shown in full, and one-line headers of LATER pieces.

List each OLD piece that is clearly no longer needed because a LATER piece replaced, finished, corrected or restated it (for example a plan draft later approved, an attempt later fixed, a file view later re-read or rewritten, an explanation later summarized). Name the LATER piece that makes it redundant. Leave out anything you are not sure about; leaving a piece in is always safe.

Reply with a JSON list of [old_number, later_number] pairs and nothing else, e.g. [[12, 87], [15, 90]], or [] if there are none."""


def _header(p: Piece) -> str:
    return f"#{p.n} {p.kind}: " + " ".join(p.text.split())[:SPOT_HEADER_LEN]


def spot(ps: list[Piece], model: str, log: list | None = None,
         should_stop=None) -> list[tuple[Piece, Piece]] | None:
    """(old, later) pairs the model proposes, chunk by chunk. None if
    `should_stop()` said to stop before the last chunk."""
    by_n = {p.n: p for p in ps}
    old = [p for p in ps if p.old and len(p.text) >= MIN_PIECE_CHARS]
    out, i = [], 0
    while i < len(old):
        if should_stop and should_stop():
            return None
        chunk, size = [], 0
        while i < len(old) and (not chunk or size + min(len(old[i].text), SPOT_PIECE_CHARS) <= SPOT_CHUNK_CHARS):
            chunk.append(old[i])
            size += min(len(old[i].text), SPOT_PIECE_CHARS)
            i += 1
        first = chunk[0].n
        later = [p for p in ps if p.n > first]
        headers, hsize = [], 0
        for p in reversed(later):                    # newest first: the end state matters most
            h = _header(p)
            if hsize + len(h) > SPOT_HEADERS_CHARS:
                break
            headers.append(h)
            hsize += len(h) + 1
        headers.reverse()
        text = ("OLD PIECES:\n\n" + "\n\n".join(f"#{p.n} {p.kind}:\n{p.text[:SPOT_PIECE_CHARS]}" for p in chunk)
                + "\n\nLATER PIECES (headers):\n" + "\n".join(headers))
        t = time.time()
        try:
            # 5 min, not ollama()'s default 30: one chunk measured 24.5 s, and an
            # overnight run that stalled without a trace cost 2 hours.
            got = bg._json_list(bg.ollama(SPOT_PROMPT, text, model, timeout=SPOT_TIMEOUT_S)) or []
        except Exception as exc:
            got = []
            if log is not None:
                log.append({"spot_error": f"{type(exc).__name__}: {exc}"[:200]})
        chunk_ns = {p.n for p in chunk}
        pairs = 0
        for pair in got:
            if (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, int) for x in pair)
                    and pair[0] in chunk_ns and pair[1] in by_n and pair[1] > pair[0]):
                out.append((by_n[pair[0]], by_n[pair[1]]))
                pairs += 1
        if log is not None:
            log.append({"spot_chunk": [chunk[0].n, chunk[-1].n], "proposed": pairs,
                        "seconds": round(time.time() - t, 1)})
        if PROGRESS:
            PROGRESS(f"spot chunk #{chunk[0].n}-#{chunk[-1].n}: {pairs} proposed in {time.time() - t:.0f}s")
    return out


# ---- running it for real: models, memory, caching, easing off ---------------
#
# Production wraps the passes above with what a background job on someone's
# laptop needs: decisions cached per session (a "superseded" verdict stays true
# as the history grows, and the newest piece in a chain is never deleted, so
# verdicts made during work are reused at the free point); a memory check
# before any model loads; each model unloaded as soon as its stage is done; and
# `machine.Yield` between decisions so the machine never strains for this.

# Measured in-memory sizes (ollama /api/ps) were larger than the downloads:
# nimble 17.7 GB for 9.3 GB, the 35B 28.5 GB for 21.9 GB.
MEMORY_FACTOR = 1.9


def _api(path: str, body: dict | None = None, timeout: int = 20) -> dict:
    url = bg.OLLAMA_URL.split("/api/")[0] + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def installed() -> dict[str, int]:
    """name -> download size, for models ollama has. {} when it is down."""
    try:
        return {m["name"]: int(m.get("size") or 0) for m in _api("/api/tags").get("models", [])}
    except Exception:
        return {}


def loaded() -> dict[str, int]:
    """name -> bytes in memory now."""
    try:
        return {m["name"]: int(m.get("size") or 0) for m in _api("/api/ps").get("models", [])}
    except Exception:
        return {}


def _name(model: str, names) -> str | None:
    for n in names:
        if n == model or n.split(":")[0] == model or n == f"{model}:latest":
            return n
    return None


def unload(model: str) -> None:
    try:
        _api("/api/generate", {"model": model, "keep_alive": 0})
    except Exception:
        pass


def can_load(model: str) -> tuple[bool, str]:
    from . import machine
    inst = installed()
    name = _name(model, inst)
    if not name:
        return False, f"{model} is not installed"
    now = loaded()
    if _name(model, now):
        return True, "already loaded"
    seen = (config_section().get("memory") or {}).get(name)
    need = int(seen or inst[name] * MEMORY_FACTOR)
    return machine.fits(need)


def config_section() -> dict:
    from . import config
    return config.section("prune")


def remember_size(model: str) -> None:
    """Record a model's real in-memory size once it is loaded."""
    from . import config
    now = loaded()
    name = _name(model, now)
    if not name:
        return
    st = config_section()
    mem = dict(st.get("memory") or {})
    if mem.get(name) != now[name]:
        mem[name] = now[name]
        st["memory"] = mem
        try:
            config.set_section("prune", st)
        except Exception:
            pass


def tier() -> tuple[str, str]:
    """(off | fast | deep, why), from `prune_tier` and what this machine has."""
    from . import config, machine
    want = config.get("prune_tier")
    if want == "off":
        return "off", "prune_tier is off"
    inst = installed()
    judge_ok = bool(_name(config.get("prune_judge"), inst))
    deep_ok = judge_ok and bool(_name(bg.LOCAL_MODEL, inst)) and machine.ram_bytes() >= 32 * 2 ** 30
    if want == "deep":
        return ("deep", "set") if deep_ok else (("fast", "deep needs the local model and 32 GB+")
                                                if judge_ok else ("off", "the judge model is not installed"))
    if want == "fast":
        return ("fast", "set") if judge_ok else ("off", "the judge model is not installed")
    if deep_ok:
        return "deep", "auto: judge and local model installed, 32 GB+"
    if judge_ok:
        return "fast", "auto: judge installed"
    return "off", f"auto: `ollama pull {config.get('prune_judge')}` to enable"


class Cache:
    """Verdicts per (earlier text, later text, judge), persisted per session."""

    def __init__(self, path=None):
        self.path, self.data = path, {}
        if path is not None:
            try:
                self.data = json.loads(path.read_text())
            except Exception:
                self.data = {}

    @staticmethod
    def key(a: Piece, b: Piece, model: str) -> str:
        import hashlib
        h = hashlib.sha256()
        for part in (model, a.text[:PIECE_CHARS], b.text[:PIECE_CHARS]):
            h.update(part.encode("utf-8", "replace") + b"\0")
        return h.hexdigest()[:32]

    def get(self, a, b, model):
        return self.data.get(self.key(a, b, model))

    def put(self, a, b, model, p: float) -> None:
        self.data[self.key(a, b, model)] = round(p, 4)

    def save(self) -> None:
        if self.path is None:
            return
        try:
            if len(self.data) > 20_000:                       # bound the file
                self.data = dict(list(self.data.items())[-20_000:])
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            tmp.replace(self.path)
        except Exception:
            pass


def _p(a: Piece, b: Piece, judge_model: str, cache: Cache) -> float:
    """The judge's p(superseded), cached. Generative judges give 1.0 or 0.0."""
    got = cache.get(a, b, judge_model)
    if got is not None:
        return float(got)
    if judge_model.startswith("nimble"):
        p = _nimble(f"EARLIER:\n{a.text[:PIECE_CHARS]}\n\nLATER:\n{b.text[:PIECE_CHARS]}")
    else:
        p = 1.0 if judge(a, b, judge_model)[0] else 0.0
    cache.put(a, b, judge_model, p)
    return p


def run_stage(messages: list, cut: int, stage: str, *, judge_model: str, min_p: float,
              cache: Cache, yielder, protect: set[str] | None = None,
              deadline: float | None = None) -> tuple[set[int] | None, dict]:
    """The deletions for `stage` ("fast", or "deep" = fast plus the spotter),
    or None when it could not finish (memory, strain, deadline), with info.
    Every finished verdict is in `cache` either way."""
    info = {"stage": stage, "judged": 0, "cached": 0, "deleted": 0}
    ps = pieces(messages, cut, protect)
    pairs = [(a, b) for a, b, _ in candidates(ps)]

    def judge_all(todo) -> set[int] | None:
        drop: set[int] = set()
        ok, why = can_load(judge_model)
        if not ok and any(cache.get(a, b, judge_model) is None for a, b in todo):
            info["stopped"] = f"judge: {why}"
            return None
        try:
            for a, b in todo:
                if cache.get(a, b, judge_model) is None:
                    if deadline and time.time() > deadline:
                        info["stopped"] = "out of time"
                        return None
                    act = yielder.step()
                    if act != "go":
                        info["stopped"] = f"machine strained ({act})"
                        return None
                    info["judged"] += 1
                else:
                    info["cached"] += 1
                if _p(a, b, judge_model, cache) >= min_p:
                    drop.update(id(x) for x in a.items)
            return drop
        finally:
            cache.save()

    drop = judge_all(pairs)
    if drop is None:
        return None, info
    if stage == "deep":
        remember_size(judge_model)
        unload(judge_model)                     # one big model in memory at a time
        ok, why = can_load(bg.LOCAL_MODEL)
        if not ok:
            info["stopped"] = f"spotter: {why}"
            return None, info
        stopped = []

        def should_stop():
            if deadline and time.time() > deadline:
                stopped.append("out of time")
            elif yielder.step() != "go":
                stopped.append("machine strained")
            return bool(stopped)
        proposed = spot(ps, bg.LOCAL_MODEL, should_stop=should_stop)
        remember_size(bg.LOCAL_MODEL)
        unload(bg.LOCAL_MODEL)
        if proposed is None:
            info["stopped"] = stopped[0] if stopped else "stopped"
            return None, info
        info["proposed"] = len(proposed)
        more = judge_all([(a, b) for a, b in proposed if id(a.items[0]) not in drop])
        if more is None:
            return None, info
        drop |= more
    remember_size(judge_model)
    unload(judge_model)
    info["deleted"] = len({id(a.items[0]) for a in ps if id(a.items[0]) in drop})
    return drop, info


def prework(messages: list, judge_model: str, cache: Cache, yielder, budget_s: float = 20.0,
            protect: set[str] | None = None) -> int:
    """While the developer works: judge a few new candidate pairs ahead of the
    next free point, within `budget_s`. Returns how many it judged."""
    cut = bg.choose_cut(messages)
    if not cut:
        return 0
    ps = pieces(messages, cut, protect)
    todo = [(a, b) for a, b, _ in candidates(ps) if cache.get(a, b, judge_model) is None]
    if not todo or not can_load(judge_model)[0]:
        return 0
    end, n = time.time() + budget_s, 0
    try:
        for a, b in todo:
            if time.time() > end or yielder.step() != "go":
                break
            _p(a, b, judge_model, cache)
            n += 1
    finally:
        cache.save()
    return n


# ---- stage 4, "lines": the 35B picks which lines of old output to keep ------
#
# Inside the compaction only (a free point), on old document views and
# tool-result excerpts longer than background.SHRINK_MIN_CHARS. The model lists
# line numbers to keep; the survivors stay byte-exact, a line carrying a detail
# Claude itself wrote is always kept, and a run of dropped lines becomes one
# "[... N lines left out]" marker. Measured on 7 real breaks
# (experiments/lossy_compare.py): 0.1-1.9% smaller compacted text, 0 details
# lost that Claude used later. Picks are cached per piece, so the same history
# gives the same bytes.

LINES_PROMPT = """You help shorten an OLD part of a coding session that an AI assistant will reread before continuing the work below.

CURRENT WORK (most recent part of the session):
{current}

OLD {kind}, numbered lines:
{numbered}

List the line numbers to KEEP: anything the assistant could need to continue the current work or answer a question about this session: names, paths, commands, numbers, errors, decisions, results, configuration values. Leave out boilerplate, repetition, progress output and details that only mattered at the time.
Answer with JSON only: {{"keep": [line numbers]}}"""


class LineCache(Cache):
    """Kept line numbers per piece text."""

    @staticmethod
    def text_key(kind: str, text: str) -> str:
        import hashlib
        return hashlib.sha256(f"lines\0{kind}\0{text}".encode("utf-8", "replace")).hexdigest()[:32]


def _pick(model: str, kind: str, lines: list[str], current: str, timeout: int = 600) -> set[int] | None:
    numbered = "\n".join(f"{i + 1}: {x[:400]}" for i, x in enumerate(lines))
    body = json.dumps({"model": model, "stream": False, "think": False, "format": "json",
                       "prompt": LINES_PROMPT.format(current=current[-3000:], kind=kind,
                                                     numbered=numbered[:24_000]),
                       "options": {"temperature": 0, "num_predict": 800, "num_ctx": 16384}}).encode()
    try:
        req = urllib.request.Request(bg.OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            keep = json.loads(json.loads(resp.read())["response"]).get("keep", [])
        return {int(x) - 1 for x in keep}
    except Exception:
        return None


def line_picker(current: str, keep_details: set[str], cache: LineCache, yielder,
                model: str | None = None, deadline: float | None = None):
    """(shrink(kind, text) -> text, state). `state["stopped"]` is set when it
    had to stop (strain, time); then pieces it had not picked stay whole."""
    from . import compress as cmp
    model = model or bg.LOCAL_MODEL
    state: dict = {"picked": 0, "cached": 0}

    def shrink(kind: str, text: str) -> str:
        lines = text.split("\n")
        if len(lines) < 4:
            return text
        key = cache.text_key(kind, text)
        keep = cache.data.get(key)
        if keep is None:
            if state.get("stopped"):
                return text
            if deadline and time.time() > deadline:
                state["stopped"] = "out of time"
                return text
            act = yielder.step()
            if act != "go":
                state["stopped"] = f"machine strained ({act})"
                return text
            got = _pick(model, kind, lines, current)
            if got is None:
                return text
            keep = sorted(got)
            cache.data[key] = keep
            state["picked"] += 1
        else:
            state["cached"] += 1
        keep = set(keep)
        out, gap = [], 0
        for i, x in enumerate(lines):
            if i in keep or i == 0 or (keep_details and cmp.details(x) & keep_details):
                if gap:
                    out.append(f"[... {gap} lines left out]")
                    gap = 0
                out.append(x)
            else:
                gap += 1
        if gap:
            out.append(f"[... {gap} lines left out]")
        res = "\n".join(out)
        return res if len(res) < len(text) else text
    return shrink, state
