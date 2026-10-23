"""Forward passes: lossless rewrites applied to a block before it is first sent.

These run over structured `Block`s (tool name, input, what came before) with one shared
`State`, in a fixed order (`FORWARD`):

1. `plan_echo`: an ExitPlanMode result that repeats its own plan verbatim
   keeps one line naming the input that holds it.
2. `heredocs`: the byte-exact command rewrite from `capture`.
3. `json_compact`: lossless JSON re-encoding.
4. `format`: cosmetic bytes in command output.
5. `repeat_runs`: a run of identical consecutive output lines, collapsed with
   its count (lossless; replaced the lossy `dedup_noise` 2026-10-22).
A fixed path alias plan (`hindsight_plan`/`plan_aliases`) can follow; the
proxy decides it at a compaction on real counts.

Every pass depends only on the block and what came before it, so the bytes of
a block never change once sent (the proxy's ledger replays them anyway, and
Opus 5.5 binds Claude's thinking to them). The user's own words, signed
thinking and file content are never edited.

Removed 2026-10-22: the pointer passes (`stale_views`, `superseded_writes`,
the pointer half of `plan_echo`) and swap mode. Opus 5.5's safeguards refused
4 of 4 requests about content behind a pointer (2026-10-13), and in forward
mode their pointers were computed on every request and thrown away. Also the
"auto" path aliases (measured -2.4 to -3.3% in real tokens).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from . import capture


@dataclass
class Block:
    """One content block, with the context a pass needs to judge it."""
    index: int                 # position in the conversation
    request: int               # index of the billed request it precedes
    role: str
    type: str                  # text | thinking | tool_use | tool_result
    text: str                  # what is resent; passes shrink this
    tool: str | None = None    # for tool_result, the paired tool_use's name
    tool_use_id: str | None = None
    input: dict | None = None  # for tool_result, the paired tool_use's input
    cwd: str = ""
    is_error: bool = False
    original: str = ""
    vaulted: bool = False      # some bytes moved out, byte-exact, to the vault
    removed_by: dict[str, int] = field(default_factory=dict)
    # Write-back (proxy): where the block lives in `messages`, whether its
    # bytes can be replaced at all, and the rewritten tool_use input.
    ref: tuple[int, int | None] | None = None
    fixed: bool = False
    out_input: dict | None = None

    def set_text(self, new: str, pass_name: str) -> None:
        saved = len(self.text) - len(new)
        if saved > 0:
            self.removed_by[pass_name] = self.removed_by.get(pass_name, 0) + saved
            self.text = new


@dataclass
class State:
    """Session-wide state shared by every pass. Never reset per message."""
    seen: set[str] = field(default_factory=set)
    known: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)   # path -> name


Pass = Callable[[list[Block], State], list[Block]]


# ---- loading ---------------------------------------------------------------

def _result_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join((x.get("text") or "") for x in content
                         if isinstance(x, dict) and isinstance(x.get("text"), str))
    return str(content)


def _tool_use_text(inp: dict) -> str:
    # The raw command, not JSON: serialised, its newlines are escaped and the
    # heredoc rewrite can never match.
    command = inp.get("command")
    if isinstance(command, str):
        rest = {k: v for k, v in inp.items() if k != "command"}
        return command + ("\n" + json.dumps(rest) if rest else "")
    return json.dumps(inp)


def from_messages(messages, cwd: str = "", request: int = 0,
                  start: int = 0) -> list[Block]:
    """Blocks from API-shaped messages. Transcripts carry the same shape."""
    blocks: list[Block] = []
    uses: dict[str, Block] = {}
    for mi, message in enumerate(messages or []):
        if not isinstance(message, dict):
            continue
        role = message.get("role") or ""
        content = message.get("content")
        # A mid-conversation role:"system" entry carries a plain string.
        items = content if isinstance(content, list) else [content]
        for ii, item in enumerate(items):
            if isinstance(item, str):
                item = {"type": "text", "text": item}
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            b = Block(index=start + len(blocks), request=request, role=role,
                      type=kind or "", text="", cwd=cwd,
                      ref=(mi, ii if isinstance(content, list) else None))
            if kind == "text":
                b.text = item.get("text") or ""
                # The user's own words are never edited (2026-10-20: dedup
                # removed blank lines and a `---` from a pasted page). Claude
                # Code's injected reminders are not the user's words.
                if role == "user" and not b.text.lstrip().startswith("<system-reminder>"):
                    b.fixed = True
            elif kind == "thinking":
                b.text = item.get("thinking") or ""
                b.fixed = True             # signed: one changed byte fails the request
            elif kind == "tool_use":
                inp = item.get("input") if isinstance(item.get("input"), dict) else {}
                b.tool, b.tool_use_id, b.input = item.get("name"), item.get("id"), inp
                b.text = _tool_use_text(inp)
                # Passes change a tool_use only through `out_input`, never
                # through its rendered text, so the input dict is always
                # replaceable. Forward passes touch only a `command` string.
                uses[b.tool_use_id] = b
            elif kind == "tool_result":
                b.tool_use_id = item.get("tool_use_id")
                use = uses.get(b.tool_use_id)
                if use:
                    b.tool, b.input = use.tool, use.input
                b.is_error = bool(item.get("is_error"))
                b.text = _result_text(item.get("content"))
                # Several text parts were joined with "\n" to make `text`;
                # that cannot be split back, so such a result stays as sent.
                c = item.get("content")
                b.fixed = not (isinstance(c, str) or (isinstance(c, list) and sum(
                    1 for x in c if isinstance(x, dict) and x.get("type") == "text") == 1))
            else:
                continue
            if b.text:
                b.original = b.text
                blocks.append(b)
    return blocks


def from_transcript(path) -> list[Block]:
    """Blocks from a Claude Code session transcript, in conversation order.

    `request` counts the billed requests that came before the block, the same
    way `transcripts` counts them. tool_use/tool_result pairing spans records, so
    pairs are joined after loading.
    """
    blocks: list[Block] = []
    request = 0
    seen_ids: set = set()
    try:
        handle = open(path, errors="replace")
    except OSError:
        return blocks
    with handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            # One request per message id: Claude Code writes a record per
            # content block, each repeating the usage (see transcripts.py).
            if message.get("usage") and message.get("id") not in seen_ids:
                if message.get("id"):
                    seen_ids.add(message["id"])
                request += 1
            if not isinstance(message.get("content"), list):
                continue
            blocks += from_messages([message], cwd=record.get("cwd") or "",
                                    request=request, start=len(blocks))
    uses = {b.tool_use_id: b for b in blocks if b.type == "tool_use"}
    for b in blocks:
        if b.type == "tool_result" and b.tool is None:
            use = uses.get(b.tool_use_id)
            if use:
                b.tool, b.input = use.tool, use.input
    return blocks


# ---- file identity ---------------------------------------------------------

INF = float("inf")


def _path(name: str) -> str:
    # One plain path, optionally quoted. Globs, substitutions and variables
    # are refused: their file cannot be known from the command text.
    return rf"""['"]?(?P<{name}>[^\s'"|;&<>()$`*?]+)['"]?"""


# Only a single, simple file view counts. Anything else - a pipe, two files,
# an unknown flag - is a *result*, not a view, and is left alone.
_BASH_VIEW = re.compile(rf"""^\s*(?:cd\s+(?P<cd>[^\s;&|]+)\s*&&\s*)?(?:
      (?:cat|nl)\s+{_path('p1')}
    | sed\s+-n\s+['"]?(?P<a>\d+),(?P<b>\d+)p['"]?\s+{_path('p2')}
    | head\s+(?:-n\s*|-)(?P<h>\d+)\s+{_path('p3')}
    )\s*$""", re.X)


def _norm(path: str, cwd: str) -> str:
    path = path.strip("'\"")
    if not os.path.isabs(path) and cwd:
        path = os.path.join(cwd, path)
    return os.path.normpath(path)


def view_of(b: Block) -> tuple[str, float, float] | None:
    """(path, first line, last line) if this tool_result shows a file's text."""
    if b.type != "tool_result" or b.is_error or not b.input:
        return None
    if b.tool == "Read":
        path = b.input.get("file_path")
        if not isinstance(path, str):
            return None
        offset = b.input.get("offset") or 1
        limit = b.input.get("limit")
        try:
            first = float(offset)
            last = first + float(limit) - 1 if limit else INF
        except (TypeError, ValueError):
            return None
        return _norm(path, b.cwd), first, last
    if b.tool == "Bash":
        command = b.input.get("command")
        if not isinstance(command, str):
            return None
        m = _BASH_VIEW.match(command)
        if not m:
            return None
        cwd = _norm(m.group("cd"), b.cwd) if m.group("cd") else b.cwd
        if m.group("p1"):
            return _norm(m.group("p1"), cwd), 1, INF
        if m.group("p2"):
            return _norm(m.group("p2"), cwd), float(m.group("a")), float(m.group("b"))
        return _norm(m.group("p3"), cwd), 1, float(m.group("h"))
    return None


# ---- passes ----------------------------------------------------------------

def plan_echo(blocks: list[Block], state: State) -> list[Block]:
    """Plan mode returns the plan twice: ExitPlanMode's `plan` input, and its
    result echoing it verbatim (measured: 9 of 10 approvals). The result keeps
    one line naming the input; the copy is exact, so nothing is lost. The
    echo comes after its call, so the change is causal at once."""
    plans = [b for b in blocks if b.type == "tool_use" and b.tool == "ExitPlanMode"
             and isinstance((b.input or {}).get("plan"), str) and b.input["plan"].strip()]
    for use in plans:
        plan = use.input["plan"]
        for b in blocks:
            if (b.type == "tool_result" and b.tool_use_id == use.tool_use_id
                    and not b.fixed and plan in b.text):
                b.set_text(b.text.replace(
                    plan, f"[TokenSaver: plan text identical to the ExitPlanMode "
                          f"input at block {use.index}]"), "plan_echo")
                b.vaulted = True
    return blocks


def heredocs(blocks: list[Block], state: State) -> list[Block]:
    """Byte-exact heredoc vaulting, as the live PreToolUse hook does it."""
    for b in blocks:
        if b.type == "tool_use" and b.tool == "Bash" and not b.fixed:
            command = (b.input or {}).get("command")
            if isinstance(command, str) and command in b.text:
                rewritten = capture.rewrite_command(command)
                if rewritten:
                    b.out_input = dict(b.input, command=rewritten[0])
                    b.set_text(_tool_use_text(b.out_input), "heredocs")
                    b.vaulted = True
    return blocks


RUN_MIN_CHARS = 12       # a shorter line saves less than its marker costs
RUN_MARK = "[the line above repeats {n} more times]"
_RUN_MARK = re.compile(r"\[the line above repeats (\d+) more times\]")
_DIFF_CMD = re.compile(r"\bgit\s+(?:diff|show)\b|\bgit\s+log\b.*\s-p\b|(?:^|[;&|]\s*)diff\s")


def repeat_runs(blocks: list[Block], state: State) -> list[Block]:
    """A run of 3+ identical consecutive lines in a tool's output becomes the
    line once plus "[the line above repeats N more times]". Lossless: the count
    is kept and the lines are exact, so the original can be rebuilt (and is, by
    `verified`). Never on file content or diffs (an Edit copies from those), the
    user's words, Claude's own text, or thinking.

    Replaced `dedup_noise` (2026-10-22). Measured on this machine's history,
    91% of what that pass removed were repeats that came back between other
    lines: closing braces and calls in code views, diff context, a repeated
    field in every record of YAML/JSON/tables. Deleting those changes what the
    code or the data says; only true runs (log spam) were redundant."""
    for b in blocks:
        if b.type != "tool_result" or b.fixed or _file_content(b):
            continue
        command = (b.input or {}).get("command")
        if isinstance(command, str) and _DIFF_CMD.search(command):
            continue
        lines, out, i, changed = b.text.split("\n"), [], 0, False
        while i < len(lines):
            j = i
            while j + 1 < len(lines) and lines[j + 1] == lines[i]:
                j += 1
            n = j - i
            mark = RUN_MARK.format(n=n)
            if n >= 2 and len(lines[i].strip()) >= RUN_MIN_CHARS and n * (len(lines[i]) + 1) > len(mark) + 1:
                out += [lines[i], mark]
                changed = True
            else:
                out += lines[i:j + 1]
            i = j + 1
        if changed:
            b.set_text("\n".join(out), "repeat_runs")
    return blocks


def expand_runs(text: str) -> str:
    """Undo `repeat_runs`."""
    out: list[str] = []
    for line in text.split("\n"):
        m = _RUN_MARK.fullmatch(line)
        if m and out:
            out += [out[-1]] * int(m.group(1))
        else:
            out.append(line)
    return "\n".join(out)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_TRAIL = re.compile(r"[ \t]+$", re.M)


def format_output(blocks: list[Block], state: State) -> list[Block]:
    """ANSI escapes and trailing blanks in command output. Never in file
    views: Edit's old_string must match the file's real whitespace."""
    for b in blocks:
        if (b.type == "tool_result" and b.tool == "Bash"
                and not b.fixed and not view_of(b)):
            b.set_text(_TRAIL.sub("", _ANSI.sub("", b.text)), "format")
    return blocks


# ---- JSON -------------------------------------------------------------------
#
# Lossless re-encoding of tool output that is entirely JSON: pretty-printing
# goes, and an array of three or more objects with the same keys becomes
# {"_cols": [...], "_rows": [[...], ...]}, so the keys appear once. Every value
# is kept exactly: numbers keep their original text (1e5 stays 1e5), and an
# object with duplicate keys, which json.loads would silently collapse, makes
# the pass back off. Never applied to file views: an Edit must match a file's
# real bytes. Measured on this machine: JSON is ~0% of content, but API and
# kubectl-style output elsewhere is mostly this.

class _Num(str):
    """A JSON number, kept as the text it was written as."""


class _DupKey(ValueError):
    pass


def _pairs(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DupKey(keys)
    return dict(pairs)


def _dump(v) -> str:
    if isinstance(v, _Num):
        return str(v)
    if isinstance(v, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _dump(x) for k, x in v.items()) + "}"
    if isinstance(v, list):
        if (len(v) >= 3 and all(isinstance(x, dict) for x in v) and len(v[0]) >= 2
                and all(list(x) == list(v[0]) for x in v)):
            cols = list(v[0])
            return ('{"_cols":' + _dump(cols) + ',"_rows":['
                    + ",".join("[" + ",".join(_dump(x[c]) for c in cols) + "]" for x in v) + "]}")
        return "[" + ",".join(_dump(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


JSON_NOTE = '[TokenSaver: JSON compacted; same-shaped object arrays shown as {"_cols","_rows"}]\n'


def compact_json(text: str) -> str | None:
    """The compact form of `text`, or None when it is not all JSON or would not shrink."""
    t = text.strip()
    if not t or t[0] not in "{[":
        return None
    try:
        obj = json.loads(t, parse_float=_Num, parse_int=_Num, object_pairs_hook=_pairs)
    except (ValueError, RecursionError):
        return None
    out = _dump(obj)
    if '"_cols"' in out and '"_cols"' not in t:
        out = JSON_NOTE + out
    return out if len(out) < len(text) else None


def json_compact(blocks: list[Block], state: State) -> list[Block]:
    """Compact JSON tool output, losslessly (see `compact_json`)."""
    for b in blocks:
        if b.type != "tool_result" or b.fixed or _file_content(b):
            continue
        out = compact_json(b.text)
        if out is not None:
            b.set_text(out, "json_compact")
    return blocks


# Directory paths at least two segments deep: absolute, ~-relative, or
# relative (`src/shared/components/X`). Relative paths were 73% of the long
# repeated tokens left after absolute aliasing (measured). The lookbehind keeps
# a match from starting inside a URL, another path or an alias.
_PATHS = re.compile(r"(?<![\w.\-/~$%:])(?:~|/[\w.\-]+|[\w.\-]+)(?:/[\w.\-]+)+")
_UUIDISH = re.compile(r"^[0-9a-f\-]{16,}$")
ALIAS_MIN = 30          # measured flat (1.40-1.46%) from 16 to 30; fewer aliases wins
ALIAS_GAIN = 20         # each later use must save at least this many chars


def _alias_token(name: str) -> str:
    # `%name%`, not `$name`: a shell expands `$$` to its PID, so a copied
    # `$$$repo$$$/src` would silently become a wrong path. `%` is literal
    # mid-word in bash and zsh, so a copied alias fails loudly instead.
    return f"%{name}%"


def _dir_prefixes(path: str):
    """Directory prefixes of `path`, longest first. A last segment with a dot
    is taken for a file name and stays visible."""
    parts = path.split("/")
    end = len(parts) - 1 if "." in parts[-1] else len(parts)
    for n in range(end, 2 if parts[0] == "" else 1, -1):
        yield "/".join(parts[:n])


ALIAS_NAME_MAX = 24


def _alias_name(path: str, aliases: dict[str, str]) -> str:
    """The folder's own name, readable. On a clash, prefixed with the first
    folder where the two paths part (`test-order_row` beside lib's),
    because a bare `order_row2` invites mixing the two up."""
    segs = [s for s in path.split("/") if s and s != "~"]
    base = next((s for s in reversed(segs) if not _UUIDISH.match(s)), "dir")
    if base.startswith("-"):
        base = "session"   # Claude Code's encoded project dir, e.g. -Users-x-Projects-y
    base = (re.sub(r"[^\w.\-]", "_", base) or "dir")[:ALIAS_NAME_MAX]
    taken = {n: p for p, n in aliases.items()}
    name = base
    if name in taken:
        other = [s for s in taken[name].split("/") if s and s != "~"]
        diverge = next((a for a, b in zip(segs, other) if a != b), None)
        if diverge and not _UUIDISH.match(diverge):
            name = f"{diverge[:ALIAS_NAME_MAX]}-{base}"
    n = 2
    while name in taken:
        name, n = f"{base}{n}", n + 1
    return name


_SUB_CACHE: dict[tuple[int, int], tuple[dict, re.Pattern]] = {}


def _substitute(text: str, aliases: dict[str, str]) -> str:
    if not aliases:
        return text
    # Aliases only ever grow, so (dict, size) names one alias set. Rebuilding
    # the pattern per block was 2/3 of a request's rewrite time (measured:
    # 1,522 compiles for 87 distinct sets on a 1,100-message session). The
    # lookbehind matches `_PATHS`, so nothing is replaced inside a URL.
    # The cache holds the dict itself: an id is reused once its object is
    # freed, and a stale pattern for another session's aliases raised
    # KeyError in the determinism gate. Held, the id cannot be reused.
    key = (id(aliases), len(aliases))
    hit = _SUB_CACHE.get(key)
    if hit is not None and hit[0] is aliases:
        pat = hit[1]
    else:
        alt = "|".join(re.escape(p) for p in sorted(aliases, key=len, reverse=True))
        pat = re.compile(rf"(?<![\w.\-/~$%:])(?:{alt})(?![\w.\-])")
        if len(_SUB_CACHE) > 256:
            _SUB_CACHE.clear()
        _SUB_CACHE[key] = (aliases, pat)
    return pat.sub(lambda m: _alias_token(aliases[m.group(0)]), text)


def _file_content(b: Block) -> bool:
    """Blocks whose bytes must match a file exactly. An Edit copies its
    old_string from these, so an alias inside one would make the Edit miss."""
    if b.type == "tool_use" and b.tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        return True
    if b.type == "tool_result" and (b.tool == "Read" or view_of(b)):
        return True
    return False


ALIAS_PLAN_MIN_USES = 3


def hindsight_plan(messages: list, min_uses: int = ALIAS_PLAN_MIN_USES) -> dict[str, str]:
    """Aliases chosen knowing the whole history: the most specific directory of
    each path that appears at least `min_uses` times and saves ALIAS_GAIN chars
    per use. For a free rebuild point only (compaction, an expired cache), where
    the history is re-sent from scratch anyway. Whether the plan is used at all
    is decided by counting real tokens with and without it."""
    blocks = from_messages(messages)
    counts: dict[str, int] = {}
    for b in blocks:
        if b.type == "thinking" or b.fixed or _file_content(b):
            continue
        for m in _PATHS.finditer(b.text):
            for p in _dir_prefixes(m.group(0)):
                counts[p] = counts.get(p, 0) + 1
    plan: dict[str, str] = {}
    for p in sorted(counts, key=lambda x: (-len(x), x)):
        if counts[p] < min_uses or len(p) < ALIAS_MIN:
            continue
        if any(p.startswith(q + "/") or q.startswith(p + "/") for q in plan):
            continue                      # keep only the most specific of a nested chain
        name = _alias_name(p, plan)
        if len(p) - len(_alias_token(name)) >= ALIAS_GAIN:
            plan[p] = name
    return plan


def plan_aliases(blocks: list[Block], plan: dict[str, str]) -> list[Block]:
    """Apply a fixed alias plan: every eligible block uses the aliases, and one
    definition line goes at the head of the first block that can carry text.
    Deterministic for a given plan and history, so a frozen plan reproduces the
    same bytes on every later request."""
    if not plan:
        return blocks
    defs_done = False
    for b in blocks:
        if b.type == "thinking" or b.fixed or _file_content(b):
            continue
        if b.type == "tool_use":
            inp = b.out_input or b.input or {}
            if not isinstance(inp.get("command"), str):
                continue
            sub = _substitute(inp["command"], plan)
            if sub != inp["command"]:
                b.out_input = dict(inp, command=sub)
                b.text = _tool_use_text(b.out_input)
            continue
        out = _substitute(b.text, plan)
        if not defs_done:
            defs = ", ".join(f"{_alias_token(n)} = {p}" for p, n in sorted(plan.items(), key=lambda x: x[1]))
            out = f"[TokenSaver path aliases, history only; always write the full path: {defs}]\n" + out
            defs_done = True
        if out != b.text:
            b.removed_by["path_aliases"] = b.removed_by.get("path_aliases", 0) + len(b.text) - len(out)
            b.text = out
    return blocks


# Order matters where two passes can touch one block: json_compact first (a
# whole-JSON output becomes one line, which nothing after it touches); format
# before repeat_runs, so lines that differ only in ANSI codes or trailing
# blanks count as one run. Measured 2026-10-22: every order saved the same
# within 84 chars of 52M on this history; the order is for correctness.
FORWARD: tuple[tuple[str, Pass], ...] = (
    ("plan_echo", plan_echo),
    ("heredocs", heredocs),
    ("json_compact", json_compact),
    ("format", format_output),
    ("repeat_runs", repeat_runs),
)


def write_back(messages: list, blocks: list[Block]) -> list:
    """`messages` with each changed block's new bytes in place.

    A copy: the caller's list is not touched. Only text, a single-text
    tool_result and a Bash command are ever written; every other key of every
    item, `cache_control` included, is carried over as is."""
    out = [dict(m) if isinstance(m, dict) else m for m in messages]
    copied: set[int] = set()
    for b in blocks:
        if b.fixed or b.ref is None or b.text == b.original and b.out_input is None:
            continue
        mi, ii = b.ref
        msg = out[mi]
        if ii is None:
            msg["content"] = b.text
            continue
        if mi not in copied:
            msg["content"] = list(msg["content"])
            copied.add(mi)
        item = dict(msg["content"][ii])
        if b.type == "text":
            item["text"] = b.text
        elif b.type == "tool_use" and b.out_input is not None:
            item["input"] = b.out_input
        elif b.type == "tool_result":
            c = item.get("content")
            if isinstance(c, str):
                item["content"] = b.text
            else:
                item["content"] = [dict(x, text=b.text) if isinstance(x, dict)
                                   and x.get("type") == "text" else x for x in c]
        else:
            continue
        msg["content"][ii] = item
    return out


def _expand_cols(v):
    if isinstance(v, dict):
        if list(v) == ["_cols", "_rows"] and isinstance(v["_cols"], list) and isinstance(v["_rows"], list):
            return [{c: _expand_cols(x) for c, x in zip(v["_cols"], row)} for row in v["_rows"]]
        return {k: _expand_cols(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand_cols(x) for x in v]
    return v


def verified(b: Block) -> bool:
    """A forward pass's change checked at runtime, after the fact, so a bug in
    a pass reverts one block instead of reaching Claude:
    - json_compact: the compact form, with `_cols/_rows` expanded, parses to
      exactly the original JSON (numbers compared as written);
    - format / repeat_runs: expanding the runs gives back exactly the original
      with only format's ANSI codes and trailing blanks removed;
    - heredocs: the command changed only through the hook's own rewrite."""
    passes_used = set(b.removed_by)
    try:
        if "json_compact" in passes_used:
            out = b.text[len(JSON_NOTE):] if b.text.startswith(JSON_NOTE) else b.text
            load = lambda t: json.loads(t.strip(), parse_float=_Num, parse_int=_Num, object_pairs_hook=_pairs)
            return _expand_cols(load(out)) == load(b.original) or load(out) == load(b.original)
        if passes_used & {"repeat_runs", "format"}:
            base = _TRAIL.sub("", _ANSI.sub("", b.original)) if "format" in passes_used else b.original
            return expand_runs(b.text) == base
        if "heredocs" in passes_used:
            return b.vaulted and b.out_input is not None
    except Exception:
        return False
    return True


def rewrite_blocks(messages: list, *, exclude: tuple[str, ...] = (), aliases="off") -> list[Block]:
    """Blocks for `messages` with the forward passes. `exclude` leaves named
    passes out (the token audit prices one pass at a time; the proxy switches
    off a pass measured as a cost). `aliases`: "off", or a fixed plan from
    `hindsight_plan`, applied last ("path_aliases" in `exclude` also turns it
    off). A change that fails its runtime check (`verified`) is reverted."""
    blocks = run(from_messages(messages), tuple(p for p in FORWARD if p[0] not in set(exclude)))
    for b in blocks:
        if b.text != b.original and not verified(b):
            b.text, b.out_input, b.vaulted = b.original, None, False
            b.removed_by = {"reverted": 0}
    if isinstance(aliases, dict) and "path_aliases" not in exclude:
        blocks = plan_aliases(blocks, aliases)
    return blocks


def rewrite_messages(messages: list, *, exclude: tuple[str, ...] = (), aliases="off") -> list:
    """An API request's `messages` with the forward passes (see `rewrite_blocks`)."""
    return write_back(messages, rewrite_blocks(messages, exclude=exclude, aliases=aliases))


def run(blocks: list[Block], passes=FORWARD, state: State | None = None) -> list[Block]:
    state = state or State()
    for _, fn in passes:
        blocks = fn(blocks, state)
    return blocks
