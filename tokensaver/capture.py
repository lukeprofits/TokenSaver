"""Full-output capture for commands whose output Claude Code would discard.

Why this exists, measured rather than assumed: when a Bash command *fails*,
Claude Code hands the result to hooks already cut to ~10,000 characters, and
in testing that cut was pure head - a 720,062-character failing run reached
the hook as lines 1-417, with the actual error at the end nowhere in it.
`PostToolUseFailure` cannot rewrite the result, so the output cannot be
recovered after the fact. The only point where it still exists is before the
command runs, and `PreToolUse` can rewrite the command.

So: for an explicitly allowlisted command, wrap it so its full output is also
written to a file; on failure, inject the tail - the part Claude never sees -
as `additionalContext`, alongside the vault id for the rest.

This rewrites the user's shell commands, which is the most intrusive thing in
this codebase. Every guard below exists because a hostile case was found in
testing, and anything not clearly safe is passed through untouched.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
from pathlib import Path

from . import vault

# Commands worth wrapping: long output, and the part that matters is the end.
# Deliberately narrow. Add to it from telemetry, not from imagination.
DEFAULT_ALLOWLIST = (
    "pytest", "python -m pytest", "python3 -m pytest",
    "mvn", "./mvnw", "gradle", "./gradlew",
    "npm test", "npm run test", "yarn test", "pnpm test", "jest", "vitest",
    "go test", "cargo test", "bundle exec rspec", "rspec", "rake test",
    "tox", "nox", "ctest", "dotnet test",
)

# Shell constructs the wrapper provably breaks or cannot reason about.
#   <<      heredoc - wrapping it in a brace group is a syntax error (measured)
#   <(, >(  process substitution - nesting ours inside is unreviewable
#   &       backgrounding - the wrapper would outlive the command
#   exec    replaces the shell; redirect bookkeeping is meaningless after it
UNSAFE = ("<<", "<(", ">(", "exec ")
UNSAFE_RE = re.compile(r"(^|[^&])&\s*$|\bexec\b")

# Flags that make a test runner wait forever. `tee` would hold the pipe open
# and the turn would never return, which is far worse than losing the output.
INTERACTIVE_FLAGS = ("--watch", "--watchAll", "-w ", "--interactive", "--ui",
                     "--serve", "--debug-brk", "--inspect-brk")

LOG_DIR = vault.VAULT_DIR / "captures"


def _log_path(tool_use_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", tool_use_id or "unknown")[:64] or "unknown"
    return LOG_DIR / f"{safe}.log"


def is_allowed(command: str, allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST) -> bool:
    """True only when wrapping this command is both useful and provably safe."""
    if not command or not command.strip():
        return False
    if any(token in command for token in UNSAFE) or UNSAFE_RE.search(command):
        return False
    if command.rstrip().endswith("&"):
        return False
    padded = command + " "
    if any(flag in padded for flag in INTERACTIVE_FLAGS):
        return False
    # Must parse as a shell word list; if it doesn't, we don't understand it.
    try:
        shlex.split(command)
    except ValueError:
        return False
    # Match the allowlist against any segment of a chain, so `cd x && pytest`
    # qualifies on its `pytest` segment.
    segments = re.split(r"&&|\|\||;|\|", command)
    for seg in segments:
        stripped = seg.strip()
        if any(stripped == a or stripped.startswith(a + " ") for a in allowlist):
            return True
    return False


def wrap(command: str, tool_use_id: str) -> tuple[str, Path]:
    """Return the rewritten command and the log path it will write.

    Form: `{ CMD ; } > >(tee -a LOG) 2>&1`

    The brace group keeps CMD in the *current* shell, so `cd` still persists
    across turns, and applies the redirect to the whole chain rather than only
    its last command - a plain `cmd > >(tee ...)` captures nothing from
    `false && echo x`. Exit status is the group's, so failures stay failures.
    """
    log = _log_path(tool_use_id)
    log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return f"{{ {command.rstrip().rstrip(';')} ; }} > >(tee -a {shlex.quote(str(log))}) 2>&1", log


# Interpreters for which reading a script from stdin and reading it from a
# file argument are equivalent. Shells are deliberately absent: `bash file`
# runs a subshell so `cd` would stop persisting, and `. file` would let a
# stray `exit` kill Claude Code's shell.
HEREDOC_INTERPRETERS = ("python3", "python", "node", "ruby", "perl", "php")

SCRIPT_DIR = vault.VAULT_DIR / "scripts"
# Below this, the vault reference costs more than the body it replaces.
MIN_BODY_CHARS = 600

EXTENSIONS = {"python3": ".py", "python": ".py", "node": ".js",
              "ruby": ".rb", "perl": ".pl", "php": ".php"}


# A quoted heredoc anywhere in the command, with whatever precedes it on that
# line. Only single-quoted tags: an unquoted tag lets the shell expand the body,
# so moving those bytes to a file would change what actually runs.
EMBEDDED_RE = re.compile(
    r"(?P<pre>[^\n;&|]*?)<<\s*'(?P<tag>[A-Za-z_][A-Za-z0-9_]*)'[ \t]*\n"
    r"(?P<body>.*?)\n(?P=tag)(?=\s|$)",
    re.DOTALL,
)
INTERP_PRE = re.compile(r"(?P<interp>" + "|".join(HEREDOC_INTERPRETERS) +
                        r")(?P<flags>(?:\s+-[A-Za-z]+)*)\s+-\s*$")
# `cat > f`, `cat >> f`, `tee f`. The body ends up on disk, so the file itself
# becomes the record and the command only needs to copy it there.
FILEWRITE_PRE = re.compile(r"(?:cat|tee)\s*(?P<op>>>|>)\s*(?P<path>[^\s<>|&;]+)\s*$")


def rewrite_command(command: str) -> tuple[str, list[dict]] | None:
    """Replace large quoted heredoc bodies with vault references, in place.

    Handles heredocs anywhere in the command, including mid-chain, which a
    whole-command match (removed) could not: measured on real transcripts, the
    whole-command form matched 1.3% of commands while quoted heredoc bodies
    account for 53.8% of all Bash command characters.

    Two shapes are rewritten, both losslessly:
      `python3 - <<'PY' body PY`   -> `python3 <vault>/<sha>.py`
      `cat > f <<'TAG' body TAG`   -> `cat <vault>/<sha>.txt > f`

    Everything else is left exactly as written. Returns None if nothing
    changed, so callers can pass the original through untouched.
    """
    out: list[str] = []
    notes: list[dict] = []
    pos = 0
    for m in EMBEDDED_RE.finditer(command):
        body = m.group("body")
        pre = m.group("pre")
        if len(body) < MIN_BODY_CHARS:
            continue

        interp_m = INTERP_PRE.search(pre)
        file_m = None if interp_m else FILEWRITE_PRE.search(pre)
        if not interp_m and not file_m:
            continue

        suffix = EXTENSIONS.get(interp_m.group("interp"), ".txt") if interp_m else ".txt"
        try:
            path = _store_script(body, suffix)
        except OSError:
            continue

        if interp_m:
            head = pre[:interp_m.start()]
            flags = interp_m.group("flags") or ""
            replacement = f"{head}{interp_m.group('interp')}{flags} {shlex.quote(str(path))}"
        else:
            head = pre[:file_m.start()]
            replacement = (f"{head}cat {shlex.quote(str(path))} "
                           f"{file_m.group('op')} {file_m.group('path')}")

        out.append(command[pos:m.start()])
        out.append(replacement)
        pos = m.end()
        notes.append({"path": str(path), "body_chars": len(body),
                      "kind": "script" if interp_m else "filewrite"})

    if not notes:
        return None
    out.append(command[pos:])
    return "".join(out), notes


def _store_script(body: str, suffix: str) -> Path:
    """Content-addressed, byte-exact, idempotent.

    The stored body gets the trailing newline back. A heredoc delivers its
    body with the newline that precedes the terminator, but the regex captures
    up to that newline and excludes it - so writing the stored bytes straight
    out produced a file missing its final newline. Harmless for a Python
    script, silent corruption for `cat > Foo.java`.
    """
    if not body.endswith("\n"):
        body += "\n"
    digest = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:16]
    path = SCRIPT_DIR / f"{digest}{suffix}"
    SCRIPT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(path)
    return path


def read_capture(tool_use_id: str) -> str | None:
    path = _log_path(tool_use_id)
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def tail_context(full: str, event_id: int | None, head_chars: int,
                 tail_chars: int = 4000) -> str | None:
    """The part of the output Claude did not get, as injectable context.

    Returns None when Claude already saw everything - no point spending tokens
    telling it what it has.
    """
    if len(full) <= head_chars:
        return None
    tail = full[-tail_chars:]
    hidden = len(full) - head_chars
    ref = f" Full output is event {event_id} (`tokensaver read {event_id}`)." if event_id else ""
    return (f"[TokenSaver] This command's output was {len(full):,} characters; Claude Code "
            f"showed only the first ~{head_chars:,}, so roughly {hidden:,} characters "
            f"including the end of the output were not visible.{ref}\n"
            f"Last {len(tail):,} characters:\n{tail}")
