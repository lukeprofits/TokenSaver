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


# Removed 2026-11-04: heredoc vaulting (`rewrite_command`), which replaced an inline
# script with `python3 ~/.tokensaver/scripts/<sha>.py`. Claude imitated that short form
# (see hook.py). Scripts already written stay in ~/.tokensaver/scripts.


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
