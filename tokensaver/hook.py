"""Claude Code hook entry point.

Observes on PostToolUse / PostToolUseFailure / PreCompact / SessionStart, and
makes exactly one change on PreToolUse: a large quoted heredoc body is moved
into the vault and the command is shortened to reference it. Verified
byte-identical - same stdout, same exit code, same resulting files. Optionally
(env TOKENSAVER_CAPTURE_OUTPUT=1) it also wraps allowlisted test commands to
record output the harness would truncate.

Contract with the harness: this runs inside Claude Code's critical path, so it
exits 0 no matter what happens. A crash, a timeout, a corrupt vault, a schema
change in the payload - all of them must leave the session exactly as it would
have been without the hook. `TOKENSAVER_DISABLE=1` short-circuits everything.

What it captures is the hook payload itself. That is a documented contract,
unlike the transcript JSONL, whose format is internal and changes between
versions.
"""

from __future__ import annotations

import json
import os
import re
import sys

from . import capture, names, vault

# Payload keys that carry the tool's own output, in the order we prefer them.
# `error` is where PostToolUseFailure puts it - measured, not documented, so
# the whole list is tried and `meta` keeps whatever is left over.
RESULT_KEYS = ("tool_response", "tool_result", "toolResponse", "error",
               "output", "stdout", "stderr")
MAX_SUMMARY = 300


def _text(value) -> str:
    """Flatten a tool response of unknown shape into text, losing nothing."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value)
    if isinstance(value, dict):
        for key in ("content", "text", "stdout", "output"):
            if key in value:
                body = _text(value[key])
                extra = {k: v for k, v in value.items()
                         if k not in (key, "content", "text", "stdout", "output")}
                return f"{body}\n{json.dumps(extra, default=str)}" if extra else body
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _looks_truncated(text: str, payload: dict) -> bool:
    """Did Claude Code already shorten this before handing it to us?

    This is the question that decides whether the vault can recover what the
    harness discards. We record the signal rather than assert an answer.
    """
    if payload.get("truncated") or payload.get("is_truncated"):
        return True
    markers = ("[truncated", "... (truncated", "output truncated",
               "tail of the output", "showing first")
    tail = text[-2000:].lower()
    return any(m in tail for m in markers)


def _pre_tool_use(payload: dict) -> dict:
    """Shorten the command where that is provably lossless.

    Anything not provably safe returns an empty decision, which leaves the
    command exactly as Claude wrote it.
    """
    if payload.get("tool_name") not in ("Bash", "BashOutput"):
        return {}
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str) or not command.strip():
        return {}

    session = payload.get("session_id")
    cwd = payload.get("cwd")
    updated = dict(tool_input)
    changed = False

    # 1. Shrink the command itself by vaulting large heredoc bodies. This is
    #    the only transformation here with no downside at all: the short form
    #    is what enters history, so there is no cache rebuild to pay back, and
    #    the vaulted bytes are byte-identical to what the heredoc delivered.
    #    Except in auto mode: the classifier sees the rewritten command, reads
    #    `python3 ~/.tokensaver/scripts/<sha>.py` as running code from outside
    #    the project, and denies it. Measured 2026-10-12: repeated "Code from
    #    External" denials of our own edits. A blocked command is a worse
    #    experience than a longer one, so auto mode keeps the original.
    rewritten = (None if payload.get("permission_mode") == "auto"
                 else capture.rewrite_command(command))
    if rewritten:
        command, notes = rewritten
        updated["command"] = command
        changed = True
        vault.record("command_vaulted", content=command, tool="Bash",
                     session_id=session, cwd=cwd,
                     summary=f"vaulted {len(notes)} heredoc body(s)",
                     meta={"scripts": notes, "tool_use_id": payload.get("tool_use_id")})

    # 2. Separately, for allowlisted test/build commands, also capture full
    #    output - Claude Code cuts a failing result to ~10k before any hook
    #    sees it, head-only, so the tail is otherwise unrecoverable.
    #    This one restructures the command with a brace group and process
    #    substitution, which is materially more invasive than vaulting a
    #    heredoc body, so it stays opt-in via TOKENSAVER_CAPTURE_OUTPUT=1.
    if names.env("CAPTURE_OUTPUT") == "1" and capture.is_allowed(command):
        wrapped, log = capture.wrap(command, payload.get("tool_use_id", ""))
        updated["command"] = wrapped
        changed = True
        vault.record("capture_armed", content=command, tool="Bash",
                     session_id=session, cwd=cwd,
                     summary=f"armed capture: {command[:200]}",
                     meta={"log": str(log), "tool_use_id": payload.get("tool_use_id")})

    if not changed:
        return {}
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "updatedInput": updated}}


def handle(payload: dict) -> dict:
    event = payload.get("hook_event_name") or payload.get("hookEventName") or "unknown"
    if event == "PreToolUse":
        return _pre_tool_use(payload)
    tool = payload.get("tool_name") or payload.get("toolName")
    session = payload.get("session_id") or payload.get("sessionId")
    cwd = payload.get("cwd")

    # Concatenate every result-bearing key rather than taking the first: a
    # failure can carry both `error` and a partial `tool_response`.
    parts = [(_text(payload[k]), k) for k in RESULT_KEYS if payload.get(k) not in (None, "")]
    raw = "\n".join(text for text, _ in parts)
    result_keys_seen = [k for _, k in parts]
    tool_input = payload.get("tool_input") or payload.get("toolInput") or {}

    kind = {"PostToolUse": "tool_result",
            "PostToolUseFailure": "tool_failure",
            "PreCompact": "compact_boundary",
            "SessionStart": "session_start",
            "UserPromptSubmit": "user_prompt",
            "Stop": "turn_end"}.get(event, event.lower())

    if kind == "user_prompt":
        raw = payload.get("prompt") or raw

    descriptor = ""
    if isinstance(tool_input, dict):
        descriptor = (tool_input.get("command") or tool_input.get("file_path")
                      or tool_input.get("pattern") or tool_input.get("path") or "")
    summary = f"{tool or event} {descriptor}".strip()[:MAX_SUMMARY]

    exit_code = None
    for key in ("exit_code", "exitCode", "returncode"):
        if isinstance(payload.get(key), int):
            exit_code = payload[key]
            break
    if exit_code is None:
        match = re.match(r"\s*Exit code (\d+)", raw)
        if match:
            exit_code = int(match.group(1))

    # The full payload minus the bulky result, so nothing contextual is lost.
    meta = {k: v for k, v in payload.items() if k not in RESULT_KEYS}

    meta["_result_keys"] = result_keys_seen
    shown_chars = len(raw)
    captured = capture.read_capture(payload.get("tool_use_id", "")) if tool == "Bash" else None
    if captured and len(captured) > shown_chars:
        # The file has what the harness cut. Keep the complete bytes.
        raw = captured
        meta["_captured_full"] = True
        meta["_chars_shown_to_claude"] = shown_chars
    event_id = vault.record(
        kind, content=raw, tool=tool, session_id=session, cwd=cwd,
        git_branch=payload.get("git_branch"), exit_code=exit_code,
        truncated=_looks_truncated(raw, payload) or bool(meta.get("_captured_full")),
        summary=summary, meta=meta)

    out: dict = {"event_id": event_id, "kind": kind, "bytes": len(raw)}
    # Give Claude the end of the output it never saw. additionalContext is the
    # only channel PostToolUseFailure has - it cannot rewrite the result.
    if meta.get("_captured_full"):
        extra = capture.tail_context(raw, event_id, shown_chars)
        if extra:
            out["hookSpecificOutput"] = {"hookEventName": event,
                                         "additionalContext": extra[:10_000]}
    return out


def main(argv: list[str] | None = None) -> int:
    if names.env("DISABLE") == "1":
        return 0
    try:
        data = sys.stdin.read()
    except Exception:
        return 0
    if not data.strip():
        # No payload at all is not an event; recording it would be pure noise.
        return 0
    try:
        payload = json.loads(data)
    except ValueError:
        # Unparseable payload is itself worth keeping - it signals format drift.
        vault.record("hook_unparsed", content=data[:100_000], summary="unparseable hook payload")
        return 0
    if not isinstance(payload, dict):
        # Valid JSON of the wrong shape is format drift too, not a crash.
        vault.record("hook_unparsed", content=data[:100_000],
                     summary=f"hook payload was {type(payload).__name__}, expected object")
        return 0
    if not (payload.get("hook_event_name") or payload.get("hookEventName")):
        return 0
    try:
        result = handle(payload)
    except Exception as exc:
        vault.record("hook_error", content=f"{type(exc).__name__}: {exc}\n\n{data[:50_000]}",
                     summary=f"hook error: {type(exc).__name__}")
        return 0
    payload_out = {k: v for k, v in result.items() if k == "hookSpecificOutput"}
    if payload_out:
        print(json.dumps(payload_out))
    if names.env("HOOK_VERBOSE") == "1":
        print(json.dumps(result), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
