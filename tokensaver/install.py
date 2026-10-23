"""Install and remove tokensaver's hooks in Claude Code settings.

Two tiers, deliberately separated:

* **capture** (default) - PostToolUse, PostToolUseFailure, PreCompact,
  SessionStart, and PreToolUse. The first four only observe. PreToolUse does
  one transformation: it moves a large quoted heredoc body into the vault and
  shortens the command to reference it. That is verified byte-identical -
  stdout, exit code and resulting files all match the original across
  interpreter scripts, `cat >` and `cat >>` writes, nested paths and
  multi-heredoc commands - and it is the only saving here that needs no cache
  rebuild, so there is nothing to pay back.

* **output capture** (opt-in, env `TOKENSAVER_CAPTURE_OUTPUT=1`) - also wraps
  allowlisted test and build commands in a brace group with process
  substitution so their full output is recorded. Materially more invasive than
  vaulting a heredoc: it restructures the command. Guarded against heredocs,
  process substitution, backgrounding, `exec`, unparseable quoting and
  watch flags, each added because that case broke in testing. Off by default.

Every edit backs up the file first and is idempotent.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import claude_settings
from .claude_settings import CONFIG_DIR

HOOK_BIN = str(Path(__file__).resolve().parent.parent / "tokensaver-hook")
CAPTURE_EVENTS = ("PostToolUse", "PostToolUseFailure", "PreCompact",
                  "SessionStart", "PreToolUse")
MARKERS = ("tokensaver-hook", "everytoken-hook")   # the second: entries from before the rename


def _entry() -> dict:
    return {"matcher": "*", "hooks": [{"type": "command", "command": HOOK_BIN}]}


def _has_ours(group: list) -> bool:
    return any(m in json.dumps(item) for item in group for m in MARKERS)


def install(path: Path | None = None) -> str:
    path = path or CONFIG_DIR / "settings.json"
    if not path.exists():
        return f"no settings file at {path}"
    backup = path.with_suffix(f".json.bak.tokensaver.{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)

    data = json.loads(path.read_text())
    hooks = data.setdefault("hooks", {})
    events = list(CAPTURE_EVENTS)
    added = []
    for event in events:
        group = hooks.setdefault(event, [])
        if _has_ours(group):
            continue
        group.append(_entry())
        added.append(event)

    claude_settings.write_json(path, data, backup=False)   # backed up above
    if not added:
        return f"already installed; backup at {backup.name}"
    return (f"installed: {', '.join(added)}\n"
            f"  backup: {backup}\n"
            f"  disable at any time with TOKENSAVER_DISABLE=1, or `uninstall`")


def uninstall(path: Path | None = None) -> str:
    path = path or CONFIG_DIR / "settings.json"
    if not path.exists():
        return f"no settings file at {path}"
    backup = path.with_suffix(f".json.bak.tokensaver.{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)

    data = json.loads(path.read_text())
    hooks = data.get("hooks", {})
    removed = []
    for event, group in list(hooks.items()):
        kept = [item for item in group if not any(m in json.dumps(item) for m in MARKERS)]
        if len(kept) != len(group):
            removed.append(event)
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    if not hooks:
        data.pop("hooks", None)
    claude_settings.write_json(path, data, backup=False)   # backed up above
    return (f"removed from {', '.join(removed)}; backup at {backup}" if removed
            else f"nothing to remove; backup at {backup}")
