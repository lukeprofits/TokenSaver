"""Route every Claude Code launch through the proxy: `~/.claude/settings.json` `env`.

Claude Code applies that `env` block to every launch: the CLI, and the VS Code
and JetBrains extensions (docs: code.claude.com, env-vars and settings). So
with three entries there, plain `claude` saves with no wrapper. What it does
not have is a fallback: if the gateway is down, Claude Code retries and then
says it cannot reach the API. Hence `watchdog`, which takes these entries out
while the proxy is down and puts them back when it is up.

Rules:
* Only keys we added are ever removed, and only while they still hold our
  value (the user or another tool may have changed one since).
* A value we replaced (for example a company gateway in ANTHROPIC_BASE_URL,
  which the proxy then chains to as its upstream) is put back when ours is
  taken out.
* Writes are atomic and re-read the file just before writing; an unparseable
  settings file is never touched. Setup and uninstall keep a backup; the
  watchdog's flips do not (they would pile up).
* State (what we added and replaced, on or off and why) is in
  ~/.tokensaver/routing.json.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from . import claude_settings, config, vault

SETTINGS = claude_settings.SETTINGS
STATE = vault.VAULT_DIR / "routing.json"


def wanted() -> dict[str, str]:
    return {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{config.get('port')}",
            "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1",
            # Claude Code turns tool search off behind a non-Anthropic base URL;
            # the proxy forwards tool_reference blocks, so it stays on (measured:
            # without it every tool definition was sent in full, ~2.5-6% of the bill).
            "ENABLE_TOOL_SEARCH": "true"}


def read_settings() -> dict | None:
    """settings.json as a dict ({} if missing); None if it will not parse."""
    return claude_settings.read_json(SETTINGS)


def write_settings(data: dict, backup: bool = True) -> None:
    claude_settings.write_json(SETTINGS, data, backup=backup)


def state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def is_on() -> bool:
    data = read_settings() or {}
    env = data.get("env") if isinstance(data.get("env"), dict) else {}
    return env.get("ANTHROPIC_BASE_URL") == wanted()["ANTHROPIC_BASE_URL"]


def enable(why: str, backup: bool = True) -> str:
    """Put our entries in. Returns what happened."""
    data = read_settings()
    if data is None:
        raise RuntimeError(f"{SETTINGS} does not parse; not touching it")
    env = dict(data.get("env") or {}) if isinstance(data.get("env"), (dict, type(None))) else None
    if env is None:
        raise RuntimeError(f"{SETTINGS} has a non-object `env`; not touching it")
    st = state()
    added, replaced = dict(st.get("added") or {}), dict(st.get("replaced") or {})
    changed = []
    for k, v in wanted().items():
        cur = env.get(k)
        if cur == v:
            added.setdefault(k, v)
            continue
        if cur is not None and k not in added:
            if k == "ENABLE_TOOL_SEARCH":
                continue                 # the user's own choice wins
            replaced[k] = cur
        env[k] = v
        added[k] = v
        changed.append(k)
    if changed:
        data["env"] = env
        write_settings(data, backup=backup)
    _save_state({"managed": True, "added": added, "replaced": replaced, "on": True, "why": why,
                 "at": time.time(), "flips": st.get("flips", 0) + (1 if changed and st.get("managed") else 0)})
    return f"set {', '.join(changed)} in {SETTINGS}" if changed else "already routed"


def disable(why: str, backup: bool = True, forget: bool = False) -> str:
    """Take our entries out (restoring anything we replaced). `forget`: uninstall."""
    st = state()
    added, replaced = dict(st.get("added") or {}), dict(st.get("replaced") or {})
    data = read_settings()
    if data is None:
        raise RuntimeError(f"{SETTINGS} does not parse; not touching it")
    env = dict(data.get("env") or {}) if isinstance(data.get("env"), dict) else {}
    changed = []
    for k, v in added.items():
        if env.get(k) != v:
            continue                    # changed by someone else since: leave it
        if k in replaced:
            env[k] = replaced[k]
        else:
            env.pop(k)
        changed.append(k)
    if changed:
        if env:
            data["env"] = env
        else:
            data.pop("env", None)
        write_settings(data, backup=backup)
    if forget:
        try:
            STATE.unlink()
        except OSError:
            pass
    else:
        _save_state(dict(st, on=False, why=why, at=time.time(), flips=st.get("flips", 0) + (1 if changed else 0)))
    return f"removed {', '.join(changed)} from {SETTINGS}" if changed else "nothing of ours to remove"


def overridden() -> list[str]:
    """Places that would win over our settings entry (doctor shows them)."""
    out = []
    if os.environ.get("ANTHROPIC_BASE_URL") and os.environ["ANTHROPIC_BASE_URL"] != wanted()["ANTHROPIC_BASE_URL"]:
        out.append(f"ANTHROPIC_BASE_URL in this shell's environment ({os.environ['ANTHROPIC_BASE_URL']})")
    for p in [Path.cwd() / ".claude" / "settings.json", Path.cwd() / ".claude" / "settings.local.json"] \
            + claude_settings.managed_paths():
        env = claude_settings.env_of(p)
        if env.get("ANTHROPIC_BASE_URL") not in (None, wanted()["ANTHROPIC_BASE_URL"]):
            out.append(f"ANTHROPIC_BASE_URL in {p}")
    return out
