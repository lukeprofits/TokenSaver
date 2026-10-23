"""Reading and writing Claude Code's settings files without disturbing anything else.

Every place TokenSaver edits a user's file goes through `write_text`:
* a symlinked file (dotfiles repos) is written through to its target, so the
  link stays a link;
* the file keeps its permissions;
* the write is atomic (temp file in the target's own folder, then rename) and
  a JSON file is parsed before it replaces the original, so a failed or
  interrupted write never leaves a broken settings file;
* a timestamped backup sits next to the target when asked.
JSON is written with non-ASCII characters as they are (not as \\u escapes), so
a diff shows only what we changed in structure.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
SETTINGS = CONFIG_DIR / "settings.json"


def managed_paths() -> list[Path]:
    """Enterprise managed settings: they win over the user's settings."""
    if sys.platform == "darwin":
        return [Path("/Library/Application Support/ClaudeCode/managed-settings.json")]
    return [Path("/etc/claude-code/managed-settings.json")]


def read_json(path: Path) -> dict | None:
    """The file as a dict ({} if missing); None if it will not parse or is not
    an object (then nothing may touch it)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def env_of(path: Path) -> dict:
    data = read_json(path) or {}
    env = data.get("env")
    return env if isinstance(env, dict) else {}


def _original_copy(target: Path) -> Path:
    from . import vault
    key = hashlib.sha256(str(target).encode()).hexdigest()[:12]
    return vault.VAULT_DIR / "originals" / f"{target.name}.{key}"


def keep_original(path: Path) -> None:
    """The file's bytes before TokenSaver first changed it (once; never
    overwritten), so uninstall can put back exactly what was there."""
    target = Path(os.path.realpath(path))
    copy = _original_copy(target)
    if copy.exists():
        return
    copy.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    copy.write_bytes(target.read_bytes() if target.exists() else b"")
    (copy.parent / (copy.name + ".path")).write_text(str(target))


def restore_original(path: Path, equal) -> str | None:
    """Put back the original bytes if `equal(original_text, current_text)` says
    nothing but our own entries changed since. Returns what it did."""
    target = Path(os.path.realpath(path))
    copy = _original_copy(target)
    if not copy.exists():
        return None
    original = copy.read_bytes()
    current = target.read_bytes() if target.exists() else b""
    if original == current:
        return None
    try:
        same = equal(original.decode("utf-8"), current.decode("utf-8"))
    except Exception:
        same = False
    if not same:
        return None                      # the user changed something since: keep their version
    if not original and target.exists():
        target.unlink()                  # we created it; it held only our entries
        return f"removed {path} (TokenSaver created it)"
    write_text(target, original.decode("utf-8"), backup=False, is_json=False)
    return f"put {path} back exactly as it was before TokenSaver"


def write_text(path: Path, text: str, backup: bool = True, is_json: bool | None = None) -> None:
    target = Path(os.path.realpath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_json is None:
        is_json = target.suffix == ".json"
    if is_json:
        json.loads(text)                 # refuse to write something unparseable
    if backup:                           # a TokenSaver change (not a restore): remember the original
        try:
            keep_original(target)
        except OSError:
            pass
    if backup and target.exists():
        shutil.copy2(target, target.with_name(target.name + f".bak.tokensaver.{time.strftime('%Y%m%d-%H%M%S')}"))
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tokensaver.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if target.exists():
            shutil.copymode(target, tmp)
        else:
            os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: Path, data: dict, backup: bool = True) -> None:
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", backup=backup, is_json=True)
