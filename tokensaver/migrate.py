"""One-time move of a machine from the old name (everytoken) to TokenSaver.

Run with no Claude Code session open (it restarts the proxy):

    tokensaver-cli migrate [--dry-run] [--yes]

What it does, each step idempotent and skipped when already done:
1. stops and removes the old service (com.everytoken.* / everytoken-*.service);
2. moves ~/.everytoken to ~/.tokensaver and leaves a link at the old place, so
   commands already in Claude's history (`python3 ~/.everytoken/scripts/...`)
   keep working;
3. rewrites our entries in ~/.claude/settings.json (hooks, status line) and
   the managed block in the shell rc (alias names claude-et -> claude-ts) to
   the new file names, with backups;
4. re-registers the MCP server as `tokensaver` (its tools become
   tokensaver_search / tokensaver_read in new sessions);
5. installs and starts the new service, if the old one was installed;
6. removes the old command names and the `everytoken` package link from the
   repo, last, once everything else worked.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import names

REPO = Path(__file__).resolve().parent.parent
from .claude_settings import SETTINGS  # noqa: E402
SCRIPTS = ("cli", "hook", "mcp", "statusline", "claude")
OLD_RC_BEGIN = "# >>> everytoken >>> (managed by `everytoken-cli setup`; removed by `everytoken-cli uninstall`)"
OLD_RC_END = "# <<< everytoken <<<"


def _swap_names(text: str) -> str:
    """Old file names, block markers and alias names in text we wrote."""
    for x in SCRIPTS:
        text = text.replace(str(REPO / f"everytoken-{x}"), str(REPO / f"tokensaver-{x}"))
    from .setup import RC_BEGIN, RC_END
    text = text.replace(OLD_RC_BEGIN, RC_BEGIN).replace(OLD_RC_END, RC_END)
    return text


def _rc_files() -> list[Path]:
    return [p for p in (Path.home() / ".zshrc", Path.home() / ".bashrc", Path.home() / ".bash_profile")
            if p.exists()]


def _rewrite_rc(text: str) -> str:
    text = _swap_names(text)
    from .setup import RC_BEGIN, RC_END

    def block(m):
        return (m.group(0).replace("alias claude-mcp-et=", "alias claude-mcp-ts=")
                .replace("alias claude-et=", "alias claude-ts="))
    return re.sub(re.escape(RC_BEGIN) + r".*?" + re.escape(RC_END), block, text, flags=re.S)


def _write(path: Path, text: str) -> None:
    from . import claude_settings
    claude_settings.write_text(path, text)


def rewrite_names(apply: bool = True) -> list[str]:
    """Step 3 alone: settings.json and the shell rc. Also run by uninstall, so
    it finds entries written before the rename."""
    done = []
    if SETTINGS.exists():
        old = SETTINGS.read_text()
        new = _swap_names(old)
        if new != old:
            json.loads(new)                       # never write a broken settings file
            if apply:
                _write(SETTINGS, new)
            done.append(f"{SETTINGS}: hooks and status line point at the tokensaver-* scripts")
    for rc in _rc_files():
        old = rc.read_text()
        new = _rewrite_rc(old)
        if new != old:
            if apply:
                _write(rc, new)
            done.append(f"{rc}: managed block renamed (claude-et -> claude-ts, claude-mcp-et -> claude-mcp-ts)")
    return done


def _data_step(apply: bool) -> list[str]:
    old, new = names.HOME_OLD, names.HOME_NEW
    if (new / names.MIGRATED).exists():
        return []
    if not old.is_dir() or old.is_symlink():
        if apply:
            new.mkdir(mode=0o700, parents=True, exist_ok=True)
            (new / names.MIGRATED).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        return [f"{new} is the data folder"]
    if new.exists() and any(new.iterdir()):
        raise RuntimeError(f"both {old} and {new} hold files; move or merge one by hand, then run migrate again")
    if apply:
        if new.exists():
            new.rmdir()
        os.rename(old, new)
        old.symlink_to(new)
        (new / names.MIGRATED).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return [f"move {old} -> {new} (a link stays at {old})"]


def _mcp_step(apply: bool) -> list[str]:
    claude = shutil.which("claude")
    if not claude:
        return []
    got = subprocess.run([claude, "mcp", "get", "everytoken"], capture_output=True, text=True, timeout=60)
    if got.returncode != 0:
        return []
    if apply:
        subprocess.run([claude, "mcp", "remove", "everytoken", "-s", "user"], capture_output=True, timeout=60)
        r = subprocess.run([claude, "mcp", "add", "--scope", "user", "tokensaver", str(REPO / "tokensaver-mcp")],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"claude mcp add tokensaver failed: {r.stderr[-300:]}")
    return ["MCP server everytoken -> tokensaver (new sessions see tokensaver_search / tokensaver_read)"]


def _old_service_installed() -> bool:
    from . import service
    if service.platform() == "macos":
        return any((service.AGENTS / f"{label}.plist").exists() for label in service.OLD_LABELS)
    return any((service.UNITS / n).exists() for n in service.OLD_UNITS)


def _cleanup_step(apply: bool) -> list[str]:
    gone = [REPO / f"everytoken-{x}" for x in SCRIPTS] + [REPO / "everytoken"]
    gone = [p for p in gone if p.is_symlink() or p.exists()]
    if apply:
        for p in gone:
            if p.is_symlink() or p.is_file():
                p.unlink()
    return [f"remove old names from the repo: {', '.join(p.name for p in gone)}"] if gone else []


def run(apply: bool) -> list[str]:
    from . import service
    out = []
    had_service = _old_service_installed()
    if had_service:
        out.append("stop and remove the old service (com.everytoken.* / everytoken-*.service)")
        if apply:
            service.remove_old()
    out += _data_step(apply)
    out += rewrite_names(apply)
    out += _mcp_step(apply)
    if had_service:
        out.append("install and start the new service (com.tokensaver.proxy)")
        if apply:
            # A fresh process, so the unit files point at the new data folder.
            r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                                "from tokensaver import service; service.install()", str(REPO)],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError(f"installing the new service failed: {r.stderr[-300:]}")
            if service.wait_health(15) is None:
                raise RuntimeError("the new service did not answer; see ~/.tokensaver/proxy.log")
    out += _cleanup_step(apply)
    return out


def main(dry_run: bool, yes: bool) -> int:
    try:
        steps = run(apply=False)
    except RuntimeError as exc:
        print(f"cannot migrate: {exc}")
        return 1
    if not steps:
        print("already migrated: nothing to do")
        return 0
    print("Migrate this machine from everytoken to TokenSaver:")
    for s in steps:
        print(f"  - {s}")
    if dry_run:
        return 0
    if not yes:
        if not sys.stdin.isatty():
            print("(no terminal: run with --yes to apply)")
            return 3
        if input("Close every Claude Code session first. Apply? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    try:
        for s in run(apply=True):
            print(f"  done: {s}")
    except Exception as exc:
        print(f"migration stopped: {type(exc).__name__}: {exc}\n(run it again: finished steps are skipped)")
        return 1
    print("Migrated. Open new Claude Code sessions; your alias is now claude-ts.")
    return 0
