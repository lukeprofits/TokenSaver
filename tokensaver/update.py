"""`tokensaver update`: the latest code from the repo, then the service refreshed.

Fast-forward only, so local edits are never overwritten (git refuses and says
why). After the pull, the service files are rewritten if the new code writes
them differently (for example a changed watchdog), the `tokensaver` command
link is checked, and the proxy restarts on the new code. Claude Code keeps
working throughout: while the proxy restarts (a few seconds) its requests are
retried, and if it does not come back the watchdog routes around it.
"""

from __future__ import annotations

import subprocess

from . import service

REPO_URL = "https://github.com/lukeprofits/TokenSaver"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(service.REPO), *args], capture_output=True, text=True, timeout=300)


def run() -> int:
    if _git("rev-parse", "--git-dir").returncode != 0:
        print(f"{service.REPO} is not a git checkout, so it cannot update itself.\n"
              f"Reinstall: git clone {REPO_URL} ~/.tokensaver/src && ~/.tokensaver/src/install.sh")
        return 1
    if _git("remote", "get-url", "origin").returncode != 0:
        print(f"no `origin` remote in {service.REPO}; add one: git -C {service.REPO} remote add origin {REPO_URL}")
        return 1
    before = _git("rev-parse", "HEAD").stdout.strip()
    r = _git("pull", "--ff-only")
    print((r.stdout + r.stderr).strip())
    if r.returncode != 0:
        print("not updated (nothing changed). If you edited files here, commit or stash them first.")
        return 1
    after = _git("rev-parse", "HEAD").stdout.strip()
    if before == after:
        print("already up to date")
        return 0
    # The new code may write different service files: let it, in a fresh process.
    if service.installed():
        r = subprocess.run([service.python(), str(service.CLI), "setup", "--yes", "--refresh"],
                           capture_output=True, text=True, timeout=600)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0:
            return 1
    print(f"updated {before[:8]} -> {after[:8]}")
    return 0
