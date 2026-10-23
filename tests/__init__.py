"""Test-wide isolation: no test may act on the real machine.

Twice a setup/uninstall test reached a real-system step nobody had stubbed
(once it rewrote ~/.claude/settings.json, once it removed the user's MCP
registration). Patching each new step is how that keeps happening, so:

1. HOME (and the env vars that pick config folders) point at a fresh temp dir
   BEFORE any tokensaver module is imported. Module-level paths (SETTINGS,
   AGENTS, UNITS, BIN_DIR, VAULT_DIR, ...) are computed at import from
   Path.home(), so they all land in the temp dir. tests/test_isolation.py
   fails the run if any of them still points under the real home.
2. Commands that change the system (claude, launchctl, systemctl, ...)
   refuse to run during tests. A test that needs one mocks the call site.

The real home stays available as REAL_HOME, for checks that the real files
were left alone.
"""

import os
import subprocess
import tempfile
from pathlib import Path

REAL_HOME = Path.home()
FAKE_HOME = Path(tempfile.mkdtemp(prefix="tokensaver-test-home-"))

os.environ["HOME"] = str(FAKE_HOME)
for _k in ("CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME", "TOKENSAVER_HOME", "EVERYTOKEN_HOME", "ZDOTDIR",
           "ANTHROPIC_BASE_URL", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
    os.environ.pop(_k, None)
assert Path.home() == FAKE_HOME, "HOME override did not take"

_REFUSE = {"claude", "launchctl", "systemctl", "loginctl", "ollama", "brew", "git", "plutil"}


def _argv0(args):
    if isinstance(args, (list, tuple)) and args:
        return os.path.basename(str(args[0]))
    if isinstance(args, str):
        return os.path.basename(args.split()[0]) if args.split() else ""
    return ""


def _guard(real):
    def wrapper(args, *a, **k):
        name = _argv0(args)
        if name in _REFUSE:
            raise RuntimeError(f"test tried to run `{name}` on the real machine: mock it")
        return real(args, *a, **k)
    return wrapper


subprocess.run = _guard(subprocess.run)
subprocess.Popen = _guard(subprocess.Popen)
