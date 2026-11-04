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

_REFUSE = {"claude", "launchctl", "systemctl", "loginctl", "ollama", "brew", "git", "plutil", "nvidia-smi"}


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
        if isinstance(args, (list, tuple)) and "local-ai" in [str(x) for x in args]:
            raise RuntimeError("test tried to start the local AI download: mock localai.start")
        return real(args, *a, **k)
    return wrapper


subprocess.run = _guard(subprocess.run)
subprocess.Popen = _guard(subprocess.Popen)

# 3. No request reaches an ollama on this machine (the real one on :11434 would
#    answer, and a pull would download gigabytes). Tests use their own stub server
#    on another port, or mock localai._tags / pull.
import urllib.request  # noqa: E402

_OLLAMA_PORTS = (":11434", ":11435")
_real_urlopen = urllib.request.urlopen


def _urlopen(url, *a, **k):
    target = url.full_url if hasattr(url, "full_url") else str(url)
    if any(p in target for p in _OLLAMA_PORTS):
        raise OSError(f"test tried to reach a real ollama ({target}): mock it")
    if "anthropic.com" in target:
        raise OSError(f"test tried to reach the real Anthropic API ({target}): use a stub upstream")
    return _real_urlopen(url, *a, **k)


urllib.request.urlopen = _urlopen
