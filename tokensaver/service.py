"""The background services: the proxy, and the watchdog that keeps Claude Code
working when the proxy is down.

macOS: two launchd agents in ~/Library/LaunchAgents. The proxy starts at login
and is restarted if it dies (KeepAlive, 5 s throttle); the watchdog runs every
15 s (StartInterval).
Linux: systemd user units in ~/.config/systemd/user: the proxy with
Restart=always, the watchdog as a oneshot service on a 15 s timer. They start
at login; with lingering (`loginctl enable-linger`, tried without sudo) they
also run without a login session.

The interpreter is pinned to the real path of the Python that ran setup
(`os.path.realpath(sys.executable)`), so a pyenv shim or a PATH change cannot
switch it later; doctor warns if that file disappears.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from . import config, vault

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "tokensaver-cli"
LABEL = "com.tokensaver.proxy"
WATCH_LABEL = "com.tokensaver.watchdog"
AGENTS = Path.home() / "Library" / "LaunchAgents"
PLIST = AGENTS / f"{LABEL}.plist"
WATCH_PLIST = AGENTS / f"{WATCH_LABEL}.plist"
UNITS = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user"
UNIT = UNITS / "tokensaver-proxy.service"
WATCH_UNIT = UNITS / "tokensaver-watchdog.service"
WATCH_TIMER = UNITS / "tokensaver-watchdog.timer"
WATCH_EVERY_S = 15


def platform() -> str:
    return "macos" if sys.platform == "darwin" else "linux" if sys.platform.startswith("linux") else sys.platform


def python() -> str:
    return os.path.realpath(sys.executable)


def _run(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(args, 1, "", str(exc))


def _domain() -> str:
    return f"gui/{os.getuid()}"


# ---- macOS ---------------------------------------------------------------------

def _plist(label: str, args: list[str], keepalive: bool, interval: int | None = None) -> str:
    log = vault.VAULT_DIR / ("proxy.log" if label == LABEL else "watchdog.log")
    arg_xml = "".join(f"\n        <string>{a}</string>" for a in args)
    extra = ("    <key>KeepAlive</key>\n    <dict><key>SuccessfulExit</key><false/></dict>\n"
             "    <key>ThrottleInterval</key><integer>5</integer>\n    <key>ExitTimeOut</key><integer>25</integer>\n"
             if keepalive else f"    <key>StartInterval</key><integer>{interval}</integer>\n")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key>
    <array>{arg_xml}
    </array>
    <key>RunAtLoad</key><true/>
{extra}    <key>ProcessType</key><string>Background</string>
    <key>StandardOutPath</key><string>{log}</string>
    <key>StandardErrorPath</key><string>{log}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key><string>{Path.home()}</string>
        <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin</string>
    </dict>
</dict>
</plist>
"""


def plist_text() -> str:
    return _plist(LABEL, [python(), str(CLI), "proxy", "--service"], keepalive=True)


# The watchdog job must keep Claude Code working even when TokenSaver itself is
# gone: the repo deleted, ~/.tokensaver deleted, or the pinned Python removed
# (a pyenv version uninstalled) without running uninstall. Then the proxy is
# dead too and the settings entries would point every launch at nothing. So the
# job's own command (stored in the launchd/systemd file, outside the repo and
# the data folder) checks that the CLI and interpreter still exist; if not, it
# takes our entries out with any Python it can find, touching only keys that
# still hold exactly our values.
_FALLBACK_PY = r'''
import json, os, sys
p, ours = sys.argv[1], json.loads(sys.argv[2])
try:
    real = os.path.realpath(p)
    d = json.load(open(real))
    env = d.get("env")
    if not isinstance(env, dict):
        raise SystemExit(0)
    hit = [k for k, v in ours.items() if env.get(k) == v]
    if not hit:
        raise SystemExit(0)
    for k in hit:
        env.pop(k)
    if not env:
        d.pop("env")
    tmp = real + ".tokensaver.tmp"
    open(tmp, "w").write(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    json.load(open(tmp))
    os.replace(tmp, real)
    print("TokenSaver is gone: removed its routing from", p)
except SystemExit:
    raise
except Exception as e:
    print("TokenSaver watchdog fallback failed:", e)
'''


def watchdog_command(settings: Path | None = None) -> str:
    """The shell command the watchdog job runs (see _FALLBACK_PY)."""
    import json
    import shlex
    from . import routing
    settings = settings or routing.SETTINGS
    q = shlex.quote
    return (f'if [ -x {q(python())} ] && [ -f {q(str(CLI))} ]; then exec {q(python())} {q(str(CLI))} watchdog; fi; '
            f'for p in {q(python())} /usr/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do '
            f'if command -v "$p" >/dev/null 2>&1 && "$p" -c "" >/dev/null 2>&1; then '
            f'exec "$p" -c {q(_FALLBACK_PY)} {q(str(settings))} {q(json.dumps(routing.wanted()))}; fi; done; '
            f'echo "TokenSaver watchdog: no Python found to remove routing from {settings}"')


def watchdog_plist_text() -> str:
    from xml.sax.saxutils import escape
    text = _plist(WATCH_LABEL, ["/bin/sh", "-c", "@CMD@"], keepalive=False, interval=WATCH_EVERY_S)
    # its log must not depend on ~/.tokensaver existing either
    log = Path.home() / "Library" / "Logs" / "tokensaver-watchdog.log"
    text = text.replace(str(vault.VAULT_DIR / "watchdog.log"), str(log))
    return text.replace("@CMD@", escape(watchdog_command()))


def _install_agent(path: Path, text: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # launchd will not start a job whose log folder is missing (a new account
    # may have no ~/Library/Logs yet: found by the clean-install simulation).
    import plistlib
    for key in ("StandardOutPath", "StandardErrorPath"):
        log = plistlib.loads(text.encode()).get(key)
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    lint = _run(["plutil", "-lint", str(tmp)])
    if lint.returncode != 0:
        tmp.unlink()
        raise RuntimeError(f"{path.name} failed lint: {(lint.stdout + lint.stderr).strip() or 'no output'}")
    tmp.replace(path)
    _reload_agent(path, label)


ALLOW_PROXY_STOP = False     # set by --allow-proxy-restart


def stopping_proxy_is_unsafe() -> str | None:
    """Why stopping the proxy from here would cut off the caller, or None.

    Measured 2026-10-23: setup run from a Claude Code session stopped the proxy
    that session's own requests go through, and every call failed until the
    user restarted it by hand. Signals, any one is enough:
    - this process's ANTHROPIC_BASE_URL is the proxy (a session's tools inherit it);
    - CLAUDECODE is set (Claude Code's own marker in the shells it starts);
    - routing is on in settings.json and there is no terminal: a Claude session
      or a script, and an open session re-reads settings.json, so it can be
      routed even though it started direct."""
    if ALLOW_PROXY_STOP:
        return None
    port = str(config.get("port"))
    base = os.environ.get("ANTHROPIC_BASE_URL", "")
    if base and ("127.0.0.1:" + port in base or "localhost:" + port in base):
        return "this process sends its Claude traffic through the proxy (ANTHROPIC_BASE_URL)"
    if os.environ.get("CLAUDECODE"):
        return "this is running inside a Claude Code session"
    try:
        from . import routing
        if routing.is_on() and not sys.stdin.isatty():
            return "routing is on and there is no terminal (a Claude session or a script)"
    except Exception:
        pass
    return None


def _refuse_proxy_stop(action: str) -> None:
    why = stopping_proxy_is_unsafe()
    if why:
        raise RuntimeError(f"not {action}: {why}, and stopping the proxy would cut that session off. "
                           f"Run it from a normal terminal: {CLI} {action}"
                           f"  (or add --allow-proxy-restart if you are sure)")


BOOTOUT_WAIT_S = 30          # the proxy drains up to ExitTimeOut (25 s) before it exits
BOOTSTRAP_TRIES = 5


def _loaded(label: str) -> bool:
    return _run(["launchctl", "print", f"{_domain()}/{label}"]).returncode == 0


def _reload_agent(path: Path, label: str, sleep=time.sleep) -> None:
    """bootout, wait until the old job is really gone, then bootstrap.

    `launchctl bootout` returns before the job has exited; a bootstrap while it
    is still shutting down fails with "Bootstrap failed: 5: Input/output
    error" (measured 2026-10-23: setup stopped half-way, the watchdog file
    never replaced). So wait for `launchctl print` to stop finding the label,
    and retry the bootstrap on that error."""
    _run(["launchctl", "bootout", f"{_domain()}/{label}"])
    end = time.time() + BOOTOUT_WAIT_S
    while _loaded(label) and time.time() < end:
        sleep(0.5)
    err = ""
    for attempt in range(BOOTSTRAP_TRIES):
        r = _run(["launchctl", "bootstrap", _domain(), str(path)])
        err = (r.stderr or r.stdout or "").strip()
        if r.returncode == 0 or "already" in err:
            return
        if _loaded(label):
            return          # loaded meanwhile (the watchdog revives a missing proxy job), from this same file
        if " 5:" not in err and "Input/output error" not in err:
            break                                   # not the still-shutting-down case: report it
        sleep(1.0 + attempt)
    raise RuntimeError(f"launchctl bootstrap {label}: {err}")


# ---- Linux ---------------------------------------------------------------------

def unit_text() -> str:
    return f"""[Unit]
Description=TokenSaver proxy (Claude Code token savings)
After=network-online.target

[Service]
ExecStart={python()} {CLI} proxy --service
Restart=always
RestartSec=5
TimeoutStopSec=25
Environment=HOME={Path.home()}
StandardOutput=append:{vault.VAULT_DIR / 'proxy.log'}
StandardError=append:{vault.VAULT_DIR / 'proxy.log'}

[Install]
WantedBy=default.target
"""


def _systemd_quote(s: str) -> str:
    """One ExecStart argument: systemd expands $ and % itself, and splits on spaces."""
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$").replace("%", "%%").replace("\n", "\\n")
    return f'"{s}"'


def watchdog_unit_text() -> str:
    return f"""[Unit]
Description=TokenSaver watchdog (routes Claude Code around a stopped proxy)

[Service]
Type=oneshot
ExecStart=/bin/sh -c {_systemd_quote(watchdog_command())}
Environment=HOME={Path.home()}
"""


def watchdog_timer_text() -> str:
    return f"""[Unit]
Description=Run the TokenSaver watchdog every {WATCH_EVERY_S} s

[Timer]
OnBootSec=30
OnUnitActiveSec={WATCH_EVERY_S}s
AccuracySec=1s

[Install]
WantedBy=timers.target
"""


def _systemctl(*args) -> subprocess.CompletedProcess:
    return _run(["systemctl", "--user", *args])


# ---- both --------------------------------------------------------------------------

def wanted_files() -> dict[Path, str]:
    if platform() == "macos":
        return {PLIST: plist_text(), WATCH_PLIST: watchdog_plist_text()}
    if platform() == "linux":
        return {UNIT: unit_text(), WATCH_UNIT: watchdog_unit_text(), WATCH_TIMER: watchdog_timer_text()}
    return {}


def supported() -> bool:
    return bool(wanted_files())


def up_to_date() -> bool:
    files = wanted_files()
    return bool(files) and all(p.exists() and p.read_text() == t for p, t in files.items()) and running()


def running() -> bool:
    if platform() == "macos":
        return _run(["launchctl", "print", f"{_domain()}/{LABEL}"]).returncode == 0
    if platform() == "linux":
        return _systemctl("is-active", "--quiet", UNIT.name).returncode == 0
    return False


def installed() -> bool:
    return any(p.exists() for p in (PLIST, UNIT))


def proxy_would_restart() -> bool:
    """Whether install() would stop the running proxy (its own file changed)."""
    if platform() == "macos":
        return _loaded(LABEL) and not (PLIST.exists() and PLIST.read_text() == plist_text())
    if platform() == "linux":
        return UNIT.exists() and UNIT.read_text() != unit_text() and running()
    return False


def install() -> list[str]:
    """Write and start both services. Raises on failure."""
    vault.VAULT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    notes = []
    if platform() == "macos":
        # Only a job whose file changed is replaced (or one not loaded): a
        # watchdog change must not restart the proxy, which is in the request path.
        for path, text, label in ((PLIST, plist_text(), LABEL), (WATCH_PLIST, watchdog_plist_text(), WATCH_LABEL)):
            if not (path.exists() and path.read_text() == text and _loaded(label)):
                if label == LABEL and _loaded(LABEL):
                    _refuse_proxy_stop("setup")
                _install_agent(path, text, label)
                notes.append(f"replaced {path.name}")
        return notes
    if platform() != "linux":
        raise RuntimeError(f"no service support on {sys.platform}")
    UNITS.mkdir(parents=True, exist_ok=True)
    if UNIT.exists() and UNIT.read_text() != unit_text() and running():
        _refuse_proxy_stop("setup")
    for p, t in wanted_files().items():
        p.write_text(t)
    r = _systemctl("daemon-reload")
    if r.returncode != 0:
        raise RuntimeError(f"systemctl --user daemon-reload: {r.stderr.strip()} (is there a user systemd session?)")
    for name in (UNIT.name, WATCH_TIMER.name):
        r = _systemctl("enable", "--now", name)
        if r.returncode != 0:
            raise RuntimeError(f"systemctl --user enable --now {name}: {r.stderr.strip()}")
    linger = _run(["loginctl", "enable-linger", os.environ.get("USER", "")])
    notes.append("runs without a login session too (lingering on)" if linger.returncode == 0 else
                 "starts at login (lingering needs an admin: `sudo loginctl enable-linger $USER`)")
    return notes


# Names from before the rename (2026-10-23): removed by `migrate` and `uninstall`.
OLD_LABELS = ("com.everytoken.watchdog", "com.everytoken.proxy")
OLD_UNITS = ("everytoken-watchdog.timer", "everytoken-watchdog.service", "everytoken-proxy.service")


def remove_old() -> list[str]:
    done = []
    if platform() == "macos":
        for label in OLD_LABELS:
            path = AGENTS / f"{label}.plist"
            if path.exists():
                _run(["launchctl", "bootout", f"{_domain()}/{label}"])
                path.unlink()
                done.append(f"removed {path}")
    elif platform() == "linux":
        for name in OLD_UNITS:
            if (UNITS / name).exists():
                _systemctl("disable", "--now", name)
                (UNITS / name).unlink()
                done.append(f"removed {UNITS / name}")
        if done:
            _systemctl("daemon-reload")
    return done


LEFT_RUNNING = False         # set by remove() when the proxy was left for open sessions


def _leave_proxy_running(path: Path) -> str:
    """For uninstall from a routed Claude session. Measured 2026-10-23: an open
    session does NOT go back to direct when the routing entries are removed
    (it failed for 16 minutes with the entries already gone), so stopping the
    proxy would cut it off. The service file goes (nothing starts it again at
    the next login) but the running job stays until logout or reboot, serving
    open sessions exactly as before; new launches already go direct."""
    global LEFT_RUNNING
    LEFT_RUNNING = True
    if platform() == "linux":
        _systemctl("disable", UNIT.name)        # no --now: keep it running
    path.unlink()
    return (f"removed {path}; the running proxy stays until you log out or restart, so open Claude "
            f"sessions keep working (new ones already go direct)")


def remove() -> list[str]:
    done = remove_old()
    keep = stopping_proxy_is_unsafe() is not None
    if platform() == "macos":
        for path, label in ((WATCH_PLIST, WATCH_LABEL), (PLIST, LABEL)):
            if path.exists():
                if label == LABEL and keep and _loaded(LABEL):
                    done.append(_leave_proxy_running(path))
                    continue
                _run(["launchctl", "bootout", f"{_domain()}/{label}"])
                path.unlink()
                done.append(f"removed {path}")
    elif platform() == "linux":
        names = (WATCH_TIMER.name,) if keep and running() else (WATCH_TIMER.name, UNIT.name)
        for name in names:
            _systemctl("disable", "--now", name)
        for p in (WATCH_TIMER, WATCH_UNIT) + (() if len(names) == 1 else (UNIT,)):
            if p.exists():
                p.unlink()
                done.append(f"removed {p}")
        if len(names) == 1 and UNIT.exists():
            done.append(_leave_proxy_running(UNIT))
        _systemctl("daemon-reload")
    return done


def restart() -> bool:
    _refuse_proxy_stop("restart")
    if platform() == "macos":
        _run(["launchctl", "kickstart", "-k", f"{_domain()}/{LABEL}"])
    elif platform() == "linux":
        _systemctl("restart", UNIT.name)
    else:
        return False
    return wait_health() is not None


def health(timeout: float = 1.0) -> dict | None:
    import json
    import urllib.request
    for prefix in ("_tokensaver", "_everytoken"):     # a proxy started before the rename
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{config.get('port')}/{prefix}/health",
                                        timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception:
            continue
    return None


def wait_health(seconds: float = 8.0) -> dict | None:
    end = time.time() + seconds
    while time.time() < end:
        h = health()
        if h:
            return h
        time.sleep(0.25)
    return None
