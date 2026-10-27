"""Keep Claude Code working when the proxy is down.

Run every 15 s by launchd (StartInterval) or a systemd timer. Claude Code has
no fallback for an unreachable ANTHROPIC_BASE_URL, so when the proxy fails two
checks in a row this takes our settings.json entries out (`routing.disable`):
new launches go straight to Anthropic, as if TokenSaver were not installed.
When the proxy answers again it puts them back. Sessions already running keep
the address they started with; the service manager restarts the proxy in
about 5 s, and Claude Code retries meanwhile.

It acts only when setup turned settings routing on (`routing.json` managed)
and never fights the user: a key someone else changed is left alone (see
`routing`). Every flip is recorded in the vault (`routing_flip`). Never raises.
"""

from __future__ import annotations

import json
import time
import urllib.request

from . import config, routing, vault

FAILS_TO_DISABLE = 2
STATE = vault.VAULT_DIR / "watchdog.json"


def healthy(timeout: float = 2.0) -> bool:
    for prefix in ("_tokensaver", "_everytoken"):     # a proxy started before the rename
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{config.get('port')}/{prefix}/health",
                                        timeout=timeout) as r:
                return bool(json.loads(r.read()).get("ok"))
        except Exception:
            continue
    return False


def _load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save(st: dict) -> None:
    try:
        STATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(STATE)
    except OSError:
        pass


def service_wait_health(seconds: float = 8.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if healthy(timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def revive() -> str | None:
    """Start the proxy job if it is installed but not loaded at all.

    launchd/systemd restart a proxy that crashes (in ~5 s); what they cannot
    undo is a job that is not loaded. Measured 2026-10-23: setup booted the old
    job out and the new bootstrap failed, so nothing restarted it and every
    open Claude session failed for 16 minutes (the routing removal does not
    reach a session that is already open). A missing service file means it was
    uninstalled on purpose: then nothing is started."""
    from . import service
    try:
        if service.platform() == "macos":
            if service.PLIST.exists() and not service._loaded(service.LABEL):
                r = service._run(["launchctl", "bootstrap", service._domain(), str(service.PLIST)])
                return "proxy job reloaded" if r.returncode == 0 else f"reload failed: {r.stderr.strip()[:120]}"
        elif service.platform() == "linux":
            if service.UNIT.exists() and not service.running() and \
                    service._systemctl("is-enabled", "--quiet", service.UNIT.name).returncode == 0:
                r = service._systemctl("start", service.UNIT.name)
                return "proxy unit started" if r.returncode == 0 else f"start failed: {r.stderr.strip()[:120]}"
    except Exception as exc:
        return f"revive error: {type(exc).__name__}"
    return None


def once(check=healthy) -> str:
    """One check. Returns what it did (for logs and tests)."""
    try:
        rs = routing.state()
        if not rs.get("managed"):
            return "not managing routing"
        st = _load()
        ok = check()
        if not ok:
            revived = revive()
            if revived:
                try:
                    vault.record("proxy_revived", tool="watchdog", index=False, summary=revived)
                except Exception:
                    pass
                if revived.startswith("proxy") and service_wait_health():
                    ok = True
        st["fails"] = 0 if ok else int(st.get("fails") or 0) + 1
        st["last"] = time.time()
        did = "no change"
        if ok and not rs.get("on") and rs.get("why", "").startswith("watchdog"):
            did = "restored: " + routing.enable("watchdog: proxy healthy again", backup=False)
        elif ok and routing.pause_due(rs):
            did = "pause over: " + routing.enable("watchdog: pause ended at midnight", backup=False)
        elif not ok and rs.get("on") and st["fails"] >= FAILS_TO_DISABLE:
            did = "bypassed: " + routing.disable("watchdog: proxy not answering", backup=False)
        _save(st)
        if did != "no change":
            try:
                vault.record("routing_flip", tool="watchdog", index=False, summary=did,
                             meta={"healthy": ok, "fails": st["fails"]})
            except Exception:
                pass
        return did
    except Exception as exc:                 # a watchdog must never be the thing that breaks
        return f"error: {type(exc).__name__}"
