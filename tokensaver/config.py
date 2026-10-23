"""Settings, in one file, each with where its value came from.

The proxy runs under launchd, which passes no shell environment, so a setting
that lives only in an env var never reaches it. Everything therefore lives in
`~/.tokensaver/config.json`. Env vars still win for a proxy started by hand,
which is what development and the gates do.

Precedence, highest first: env var > user > tuned > default. Only tuned values
are clamped to a setting's bounds: a user value or env var is the user's call
and is honoured as given (doctor warns when it is out of bounds). `tune` writes
only the tuned slot, never the user slot.

The file is read cheaply and often. `load()` never raises: a file that will not
parse is moved aside to `config.json.corrupt.<time>` and the defaults are used,
because a broken settings file must not take the proxy down with it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass

from . import names, vault

PATH = vault.VAULT_DIR / "config.json"


@dataclass(frozen=True)
class Spec:
    name: str
    env: str | None
    type: type
    default: object
    lo: object = None          # bounds for tuned values; None = unbounded
    hi: object = None
    doc: str = ""


SPECS: dict[str, Spec] = {s.name: s for s in (
    Spec("rewrite", "TOKENSAVER_PROXY_REWRITE", str, "off", doc="off | forward (swap was removed 2026-10-22; any other value relays unchanged)"),
    Spec("bg_compact", "TOKENSAVER_BG_COMPACT", bool, False, doc="background compaction"),
    Spec("binding_monitor", None, bool, True,
         doc="ask the API to report thinking blocks that fail the preserved-thinking check (header only)"),
    Spec("bg_yield", None, bool, True, doc="ease off background model work when the machine strains"),
    Spec("prune_tier", None, str, "auto", doc="auto | off | fast | deep: local-model pruning of compactions"),
    Spec("prune_min_p", None, float, 0.8, 0.7, 0.99, "delete a piece only at this judge probability"),
    Spec("prune_judge", None, str, "nimble", doc="decision model for the prune judge"),
    Spec("bg_after_min", "TOKENSAVER_BG_AFTER_MIN", float, 15.0, 10.0, 61.0,
         "apply a prepared compaction after this many idle minutes"),
    Spec("bg_warm_payback_max", None, float, 70.0, 20.0, 200.0,
         "apply on a warm cache only if the rewrite repays within this many requests"),
    Spec("bg_prep_idle_min", None, float, 10.0, 5.0, 57.0,
         "start preparing after this many idle minutes"),
    Spec("bg_min_removable", None, int, 15_000, 5_000, 60_000,
         "compact only if at least this many tokens of old history would be replaced"),
    Spec("bg_keep_tokens", "TOKENSAVER_BG_KEEP", int, 30_000, 20_000, 60_000),
    Spec("bg_index", "TOKENSAVER_BG_INDEX", int, 1_000, 500, 2_000),
    Spec("local_model", "TOKENSAVER_LOCAL_MODEL", str, "qwen3.6:35b-mlx"),
    Spec("ollama_url", "TOKENSAVER_OLLAMA_URL", str, "http://127.0.0.1:11434/api/generate"),
    Spec("upstream", "TOKENSAVER_UPSTREAM", str, "https://api.anthropic.com"),
    Spec("port", "TOKENSAVER_PROXY_PORT", int, 8787),
    Spec("audit", "TOKENSAVER_AUDIT", str, "off", doc="off | on: count_tokens samples"),
    # count_tokens is free with its own rate limits (Anthropic docs), so every
    # request is counted: the real-token numbers and the tuning that uses them
    # need the whole picture, not a sample. 429s back off on their own. Measured
    # on a real ticket: ~340 requests a day, ~2-3 calls each.
    Spec("audit_every", None, int, 1, 1, 200, "count every Nth request"),
    Spec("audit_min_gap_s", None, float, 1.0, 0.5, 3600.0, "seconds between audit call pairs"),
    Spec("audit_max_per_day", None, int, 5_000, 0, 20_000, "count_tokens calls per day"),
    Spec("vault_keep_days", None, int, 7, 1, 365, "days of request bodies to keep"),
    Spec("vault_max_mb", None, int, 2_048, 100, 100_000, "cap on stored request bodies"),
)}

_cache: dict = {"mtime": None, "data": {}, "checked": 0.0}


def _parse(spec: Spec, raw):
    if spec.type is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    return spec.type(raw)


def load(force: bool = False) -> dict:
    """The file's contents, re-read when it changed. Never raises."""
    now = time.time()
    if not force and now - _cache["checked"] < 2.0:
        return _cache["data"]
    _cache["checked"] = now
    try:
        mtime = PATH.stat().st_mtime
    except OSError:
        _cache.update(mtime=None, data={})
        return _cache["data"]
    if mtime == _cache["mtime"] and not force:
        return _cache["data"]
    try:
        data = json.loads(PATH.read_text())
        if not isinstance(data, dict):
            raise ValueError("config is not an object")
    except (OSError, ValueError):
        try:
            PATH.rename(PATH.with_name(f"config.json.corrupt.{int(now)}"))
        except OSError:
            pass
        data = {}
    _cache.update(mtime=mtime, data=data)
    return data


def resolve(name: str) -> tuple[object, str]:
    """(value, source) where source is env | user | tuned | default."""
    spec = SPECS[name]
    raw = names.env(spec.env.removeprefix("TOKENSAVER_")) if spec.env else None
    if raw is not None:     # TOKENSAVER_*, or the old EVERYTOKEN_* name
        try:
            return _parse(spec, raw), "env"
        except (TypeError, ValueError):
            pass
    entry = (load().get("settings") or {}).get(name) or {}
    for source in ("user", "tuned"):
        slot = entry.get(source)
        if isinstance(slot, dict) and "value" in slot:
            try:
                value = _parse(spec, slot["value"])
            except (TypeError, ValueError):
                continue
            if source == "tuned":
                value = clamp(spec, value)
            return value, source
    return spec.default, "default"


def get(name: str):
    return resolve(name)[0]


def clamp(spec: Spec, value):
    if spec.lo is not None and value < spec.lo:
        return spec.lo
    if spec.hi is not None and value > spec.hi:
        return spec.hi
    return value


def _write(data: dict) -> None:
    PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=PATH.parent, prefix=".config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    load(force=True)


def _locked(fn):
    import fcntl
    PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(PATH.with_name("config.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def set_value(name: str, value, source: str = "user", why: str = "",
              evidence: dict | None = None) -> None:
    """Record `value` in the `source` slot (user or tuned) with provenance."""
    spec = SPECS[name]
    value = _parse(spec, value)
    if source == "tuned":
        value = clamp(spec, value)

    def update():
        data = dict(load(force=True))
        settings = dict(data.get("settings") or {})
        entry = dict(settings.get(name) or {})
        entry[source] = {"value": value, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "why": why, "evidence": evidence or {}}
        settings[name] = entry
        data["settings"] = settings
        data.setdefault("schema", 1)
        _write(data)
    _locked(update)


def unset(name: str, source: str = "user") -> None:
    def update():
        data = dict(load(force=True))
        entry = dict((data.get("settings") or {}).get(name) or {})
        if entry.pop(source, None) is not None:
            data.setdefault("settings", {})[name] = entry
            _write(data)
    _locked(update)


def section(key: str) -> dict:
    """A non-settings section (install, tune, fallback), read-only copy."""
    return dict(load().get(key) or {})


def set_section(key: str, value: dict) -> None:
    def update():
        data = dict(load(force=True))
        data[key] = value
        data.setdefault("schema", 1)
        _write(data)
    _locked(update)


def report() -> list[tuple[str, object, str, str]]:
    """(name, value, source, note) for every setting; note flags out-of-bounds."""
    rows = []
    for name, spec in SPECS.items():
        value, source = resolve(name)
        note = ""
        if spec.lo is not None and isinstance(value, (int, float)) and not (spec.lo <= value <= spec.hi):
            note = f"outside {spec.lo}-{spec.hi}"
        rows.append((name, value, source, note))
    return rows
