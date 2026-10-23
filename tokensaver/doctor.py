"""Configuration check: what is set, what it resolved to, what is assumed.

The point of this command is to make every input to the cost numbers visible.
A model alias, a stale price row, or an unset threshold each silently changes
what `report` and `simulate` mean.
"""

from __future__ import annotations

import time

import json
import os
from pathlib import Path

from .pricing import STALE_AFTER_DAYS, VERIFIED, _family, resolve, stale_days
from .transcripts import Session

from .claude_settings import CONFIG_DIR  # noqa: E402
# Range accepted by Claude Code's autoCompactWindow setting.
WINDOW_MIN, WINDOW_MAX = 100_000, 1_000_000


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _settings_files(cwd: Path) -> list[Path]:
    return [CONFIG_DIR / "settings.json",
            CONFIG_DIR / "settings.local.json",
            cwd / ".claude" / "settings.json",
            cwd / ".claude" / "settings.local.json"]


def _observed_models() -> dict[str, int]:
    """Models that actually appeared in proxied requests, counted.

    Empty when the proxy has never run, in which case the cross-check below
    is simply skipped rather than guessed at.
    """
    import sqlite3

    from . import vault
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        rows = conn.execute(
            "SELECT meta FROM events WHERE kind='api_request'").fetchall()
        conn.close()
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for (meta_json,) in rows:
        try:
            mid = (json.loads(meta_json or "{}") or {}).get("model")
        except ValueError:
            continue
        if mid:
            counts[mid] = counts.get(mid, 0) + 1
    return counts


def render(sessions: list[Session], cwd: Path | None = None) -> str:
    cwd = cwd or Path.cwd()
    L = ["TokenSaver doctor", ""]

    merged: dict = {}
    L.append("  settings files (later wins):")
    for path in _settings_files(cwd):
        if not path.exists():
            continue
        data = _load(path)
        if not data:
            L.append(f"    {path}  (unreadable or empty)")
            continue
        merged.update(data)
        keys = [k for k in ("model", "autoCompactWindow", "promptCacheTtl") if k in data]
        L.append(f"    {path}  {', '.join(keys) if keys else '(nothing relevant)'}")
    L.append("")

    model = merged.get("model")
    if model:
        r = resolve(model)
        note = f"  <-- ASSUMED: {r.reason}" if r.assumed else ""
        L.append(f"  model setting:      {model!r} -> prices as {r.model_id}{note}")
    else:
        L.append("  model setting:      <not set> (Claude Code default)")

    window = merged.get("autoCompactWindow")
    if window is None:
        L.append(f"  autoCompactWindow:  <not set> -> compacts near the model's limit "
                 f"(~967k on 1M-context models)")
    elif not isinstance(window, int) or not (WINDOW_MIN <= window <= WINDOW_MAX):
        L.append(f"  autoCompactWindow:  {window!r}  <-- OUT OF RANGE "
                 f"({WINDOW_MIN:,}-{WINDOW_MAX:,}); Claude Code may ignore it")
    else:
        L.append(f"  autoCompactWindow:  {window:,}")
    L.append("")

    # What the alias actually resolved to, from the transcripts themselves.
    seen: dict[str, int] = {}
    assumed: dict[str, str] = {}
    unpriced: dict[str, str] = {}
    for sess in sessions:
        for req in sess.requests:
            seen[req.model] = seen.get(req.model, 0) + 1
    L.append("  models actually used (from transcripts):")
    for mid, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        r = resolve(mid)
        tag = ""
        if r.assumed:
            assumed[mid] = r.reason
            tag = "  [assumed price]"
        elif not r.billable:
            unpriced[mid] = r.reason
            tag = "  [not billed]"
        L.append(f"    {n:6,} requests  {mid or '<none>':32s}{tag}")
    L.append("")

    # Only models in the alias's own family tell us anything about how the
    # alias resolved; a sonnet or local-model session was an explicit choice.
    if model:
        fam = _family(resolve(model).model_id)
        kin = sorted({m for m in seen if m and _family(m) == fam})
        if fam and kin:
            L.append(f"  NOTE: {model!r} is a {fam} alias; the {fam} models actually seen were "
                     f"{', '.join(kin)}.")
            if len(kin) == 1 and kin[0] != resolve(model).model_id:
                L.append(f"        It has NOT resolved to {resolve(model).model_id} here - costs "
                         f"above are priced as {kin[0]} where the transcript names it.")
            L.append("        Pin an exact id in settings.json if you need a specific model.")
            L.append("")

    # Cross-check the setting against what the wire actually carried. A model
    # id this Claude Code version does not know is accepted into settings.json
    # and then silently falls back, so the setting claims one model while the
    # requests name another - and every cost figure here would be priced for
    # the wrong one. Measured: `claude-opus-5-5[1m]` on v2.1.274 resolved to
    # claude-opus-5 on all three live requests.
    observed = _observed_models()
    if observed:
        L.append("  models observed on the wire (via the proxy):")
        for mid, n in sorted(observed.items(), key=lambda kv: -kv[1]):
            L.append(f"    {n:6,} requests  {mid}")
        if model:
            claimed_res = resolve(model)
            claimed = claimed_res.model_id
            if claimed and claimed not in observed:
                # Direction matters. Pricing a cheaper model than you run
                # understates the bill and is a real error; pricing a more
                # expensive one is the conservative side of the rule in
                # pricing.py and only needs noting.
                actual = max(observed, key=observed.get)
                actual_price = resolve(actual).price
                claimed_price = claimed_res.price
                understated = (actual_price and claimed_price
                               and actual_price.read > claimed_price.read)
                if understated:
                    L.append(f"    <-- WRONG: priced as {claimed}, but requests "
                             f"used {actual}, which costs more.")
                    L.append("        Every cost figure above understates the bill. "
                             "Pin the exact id in settings.json.")
                else:
                    L.append(f"    note: {model!r} resolves to {actual} here, not the "
                             f"{claimed} this prices it as - so the figures above "
                             f"are conservative.")
                    L.append(f"        Pin {actual} or pass "
                             f"`--reprice-as {actual}` for exact costs.")
        L.append("")

    days = stale_days()
    flag = "  <-- RECHECK" if days > STALE_AFTER_DAYS else ""
    L.append(f"  prices verified:    {VERIFIED} ({days} days ago){flag}")
    if days > STALE_AFTER_DAYS:
        L.append("        A stale price row silently invalidates every cost number.")
        L.append("        Recheck platform.claude.com/docs/en/about-claude/pricing.")

    for mid, reason in assumed.items():
        L.append(f"  assumed  {mid}: {reason}")
    for mid, reason in unpriced.items():
        L.append(f"  excluded {mid}: {reason}")
    L.extend(_proxy_lines())
    return "\n".join(L)


def _proxy_lines() -> list[str]:
    """Is the proxy running, current, rewriting, and is the vault bounded?"""
    from . import background, config, proxy, routing, service, vault
    L = ["", "  proxy:"]
    from . import names
    if vault.VAULT_DIR == names.HOME_OLD:
        L.append("    NOT MIGRATED from the old name: run `tokensaver migrate` with no Claude Code session "
                 "open (until then the old service and ~/.everytoken stay in use)")
    h = service.health()
    if not h:
        L.append(f"    not answering on port {config.get('port')}"
                 + (" (service installed: `tokensaver restart`)" if service.installed() else
                    " (not installed: `tokensaver setup`)"))
    else:
        stale = "  <-- RESTART: code on disk changed" if h.get("code") != proxy.code_id() else ""
        L.append(f"    running pid {h.get('pid')}, rewrite={h.get('rewrite')}, bg_compact={h.get('bg_compact')}, "
                 f"audit={'on' if (h.get('audit') or {}).get('on') else 'off'}{stale}")
        if (h.get("audit") or {}).get("disabled"):
            L.append(f"    audit disabled: {h['audit']['disabled']}")
    slot = ((config.load().get("settings") or {}).get("bg_compact") or {}).get("user") or {}
    if not config.get("bg_compact") and str(slot.get("why", "")).startswith("quality signals"):
        L.append(f"    COMPACTION TURNED ITSELF OFF ({slot.get('at')}): {slot['why']}")
        for e in (h.get("strain") or [])[-3:]:
            L.append(f"    eased off {time.strftime('%m-%d %H:%M', time.localtime(e.get('t', 0)))}: "
                     f"{e.get('why')}{' (' + e['action'] + ')' if e.get('action') else ''}")
    try:
        from . import prune
        t, why = prune.tier()
        inst = prune.installed()
        mem = (config.section("prune").get("memory") or {})
        L.append(f"    smart prune: tier {t} ({why}); judge {config.get('prune_judge')} "
                 f"{'installed' if prune._name(config.get('prune_judge'), inst) else 'NOT installed'}, "
                 f"delete at p >= {config.get('prune_min_p')}")
        L.append(f"    final check on pruned compactions: "
                 + (f"{background.LOCAL_MODEL}" if prune._name(background.LOCAL_MODEL, inst) else
                    "none (no 35B: fast-prune results apply unchecked; compaction itself needs no model)"))
        L.append(f"    ease off when the machine strains: "
                 f"{'on' if config.get('bg_yield') else 'OFF: never pauses (config bg_yield)'}"
                 + (f"; measured model memory: " + ", ".join(f"{k} {v / 2**30:.1f} GB" for k, v in mem.items())
                    if mem else ""))
    except Exception:
        pass
    fb = config.section("fallback")
    if fb.get("global_off"):
        L.append(f"    REWRITING OFF by automatic fallback ({fb.get('at')}): {fb.get('why')}")
    L.append(f"    service: {'installed' if service.installed() else 'not installed'} ({service.platform()}); "
             f"interpreter {service.python()}"
             + ("" if Path(service.python()).exists() else "  <-- MISSING: run setup again"))
    rs = routing.state()
    if rs.get("managed"):
        L.append(f"    routing: settings.json env {'ON' if routing.is_on() else 'off'} "
                 f"({rs.get('why')}, {time.strftime('%m-%d %H:%M', time.localtime(rs.get('at', 0)))}; "
                 f"{rs.get('flips', 0)} watchdog flips)")
    else:
        L.append("    routing: shell aliases only (claude-ts); IDE sessions go direct")
    for o in routing.overridden():
        L.append(f"    routing overridden here by {o}")
    try:
        size = sum(f.stat().st_size for f in vault.VAULT_DIR.rglob("*") if f.is_file())
        L.append(f"    vault: {size / 2**20:,.0f} MB (cap {config.get('vault_max_mb'):,} MB of request bodies, "
                 f"{config.get('vault_keep_days')} days)")
    except OSError:
        pass
    for name, value, source, note in config.report():
        if note:
            L.append(f"    setting {name}={value} ({source}) is {note}")
    return L
