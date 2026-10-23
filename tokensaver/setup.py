"""One-command setup and uninstall, sized to the machine, shown before it is done.

What it sets up: the proxy (forward rewriting, background compaction, free
token counts) as a background service; a watchdog that routes Claude Code
around the proxy if it stays down; and routing, by default through the `env`
block of ~/.claude/settings.json, so every Claude Code launch (terminal and
IDE extensions) goes through it with no wrapper. Background compaction is
rules only and needs no model, so every machine gets it. `--with-prune` adds
local models (ollama): Nimble for the fast prune (24 GB+), and on a machine
that fits it (`background.gate_model_for`) the 35B for the deep prune and the
one check before a pruned compaction is used. `--route alias` keeps the
older shell-alias route (`claude-ts`).

Rules:
* Nothing outside this repo changes until the whole plan has been shown, and
  it is confirmed once. Without a terminal and without `--yes`, setup prints
  the plan and exits 3, so a Claude session running it shows the plan and asks.
* Every step is idempotent: running setup again reports "already".
* Uninstall removes only what setup marked, and puts back anything it replaced.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import claude_settings, config, vault

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "tokensaver-claude"
CLI = REPO / "tokensaver-cli"
RC_BEGIN = "# >>> TokenSaver >>> (managed by `tokensaver-cli setup`; removed by `tokensaver-cli uninstall`)"
RC_END = "# <<< TokenSaver <<<"
SETUP_WHY = "setup"
SETTINGS = claude_settings.SETTINGS
STATUSLINE = REPO / "tokensaver-statusline"
STATUS_ORIG = vault.VAULT_DIR / "statusline-orig"
# The badge's % grows while the developer is idle (a compaction becoming
# ready), so the status line must re-run on a timer, not only on events.
STATUS_REFRESH_S = 10
STATUS_REFRESH_ADDED = vault.VAULT_DIR / "statusline-refresh-added"


def _our_status_command() -> str:
    return f"bash {shlex.quote(str(STATUSLINE))}"


def _read_settings() -> dict | None:
    """settings.json as a dict; None if it exists but will not parse (then we
    must not touch it at all)."""
    return claude_settings.read_json(SETTINGS)


def _write_settings(data: dict) -> None:
    claude_settings.write_json(SETTINGS, data)


def statusline_step(mode: str) -> Step:
    data = _read_settings()
    if data is None:
        return Step("statusbar", "skip", f"{SETTINGS} does not parse; not touching it")
    current = data.get("statusLine") or {}
    ours = _our_status_command()
    if current.get("command") == ours and "refreshInterval" in current:
        return Step("statusbar", "already", "status bar shows ⚡ and the % saved in TokenSaver sessions")
    if current.get("command") == ours:
        note = f"adds a {STATUS_REFRESH_S} s refresh so the % saved updates while you are idle"
    elif current.get("command"):
        note = (f"keeps your status line ({current.get('command')}) and adds ⚡ with the % saved after it, "
                f"refreshed every {STATUS_REFRESH_S} s")
    else:
        note = f"adds ⚡ with the % saved (you have no status line now), refreshed every {STATUS_REFRESH_S} s"

    def apply():
        d = _read_settings()
        if d is None:
            raise RuntimeError(f"{SETTINGS} does not parse")
        cur = d.get("statusLine") or {}
        if cur.get("command") and cur.get("command") != ours:
            STATUS_ORIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            STATUS_ORIG.write_text(cur["command"])
        new = dict(cur, type="command", command=ours)
        if "refreshInterval" not in cur:        # the user's own interval wins
            new["refreshInterval"] = STATUS_REFRESH_S
            STATUS_REFRESH_ADDED.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            STATUS_REFRESH_ADDED.write_text(str(STATUS_REFRESH_S))
        d["statusLine"] = new
        _write_settings(d)
    if mode == "no":
        return Step("statusbar", "skip", "status bar badge not added (--statusbar no)")
    return Step("statusbar", "todo" if mode == "yes" else "ask",
                f"status bar in {SETTINGS}: {note}", apply,
                ["shows nothing in sessions that do not go through tokensaver"])


def _restore_statusline() -> str | None:
    data = _read_settings()
    if not data or (data.get("statusLine") or {}).get("command") != _our_status_command():
        return None
    if STATUS_REFRESH_ADDED.exists():
        data["statusLine"] = {k: v for k, v in data["statusLine"].items() if k != "refreshInterval"}
        STATUS_REFRESH_ADDED.unlink()
    if STATUS_ORIG.exists() and STATUS_ORIG.read_text().strip():
        data["statusLine"] = dict(data["statusLine"], command=STATUS_ORIG.read_text().strip())
        STATUS_ORIG.unlink()
        _write_settings(data)
        return "restored your status line"
    data.pop("statusLine", None)
    _write_settings(data)
    return "removed the status bar badge"


@dataclass
class Step:
    name: str
    status: str            # todo | already | skip | done | failed
    detail: str
    apply: object = None   # callable, run when status == "todo"
    notes: list = field(default_factory=list)


# ---- facts about this machine -------------------------------------------------

def _claude_json() -> dict:
    try:
        return json.loads((Path.home() / ".claude.json").read_text())
    except (OSError, ValueError):
        return {}


def billing_type() -> str:
    acct = _claude_json().get("oauthAccount")
    if isinstance(acct, dict) and acct.get("billingType"):
        return str(acct["billingType"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api_key"
    return "unknown"


def _enable_routing() -> str:
    """Point Claude Code at the proxy only once it answers: a dead port would
    fail every new session until the watchdog noticed."""
    from . import routing, service
    if service.wait_health(15) is None:
        raise RuntimeError("the proxy is not answering, so settings.json was not changed "
                           "(port in use? see `tokensaver doctor`)")
    return routing.enable("setup")


def _cloud_provider() -> str | None:
    """Bedrock / Vertex / Foundry, from the shell or any settings file Claude Code reads."""
    envs = [dict(os.environ), claude_settings.env_of(SETTINGS)] + \
        [claude_settings.env_of(p) for p in claude_settings.managed_paths()]
    for env in envs:
        for var, name in (("CLAUDE_CODE_USE_BEDROCK", "Amazon Bedrock"), ("CLAUDE_CODE_USE_VERTEX", "Google Vertex AI"),
                          ("CLAUDE_CODE_USE_FOUNDRY", "Microsoft Foundry")):
            if str(env.get(var, "")).lower() in ("1", "true", "yes"):
                return name
    return None


def _managed_base_url() -> tuple[str, str] | None:
    for p in claude_settings.managed_paths():
        url = claude_settings.env_of(p).get("ANTHROPIC_BASE_URL")
        if url:
            return str(p), str(url)
    return None


def existing_base_url() -> str | None:
    """An ANTHROPIC_BASE_URL already in use: the proxy chains to it."""
    if os.environ.get("ANTHROPIC_BASE_URL"):
        return os.environ["ANTHROPIC_BASE_URL"]
    return claude_settings.env_of(SETTINGS).get("ANTHROPIC_BASE_URL")


def _shell() -> str:
    return os.path.basename(os.environ.get("SHELL") or "zsh")


def rc_file() -> Path:
    return Path.home() / (".zshrc" if _shell() == "zsh" else ".bashrc")


def user_aliases() -> dict[str, str]:
    """The user's aliases, read from an interactive shell (read-only)."""
    try:
        out = subprocess.run([os.environ.get("SHELL") or "/bin/zsh", "-ic", "alias"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    aliases = {}
    for line in out.splitlines():
        line = line.removeprefix("alias ")
        name, sep, value = line.partition("=")
        if not sep:
            continue
        try:
            value = shlex.split(value)[0] if value[:1] in "'\"" else value
        except ValueError:
            pass
        aliases[name.strip()] = value
    return aliases


_CLAUDE_CALL = re.compile(r"(^|\s)(?:command\s+)?claude(?=\s|$)")


TRY_SUFFIX = "-ts"


def compose_aliases(aliases: dict[str, str], mode: str = "replace") -> dict[str, str]:
    """Each alias that runs the claude binary, rerouted through the launcher
    with everything else kept: env prefixes, flags. No claude alias at all
    gets a plain one.

    mode "try" leaves every existing alias alone and adds `<name>-ts` twins,
    so the user's `claude` cannot break while they try it. mode "replace"
    reroutes the originals and adds `claude-direct`, the original `claude`,
    as a permanent way around tokensaver."""
    launcher = shlex.quote(str(LAUNCHER))
    out = {}
    base = {n: v for n, v in aliases.items() if not n.endswith(TRY_SUFFIX) and n != "claude-direct"}
    for name, value in base.items():
        if "tokensaver-claude" in value or "ollama" in value:
            continue
        if _CLAUDE_CALL.search(value):
            routed = _CLAUDE_CALL.sub(lambda m: m.group(1) + launcher, value, count=1)
            out[name + TRY_SUFFIX if mode == "try" else name] = routed
    if "claude" not in base:
        out["claude" + TRY_SUFFIX if mode == "try" else "claude"] = launcher
    if mode == "replace":
        direct = base.get("claude", "command claude")
        if "tokensaver-claude" not in direct:
            out["claude-direct"] = direct if direct.startswith(("command ", "ENABLE", "env ")) or \
                "command claude" in direct else direct.replace("claude", "command claude", 1)
    return out


def rc_block(composed: dict[str, str]) -> str:
    lines = [RC_BEGIN]
    for name, value in sorted(composed.items()):
        lines.append(f"alias {name}={shlex.quote(value)}")
    lines.append(RC_END)
    return "\n".join(lines) + "\n"


def _strip_block(text: str) -> str:
    # also the blank line setup put in front of the block
    return re.sub(r"\n?" + re.escape(RC_BEGIN) + r".*?" + re.escape(RC_END) + r"\n?", "", text, flags=re.S)


# ---- ollama and the gate model -------------------------------------------------

def _ollama_base() -> str:
    return config.get("ollama_url").split("/api/")[0]


def ollama_models() -> list[str] | None:
    """Installed ollama model names, or None when ollama is not answering."""
    try:
        with urllib.request.urlopen(_ollama_base() + "/api/tags", timeout=2) as resp:
            return [m.get("name") for m in json.loads(resp.read()).get("models") or []]
    except Exception:
        return None


def ollama_step(model: str | None) -> Step:
    """Install ollama if needed (for `--with-prune`), and pull the 35B when
    this machine can run it (deep prune and its check); Nimble is pulled by
    `prune_step`."""
    from . import machine, service
    names = ollama_models()
    have_bin = bool(shutil.which("ollama"))
    fits = (f"the {model} for the deep prune and its check" if model else
            f"no 35B (it needs a 48 GB+ Mac; this is {machine.ram_bytes() / 2 ** 30:.0f} GB, "
            f"{service.platform()}): fast prune only")
    if names is not None and (model is None or model in names):
        return Step("ollama", "already", f"ollama running; {fits}")
    actions, notes = [], []
    if not have_bin:
        if service.platform() == "macos" and shutil.which("brew"):
            actions.append(["brew", "install", "ollama"])
            notes.append("installs ollama with Homebrew and starts it as a login service")
        elif service.platform() == "linux":
            return Step("ollama", "skip", "ollama is not installed. Install it (needs sudo): "
                        "`curl -fsSL https://ollama.com/install.sh | sh`, then run setup again; until then "
                        "no pruning (compaction still runs)")
        else:
            return Step("ollama", "skip", "ollama is not installed: get it from https://ollama.com/download, "
                        "then run setup again; until then no pruning (compaction still runs)")

    def apply():
        for cmd in actions:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd)}: {r.stderr[-300:]}")
        if ollama_models() is None:
            if shutil.which("brew") and service.platform() == "macos":
                subprocess.run(["brew", "services", "start", "ollama"], capture_output=True, timeout=60)
            else:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
            for _ in range(40):
                if ollama_models() is not None:
                    break
                time.sleep(0.5)
        if model is None:
            return
        r = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=3 * 3600)
        if r.returncode != 0:
            raise RuntimeError(f"`ollama pull {model}` failed: {r.stderr[-300:]}")
    what = f"download {model} (~22 GB) for the deep prune and its check" if model else "start ollama"
    return Step("ollama", "todo", f"{what} (local, free); {fits}" if model else f"{what}; {fits}", apply, notes)


# ---- the plan -------------------------------------------------------------------

def plan(route: str = "settings", shell_rc: str = "yes", audit: str | None = None, service: bool = True,
         with_hooks: bool = False, aliases: str = "try", statusbar: str = "yes",
         with_prune: bool = False) -> list[Step]:
    from . import background, machine, routing
    from . import service as svc
    steps: list[Step] = []
    notes = []
    if sys.version_info < (3, 9):
        return [Step("preflight", "failed", f"Python 3.9+ needed, have {platform.python_version()}")]
    if not shutil.which("claude"):
        return [Step("preflight", "failed", "`claude` is not on PATH")]
    cloud = _cloud_provider()
    if cloud:
        return [Step("preflight", "skip", f"{cloud} is in use: Claude Code does not send "
                                          "API traffic through ANTHROPIC_BASE_URL, so the proxy cannot help")]
    managed = _managed_base_url()
    if managed and route == "settings":
        return [Step("preflight", "skip", f"your organisation's managed settings ({managed[0]}) set "
                                          f"ANTHROPIC_BASE_URL={managed[1]}, which wins over yours: TokenSaver "
                                          "would never be used. Nothing was changed.")]
    if service and not svc.supported():
        return [Step("preflight", "failed", f"no background service support on {sys.platform} (macOS and Linux only)")]
    upstream = existing_base_url()
    local = f"127.0.0.1:{config.get('port')}"
    chain = bool(upstream) and local not in upstream and "localhost" not in upstream
    if chain:
        notes.append(f"chaining to your existing ANTHROPIC_BASE_URL ({upstream})")
    if route == "settings" and os.environ.get("ANTHROPIC_BASE_URL") and local not in os.environ["ANTHROPIC_BASE_URL"]:
        notes.append(f"ANTHROPIC_BASE_URL is exported in your shell ({os.environ['ANTHROPIC_BASE_URL']}): shell "
                     "values win over settings.json, so terminal sessions would skip TokenSaver until you "
                     "remove that export (the proxy is set to chain to it)")
    if REPO != Path.home() / ".tokensaver" / "src":
        notes.append(f"installed from {REPO}: keep it there (services and settings point at it)")
    steps.append(Step("preflight", "already", f"Python {platform.python_version()} ({svc.python()}); "
                      f"claude at {shutil.which('claude')}; {svc.platform()}, "
                      f"{machine.ram_bytes() / 2 ** 30:.0f} GB memory", notes=notes))

    gate_model = background.gate_model_for(machine.ram_bytes(), svc.platform())
    billing = billing_type()
    audit_on = audit or "on"
    want = {"rewrite": "forward", "bg_compact": True, "audit": audit_on}
    if with_prune and gate_model:
        want["local_model"] = gate_model
    if chain:
        want["upstream"] = upstream
    cfg_notes = ["background compaction on (rules only, no model needed)" +
                 ("; --with-prune adds local-model pruning" if not with_prune else
                  f"; pruning with Nimble" + (f" and {gate_model} (deep prune + its check)" if gate_model else
                                              " (fast prune; no 35B fits, so unchecked)")),
                 f"free token counts {audit_on} (Anthropic's count_tokens on your own login, {billing}): "
                 "they measure the saving and decide each compaction"]
    current = {k: config.get(k) for k in want}
    todo = {k: v for k, v in want.items() if current[k] != v}

    def apply_config():
        for k, v in todo.items():
            config.set_value(k, v, "user", why=SETUP_WHY)
    steps.append(Step("config", "todo" if todo else "already",
                      ", ".join(f"{k}={v}" for k, v in want.items()), apply_config, cfg_notes))
    if with_prune:
        ollama = ollama_step(gate_model)
        steps.append(ollama)

    if with_hooks:
        from . import install
        steps.append(Step("hooks", "todo", "add TokenSaver hooks to ~/.claude/settings.json (vault features)",
                          lambda: install.install()))

    if service:
        where = (f"launchd agents in {svc.AGENTS}" if svc.platform() == "macos"
                 else f"systemd user units in {svc.UNITS}")
        detail = (f"{where}: the proxy (starts at login, restarts if it dies) and a watchdog "
                  f"that routes Claude Code around it if it stays down")
        unsafe = svc.stopping_proxy_is_unsafe()
        if not svc.up_to_date() and unsafe and svc.proxy_would_restart():
            steps.append(Step("service", "skip", detail, notes=[
                f"NOT from here: {unsafe}; replacing the proxy would cut this session off",
                f"run in a normal terminal: {CLI} setup"]))
        else:
            steps.append(Step("service", "already" if svc.up_to_date() else "todo", detail, svc.install))

    if route == "settings":
        on = routing.is_on()
        steps.append(Step("routing", "already" if on else "todo",
                          f"every Claude Code launch (terminal and IDE extensions) goes through TokenSaver: "
                          f"ANTHROPIC_BASE_URL and two related entries in the `env` block of {routing.SETTINGS}",
                          _enable_routing,
                          ["sessions already open pick this up too (Claude Code re-reads the file)",
                           "if the proxy crashes, it is restarted in ~5 s (open sessions retry meanwhile); "
                           "if it stays down, the watchdog takes these entries out so NEW sessions go direct "
                           "(an open session keeps the proxy address: restart it)",
                           "if TokenSaver's files are deleted without uninstalling, the watchdog still takes "
                           "these out",
                           "Remote Control is unavailable through a custom API address (Claude Code's rule)"]))
        steps.append(command_step(shell_rc))
    else:
        steps.append(_alias_step(shell_rc, aliases))

    if with_prune:
        steps.append(prune_step(ollama_first=ollama.status == "todo"))
    steps.append(statusline_step(statusbar))
    steps.append(Step("verify", "todo", "conformance suite (local, free) and the live health check",
                      lambda: verify(route)))
    return steps


BIN_DIR = Path.home() / ".local" / "bin"
PATH_LINE = 'export PATH="$HOME/.local/bin:$PATH"'


def _bin_on_path() -> bool:
    return any(Path(p).expanduser().resolve() == BIN_DIR.resolve()
               for p in os.environ.get("PATH", "").split(os.pathsep) if p)


def command_step(shell_rc: str = "yes") -> Step:
    """`tokensaver` as a terminal command: a link in ~/.local/bin, plus that
    folder on PATH through a marked block in the shell rc if it is not there
    already. The same block held the claude-ts aliases of the older alias
    route; it is replaced, the user's own lines are never touched."""
    link = BIN_DIR / "tokensaver"
    rc = rc_file()
    rc_text = rc.read_text() if rc.exists() else ""
    current = re.search(re.escape(RC_BEGIN) + r".*?" + re.escape(RC_END) + r"\n?", rc_text, re.S)
    need_path = not _bin_on_path() and not (current and PATH_LINE in current.group(0))
    want_block = f"{RC_BEGIN}\n{PATH_LINE}\n{RC_END}\n" if (need_path or (current and PATH_LINE in current.group(0))) else ""
    has_link = link.is_symlink() and Path(os.path.realpath(link)) == Path(os.path.realpath(CLI))
    if link.exists() and not has_link:
        return Step("command", "skip", f"{link} already exists and is not TokenSaver's; "
                                       f"use {CLI} directly (or remove that file and run setup again)")
    block_ok = (current.group(0) if current else "") == want_block
    if has_link and block_ok:
        return Step("command", "already", f"`tokensaver` command at {link}")
    notes = []
    if want_block and not block_ok:
        notes.append(f"adds {BIN_DIR} to PATH in {rc} (a marked block; your own lines stay)")
    if current and "alias " in current.group(0):
        notes.append("replaces the older claude-ts aliases: plain `claude` is routed now")
    if shell_rc == "no" and want_block and not block_ok:
        notes.append(f"not editing {rc}: add `{PATH_LINE}` yourself")

    def apply():
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        if not has_link:
            link.symlink_to(CLI)
        if not block_ok and shell_rc != "no":
            new = _strip_block(rc_text)
            if want_block:
                new = new.rstrip("\n") + "\n\n" + want_block
            claude_settings.write_text(rc, new, is_json=False)
    return Step("command", "todo", "`tokensaver` in your terminal (tokensaver uninstall, tokensaver --benchmark, "
                                   "tokensaver update, ...)", apply, notes)


def _alias_step(shell_rc: str, aliases: str) -> Step:
    composed = compose_aliases(user_aliases(), aliases)
    block = rc_block(composed)
    rc = rc_file()
    rc_text = rc.read_text() if rc.exists() else ""
    current_block = re.search(re.escape(RC_BEGIN) + r".*?" + re.escape(RC_END) + r"\n?", rc_text, re.S)
    rc_notes = [f"  {line}" for line in block.splitlines()[1:-1]]
    if aliases == "try":
        rc_notes.append("your existing aliases are NOT changed: try the -ts ones; switch later with --aliases replace")
    rc_notes.append("IDE extensions do not read shell aliases; they keep using Claude Code directly")
    rc_notes.append("way out, any time: `command claude` (plain binary) or `tokensaver uninstall`")
    if current_block and current_block.group(0) == block:
        return Step("shell", "already", f"aliases in {rc}", notes=rc_notes)
    if shell_rc == "no":
        return Step("shell", "skip", f"not editing {rc}; add these yourself:", notes=rc_notes)

    def apply_rc():
        new = _strip_block(rc_text).rstrip("\n") + "\n\n" + block
        claude_settings.write_text(rc, new, is_json=False)
    return Step("shell", "todo", f"{'replace' if current_block else 'append'} a marked block at the end of {rc}",
                apply_rc, rc_notes)


def prune_step(ollama_first: bool = False) -> Step:
    """The decision model smart pruning uses (`--with-prune`, ~9 GB; ~0.5-1% more)."""
    from . import machine, prune
    judge = config.get("prune_judge")
    inst = prune.installed()
    if not inst and not ollama_first and ollama_models() is None:
        return Step("prune", "skip", "smart prune needs ollama (not running)")
    if prune._name(judge, inst):
        t, why = prune.tier()
        return Step("prune", "already", f"smart prune: tier {t} ({why})")
    ram = machine.ram_bytes()
    if ram and ram < 24 * 2 ** 30:        # Nimble measured ~16 GB loaded, plus 6 GB headroom
        return Step("prune", "skip", f"smart prune needs 24 GB+ of memory; this machine has {ram / 2**30:.0f} GB")

    def pull():
        r = subprocess.run(["ollama", "pull", judge], capture_output=True, text=True, timeout=3 * 3600)
        if r.returncode != 0:
            raise RuntimeError(f"`ollama pull {judge}` failed: {r.stderr[-300:]}")
    return Step("prune", "todo", f"download {judge} (~9 GB, Apache-2.0) so idle compactions also drop "
                                 f"superseded history (eases off when the machine strains)", pull)


def verify(route: str = "settings") -> None:
    from . import service as svc
    r = subprocess.run([sys.executable, str(REPO / "experiments" / "proxy_conformance.py")],
                       capture_output=True, text=True, timeout=600)
    if "all conformance checks passed" not in r.stdout:
        config.set_value("rewrite", "off", "user", why="setup: conformance failed")
        raise RuntimeError("conformance failed; rewriting turned off. Output:\n" + r.stdout[-2000:])
    # The service step already started it; restarting here would only cut off
    # a session routed through it. Wait for it to answer instead.
    if svc.installed() and svc.wait_health(20) is None:
        raise RuntimeError(f"the service did not answer; see {vault.VAULT_DIR / 'proxy.log'}")
    # Claude Code itself must start. `--version` makes no API call.
    cmd = ["claude", "--version"] if route == "settings" else [str(LAUNCHER), "--version"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, TOKENSAVER_AUTOSTART="0"))
    if r.returncode != 0 or "Claude Code" not in r.stdout:
        raise RuntimeError(f"Claude Code did not start: {r.stdout[-300:]} {r.stderr[-300:]}")


def render(steps: list[Step]) -> str:
    lines = []
    for s in steps:
        mark = {"todo": "will do", "already": "already", "skip": "skipped",
                "done": "done", "failed": "FAILED"}.get(s.status, s.status)
        lines.append(f"  [{mark:8s}] {s.name:9s} {s.detail}")
        for n in s.notes:
            lines.append(f"             {n}")
    return "\n".join(lines)


def run(dry_run: bool = False, yes: bool = False, route: str = "settings", shell_rc: str = "yes",
        audit: str | None = None, service: bool = True, with_hooks: bool = False, aliases: str = "try",
        statusbar: str = "yes", with_prune: bool = False, refresh: bool = False) -> int:
    """Show the whole plan, ask once, apply. Without a terminal and without
    --yes, print the plan and exit 3 (a Claude session shows it and asks).
    `refresh` (after `update`): the recorded install again, no questions, and
    the proxy restarted so it runs the new code."""
    if refresh:
        rec = config.section("install") or {}
        route, yes, with_prune = rec.get("route") or route, True, bool(rec.get("with_prune"))
    steps = plan(route=route, shell_rc=shell_rc, audit=audit, service=service, with_hooks=with_hooks,
                 aliases=aliases, statusbar=statusbar, with_prune=with_prune)
    print("TokenSaver setup\n")
    print(render(steps))
    if any(s.status == "failed" for s in steps) or (steps and steps[0].status == "skip"):
        return 1
    if dry_run:
        print("\n(dry run: nothing changed)")
        return 0
    if not yes and not sys.stdin.isatty():
        print("\nNothing changed. Show this plan to the user, then rerun with --yes.")
        return 3
    if not yes and input("\nApply all of the above? [y/N] ").strip().lower() not in ("y", "yes"):
        print("nothing changed")
        return 1
    print()
    for s in steps:
        if s.status != "todo":
            continue
        try:
            s.apply()
            s.status = "done"
            print(f"  [done    ] {s.name}")
        except Exception as exc:
            s.status = "failed"
            print(f"  [FAILED  ] {s.name}: {exc}")
            return 1
    from . import service as svc
    config.set_section("install", {"repo": str(REPO), "python": svc.python(), "route": route,
                                   "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "service": svc.installed(), "rc": str(rc_file()) if route == "alias" else None,
                                   "with_prune": with_prune})
    if refresh:
        print("proxy restarted on the new code" if svc.restart() else
              "the proxy did not come back after the update (the watchdog keeps Claude Code working direct); "
              "see `tokensaver doctor`")
        return 0
    if route == "settings":
        print("\nDone. Claude Code (terminal and IDE, sessions already open included) now goes through")
        print("TokenSaver. Just use `claude` as usual. Open a new terminal for the `tokensaver` command.")
    else:
        use = "`claude-ts` (your `claude` is unchanged)" if aliases == "try" else "`claude` as usual"
        print("\nDone. Open a new terminal (or `source " + str(rc_file()) + "`), then use " + use + ".")
    print("See what it saves: the ⚡ in the status line, `tokensaver proxy --stats`, `tokensaver doctor`.")
    print("Remove it completely: `tokensaver uninstall`.")
    return 0


def _mcp_remove() -> str | None:
    claude = shutil.which("claude")
    if not claude:
        return None
    got = subprocess.run([claude, "mcp", "get", "tokensaver"], capture_output=True, text=True, timeout=60)
    if got.returncode != 0 or "tokensaver-mcp" not in got.stdout:
        return None                     # not registered, or someone else's server of that name
    subprocess.run([claude, "mcp", "remove", "tokensaver", "-s", "user"], capture_output=True, timeout=60)
    return "removed the tokensaver MCP server from Claude Code"


def uninstall(purge: bool | None = None, yes: bool = False) -> int:
    """Undo everything setup did. Each step runs even if an earlier one failed,
    and routing comes out first: whatever else goes wrong, Claude Code then
    talks to Anthropic directly. Only TokenSaver's own entries are removed;
    anything we replaced is put back. `purge` (asked when None and on a
    terminal) also deletes ~/.tokensaver: the vault, settings and logs."""
    from . import install, migrate, routing
    from . import service as svc
    done: list[str] = []

    def step(what, fn):
        try:
            out = fn()
            if isinstance(out, list):
                done.extend(out)
            elif out:
                done.append(out)
        except Exception as exc:
            done.append(f"FAILED to {what}: {type(exc).__name__}: {exc}")

    def routing_off():
        msg = routing.disable("uninstall", forget=True)
        return None if msg == "nothing of ours to remove" else msg

    def rc_block():
        rc = rc_file()
        if rc.exists() and RC_BEGIN in rc.read_text():
            claude_settings.write_text(rc, _strip_block(rc.read_text()), is_json=False)
            return f"removed TokenSaver's block from {rc} (backup kept)"

    def command_link():
        link = BIN_DIR / "tokensaver"
        if link.is_symlink() and Path(os.path.realpath(link)) == Path(os.path.realpath(CLI)):
            link.unlink()
            return f"removed the `tokensaver` command ({link})"

    def config_reset():
        data = config.load(force=True)
        for name, entry in (data.get("settings") or {}).items():
            if (entry.get("user") or {}).get("why", "").startswith(SETUP_WHY):
                config.unset(name, "user")
        config.set_section("install", {})

    step("update entries written before the rename", lambda: (migrate.rewrite_names(), None)[1])
    step("take routing out of settings.json", routing_off)
    step("stop and remove the service", svc.remove)
    step("remove the old-name service", svc.remove_old)
    step("restore the status line", _restore_statusline)
    step("remove the hooks", lambda: (lambda m: None if m.startswith("nothing to remove") else m)(install.uninstall()))
    step("remove the MCP server", _mcp_remove)
    step("remove the shell block", rc_block)
    step("remove the `tokensaver` command", command_link)
    step("reset setup's settings", config_reset)

    def originals():
        # Byte for byte what was there before, when nothing but our entries changed
        # (our edits re-indent JSON; the user's own formatting comes back).
        def same_json(a, b):
            return (json.loads(a) if a.strip() else {}) == json.loads(b)
        out = [claude_settings.restore_original(SETTINGS, same_json),
               claude_settings.restore_original(rc_file(), lambda a, b: a.rstrip("\n") == b.rstrip("\n"))]
        return [x for x in out if x]
    step("put files back exactly as they were", originals)
    for line in done:
        print("  " + line)
    failed = any(line.startswith("FAILED") for line in done)
    home = vault.VAULT_DIR
    if svc.LEFT_RUNNING:
        # The proxy still serving open sessions writes its state there: deleting
        # it now would only have it recreated. After logout it is safe.
        if purge:
            print(f"  not deleting {home} yet: the proxy is still running for open sessions. "
                  f"After you log out: rm -rf {home}")
        purge = False
    elif purge is None:
        purge = yes is False and sys.stdin.isatty() and input(
            f"\nAlso delete {home} (the vault of recorded sessions, settings, logs)? [y/N] "
        ).strip().lower() in ("y", "yes")
    if purge and home.exists():
        inside = Path(os.path.realpath(REPO)).is_relative_to(Path(os.path.realpath(home)))
        shutil.rmtree(home, ignore_errors=True)
        old = Path.home() / ".everytoken"
        if old.is_symlink():
            old.unlink()
        print(f"  deleted {home}" + (" (including this copy of TokenSaver)" if inside else ""))
    elif not svc.LEFT_RUNNING:
        print(f"  kept {home} (your recorded sessions); delete it any time")
    print("  left in place: ollama and any models it pulled (`ollama rm <model>` if unused)")
    if not purge and Path(os.path.realpath(REPO)) != Path(os.path.realpath(home / "src")):
        print(f"  the code itself is still at {REPO}; delete that folder to finish")
    print("\nTokenSaver is removed" + (" (with the failures above)" if failed else "") +
          ". New Claude Code sessions talk to Anthropic directly"
          + ("; sessions open now keep using the proxy until you log out." if svc.LEFT_RUNNING else "."))
    return 1 if failed else 0
