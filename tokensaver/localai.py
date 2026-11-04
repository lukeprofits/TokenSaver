"""Local AI: models on this machine that trim a little more during breaks.

Everything else in TokenSaver needs no model. On a machine that can run them,
two local models (through ollama, free) add the prune stages of background
compaction (see `proxy._prune_step`): Nimble drops pieces a later piece
replaced, and on bigger machines the 35B finds more and runs the one final
check. About 0.5-1% more saving.

Default `local_ai` = auto: a machine gets them only if it passes both checks:
- specs: enough memory for the model (`TIERS`: RAM on an Apple-silicon Mac,
  the graphics card's own memory on Linux; none on an Intel Mac or a Linux
  machine without a usable card, where they would be too slow);
- disk: more than DISK_MIN free where ~/.tokensaver lives (skipped when the
  models are already there).
A machine that fails gets nothing, not even ollama. `on` skips the disk check
(never the specs check); `off` downloads nothing and stops our ollama.

Install stays instant: `start()` launches `sync()` in the background
(detached, low priority, one at a time). It uses an ollama already running on
:11434 if there is one; otherwise it downloads ollama's own standalone release
(pinned, checked against the release's sha256 list) into ~/.tokensaver/ollama
and runs it as a background service on its own port (no sudo, no Homebrew,
nothing system-wide). Then it pulls the models; ollama resumes partial pulls,
and the archive download resumes too. The watchdog restarts an unfinished
sync (sleep, offline, logout). The proxy needs nothing new: `prune.tier()`
turns on once the models are installed, at the next break.

Progress and the last decision are in ~/.tokensaver/local_ai.json, shown by
`tokensaver status`, `tokensaver local-ai` and doctor.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, machine, vault

GB = 2 ** 30
DISK_MIN = 100 * GB                 # "more than 100 GB free" (the user's rule)
OLLAMA_VERSION = "v0.40.14"         # tested with TokenSaver; bump after testing a newer one
RELEASE = f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/"
SYSTEM_URL = "http://127.0.0.1:11434"
OWN_PORT = 11435
DIR = vault.VAULT_DIR / "ollama"
BIN = DIR / "bin" / "ollama"
MODELS = DIR / "models"
STATE = vault.VAULT_DIR / "local_ai.json"
LOCK = vault.VAULT_DIR / "local_ai.lock"
RETRY_S = 600                       # the watchdog restarts an unfinished sync at most this often

# Per platform, smallest first: (role, ollama tag, download bytes, memory it needs in GB).
# Mac: unified memory; Nimble measured 17.7 GB loaded, the 35B 28.5 GB (prune.MEMORY_FACTOR).
# Linux: the GGUF builds of the same models, on the graphics card's memory.
TIERS = {
    "macos": [("judge", "nimble", int(9.3e9), 24), ("check", "qwen3.6:35b-mlx", int(21.9e9), 48)],
    "linux": [("judge", "nimble:9b-q8_0", int(9.5e9), 16), ("check", "qwen3.6:35b", int(23e9), 32)],
}
# A Linux build is offered only once its quality matched the Mac build on real breaks
# (experiments/gate_models.py for the check model, smart_prune.py for the judge).
LINUX_MEASURED = {"judge": False, "check": False}
# TokenSaver's own ollama (downloaded, run as a service) is used only once that path ran for real
# on the platform: the pinned build starts, answers on OWN_PORT and loads these models. Until then
# a machine without its own ollama gets no local AI (nothing downloaded) and is told why.
OWN_OLLAMA_TESTED = {"macos": False, "linux": False}
ROLE_NAMES = {"judge": "Nimble", "check": "35B"}
ROLE_CONFIG = {"judge": "prune_judge", "check": "local_model"}


def _platform_name() -> str:
    return "macos" if sys.platform == "darwin" else "linux" if sys.platform.startswith("linux") else sys.platform


def _arch() -> str:
    m = _platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "amd64" if m in ("x86_64", "amd64") else m


def own_url() -> str:
    return f"http://127.0.0.1:{OWN_PORT}"


def _tags(base: str, timeout: float = 2.0) -> list[str] | None:
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=timeout) as r:
            return [m.get("name") for m in json.loads(r.read()).get("models") or []]
    except Exception:
        return None


def _has(tag: str, names) -> bool:
    return any(n == tag or n == f"{tag}:latest" for n in names or [])


def _own_has(tag: str) -> bool:
    """A model in our ollama's folder, read from disk (works while its service is stopped)."""
    name, _, version = tag.partition(":")
    return (MODELS / "manifests" / "registry.ollama.ai" / "library" / name / (version or "latest")).is_file()


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(st: dict) -> None:
    try:
        STATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(STATE)
    except OSError:
        pass


def _update(**kw) -> dict:
    st = load_state()
    st.update(kw, at=time.time())
    _save_state(st)
    return st


# ---- the decision -------------------------------------------------------------------

def decide(mode: str | None = None) -> dict:
    """What this machine gets. Never downloads anything.
    {mode, models: [(role, tag, bytes)], missing: [tag], ollama: system|ours|None,
     need_bytes, skip: reason or None}"""
    mode = mode or config.get("local_ai")
    plat = _platform_name()
    out = {"mode": mode, "models": [], "missing": [], "ollama": None, "need_bytes": 0, "skip": None,
           "platform": plat}
    if mode == "off":
        out["skip"] = "you turned it off (`tokensaver local-ai auto` to allow it)"
        return out
    tiers = TIERS.get(plat)
    if not tiers:
        out["skip"] = f"not available on {plat}"
        return out
    mem = machine.model_memory_bytes()
    if plat == "macos" and _arch() != "arm64":
        out["skip"] = "needs an Apple-silicon Mac (on an Intel Mac the models are too slow)"
        return out
    if plat == "linux" and not mem:
        out["skip"] = ("needs a graphics card with its own memory and its driver installed "
                       "(on the CPU alone the models are too slow)")
        return out
    what = "graphics-card memory" if plat == "linux" else "memory"
    fit = [(r, t, b) for r, t, b, need in tiers if mem >= need * GB]
    if not fit:
        out["skip"] = f"{mem / GB:.0f} GB {what}; needs {tiers[0][3]} GB+"
        return out
    if plat == "linux":
        fit = [x for x in fit if LINUX_MEASURED[x[0]]]
        if not fit:
            out["skip"] = "not available on Linux yet (its model builds are still being measured)"
            return out
    out["models"] = fit
    if BIN.exists() or config.get("ollama_url").startswith(own_url()):
        out["ollama"] = "ours"
        out["missing"] = [t for _, t, _ in fit if not _own_has(t)]
    else:
        names = _tags(SYSTEM_URL)
        out["ollama"] = "system" if names is not None else "ours"
        if out["ollama"] == "ours" and not OWN_OLLAMA_TESTED.get(plat):
            out["skip"] = ("needs ollama for now: install it from ollama.com and run `tokensaver update` "
                           "(TokenSaver's own copy is still being tested on this platform)")
            return out
        out["missing"] = [t for _, t, _ in fit if not _has(t, names)]
    need = sum(b for _, t, b in fit if t in out["missing"])
    if out["ollama"] == "ours" and not BIN.exists():
        need += _archive_bytes(plat)
    out["need_bytes"] = need
    if out["missing"] and mode == "auto":
        free = machine.disk_free_bytes(vault.VAULT_DIR)
        if free is None or free <= DISK_MIN:
            out["skip"] = (f"{(free or 0) / GB:.0f} GB free disk; needs more than {DISK_MIN // GB} GB "
                           "(`tokensaver local-ai on` to download anyway)")
            return out
    if out["ollama"] == "ours" and not BIN.exists() and plat == "linux" and not _can_unzstd():
        out["skip"] = "needs `zstd` to unpack ollama (install it with your package manager, then `tokensaver update`)"
    return out


def describe(d: dict | None = None) -> str:
    """One plain line: what local AI is doing on this machine."""
    d = d or decide()
    st = load_state()
    if d["skip"]:
        return f"skipped ({d['skip']})" if d["mode"] != "off" else "off (you turned it off)"
    names = " + ".join(ROLE_NAMES[r] for r, _, _ in d["models"])
    if not d["missing"]:
        return f"on ({names}" + (", using your ollama)" if d["ollama"] == "system" else ")")
    if running():
        done, total = st.get("done_bytes") or 0, st.get("total_bytes") or d["need_bytes"]
        return f"downloading ({done / GB:.1f} of {total / GB:.0f} GB): TokenSaver works without it meanwhile"
    if st.get("error"):
        return f"download stopped ({st['error']}); retries by itself"
    return f"waiting to download {names} ({d['need_bytes'] / GB:.0f} GB) in the background"


def configure(d: dict, dry: bool = False) -> list[str]:
    """Point the prune stages at this machine's ollama and model tags (the `tuned`
    slot, so a user's own setting still wins). Cheap; changes only what differs.
    Returns the names that differ(ed)."""
    if d["skip"] or not d["models"]:
        return []
    base = own_url() if d["ollama"] == "ours" else SYSTEM_URL
    want = {"ollama_url": base + "/api/generate"}
    want.update({ROLE_CONFIG[r]: t for r, t, _ in d["models"]})
    differ = [n for n, v in want.items() if config.get(n) != v and config.resolve(n)[1] not in ("user", "env")]
    if not dry:
        for name in differ:
            config.set_value(name, want[name], "tuned", why="local AI setup")
    return differ


# ---- the background download ------------------------------------------------------

def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def running() -> bool:
    return _pid_alive(load_state().get("pid"))


def start(cli: str | None = None) -> bool:
    """Start `sync` detached if this machine wants local AI and it is unfinished.
    Returns whether one was started. Never raises, never waits."""
    try:
        if running():
            return False
        d = decide()
        if d["skip"] or not d["missing"] and (d["ollama"] != "ours" or _service_ok()):
            return False
        from . import service
        cli = cli or str(service.CLI)
        vault.VAULT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        log = open(vault.VAULT_DIR / "local_ai.log", "a")
        subprocess.Popen([service.python(), cli, "local-ai", "--sync"], stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        _update(started=time.time(), error=None, want=[t for _, t, _ in d["models"]], finished=False)
        return True
    except Exception:
        return False


def kick(now: float | None = None) -> str | None:
    """For the watchdog: restart an unfinished sync, at most every RETRY_S."""
    st = load_state()
    if not st.get("want") or st.get("finished") or running():
        return None
    now = now or time.time()
    if now - float(st.get("started") or 0) < RETRY_S:
        return None
    return "local AI download restarted" if start() else None


def _lock():
    import fcntl
    LOCK.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    f = open(LOCK, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def sync(log=print) -> int:
    """Install what `decide()` says, in this process (run detached by `start`)."""
    lock = _lock()
    if lock is None:
        log("another local AI download is running")
        return 0
    try:
        os.nice(10)
    except OSError:
        pass
    d = decide()
    _update(pid=os.getpid(), want=[t for _, t, _ in d["models"]], finished=False, error=None,
            skip=d["skip"], total_bytes=d["need_bytes"], done_bytes=0)
    if d["skip"]:
        _update(pid=None, finished=True)
        log(f"local AI: skipped ({d['skip']})")
        return 0
    try:
        try:
            _install_models(d, log)
        except RuntimeError as exc:
            if d["ollama"] != "system" or "newer version" not in str(exc):
                raise
            # The ollama already on this machine is too old for this model: use our own
            # (pinned, current) one instead, and leave the user's untouched.
            log("your ollama is too old for these models; installing TokenSaver's own")
            d = dict(d, ollama="ours", missing=[t for _, t, _ in d["models"] if not _own_has(t)])
            _install_models(d, log)
        _update(pid=None, finished=True, done_bytes=d["need_bytes"])
        log("local AI ready")
        try:
            vault.record("local_ai", tool="setup", index=False, summary="local AI ready",
                         meta={"models": [t for _, t, _ in d["models"]], "ollama": d["ollama"]})
        except Exception:
            pass
        return 0
    except Exception as exc:
        _update(pid=None, error=str(exc)[:200])
        log(f"local AI download stopped: {exc}")
        return 1
    finally:
        lock.close()


def _install_models(d: dict, log) -> None:
    if d["ollama"] == "ours":
        install_ollama(log)
    configure(d)
    base = own_url() if d["ollama"] == "ours" else SYSTEM_URL
    done_before = 0
    for _, tag, size in d["models"]:
        if tag in d["missing"]:
            log(f"pulling {tag}")
            pull(base, tag, lambda done, before=done_before: _update(done_bytes=before + done))
            done_before += size
            _update(pulled=sorted(set(load_state().get("pulled") or []) | {tag}))


def pull(base: str, tag: str, progress=lambda done: None) -> None:
    """ollama's /api/pull, streamed; ollama keeps partial layers, so a retry resumes."""
    req = urllib.request.Request(base + "/api/pull", data=json.dumps({"model": tag}).encode(),
                                 headers={"Content-Type": "application/json"})
    layers: dict = {}
    last = 0.0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("error"):
                raise RuntimeError(f"ollama pull {tag}: {ev['error'][:160]}")
            if ev.get("digest") and ev.get("completed") is not None:
                layers[ev["digest"]] = int(ev["completed"])
            if time.time() - last > 2:
                last = time.time()
                progress(sum(layers.values()))
            if ev.get("status") == "success":
                return
    raise RuntimeError(f"ollama pull {tag}: ended without success")


# ---- our own ollama -----------------------------------------------------------------

def _archives(plat: str | None = None) -> list[str]:
    plat = plat or _platform_name()
    if plat == "macos":
        return ["ollama-darwin.tgz"]
    names = [f"ollama-linux-{_arch()}.tar.zst"]
    if machine.gpu()[0] == "amd" and _arch() == "amd64":
        names.append("ollama-linux-amd64-rocm.tar.zst")
    return names


# Download sizes of the pinned release, for the "how much" line before anything is fetched.
_ARCHIVE_BYTES = {"ollama-darwin.tgz": 178 * 2 ** 20, "ollama-linux-amd64.tar.zst": 1454 * 2 ** 20,
                  "ollama-linux-arm64.tar.zst": 1508 * 2 ** 20, "ollama-linux-amd64-rocm.tar.zst": 1003 * 2 ** 20}


def _archive_bytes(plat: str) -> int:
    return sum(_ARCHIVE_BYTES.get(n, 1500 * 2 ** 20) for n in _archives(plat))


def _can_unzstd() -> bool:
    return bool(shutil.which("zstd")) or _tar_zstd()


def _tar_zstd() -> bool:
    try:
        r = subprocess.run(["tar", "--help"], capture_output=True, text=True, timeout=5)
        return "--zstd" in (r.stdout + r.stderr)
    except Exception:
        return False


def _fetch(url: str, dest: Path) -> None:
    """Download with resume (HTTP Range into dest.part)."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:               # already complete
            part.replace(dest)
            return
        raise
    with r:
        mode = "ab" if have and r.status == 206 else "wb"
        with open(part, mode) as f:
            shutil.copyfileobj(r, f, 1 << 20)
    part.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected(sums: str, name: str) -> str | None:
    for line in sums.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*").lstrip("./") == name:
            return parts[0].lower()
    return None


def _safe_members(tf: tarfile.TarFile, dest: Path):
    root = dest.resolve()
    for m in tf.getmembers():
        target = (dest / m.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"archive entry outside the folder: {m.name}")
        if m.issym() or m.islnk():
            link = (target.parent / m.linkname).resolve()
            if root not in link.parents:
                raise RuntimeError(f"archive link outside the folder: {m.name}")
        if m.isdev():
            raise RuntimeError(f"device file in archive: {m.name}")
        yield m


def _unpack(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(dest, members=list(_safe_members(tf, dest)))
        return
    # .tar.zst: Python 3.9 has no zstd. List first, check every path, then extract.
    cmd = ["tar", "--zstd"] if _tar_zstd() else None
    if cmd:
        listing = subprocess.run(cmd + ["-tf", str(path)], capture_output=True, text=True, timeout=600)
    else:
        z = subprocess.run(["zstd", "-dc", str(path)], capture_output=True, timeout=1800)
        if z.returncode != 0:
            raise RuntimeError(f"zstd: {z.stderr[-200:]!r}")
        plain = path.with_suffix("")          # .tar
        plain.write_bytes(z.stdout)
        try:
            with tarfile.open(plain) as tf:
                tf.extractall(dest, members=list(_safe_members(tf, dest)))
        finally:
            plain.unlink()
        return
    for name in listing.stdout.splitlines():
        if name.startswith("/") or ".." in Path(name).parts:
            raise RuntimeError(f"archive entry outside the folder: {name}")
    r = subprocess.run(cmd + ["-xf", str(path), "-C", str(dest), "--no-same-owner"],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"tar: {r.stderr[-200:]}")


def install_ollama(log=print) -> None:
    """Our own ollama in ~/.tokensaver/ollama, run as a background service on OWN_PORT."""
    from . import service
    DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not BIN.exists():
        dl = DIR / "download"
        dl.mkdir(exist_ok=True)
        sums_path = dl / "sha256sum.txt"
        _fetch(RELEASE + "sha256sum.txt", sums_path)
        sums = sums_path.read_text()
        for name in _archives():
            log(f"downloading {name} ({OLLAMA_VERSION})")
            path = dl / name
            if not path.exists():
                _fetch(RELEASE + name, path)
            want = _expected(sums, name)
            if not want or _sha256(path) != want:
                path.unlink()
                raise RuntimeError(f"{name} failed its checksum (deleted; will download again)")
        staging = DIR / "staging"
        shutil.rmtree(staging, ignore_errors=True)
        for name in _archives():
            # the Mac archive is flat (binary + libraries); the Linux one has bin/ and lib/
            _unpack(dl / name, staging / "bin" if name.endswith(".tgz") else staging)
        if not (staging / "bin" / "ollama").exists():
            raise RuntimeError("the ollama archive had no bin/ollama")
        for sub in ("bin", "lib"):
            if (staging / sub).exists():
                shutil.rmtree(DIR / sub, ignore_errors=True)
                (staging / sub).replace(DIR / sub)
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(dl, ignore_errors=True)
    MODELS.mkdir(mode=0o700, parents=True, exist_ok=True)
    service.install_ollama()
    for _ in range(60):
        if _tags(own_url()) is not None:
            return
        time.sleep(0.5)
    raise RuntimeError(f"our ollama did not answer on port {OWN_PORT} (see {vault.VAULT_DIR / 'ollama.log'})")


def _service_ok() -> bool:
    return _tags(own_url(), timeout=1.0) is not None


def set_mode(mode: str) -> str:
    """`tokensaver local-ai auto|on|off`."""
    from . import service
    config.set_value("local_ai", mode, "user", why="tokensaver local-ai")
    if mode == "off":
        st = load_state()
        if _pid_alive(st.get("pid")):
            try:
                os.kill(int(st["pid"]), 15)
            except OSError:
                pass
        _update(pid=None, want=[], finished=True)
        stopped = service.remove_ollama()
        size = _du(DIR) if DIR.exists() else 0
        return ("local AI off: nothing more will be downloaded" + ("; our ollama stopped" if stopped else "") +
                (f". Its files ({size / GB:.0f} GB) are kept in {DIR}; `tokensaver uninstall` removes them, "
                 f"or delete that folder" if size else ""))
    d = decide(mode)
    if d["skip"]:
        return f"local AI {mode}: skipped on this machine ({d['skip']})"
    if not d["missing"]:
        if d["ollama"] == "ours" and BIN.exists() and not _service_ok():
            service.install_ollama()
        return f"local AI {mode}: {describe(d)}"
    start()
    return (f"local AI {mode}: downloading {' + '.join(ROLE_NAMES[r] for r, _, _ in d['models'])} "
            f"({d['need_bytes'] / GB:.0f} GB) in the background; TokenSaver works without it meanwhile "
            "(`tokensaver status` shows progress)")


def _du(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def remove(keep_files: bool = False) -> list[str]:
    """For uninstall: stop our ollama and (unless kept) delete its folder.
    A system ollama is never touched."""
    from . import service
    done = []
    st = load_state()
    if _pid_alive(st.get("pid")):
        try:
            os.kill(int(st["pid"]), 15)
        except OSError:
            pass
    if service.remove_ollama():
        done.append("stopped TokenSaver's ollama")
    if DIR.exists() and not keep_files:
        size = _du(DIR)
        shutil.rmtree(DIR, ignore_errors=True)
        done.append(f"deleted {DIR} ({size / GB:.0f} GB: ollama and its models)")
    return done
