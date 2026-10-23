"""Is the machine strained? Background model work eases off when it is.

The local models are large (the 35B judge takes ~28 GB in memory, Nimble
~18 GB; measured with both loaded: 46 GB of 64 GB, 6 GB left). Working while
the developer uses Claude is fine and wanted; making the computer slow for
whatever else they are doing is not. So background work checks the machine
between decisions and eases off in steps when it strains:

  1. pause a few seconds between decisions;
  2. still strained: unload the big model and keep to the small tier;
  3. still strained: stop until the machine is calm again.

Signals (no admin rights needed; macOS, and Linux from /proc and /sys):
- memory pressure level (kern.memorystatus_vm_pressure_level: 1 normal,
  2 warn, 4 critical);
- free memory percentage (`memory_pressure -Q`);
- swap GROWTH between checks (total swap is not a signal: macOS keeps swap
  around; measured 31.7 of 32 GB used at "normal" pressure);
- CPU speed limit from heat (`pmset -g therm`);
- battery under BATTERY_MIN percent while not on AC.

`bg_yield` (config, default on) turns easing off; when it is off, nothing
pauses for strain, and doctor/stats say so. Every read fails safe: a signal
that cannot be read counts as calm, not strained, except that `fits()` refuses
to load a model whose size it cannot check against free memory.
"""

from __future__ import annotations

import glob
import re
import subprocess
import sys
import time

from . import config

BATTERY_MIN = 30
FREE_MIN_PCT = 10                 # below this free share of RAM: strained
SWAP_GROWTH_MB = 512              # swap grown by more than this since the last check
HEADROOM_BYTES = 6 * 2 ** 30      # free memory that must remain after loading a model
PAUSE_S = (5, 30, 120)            # step 1 pause; how long strain lasts before step 2; then step 3

_last = {"swap_mb": None, "t": 0.0}
_log: list[dict] = []             # recent ease-off events, for doctor


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


LINUX = sys.platform.startswith("linux")


def _meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    """Linux /proc/meminfo in bytes; {} where it cannot be read."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                name, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    out[name.strip()] = int(parts[0]) * (1024 if parts[1:2] == ["kB"] else 1)
    except OSError:
        pass
    return out


def ram_bytes() -> int:
    if LINUX:
        return _meminfo().get("MemTotal", 0)
    try:
        return int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
    except ValueError:
        return 0


def _linux_pressure(path: str = "/proc/pressure/memory") -> int:
    """PSI "some avg10" mapped onto macOS's levels: 1 normal, 2 warn, 4 critical."""
    try:
        m = re.search(r"some avg10=([\d.]+)", open(path).read())
    except OSError:
        return 1
    v = float(m.group(1)) if m else 0.0
    return 4 if v >= 40 else 2 if v >= 10 else 1


def pressure_level() -> int:
    if LINUX:
        return _linux_pressure()
    try:
        return int(_run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip() or 1)
    except ValueError:
        return 1


def free_pct() -> int | None:
    if LINUX:
        mi = _meminfo()
        return round(100 * mi["MemAvailable"] / mi["MemTotal"]) if mi.get("MemTotal") and "MemAvailable" in mi else None
    m = re.search(r"free percentage:\s*(\d+)%", _run(["memory_pressure", "-Q"]))
    return int(m.group(1)) if m else None


def available_bytes() -> int | None:
    """Free plus quickly reclaimable memory (inactive, speculative, purgeable
    pages), from vm_stat. Measured: `memory_pressure`'s free percentage read
    23% while ollama still held both models and 86% once it unloaded them,
    while this read 45.6 GB available. Linux: MemAvailable."""
    if LINUX:
        return _meminfo().get("MemAvailable")
    out = _run(["vm_stat"])
    page = re.search(r"page size of (\d+) bytes", out)
    if not page:
        return None
    total = 0
    for name in ("free", "inactive", "speculative", "purgeable"):
        m = re.search(rf"Pages {name}:\s+(\d+)", out)
        if m:
            total += int(m.group(1))
    return total * int(page.group(1))


def swap_used_mb() -> float | None:
    if LINUX:
        mi = _meminfo()
        return (mi["SwapTotal"] - mi.get("SwapFree", 0)) / 2 ** 20 if "SwapTotal" in mi else None
    m = re.search(r"used = ([\d.]+)M", _run(["sysctl", "-n", "vm.swapusage"]))
    return float(m.group(1)) if m else None


def cpu_speed_limit() -> int | None:
    if LINUX:
        return None               # no portable reading; thermal throttling is not a signal there
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", _run(["pmset", "-g", "therm"]))
    return int(m.group(1)) if m else None


def _linux_battery(root: str = "/sys/class/power_supply") -> bool:
    for bat in glob.glob(f"{root}/BAT*"):
        try:
            status = open(f"{bat}/status").read().strip()
            capacity = int(open(f"{bat}/capacity").read().strip())
        except (OSError, ValueError):
            continue
        if status == "Discharging" and capacity < BATTERY_MIN:
            return True
    return False


def low_battery() -> bool:
    if LINUX:
        return _linux_battery()
    out = _run(["pmset", "-g", "batt"])
    if "AC Power" in out:
        return False
    m = re.search(r"(\d+)%", out)
    return bool(m and int(m.group(1)) < BATTERY_MIN)


def strained() -> tuple[bool, str]:
    """(strained, why). Calm when a signal cannot be read."""
    level = pressure_level()
    if level >= 2:
        return True, f"memory pressure {'critical' if level >= 4 else 'warn'}"
    pct = free_pct()
    if pct is not None and pct < FREE_MIN_PCT:
        return True, f"only {pct}% of memory free"
    swap = swap_used_mb()
    if swap is not None:
        prev, _last["swap_mb"] = _last["swap_mb"], swap
        if prev is not None and swap - prev > SWAP_GROWTH_MB:
            return True, f"swap grew {swap - prev:.0f} MB"
    limit = cpu_speed_limit()
    if limit is not None and limit < 100:
        return True, f"CPU limited to {limit}% by heat"
    if low_battery():
        return True, f"battery under {BATTERY_MIN}%"
    return False, ""


def fits(model_bytes: int, loaded: bool = False) -> tuple[bool, str]:
    """Whether loading a model of this size leaves HEADROOM_BYTES available. A
    model already in memory always fits. Unknown memory: does not fit."""
    if loaded:
        return True, "already loaded"
    free = available_bytes()
    if not free or not model_bytes:
        return False, "available memory unknown"
    if free - model_bytes < HEADROOM_BYTES:
        return False, (f"needs {model_bytes / 2 ** 30:.0f} GB plus {HEADROOM_BYTES / 2 ** 30:.0f} GB headroom; "
                       f"{free / 2 ** 30:.0f} GB free")
    return True, "fits"


class Yield:
    """Call `step()` between decisions. Returns "go", "small" (drop to the small
    tier: the caller unloads the big model) or "stop" (stop for now)."""

    def __init__(self, sleep=time.sleep, now=time.time, check=None):
        self.sleep, self.now, self.check = sleep, now, check or strained
        self.since = None
        self.events = 0

    def step(self) -> str:
        if not config.get("bg_yield"):
            return "go"
        bad, why = self.check()
        if not bad:
            self.since = None
            return "go"
        t = self.now()
        if self.since is None:
            self.since = t
        self.events += 1
        held = t - self.since
        action = "pause" if held < PAUSE_S[1] else ("small" if held < PAUSE_S[2] else "stop")
        _log.append({"t": t, "why": why, "action": action})
        del _log[:-50]
        if action == "pause":
            self.sleep(PAUSE_S[0])
            return "go"
        return action


def recent_events() -> list[dict]:
    return list(_log)
