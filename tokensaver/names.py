"""The product's name, its data folder, and its env vars, in one place.

Renamed from "everytoken" to TokenSaver on 2026-10-23. Until `tokensaver-cli
migrate` has moved this machine over, the old data folder (~/.everytoken) is
the live one and old env vars (EVERYTOKEN_*) are still honoured, so a
half-migrated machine never loses data or a kill switch.
"""

from __future__ import annotations

import os
from pathlib import Path

NAME = "TokenSaver"
HOME_NEW = Path.home() / ".tokensaver"
HOME_OLD = Path.home() / ".everytoken"
MIGRATED = ".migrated"          # marker in HOME_NEW, written by `migrate`


def env(suffix: str, default: str | None = None) -> str | None:
    """TOKENSAVER_<suffix>, else the old EVERYTOKEN_<suffix>, else default."""
    for prefix in ("TOKENSAVER_", "EVERYTOKEN_"):
        value = os.environ.get(prefix + suffix)
        if value not in (None, ""):
            return value
    return default


def home() -> Path:
    """The data folder: the env override, else ~/.tokensaver once migrated (or
    on a fresh machine), else the old ~/.everytoken."""
    override = env("HOME")
    if override:
        return Path(override)
    if (HOME_NEW / MIGRATED).exists():
        return HOME_NEW
    if HOME_OLD.is_dir() and not HOME_OLD.is_symlink():
        return HOME_OLD
    return HOME_NEW
