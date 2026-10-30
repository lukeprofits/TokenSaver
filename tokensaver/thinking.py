"""Thinking-rate samples: a few exact sent requests, kept to be measured weekly.

How many tokens Anthropic bills per character of a thinking signature can only be
measured on a request exactly as it was sent: on Opus 5.5+ thinking is bound to
the bytes before it (system, tools, every earlier message), so a request rebuilt
from transcripts reads ~0. So when a request with enough thinking goes by, its
count body is kept here, and once a week the audit worker measures what was kept
(`measure_due`), on an in-flight request's credentials: the proxy keeps no auth.

Kept small on purpose:
- one sample per session, the newest (a later request holds everything an
  earlier one did), refreshed at most every REFRESH_S;
- system, tools and the thinking config are the same across sessions: stored
  once under their hash (`shared/`), the session file holds only its messages;
- deleted after KEEP_DAYS, and at most PER_MODEL sessions per model;
- each result is a small `thinking_factor` vault row; a sample is measured once
  (again only after it is refreshed).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import threading
import time

from . import vault

DIR = vault.VAULT_DIR / "samples"
SHARED_KEYS = ("model", "system", "tools", "tool_choice", "thinking")
MIN_SIG_CHARS = 20_000           # smaller signatures measure too coarsely
REFRESH_S = 6 * 3600
KEEP_DAYS = 30
PER_MODEL = 20
MEASURE_EVERY_S = 7 * 86400
_lock = threading.Lock()


def _safe(session: str) -> str:
    return "".join(c for c in session if c.isalnum() or c in "-_")[:80]


def _write(path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _index() -> dict:
    try:
        return json.loads((DIR / "index.json").read_text())
    except (OSError, ValueError):
        return {"samples": {}, "measured_at": 0}


def _save_index(idx: dict) -> None:
    _write(DIR / "index.json", json.dumps(idx).encode())


def _thinking_on(body: dict) -> bool:
    th = body.get("thinking")
    return isinstance(th, dict) and th.get("type") not in (None, "disabled")


def store(session: str | None, body: dict | None, now: float | None = None) -> bool:
    """Keep `body` (a count_tokens body as sent, thinking kept) as this session's
    sample if it is worth measuring and the last one is older than REFRESH_S."""
    from .background import _signature_chars
    now = time.time() if now is None else now
    if not session or not isinstance(body, dict) or not _thinking_on(body):
        return False
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return False
    sig = _signature_chars(msgs)
    if sig < MIN_SIG_CHARS:
        return False
    with _lock:
        idx = _index()
        old = idx["samples"].get(session)
        if old and now - old.get("stored_at", 0) < REFRESH_S:
            return False
        shared = {k: body[k] for k in SHARED_KEYS if k in body}
        raw = json.dumps(shared, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        sha = hashlib.sha256(raw).hexdigest()[:32]
        if not (DIR / "shared" / f"{sha}.json.gz").exists():
            _write(DIR / "shared" / f"{sha}.json.gz", gzip.compress(raw))
        _write(DIR / f"{_safe(session)}.json.gz",
               gzip.compress(json.dumps({"shared": sha, "messages": msgs}, ensure_ascii=False).encode()))
        idx["samples"][session] = {"model": body.get("model"), "sig_chars": sig, "stored_at": now,
                                   "shared": sha, "measured_at": None}
        _prune(idx, now)
        _save_index(idx)
    return True


def load(session: str) -> dict | None:
    """The stored count body for `session`, or None."""
    try:
        part = json.loads(gzip.decompress((DIR / f"{_safe(session)}.json.gz").read_bytes()))
        shared = json.loads(gzip.decompress((DIR / "shared" / f"{part['shared']}.json.gz").read_bytes()))
    except (OSError, ValueError, KeyError):
        return None
    return dict(shared, messages=part["messages"])


def _prune(idx: dict, now: float) -> None:
    """Drop samples past KEEP_DAYS or beyond PER_MODEL newest for their model,
    and shared files nothing points at."""
    samples = idx["samples"]
    for s, e in list(samples.items()):
        if now - e.get("stored_at", 0) > KEEP_DAYS * 86400:
            samples.pop(s)
    by_model: dict = {}
    for s, e in samples.items():
        by_model.setdefault(e.get("model"), []).append((e.get("stored_at", 0), s))
    for items in by_model.values():
        for _, s in sorted(items, reverse=True)[PER_MODEL:]:
            samples.pop(s)
    keep = {_safe(s) + ".json.gz" for s in samples}
    used = {e.get("shared") + ".json.gz" for e in samples.values() if e.get("shared")}
    try:
        for p in DIR.glob("*.json.gz"):
            if p.name not in keep:
                p.unlink()
        for p in (DIR / "shared").glob("*.json.gz"):
            if p.name not in used:
                p.unlink()
    except OSError:
        pass


def due(now: float | None = None) -> bool:
    """A week since the last measuring, and something is waiting to be measured."""
    now = time.time() if now is None else now
    idx = _index()
    if now - idx.get("measured_at", 0) < MEASURE_EVERY_S:
        return False
    return any(not e.get("measured_at") for e in idx["samples"].values())


def _transient(why: str) -> bool:
    return why.startswith(("count refused: 429", "count refused: 5", "count refused: 0",
                           "count refused: no response"))


def measure_due(count, now: float | None = None) -> int:
    """Measure every sample not yet measured. `count(body, beta) -> (tokens | None, why)`.
    Stops at a rate-limit or network error (the rest wait for the next job); the
    week restarts only when everything waiting was measured. Returns how many."""
    from .audit import measure_thinking
    now = time.time() if now is None else now
    done = 0
    for session, e in list(_index()["samples"].items()):
        if e.get("measured_at"):
            continue
        body = load(session)
        if body is None:
            continue
        got = measure_thinking(body, count)
        if got["with"] is None and _transient(got["why"]):
            return done
        vault.record("thinking_factor", tool="proxy", session_id=session, index=False,
                     summary=(f"thinking {got['factor']:.3f} tokens per signature char "
                              f"({got['sig_chars']:,} chars)" if got["factor"]
                              else f"thinking not measured: {got['why']}"),
                     meta=dict(got, model=body.get("model"), stored_at=e.get("stored_at")))
        with _lock:
            idx = _index()
            if session in idx["samples"] and idx["samples"][session].get("stored_at") == e.get("stored_at"):
                idx["samples"][session]["measured_at"] = now
                _save_index(idx)
        done += 1
    with _lock:
        idx = _index()
        idx["measured_at"] = now
        _save_index(idx)
    return done


def summary() -> dict:
    """{"sessions", "bytes", "next_in_s", "waiting"} for stats and doctor."""
    idx = _index()
    size = 0
    try:
        size = sum(p.stat().st_size for p in DIR.rglob("*.json.gz"))
    except OSError:
        pass
    waiting = sum(1 for e in idx["samples"].values() if not e.get("measured_at"))
    return {"sessions": len(idx["samples"]), "bytes": size, "waiting": waiting,
            "next_in_s": max(0.0, idx.get("measured_at", 0) + MEASURE_EVERY_S - time.time())}
