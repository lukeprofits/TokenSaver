"""Append-only local event store: SQLite index + content-addressed blobs.

Design constraints, in order of importance:

1. **Never break the session.** Every public function here is called from a
   hook in Claude Code's critical path. Any failure must be swallowed and
   reported, never raised.
2. **Append-only.** Nothing is updated or deleted. The index is a cache over
   the blobs; if they disagree, the blob wins.
3. **Independent of the transcript.** Claude Code deletes transcripts after
   `cleanupPeriodDays` and its JSONL format is internal. The vault captures
   hook payloads, which are a documented contract.

Storage is NOT encrypted. It holds source code, command output, and possibly
secrets - the same material already sitting in plaintext under ~/.claude, so
this does not change the exposure, but it does concentrate it. The directory
is created 0700. Whole-disk encryption is the intended protection; per-file
encryption needs a key-management story this does not yet have.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .names import home as _home  # noqa: E402

VAULT_DIR = _home()
DB_PATH = VAULT_DIR / "vault.db"
BLOB_DIR = VAULT_DIR / "blobs"
# Payloads at or under this size live inline in the index; larger ones become blobs.
INLINE_MAX = 4096
# Full-text index cap per event. The blob is always complete; this only bounds
# what is searchable, so a pathological payload cannot bloat the index without
# limit. Recorded as `fts_capped` so a miss is explainable rather than silent.
FTS_MAX_CHARS = 1_000_000
# Substring index is heavier per byte than the word index, so it gets a
# smaller cap. It exists for the half-remembered-identifier case, which is
# usually answered by the first screenful of a payload anyway.
SUBSTR_MAX_CHARS = 262_144
# Trigram matching needs at least three characters.
SUBSTR_MIN_QUERY = 3
# Beyond this, metadata moves to a blob. Truncating JSON mid-structure makes it
# unparseable, which is worse than not storing it - an earlier version of this
# file did exactly that and silently destroyed a payload under investigation.
META_INLINE_MAX = 8000

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    ts          TEXT    NOT NULL,
    session_id  TEXT,
    kind        TEXT    NOT NULL,
    tool        TEXT,
    cwd         TEXT,
    git_branch  TEXT,
    exit_code   INTEGER,
    bytes       INTEGER NOT NULL DEFAULT 0,
    truncated   INTEGER NOT NULL DEFAULT 0,
    fts_capped  INTEGER NOT NULL DEFAULT 0,
    blob_sha    TEXT,
    inline      TEXT,
    summary     TEXT,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, id);
CREATE INDEX IF NOT EXISTS events_kind    ON events(kind, id);
CREATE INDEX IF NOT EXISTS events_sha     ON events(blob_sha);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    summary, body, tool, event_id UNINDEXED,
    tokenize = "unicode61 tokenchars '_.-'"
);

-- Substring fallback. A word tokenizer cannot match the middle of an
-- identifier, so "proration" never finds
-- tests/test_billing.py::test_proration_rounding. Trigram can.
CREATE VIRTUAL TABLE IF NOT EXISTS events_sub USING fts5(
    body, event_id UNINDEXED, tokenize = 'trigram'
);
"""


@dataclass
class Event:
    id: int
    ts: str
    kind: str
    tool: str | None
    exit_code: int | None
    bytes: int
    truncated: bool
    blob_sha: str | None
    summary: str
    session_id: str | None = None
    cwd: str | None = None


_SCHEMA_READY: set[str] = set()
_SCHEMA_LOCK = __import__("threading").Lock()


def _connect() -> sqlite3.Connection:
    VAULT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    BLOB_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    # The schema script takes a write lock. Running it on every connection made
    # concurrent writers (relay threads, retention) collide and drop rows, so
    # it runs once per process per database.
    key = str(DB_PATH)
    if key not in _SCHEMA_READY:
        with _SCHEMA_LOCK:
            if key not in _SCHEMA_READY:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(SCHEMA)
                _SCHEMA_READY.add(key)
    return conn


def _blob_path(sha: str) -> Path:
    return BLOB_DIR / sha[:2] / sha[2:4] / sha


def put_blob(data: bytes) -> str:
    """Write content-addressed. Returns sha256. Idempotent by construction."""
    sha = hashlib.sha256(data).hexdigest()
    path = _blob_path(sha)
    if not path.exists():
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.chmod(tmp, 0o600)
        tmp.replace(path)
    return sha


def get_blob(sha: str) -> bytes | None:
    path = _blob_path(sha)
    try:
        return path.read_bytes()
    except OSError:
        return None


def _head_tail(text: str, limit: int) -> str:
    """Keep both ends when text exceeds `limit`.

    Indexing only the head is the same mistake Claude Code's failure
    truncation makes: in command output the invocation is at the top and the
    error is at the bottom, so a head-only index cannot find the failure that
    prompted the search. Measured: a 720k pytest log indexed head-only did not
    match its own failing test name.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n[... {omitted:,} characters omitted from index ...]\n{text[-half:]}"


def _meta_json(meta: dict) -> str:
    """Serialise metadata, spilling to a blob rather than truncating."""
    try:
        text = json.dumps(meta, default=str)
    except Exception:
        return json.dumps({"_meta_error": "unserialisable"})
    if len(text) <= META_INLINE_MAX:
        return text
    try:
        return json.dumps({"_meta_blob": put_blob(text.encode("utf-8", "replace")),
                           "_meta_bytes": len(text)})
    except Exception:
        return json.dumps({"_meta_error": "spill failed", "_meta_bytes": len(text)})


def read_meta(event_id: int) -> dict:
    """Metadata for an event, following a blob spill if there was one."""
    try:
        conn = _connect()
        row = conn.execute("SELECT meta FROM events WHERE id=?", (event_id,)).fetchone()
        conn.close()
        meta = json.loads(row[0]) if row and row[0] else {}
    except Exception:
        return {}
    sha = meta.get("_meta_blob")
    if sha:
        data = get_blob(sha)
        if data is not None:
            try:
                return json.loads(data.decode("utf-8", "replace"))
            except ValueError:
                return meta
    return meta


def record(kind: str, *, content: str = "", tool: str | None = None,
           session_id: str | None = None, cwd: str | None = None,
           git_branch: str | None = None, exit_code: int | None = None,
           truncated: bool = False, summary: str = "",
           meta: dict | None = None, index: bool = True) -> int | None:
    """Append one event. Returns its id, or None if anything went wrong.

    Callers in the hook path must treat None as 'carry on regardless'.
    `index=False` skips full-text indexing: whole API request bodies repeat
    the tool results already indexed, and indexing them was most of the
    vault's growth.
    """
    for attempt in (0, 1):
        rid = _record_once(kind, content, tool, session_id, cwd, git_branch, exit_code,
                           truncated, summary, meta, index)
        if rid is not None or attempt:
            return rid
        time.sleep(0.2)               # one retry: a busy moment, not a broken vault
    return None


def _record_once(kind, content, tool, session_id, cwd, git_branch, exit_code,
                 truncated, summary, meta, index) -> int | None:
    try:
        raw = content.encode("utf-8", "replace")
        sha = inline = None
        if len(raw) > INLINE_MAX:
            sha = put_blob(raw)
        else:
            inline = content
        body = _head_tail(content, FTS_MAX_CHARS)
        capped = len(content) > FTS_MAX_CHARS
        conn = _connect()
        with conn:
            cur = conn.execute(
                "INSERT INTO events (ts, session_id, kind, tool, cwd, git_branch,"
                " exit_code, bytes, truncated, fts_capped, blob_sha, inline, summary, meta)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), session_id, kind,
                 tool, cwd, git_branch, exit_code, len(raw), int(truncated), int(capped), sha,
                 inline, summary[:2000], _meta_json(meta or {})))
            if index:
                conn.execute("INSERT INTO events_fts (summary, body, tool, event_id)"
                             " VALUES (?,?,?,?)", (summary, body, tool or "", cur.lastrowid))
                conn.execute("INSERT INTO events_sub (body, event_id) VALUES (?,?)",
                             (f"{summary}\n{_head_tail(content, SUBSTR_MAX_CHARS)}", cur.lastrowid))
        conn.close()
        return cur.lastrowid
    except Exception:
        return None


def read(event_id: int) -> tuple[Event, str] | None:
    """Exact stored bytes for one event. Never a paraphrase."""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT id, ts, kind, tool, exit_code, bytes, truncated, blob_sha,"
            " summary, session_id, cwd, inline FROM events WHERE id=?", (event_id,)).fetchone()
        conn.close()
    except Exception:
        return None
    if not row:
        return None
    ev = Event(*row[:11])
    ev.truncated = bool(ev.truncated)
    if ev.blob_sha:
        data = get_blob(ev.blob_sha)
        body = data.decode("utf-8", "replace") if data is not None else ""
    else:
        body = row[11] or ""
    return ev, body


PRUNED = "[TokenSaver: body removed by retention; metadata kept]"


def prune(keep_days: int, max_bytes: int, kinds: tuple[str, ...] = ("api_request",),
          now: float | None = None) -> dict:
    """Drop stored bodies of `kinds` older than `keep_days`, then oldest first
    until the rest fit in `max_bytes`. The rows and their metadata stay, so
    stats and cache history survive; only the body goes. A blob is deleted only
    when no kept row still points at it (blobs are content-addressed and can be
    shared). Never raises: returns what it did, or the error."""
    done = {"pruned": 0, "bytes": 0, "blobs": 0}
    try:
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((now or time.time()) - keep_days * 86400))
        q = ",".join("?" * len(kinds))
        conn = _connect()
        rows = conn.execute(
            f"SELECT id, ts, bytes, blob_sha FROM events WHERE kind IN ({q})"
            f" AND (blob_sha IS NOT NULL OR (inline IS NOT NULL AND inline != ?)) ORDER BY id",
            (*kinds, PRUNED)).fetchall()
        total = sum(r[2] or 0 for r in rows)
        drop = []
        for rid, ts, size, sha in rows:
            if ts < cutoff or total > max_bytes:
                drop.append((rid, sha))
                total -= size or 0
                done["bytes"] += size or 0
        with conn:
            for rid, sha in drop:
                conn.execute("UPDATE events SET blob_sha=NULL, inline=? WHERE id=?", (PRUNED, rid))
                conn.execute("DELETE FROM events_fts WHERE event_id=?", (rid,))
                conn.execute("DELETE FROM events_sub WHERE event_id=?", (rid,))
        for _, sha in drop:
            if sha and not conn.execute("SELECT 1 FROM events WHERE blob_sha=? LIMIT 1", (sha,)).fetchone():
                try:
                    _blob_path(sha).unlink()
                    done["blobs"] += 1
                except OSError:
                    pass
        conn.close()
        done["pruned"] = len(drop)
    except Exception as exc:
        done["error"] = f"{type(exc).__name__}: {exc}"
    return done


SPILL_SCAN_MAX = 500


def find_tool_use(tool_use_id: str) -> int | None:
    """The newest result/failure event for one tool call, or None.

    Compression pointers name the `tool_use_id` because that is what the
    conversation carries; the vault keys events by its own id. The hook keeps
    `tool_use_id` in `meta`, so look there, then in metadata that was spilled
    to a blob, newest first and bounded. Never raises.
    """
    if not isinstance(tool_use_id, str) or not tool_use_id:
        return None
    kinds = ("tool_result", "tool_failure")
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT id FROM events WHERE kind IN (?,?) AND json_valid(meta)"
            " AND json_extract(meta, '$.tool_use_id') = ? ORDER BY id DESC LIMIT 1",
            (*kinds, tool_use_id)).fetchone()
        spilled = [] if row else conn.execute(
            "SELECT id FROM events WHERE kind IN (?,?) AND meta LIKE '%\"_meta_blob\"%'"
            " ORDER BY id DESC LIMIT ?", (*kinds, SPILL_SCAN_MAX)).fetchall()
        conn.close()
    except Exception:
        return None
    if row:
        return row[0]
    for (event_id,) in spilled:
        if read_meta(event_id).get("tool_use_id") == tool_use_id:
            return event_id
    return None


class QueryError(Exception):
    """The query itself was invalid - distinct from 'no results'."""


def escape_query(query: str) -> str:
    """Make arbitrary user text safe for FTS5 MATCH.

    FTS5 treats bare input as query syntax, so a path or an error message with
    a quote, colon or asterisk raises rather than matching. Each whitespace-
    separated term becomes a quoted phrase, ANDed together. A term ending in
    `*` keeps its prefix search; `OR` and `NOT` in caps stay operators.
    """
    terms, out = query.split(), []
    for term in terms:
        if term in ("OR", "AND", "NOT"):
            out.append(term)
            continue
        prefix = term.endswith("*")
        core = term[:-1] if prefix else term
        core = core.replace('"', '""')
        out.append(f'"{core}"' + ("*" if prefix else ""))
    return " ".join(out) or '""'


_SELECT = ("SELECT e.id, e.ts, e.kind, e.tool, e.exit_code, e.bytes, e.truncated,"
           " e.blob_sha, e.summary, e.session_id, e.cwd FROM {table} f"
           " JOIN events e ON e.id = f.event_id WHERE {table} MATCH ?")


def _run(conn, table: str, match: str, kind: str | None, limit: int) -> list[Event]:
    sql = _SELECT.format(table=table)
    params: list = [match]
    if kind:
        sql += " AND e.kind = ?"
        params.append(kind)
    sql += " ORDER BY e.id DESC LIMIT ?"
    params.append(limit)
    out = []
    for row in conn.execute(sql, params).fetchall():
        ev = Event(*row)
        ev.truncated = bool(ev.truncated)
        out.append(ev)
    return out


def search(query: str, limit: int = 20, kind: str | None = None) -> list[Event]:
    """Word search, falling back to substring search when it finds nothing.

    Raises QueryError only if the query is unusable - an empty list always
    means "no such content", never "your query was malformed".
    """
    try:
        conn = _connect()
        try:
            hits = _run(conn, "events_fts", escape_query(query), kind, limit)
            if not hits and len(query.strip()) >= SUBSTR_MIN_QUERY:
                hits = _run(conn, "events_sub", f'"{query.strip().replace(chr(34), chr(34) * 2)}"',
                            kind, limit)
            return hits
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        raise QueryError(str(exc)) from exc
    except Exception:
        return []


def stats() -> dict:
    try:
        conn = _connect()
        n, total = conn.execute("SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM events").fetchone()
        kinds = dict(conn.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind"
                                  " ORDER BY 2 DESC").fetchall())
        trunc = conn.execute("SELECT COUNT(*) FROM events WHERE truncated=1").fetchone()[0]
        capped = conn.execute("SELECT COUNT(*) FROM events WHERE fts_capped=1").fetchone()[0]
        conn.close()
        blobs = sum(1 for _ in BLOB_DIR.rglob("*") if _.is_file()) if BLOB_DIR.exists() else 0
        return {"events": n, "bytes": total, "kinds": kinds, "blobs": blobs,
                "truncated_captures": trunc, "fts_capped": capped, "path": str(VAULT_DIR)}
    except Exception as exc:
        return {"error": str(exc), "path": str(VAULT_DIR)}
