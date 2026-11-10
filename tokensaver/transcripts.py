"""Defensive reader for Claude Code session transcripts.

The on-disk JSONL format is internal to Claude Code and changes between
versions, so every field access here is optional and every record that fails
to parse is counted rather than raised. Nothing downstream may assume a field
exists. The authoritative record for TokenSaver is the vault; this reader
exists to establish a baseline from history that predates it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "projects"


@dataclass
class Request:
    """One billed model request."""
    model: str
    inp: int          # uncached input tokens
    read: int         # cache read tokens
    write: int        # total cache creation tokens
    write_5m: int     # of which 5-minute TTL (0 if not reported)
    write_1h: int     # of which 1-hour TTL (0 if not reported)
    out: int
    timestamp: str
    effort: str | None = None
    # When the request was sent, as near as the transcript tells: the time of the user or
    # tool-result record just before the reply ("" if none). `timestamp` is when the reply
    # was written; a cache entry's lifetime runs from the request.
    start: str = ""

    @property
    def context(self) -> int:
        """Total prompt size: what was sent this turn, cached or not."""
        return self.inp + self.read + self.write


@dataclass
class Compaction:
    """A compact_boundary event, with Claude Code's own measurements."""
    trigger: str
    pre_tokens: int
    post_tokens: int
    duration_ms: int
    preserved_count: int
    timestamp: str


@dataclass
class Session:
    path: Path
    project: str
    session_id: str = ""
    cwd: str = ""
    git_branch: str = ""
    versions: set[str] = field(default_factory=set)
    requests: list[Request] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
    unparsed_lines: int = 0

    @property
    def started(self) -> str:
        return self.requests[0].timestamp if self.requests else ""

    @property
    def ended(self) -> str:
        return self.requests[-1].timestamp if self.requests else ""

    @property
    def primary_model(self) -> str:
        counts: dict[str, int] = {}
        for r in self.requests:
            counts[r.model] = counts.get(r.model, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    @property
    def floor(self) -> int:
        """Context of the first billed request: system + tools + project context.

        This is the irreducible per-epoch cost. A compaction threshold below it
        is meaningless, and one close to it leaves no room to work.
        """
        return self.requests[0].context if self.requests else 0


def _usage(rec: dict) -> dict | None:
    msg = rec.get("message")
    if isinstance(msg, dict):
        u = msg.get("usage")
        if isinstance(u, dict):
            return u
    u = rec.get("usage")
    return u if isinstance(u, dict) else None


def _int(d: dict, key: str) -> int:
    v = d.get(key)
    return v if isinstance(v, int) else 0


def read_session(path: Path) -> Session:
    sess = Session(path=path, project=path.parent.name)
    # Claude Code writes one record per content block of a response (thinking,
    # text, each tool call), every one repeating the response's usage. Counted
    # once per message id. Measured: 10,895 records were 5,185 requests, so
    # every figure from transcripts was inflated about 2x.
    seen_ids: set[str] = set()
    last_user = ""
    with path.open(errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                sess.unparsed_lines += 1
                continue
            if not isinstance(rec, dict):
                sess.unparsed_lines += 1
                continue

            sess.session_id = rec.get("sessionId") or rec.get("session_id") or sess.session_id
            sess.cwd = rec.get("cwd") or sess.cwd
            sess.git_branch = rec.get("gitBranch") or sess.git_branch
            if rec.get("version"):
                sess.versions.add(rec["version"])

            if rec.get("subtype") == "compact_boundary":
                meta = rec.get("compactMetadata") or {}
                preserved = (meta.get("preservedMessages") or {}).get("uuids") or []
                sess.compactions.append(Compaction(
                    trigger=meta.get("trigger", "?"),
                    pre_tokens=_int(meta, "preTokens"),
                    post_tokens=_int(meta, "postTokens"),
                    duration_ms=_int(meta, "durationMs"),
                    preserved_count=len(preserved),
                    timestamp=rec.get("timestamp", ""),
                ))
                continue

            u = _usage(rec)
            if not u:
                if rec.get("type") == "user" and rec.get("timestamp"):
                    last_user = rec["timestamp"]
                continue
            mid = (rec.get("message") or {}).get("id")
            if mid:
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
            creation = u.get("cache_creation") or {}
            req = Request(
                model=(rec.get("message") or {}).get("model") or "",
                inp=_int(u, "input_tokens"),
                read=_int(u, "cache_read_input_tokens"),
                write=_int(u, "cache_creation_input_tokens"),
                write_5m=_int(creation, "ephemeral_5m_input_tokens") if isinstance(creation, dict) else 0,
                write_1h=_int(creation, "ephemeral_1h_input_tokens") if isinstance(creation, dict) else 0,
                out=_int(u, "output_tokens"),
                timestamp=rec.get("timestamp", ""),
                effort=rec.get("effort"),
                start=last_user,
            )
            # Records with no tokens at all are bookkeeping, not billed requests.
            if req.context or req.out:
                sess.requests.append(req)
    return sess


def _request(rec: dict, u: dict) -> Request:
    creation = u.get("cache_creation") or {}
    return Request(
        model=(rec.get("message") or {}).get("model") or "",
        inp=_int(u, "input_tokens"),
        read=_int(u, "cache_read_input_tokens"),
        write=_int(u, "cache_creation_input_tokens"),
        write_5m=_int(creation, "ephemeral_5m_input_tokens") if isinstance(creation, dict) else 0,
        write_1h=_int(creation, "ephemeral_1h_input_tokens") if isinstance(creation, dict) else 0,
        out=_int(u, "output_tokens"),
        timestamp=rec.get("timestamp", ""),
        effort=rec.get("effort"),
    )


def api_history(path: Path, info: dict | None = None) -> tuple[list[dict], list[tuple[int, Request, int]]]:
    """(messages, requests) for the main conversation, in one pass so they
    always line up: `messages` as the API sees them (consecutive same-role
    records merged), and for each billed main request the number of messages
    it sent with its usage. Subagent records (`isSidechain`) are left out of
    both. At Claude Code's own compaction (`compact_boundary`) the history
    starts over, as the API's does; the index restarts with it, so a request
    sends `messages[epoch_start:epoch_start + n]` (see `request_messages`).
    `info`, if given, gets "duplicates" (repeated records or tool calls left out)
    and "boundaries" ([(epoch start, time of Claude Code's compaction)])."""
    messages: list[dict] = []
    ids: list = []                 # each message's reply id (assistant), kept beside it, never sent
    requests: list[tuple[int, Request, int]] = []
    seen: set[str] = set()
    start = 0
    # Some transcripts hold the same record, or the same tool call, twice (seen on another
    # machine: 210 of 1,776 counts refused with "`tool_use` ids must be unique"). The API saw
    # each once, so a repeat is left out: by record uuid, and by tool call / result id
    # within the current history.
    uuids: set = set()
    calls: set = set()
    results: set = set()
    dup = 0
    bounds: list = []
    last_user = ""
    if info is not None:
        info["duplicates"], info["boundaries"] = 0, bounds
    try:
        fh = path.open(errors="replace")
    except OSError:
        return messages, requests
    with fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or rec.get("isSidechain"):
                continue
            if rec.get("subtype") == "compact_boundary":
                start = len(messages)
                calls, results = set(), set()
                bounds.append((start, rec.get("timestamp")))
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            role, content = msg.get("role"), msg.get("content")
            if role not in ("user", "assistant") or content is None:
                continue
            uid = rec.get("uuid")
            if uid:
                if uid in uuids:
                    dup += 1
                    continue
                uuids.add(uid)
            mid = msg.get("id")
            if role == "user" and rec.get("timestamp"):
                last_user = rec["timestamp"]
            if role == "assistant" and mid and mid not in seen:
                seen.add(mid)
                u = _usage(rec)
                if u:
                    req = _request(rec, u)
                    req.start = last_user
                    if req.context or req.out:
                        requests.append((len(messages), req, start))
            items = content if isinstance(content, list) else [{"type": "text", "text": content}]
            kept = []
            for b in items:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id"):
                    if b["id"] in calls:
                        dup += 1
                        continue
                    calls.add(b["id"])
                elif isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id"):
                    if b["tool_use_id"] in results:
                        dup += 1
                        continue
                    results.add(b["tool_use_id"])
                kept.append(b)
            if not kept:
                continue
            items = kept
            # Parallel tool calls: Claude Code writes a tool's result as soon as
            # that tool finishes, while later tool_use blocks of the SAME reply
            # are still being written (file order: call 1, call 2, result 1,
            # call 3, ... result 2). The API saw one reply with every call,
            # then every result. Rebuilt in file order, a call lost its result
            # and every later count_tokens call failed with "tool_use ids were
            # found without tool_result blocks" (one tester: 410 of 2,084).
            if (role == "assistant" and mid and len(messages) >= start + 2
                    and messages[-2]["role"] == "assistant" and ids[-2] == mid
                    and all(isinstance(b, dict) and b.get("type") == "tool_result"
                            for b in messages[-1]["content"])):
                messages[-2]["content"] = messages[-2]["content"] + items
                continue
            if len(messages) > start and messages[-1]["role"] == role:
                messages[-1]["content"] = messages[-1]["content"] + items
                if role == "assistant" and mid:
                    ids[-1] = mid
            else:
                messages.append({"role": role, "content": list(items)})
                ids.append(mid if role == "assistant" else None)
    if info is not None:
        info["duplicates"] = dup
    return messages, requests


def request_messages(messages: list[dict], entry) -> list[dict]:
    """What one request from `api_history` sent."""
    end, _, start = entry
    return messages[start:end]


def load(root: Path | None = None, since: str | None = None,
         min_requests: int = 1) -> list[Session]:
    root = root or DEFAULT_ROOT
    out = []
    for path in sorted(root.glob("*/*.jsonl")):
        sess = read_session(path)
        if len(sess.requests) < min_requests:
            continue
        if since and sess.ended and sess.ended[:10] < since:
            continue
        out.append(sess)
    out.sort(key=lambda s: s.started)
    return out
