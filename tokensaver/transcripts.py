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


def api_history(path: Path) -> tuple[list[dict], list[tuple[int, Request, int]]]:
    """(messages, requests) for the main conversation, in one pass so they
    always line up: `messages` as the API sees them (consecutive same-role
    records merged), and for each billed main request the number of messages
    it sent with its usage. Subagent records (`isSidechain`) are left out of
    both. At Claude Code's own compaction (`compact_boundary`) the history
    starts over, as the API's does; the index restarts with it, so a request
    sends `messages[epoch_start:epoch_start + n]` (see `request_messages`)."""
    messages: list[dict] = []
    requests: list[tuple[int, Request, int]] = []
    seen: set[str] = set()
    start = 0
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
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            role, content = msg.get("role"), msg.get("content")
            if role not in ("user", "assistant") or content is None:
                continue
            mid = msg.get("id")
            if role == "assistant" and mid and mid not in seen:
                seen.add(mid)
                u = _usage(rec)
                if u:
                    req = _request(rec, u)
                    if req.context or req.out:
                        requests.append((len(messages), req, start))
            items = content if isinstance(content, list) else [{"type": "text", "text": content}]
            if len(messages) > start and messages[-1]["role"] == role:
                messages[-1]["content"] = messages[-1]["content"] + items
            else:
                messages.append({"role": role, "content": list(items)})
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
