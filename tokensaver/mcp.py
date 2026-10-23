"""MCP server exposing the vault to Claude, over stdio. No dependencies.

Without this, tokensaver's breadcrumbs are decorative: a stub that says
"full output in vault, event 412" is useless if the model has no way to read
event 412. This gives it `tokensaver_search` and `tokensaver_read`, which
return the exact stored bytes - never a paraphrase.

Protocol: JSON-RPC 2.0 over stdin/stdout, one message per line. Only the three
methods Claude Code needs are implemented (`initialize`, `tools/list`,
`tools/call`); anything else gets a proper "method not found" rather than a
crash, because a server that dies takes its tools with it.
"""

from __future__ import annotations

import json
import sys

from . import vault

PROTOCOL_VERSION = "2025-06-18"
MAX_CHARS = 30_000          # per read, so one call cannot flood the context

TOOLS = [
    {
        "name": "tokensaver_search",
        "description": (
            "Search this machine's local TokenSaver vault for past tool output, "
            "commands, file contents and failures from earlier in this or any "
            "previous session. Returns exact stored text, never a summary. Use "
            "it whenever a tool result refers to a vault event id, when output "
            "was truncated, or when you need something from earlier that is no "
            "longer in context. Matches whole words and also substrings, so a "
            "partly-remembered identifier works."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Text to find: a path, error code, test name, identifier."},
                "kind": {"type": "string",
                         "description": "Optional filter: tool_result, tool_failure, "
                                        "command_vaulted, user_prompt."},
                "limit": {"type": "integer", "description": "Max results, default 20."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "tokensaver_read",
        "description": (
            "Read one vault event's exact stored content, by event_id (from "
            "tokensaver_search or a truncation breadcrumb) or by tool_use_id "
            "(named in a TokenSaver pointer that replaced an earlier tool "
            "result or Write/Edit body). input=true returns what the tool was "
            "called with - a Write's content - instead of its output. Large "
            "entries can be read in parts with offset, or from the end with "
            "from_end=true - a failing command's error is usually at the end."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "event_id": {"type": "integer"},
                "tool_use_id": {"type": "string",
                                "description": "Alternative to event_id: the tool call's id."},
                "input": {"type": "boolean",
                          "description": "Return the tool call's input instead of its output."},
                "offset": {"type": "integer", "description": "Start character, default 0."},
                "limit": {"type": "integer",
                          "description": f"Characters to return, default and max {MAX_CHARS}."},
                "from_end": {"type": "boolean",
                             "description": "Read the last `limit` characters instead."},
            },
        },
    },
]


def _text(body: str) -> dict:
    return {"content": [{"type": "text", "text": body}]}


def _search(args: dict) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return _text("TokenSaver: query is required")
    try:
        hits = vault.search(query, limit=min(int(args.get("limit") or 20), 100),
                            kind=args.get("kind") or None)
    except vault.QueryError as exc:
        return _text(f"TokenSaver: that query could not be parsed ({exc}). "
                     f"Try plain words without punctuation.")
    if not hits:
        return _text(f"TokenSaver: nothing in the vault matches {query!r}. "
                     f"This means no such content was recorded, not that the "
                     f"search failed.")
    lines = [f"{len(hits)} match(es). Read one with tokensaver_read(event_id=N)."]
    for ev in hits:
        flag = " [Claude Code truncated this; the vault has all of it]" if ev.truncated else ""
        lines.append(
            f"  event {ev.id}: {ev.kind} tool={ev.tool or '-'} "
            f"exit={ev.exit_code} {ev.bytes:,} bytes {ev.ts}{flag}\n"
            f"    {ev.summary}")
    return _text("\n".join(lines))


def _input_text(tool_input) -> str:
    """A tool call's input, with long string fields shown raw, not escaped."""
    if not isinstance(tool_input, dict):
        return json.dumps(tool_input, default=str)
    return "\n".join(f"== {key} ==\n{value if isinstance(value, str) else json.dumps(value, default=str)}"
                     for key, value in tool_input.items())


def _read(args: dict) -> dict:
    tool_use_id = args.get("tool_use_id")
    if tool_use_id and args.get("event_id") is None:
        event_id = vault.find_tool_use(str(tool_use_id))
        if event_id is None:
            return _text(f"TokenSaver: no vault event for tool_use_id {tool_use_id!r}. "
                         f"It may predate the vault or have been recorded with "
                         f"TOKENSAVER_DISABLE=1.")
    else:
        try:
            event_id = int(args.get("event_id"))
        except (TypeError, ValueError):
            return _text("TokenSaver: pass event_id (integer) or tool_use_id")
    got = vault.read(event_id)
    if not got:
        return _text(f"TokenSaver: no event {event_id}")
    ev, body = got
    if args.get("input"):
        meta = vault.read_meta(event_id)
        tool_input = meta.get("tool_input", meta.get("toolInput"))
        if tool_input is None:
            return _text(f"TokenSaver: event {event_id} has no recorded tool input")
        body = _input_text(tool_input)
    limit = min(int(args.get("limit") or MAX_CHARS), MAX_CHARS)
    if args.get("from_end"):
        chunk = body[-limit:]
        start = max(0, len(body) - limit)
    else:
        start = max(0, int(args.get("offset") or 0))
        chunk = body[start:start + limit]
    header = (f"event {ev.id} | {ev.kind} | tool={ev.tool or '-'} | "
              f"exit={ev.exit_code} | {ev.ts}\n{ev.summary}\n"
              f"characters {start:,}-{start + len(chunk):,} of {len(body):,}")
    if start + len(chunk) < len(body):
        part = ", input=true" if args.get("input") else ""
        header += (f"\nMore follows: tokensaver_read(event_id={ev.id}{part}, "
                   f"offset={start + len(chunk)}), or from_end=true for the end.")
    return _text(f"{header}\n{'-' * 60}\n{chunk}")


def handle(message: dict) -> dict | None:
    """Return a JSON-RPC response, or None for a notification."""
    method = message.get("method")
    msg_id = message.get("id")

    if method == "initialize":
        result = {"protocolVersion": PROTOCOL_VERSION,
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "tokensaver", "version": "0.1.0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "tokensaver_search":
            result = _search(args)
        elif name == "tokensaver_read":
            result = _read(args)
        else:
            result = {"content": [{"type": "text", "text": f"unknown tool {name!r}"}],
                      "isError": True}
    elif method in ("notifications/initialized", "notifications/cancelled"):
        return None                      # notifications take no response
    elif msg_id is None:
        return None                      # any other notification
    else:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"method not found: {method}"}}

    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        try:
            response = handle(message)
        except Exception as exc:
            # A tool failure must not kill the server; the model loses every
            # tool if this process exits.
            response = {"jsonrpc": "2.0", "id": message.get("id"),
                        "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
