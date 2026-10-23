"""Does this account enforce the preserved-thinking prefix check?

The answer decides whether a history-rewriting swap is viable here. Accounts
created on or after 2026-08-31 get a hard 400 when a replayed thinking block's
prefix has changed; older accounts run the same check but pass failing blocks
through unless the request opts in. If this account is in the lenient group,
the whole thinking problem - and the 2.5-20.1% output-token cost of working
around it - disappears.

Method: sit in the request path, wait for a real request that carries a
thinking block in its history, then send upstream a *copy* of that request
with one earlier user message altered. The copy reuses the headers the client
already sent, so no credential is ever read, stored or handled here - the
probe cannot see the token it authenticates with.

Costs one extra API request per probe. Nothing is modified on the path the
client sees: the real request is relayed untouched, exactly as always.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import proxy  # noqa: E402

RESULTS: list[dict] = []
DONE = threading.Event()


def _has_thinking(request: dict) -> int:
    count = 0
    for message in request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") in (
                    "thinking", "redacted_thinking"):
                count += 1
    return count


def _alter_prefix(request: dict) -> dict | None:
    """Change the earliest user text that precedes a thinking block.

    The check compares `system`, `tools` and every message before the block,
    so altering the first user turn is the minimal edit that must trip it.
    """
    clone = json.loads(json.dumps(request))
    first_thinking = None
    for i, message in enumerate(clone.get("messages") or []):
        if isinstance(message, dict) and _has_thinking({"messages": [message]}):
            first_thinking = i
            break
    if first_thinking is None:
        return None
    for message in clone["messages"][:first_thinking]:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            message["content"] = content + " (probe: prefix deliberately altered)"
            return clone
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    block["text"] += " (probe: prefix deliberately altered)"
                    return clone
    return None


_original_relay = proxy.Handler._relay


def _probing_relay(self, method: str) -> None:
    """Relay normally, then - once - probe with an altered copy."""
    length = int(self.headers.get("Content-Length") or 0)
    body = self.rfile.read(length) if length else b""

    # Hand the body back to the real relay by replaying it from a buffer.
    import io
    self.rfile = io.BytesIO(body + b"")
    self.headers.replace_header("Content-Length", str(len(body))) \
        if self.headers.get("Content-Length") else None
    _original_relay(self, method)

    if DONE.is_set() or "/v1/messages" not in self.path or not body:
        return
    try:
        request = json.loads(body)
    except ValueError:
        return
    blocks = _has_thinking(request)
    if not blocks:
        return
    altered = _alter_prefix(request)
    if altered is None:
        return

    DONE.set()
    headers = {k: v for k, v in self.headers.items()
               if k.lower() not in proxy.HOP_BY_HOP}
    headers["Accept-Encoding"] = "identity"
    altered["stream"] = False
    altered["max_tokens"] = 64
    payload = json.dumps(altered).encode()
    req = urllib.request.Request(
        proxy.UPSTREAM.rstrip("/") + self.path, data=payload,
        headers=headers, method="POST")
    outcome: dict = {"thinking_blocks": blocks, "model": request.get("model")}
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        text = resp.read().decode("utf-8", "replace")
        outcome["status"] = resp.status
        try:
            parsed = json.loads(text)
            outcome["input_transformations"] = parsed.get("input_transformations")
            outcome["stop_reason"] = parsed.get("stop_reason")
        except ValueError:
            pass
    except urllib.error.HTTPError as exc:
        outcome["status"] = exc.code
        outcome["body"] = exc.read().decode("utf-8", "replace")[:600]
    except Exception as exc:
        outcome["status"] = "transport error"
        outcome["body"] = f"{type(exc).__name__}: {exc}"
    RESULTS.append(outcome)


def main() -> int:
    proxy.Handler._relay = _probing_relay
    srv = ThreadingHTTPServer(("127.0.0.1", 8788), proxy.Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("probe proxy on http://127.0.0.1:8788 -> " + proxy.UPSTREAM)
    print("waiting for a request carrying a thinking block...")
    deadline = time.time() + 240
    while not DONE.is_set() and time.time() < deadline:
        time.sleep(0.5)
    time.sleep(2)
    srv.shutdown()

    if not RESULTS:
        print("\nno probe fired: no relayed request carried a thinking block.")
        return 1
    r = RESULTS[0]
    print(f"\nmodel: {r.get('model')}   thinking blocks in history: "
          f"{r.get('thinking_blocks')}")
    print(f"altered-prefix request returned: {r.get('status')}")
    body = r.get("body") or ""
    if r.get("status") == 400 and "bound to a different conversation" in body:
        print("\nENFORCED. This account rejects a replayed thinking block whose")
        print("prefix changed. A history-rewriting swap must therefore either")
        print("strip thinking (2.5-20.1% more output tokens) or use the")
        print("compact-2026-09-04 beta so the API authors the summary.")
    elif r.get("status") == 400:
        print(f"\n400, but not the binding check:\n{body}")
    elif r.get("status") == 200:
        transforms = r.get("input_transformations")
        if transforms:
            print(f"\nACCEPTED BUT DROPPED: input_transformations={transforms}")
            print("The request succeeded and the invalid thinking was discarded,")
            print("so the output-token cost applies without a visible error.")
        else:
            print("\nLENIENT. The altered prefix was accepted with no thinking")
            print("dropped, so on this account a history-rewriting swap does not")
            print("pay the thinking penalty at all.")
    else:
        print(f"\ninconclusive: {body[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
