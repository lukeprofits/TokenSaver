"""Cache-safety gate for the proxy's forward rewrite.

The prompt cache matches on the bytes already sent. A forward rewrite is only
free if every message it rewrote for request k comes out identical when
request k+n rewrites the longer history. One differing byte in message m
means every token after m is billed as a cache write: about 40x a read.

This rebuilds API-shaped `messages` for each billed request of every local
transcript, rewrites a sample of prefixes and the full history with
`passes.rewrite_messages`, and requires each prefix's messages to match the
full rewrite exactly. It also checks the rewrite never touches thinking,
`cache_control` or any key other than text / tool_result content / Bash
command, and reports the chars saved.

Local only, zero API calls. Usage: python3 experiments/rewrite_determinism.py [prefixes per session]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import passes  # noqa: E402


def conversation(path: Path) -> tuple[list[dict], list[int]]:
    """(messages, request starts): messages as the API sees them, and for each
    billed request the number of messages it sent."""
    messages: list[dict] = []
    starts: list[int] = []
    seen_ids: set[str] = set()
    for line in path.open(errors="replace"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict) or rec.get("isSidechain"):
            continue
        role, content = msg.get("role"), msg.get("content")
        if role not in ("user", "assistant") or content is None:
            continue
        mid = msg.get("id")
        if role == "assistant" and mid and mid not in seen_ids:
            seen_ids.add(mid)
            starts.append(len(messages))
        items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = messages[-1]["content"] + items
        else:
            messages.append({"role": role, "content": list(items)})
    return messages, starts


def dumps(m) -> str:
    return json.dumps(m, sort_keys=True, ensure_ascii=False)


def only_allowed_changes(orig: dict, new: dict) -> str | None:
    if orig.get("role") != new.get("role") or set(orig) != set(new):
        return "message keys changed"
    a, b = orig.get("content"), new.get("content")
    if isinstance(a, str) or isinstance(b, str):
        return None if isinstance(a, str) and isinstance(b, str) else "content type changed"
    if len(a) != len(b):
        return "item count changed"
    for x, y in zip(a, b):
        if x == y:
            continue
        if not isinstance(x, dict) or x.get("type") != y.get("type") or set(x) != set(y):
            return "item shape changed"
        if x.get("type") == "thinking":
            return "thinking changed"
        if x.get("cache_control") != y.get("cache_control"):
            return "cache_control changed"
        changed = {k for k in x if x[k] != y[k]}
        allowed = {"text": {"text"}, "tool_result": {"content"}, "tool_use": {"input"}}.get(x["type"], set())
        if changed - allowed:
            return f"{x['type']} key changed: {sorted(changed - allowed)}"
        if x["type"] == "tool_use":
            ci = {k for k in x["input"] if x["input"].get(k) != y["input"].get(k)}
            allowed_in = {"command"}          # forward passes change a Bash command only
            if ci - allowed_in or set(x["input"]) != set(y["input"]):
                return f"tool_use input changed: {sorted(ci - allowed_in)}"
    return None


def main() -> int:
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sessions = checked = mismatches = shape = 0
    control = None
    o_chars = c_chars = 0
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl")):
        messages, starts = conversation(path)
        starts = [s for s in starts if s > 0]
        if len(starts) < 2:
            continue
        sessions += 1
        # The proxy sends aliases "off" or a fixed hindsight plan; both must
        # reproduce every earlier request byte for byte.
        plan = passes.hindsight_plan(messages)
        modes = ("off",) + ((plan,) if plan else ())
        for mode in modes[1:]:
            full_p = passes.rewrite_messages(messages, aliases=mode)
            for k in starts[::max(1, len(starts) // per)]:
                prefix = passes.rewrite_messages(messages[:k], aliases=mode)
                checked += 1
                for i, (a, b) in enumerate(zip(prefix, full_p)):
                    if dumps(a) != dumps(b):
                        mismatches += 1
                        print(f"  {path.name[:8]} plan: request with {k} messages differs at message {i}")
                        break
        full = passes.rewrite_messages(messages)
        for m, n in zip(messages, full):
            why = only_allowed_changes(m, n)
            if why:
                shape += 1
                print(f"  {path.name[:8]}: {why}")
                break
        o_chars += len(dumps(messages))
        c_chars += len(dumps(full))
        step = max(1, len(starts) // per)
        for k in starts[::step]:
            prefix = passes.rewrite_messages(messages[:k])
            checked += 1
            for i, (a, b) in enumerate(zip(prefix, full)):
                if dumps(a) != dumps(b):
                    mismatches += 1
                    print(f"  {path.name[:8]}: request with {k} messages differs at message {i}")
                    break
        if control is None and len(starts) > 3:
            control = (messages, starts)
    # Negative control: a rewrite that depends on how long the history is (as
    # the removed pointer passes did) must be caught, or this gate proves nothing.
    caught = 0
    if control:
        msgs, st = control
        noisy = lambda ms: [dict(ms[0], content=f"[{len(ms)}]")] + list(ms[1:])
        full_n = noisy(msgs)
        caught = sum(1 for k in st[1:4] if dumps(noisy(msgs[:k])[0]) != dumps(full_n[0]))
    print(f"\n{sessions} sessions, {checked} prefixes checked")
    print(f"prefix mismatches (cache breaks): {mismatches}")
    print(f"disallowed changes (thinking, cache_control, other keys): {shape}")
    print(f"serialised messages {o_chars:,} -> {c_chars:,} chars: {100 * (1 - c_chars / o_chars):.1f}% smaller")
    print(f"negative control (a history-length-dependent rewrite) caught: {caught} of 3")
    return 1 if mismatches or shape or caught < 3 else 0


if __name__ == "__main__":
    raise SystemExit(main())
