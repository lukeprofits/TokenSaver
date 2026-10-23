"""Side requests must share the main request's cache: replayed on real traffic.

Claude Code sends `auxiliary` requests (session recap, prompt suggestion) that
carry the whole conversation plus one instruction. If the proxy does not
recognise one as following the main history, it goes out unrewritten and
writes a cache of its own: in the first real trial that cost $3.5-4.4 each.

For every vaulted auxiliary request this finds the main request just before it
in the same session and reports:
- whether `proxy._follows_main` matches it, and if not, the first difference;
- for each match, whether the rewritten auxiliary messages reproduce the
  rewritten main messages up to the last shared one, so the cache is shared.
  Both are rewritten with the same alias arguments ("off" and the session's
  current plan); compaction state at the time is not stored, and apply() only
  prepends text at the cut, so it cannot make two equal prefixes differ.
- warm cache writes over 50K tokens on each auxiliary request, as billed.

Local, free (reads ~/.tokensaver/vault.db and request bodies).
Usage: python3 experiments/aux_replay.py [session-prefix]
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import passes, proxy, vault  # noqa: E402


def first_diff(a: list, b: list):
    for i in range(min(len(a), len(b))):
        x, y = json.dumps(a[i], sort_keys=True), json.dumps(b[i], sort_keys=True)
        if x != y:
            return i, x[:120], y[:120]
    return None


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    conn = sqlite3.connect(vault.DB_PATH)
    rows = conn.execute("SELECT id, ts, session_id, blob_sha FROM events WHERE kind='api_request' "
                        "AND session_id LIKE ? ORDER BY id", (prefix + "%",)).fetchall()
    last_main: dict[str, list] = {}
    n = matched = shared = 0
    unmatched = []
    for rid, ts, session, sha in rows:
        meta = vault.read_meta(rid)
        raw = vault.get_blob(sha) if sha else None
        if not raw or meta.get("status") != 200 or "/v1/messages" != (meta.get("path") or "").split("?")[0]:
            continue
        try:
            messages = json.loads(raw).get("messages")
        except ValueError:
            continue
        if not isinstance(messages, list):
            continue
        cls = meta.get("request_class")
        if cls == "main":
            last_main[session] = messages
            continue
        if cls != "auxiliary" or session not in last_main:
            continue
        n += 1
        main_msgs = last_main[session]
        proxy._MAIN_HIST[session] = main_msgs
        usage = meta.get("usage") or {}
        wrote = int(usage.get("cache_creation_input_tokens") or 0)
        last = messages[-1].get("content")
        kind = (last if isinstance(last, str) else json.dumps(last))[:40].replace("\n", " ")
        if not proxy._follows_main(session, messages):
            d = first_diff(proxy._canon(main_msgs), proxy._canon(messages[:len(main_msgs)]))
            unmatched.append((ts, len(main_msgs), len(messages), wrote, kind, d))
            continue
        matched += 1
        plan = (proxy._bg_load(session).get("aliases") or {}).get("plan")
        ok = True
        for arg in ("off",) + ((plan,) if plan else ()):
            rm = passes.rewrite_messages(main_msgs, aliases=arg)
            ra = passes.rewrite_messages(messages, aliases=arg)
            k = len(main_msgs) - 1          # the newest main message is the one that changes form
            # cache_control markers say where to cache, not what: real side
            # requests with different markers read the main cache in full.
            if (json.dumps(proxy._canon(rm[:k]), sort_keys=True)
                    != json.dumps(proxy._canon(ra[:k]), sort_keys=True)):
                ok = False
        shared += ok
        if not ok:
            print(f"  PREFIX DIFFERS {ts} {kind!r}")
    print(f"auxiliary requests with a main before them: {n}")
    print(f"  matched: {matched}  (rewritten prefix identical: {shared})")
    print(f"  unmatched: {len(unmatched)}")
    for ts, lm, la, wrote, kind, d in unmatched:
        print(f"    {ts} main {lm} aux {la} wrote {wrote:,} {kind!r}")
        if d:
            print(f"      first diff at message {d[0]}:\n        main {d[1]}\n        aux  {d[2]}")
    return 0 if shared == matched else 1


if __name__ == "__main__":
    sys.exit(main())
