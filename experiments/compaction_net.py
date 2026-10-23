"""Net value of background compaction: the saving minus the re-reads it causes.

Compaction cuts old file views to their first lines. If Claude later needs one
of those files and did not re-read it in the real session, the compacted Claude
would have to: that re-read then stays in context and is resent on every later
request. This charges exactly that, conservatively:

* every file view in the compacted (old) part, cut to a head, is a candidate;
* after the break, the first time Claude touches the file decides: a Read of it
  costs nothing extra (the real session re-read it anyway); an Edit/Write of it,
  or Claude naming it in its text, costs a re-read of the file's last old view,
  carried from that request to the end of the session.

Not modelled: re-running commands to see old output again.
Local only. Usage: python3 experiments/compaction_net.py <bg_compact.jsonl>
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import passes, background as bg  # noqa: E402
from tokensaver.pricing import resolve  # noqa: E402
from tokensaver.transcripts import load  # noqa: E402

AFTER_S, KEEP_CHARS, HEAD = 15 * 60, bg.KEEP_TOKENS * 4, bg.EXTRACT_RESULT_CHARS


# Moments are selected by total context size, as when these were measured
# (the package dropped this setting for `bg_min_removable`, 2026-10-22).
MIN_TOKENS = 100_000

def first_uuid(p):
    for l in open(p, errors="replace"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("uuid"):
            return r["uuid"]


def t(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


def main() -> int:
    files = sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: os.stat(p).st_birthtime)
    owner, forks = {}, set()
    for p in files:
        u = first_uuid(p)
        (forks.add(p.stem[:8]) if u in owner else owner.__setitem__(u, p.stem[:8]))
    shrink = {}
    for l in open(sys.argv[1]):
        r = json.loads(l)
        if r["gate"]["passed"]:
            sid = r["session"]
            if sid in forks:
                sid = owner[first_uuid(next(f for f in files if f.stem.startswith(sid)))]
            shrink.setdefault(sid, 1 - (r["total_tokens"] - r["old_tokens"] + r["new_tokens"]) / r["total_tokens"])
    base = gross = net = 0.0
    rereads = edits = mentions = free = 0
    for s in load():
        if not s.requests or s.path.stem[:8] in forks:
            continue
        p = resolve(s.primary_model).price
        rq = [r for r in s.requests if r.timestamp]
        if not p or not rq:
            continue
        k = shrink.get(s.path.stem[:8])
        blocks = passes.from_transcript(s.path)
        red = 0
        prev = 0
        extra_ctx = [0] * (len(rq) + 2)   # re-read tokens carried from request j on
        writes_at = [0] * (len(rq) + 2)
        for i, r in enumerate(rq):
            c = r.context
            if prev and c < 0.5 * prev:
                red = 0
            prev = c
            sim = c - red
            gap = t(r.timestamp) - t(rq[i - 1].timestamp) if i else 0
            if k and gap > AFTER_S and sim > MIN_TOKENS:
                new = sim * (1 - k)
                red += sim - new
                # which file views does this compaction cut, and are they needed later?
                before = [b for b in blocks if b.request < i + 1]
                after = [b for b in blocks if b.request >= i + 1]
                acc, cut = 0, len(before)
                for j in range(len(before) - 1, -1, -1):
                    acc += len(before[j].text)
                    if acc > KEEP_CHARS:
                        cut = j
                        break
                old_views, recent_paths = {}, set()
                for j, b in enumerate(before):
                    v = passes.view_of(b) if b.type == "tool_result" else None
                    if not v:
                        continue
                    if j < cut and len(b.original) > HEAD:
                        old_views[v[0]] = len(b.original)
                    elif j >= cut:
                        recent_paths.add(v[0])
                for path, size in old_views.items():
                    if path in recent_paths:
                        continue
                    name = os.path.basename(path)
                    for b in after:
                        touched = None
                        if b.type == "tool_use" and isinstance(b.input, dict):
                            fp = b.input.get("file_path") or ""
                            if b.tool == "Read" and fp.endswith(name):
                                touched = "read"
                            elif b.tool in ("Edit", "Write", "MultiEdit") and fp.endswith(name):
                                touched = "edit"
                            elif b.tool == "Bash" and name in (b.input.get("command") or "") and passes.view_of(b) is None:
                                touched = None
                        elif b.type == "tool_result" and passes.view_of(b) and passes.view_of(b)[0] == path:
                            touched = "read"
                        elif b.role == "assistant" and b.type == "text" and name in b.text:
                            touched = "mention"
                        if touched:
                            if touched == "read":
                                free += 1
                            else:
                                rereads += 1
                                edits += touched == "edit"
                                mentions += touched == "mention"
                                at = next((q for q, rr in enumerate(rq) if q > i and rr and b.request <= q + 1), len(rq))
                                extra_ctx[at] += size // 4
                                writes_at[at] += size // 4
                            break
            carried = sum(extra_ctx[:i + 1])
            cost = (r.inp * p.inp + r.read * p.read + r.write * p.write_1h + r.out * p.out) / 1e6
            base += cost
            read_g = max(sim - r.write - r.inp, 0)
            g = (r.inp * p.inp + read_g * p.read + r.write * p.write_1h + r.out * p.out) / 1e6
            gross += g
            net += g + (carried - writes_at[i]) * p.read / 1e6 + writes_at[i] * p.write_1h / 1e6
    print(f"gross compaction saving: {100 * (1 - gross / base):.1f}% of bill")
    print(f"re-reads it would cause: {rereads} (first touch was an edit {edits}, a mention {mentions}); "
          f"already re-read in the real session: {free}")
    print(f"NET compaction saving:   {100 * (1 - net / base):.1f}% of bill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
