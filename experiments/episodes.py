"""Episode compaction on real breaks: how much smaller, and at what cost.

For each real break (bg_compact.moments), builds today's compaction and the
episode-collapsed versions E1 (developer messages + the assistant's own final
message per finished, unneeded episode) and E2 (developer messages + a checked
one-line outcome), then reports:
- extra shrink of the compacted text over today's;
- used-later loss: identifiers Claude used after the break that today's text
  had and the collapsed one lacks (body only), and the deleted pieces they came
  from (each one is a likely re-read);
- dollars: tokens saved per request x requests that followed the break, at the
  read price, against an estimated re-read cost (one extra full-context request
  plus the re-read output) per lost piece;
- the QA gate, with TOKENSAVER_EPISODES_GATE=1.

Local only (the 35B segments, Nimble scores), $0 API. Usage:
  python3 experiments/episodes.py [max_moments]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import names  # noqa: E402
from tokensaver import background as bg, prune  # noqa: E402
import episodes_lib as ep  # noqa: E402
from bg_compact import moments  # noqa: E402
from smart_prune import body_of  # noqa: E402
from stack import indep_ident  # noqa: E402

OUT = names.home() / "experiments" / "episodes.jsonl"
READ, WRITE = 0.20 / 1e6, 8.0 / 1e6           # Opus 5.5, $/token


def log(msg):
    print(f"    {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _msg_of(messages: list, piece) -> int:
    """The index of the message holding a piece's first item."""
    target = id(piece.items[0])
    for i, m in enumerate(messages):
        if any(id(x) == target for x in bg._items(m)):
            return i
    return -1


def with_notes(text: str, notes: list[str]) -> str:
    if not notes:
        return text
    head, sep, idx = text.partition("\n\n## Identifiers from the earlier part\n\n")
    return head + "\n\n## Finished earlier work (outcomes)\n\n" + "\n".join(notes) + sep + idx


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    gate_on = os.environ.get("TOKENSAVER_EPISODES_GATE") == "1"
    only = set(filter(None, os.environ.get("TOKENSAVER_BG_SESSIONS", "").split(",")))
    OUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    seen, done = set(), 0
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: p.stat().st_size):
        if only and path.stem[:8] not in only:
            continue
        for before, after in moments(path):
            if done >= limit:
                return 0
            fp = bg.fingerprint(before[:50])
            if fp in seen:
                continue
            seen.add(fp)
            cut = bg.choose_cut(before)
            if not cut:
                continue
            done += 1
            old, recent = before[:cut], before[cut:]
            base, _ = bg._extract(before, cut)
            later = set().union(*(indep_ident(json.dumps(m)) for m in after if m["role"] == "assistant"))
            need = ((set().union(*(indep_ident(json.dumps(m)) for m in old))
                     - set().union(*(indep_ident(json.dumps(m)) for m in recent))) & later)
            base_body = indep_ident(body_of(base["text"]))
            n_after = sum(1 for m in after if m["role"] == "assistant")
            print(f"\n{path.stem[:8]}: {cut} old messages; today {base['new_tokens']:,} tokens; "
                  f"{len(need & base_body)} identifiers used later in today's text; ~{n_after} requests followed", flush=True)
            t = time.time()
            eps = ep.segment(before, cut, progress=log)
            prune.unload(bg.LOCAL_MODEL)
            cur = ep.current_digest(before, cut)
            # The 35B's own "finished" flag proved too strict (it marked 9 of 9
            # episodes unfinished, including "committed changes" and "resolved
            # the 401"), so every episode but the last in the old part is a
            # candidate and Nimble's p(needed) decides.
            last_start = max((e["start"] for e in eps), default=0)
            finished = [e for e in eps if e["start"] < last_start]
            for e in finished:
                e["p_needed"] = round(ep.needed(ep._episode_text(before, e), cur), 3)
            prune.unload("nimble")
            log(f"{len(eps)} episodes, {len(finished)} finished; scored in {time.time() - t:.0f}s")
            for e in finished:
                log(f"  #{e['start']}-#{e['end']} p(needed)={e['p_needed']:.2f}  {e['outcome'][:110]}")
            collapse = [e for e in finished if e["p_needed"] < ep.NEEDED_MAX_P]
            row = {"session": path.stem[:8], "cut": cut, "episodes": eps, "base_tokens": base["new_tokens"],
                   "need": len(need & base_body), "after_requests": n_after, "variants": {}}
            # E3 (Nimble labelling each piece) is dropped: Nimble never chose
            # "learning" (0 of 63) and E3 failed the gate (22 -> 17). E4 replaces it.
            all_pieces = prune.pieces(before, cut, bg.latest_doc_views(old))
            protect = ep.recent_names(before, cut)
            # E4: the 35B labels every candidate piece; Nimble verifies each drop.
            t = time.time()
            inside_all = [p for p in all_pieces if p.old and any(
                e["start"] <= _msg_of(before, p) <= e["end"] for e in finished)]
            labels35 = ep.label_35b(inside_all, cur)
            prune.unload(bg.LOCAL_MODEL)
            e4_drop, verified = set(), {}
            for p in inside_all:
                if labels35.get(p.n) not in ("plan", "process", "obsolete"):
                    continue
                mine = {x.group(0) for x in ep._IDENT.finditer(p.text) if len(x.group(0)) > 4}
                if mine & protect:
                    continue
                verified[p.n] = round(ep.verify_removable(p, cur), 3)
                if verified[p.n] >= ep.DROP_MIN_P:
                    e4_drop.update(id(x) for x in p.items)
            prune.unload("nimble")
            counts = {}
            for v in labels35.values():
                counts[v] = counts.get(v, 0) + 1
            log(f"E4: 35B labels {counts}; {len(verified)} verified by Nimble; dropping {len(e4_drop)} items "
                f"({time.time() - t:.0f}s)")
            row["e4"] = {"labels": {str(k): v for k, v in labels35.items()},
                         "verified": {str(k): v for k, v in verified.items()}}
            for variant in ("E1", "E2", "E4"):
                if variant == "E4":
                    skip, notes = e4_drop, []
                else:
                    skip, notes = ep.collapse(before, collapse, variant)
                res, cand_for = bg._extract(before, cut, skip)
                text = with_notes(res["text"], notes)
                new_tokens = len(text) // bg.CHARS_PER_TOKEN
                lost = (need & base_body) - indep_ident(body_of(text))
                # pieces that carried a lost identifier: each a likely re-read
                pieces = prune.pieces(before, cut)
                lost_pieces = [p for p in pieces if p.old and any(id(x) in skip for x in p.items)
                               and any(l.strip("'\"") in p.text for l in lost)]
                saved_tok = base["new_tokens"] - new_tokens
                saved = saved_tok * n_after * READ
                # One re-read = one extra request at the whole request's size (the
                # compacted text, the recent part, ~46K of system and tools) read
                # from cache, plus ~2K tokens of re-read output written.
                request_tokens = new_tokens + bg.size_tokens(recent) + 46_000
                reread = len(lost_pieces) * (request_tokens * READ + 2_000 * WRITE)
                gate = None
                if gate_on:
                    def cand(i, j, _f=cand_for, _n=notes):
                        return with_notes(_f(i, j), _n)
                    gate = {k: v for k, v in bg.gate(old, cand).items() if k != "items"}
                    gate["current"] = {k: v for k, v in bg.gate(old, cand, current=cur).items() if k != "items"}
                    prune.unload(bg.LOCAL_MODEL)
                row["variants"][variant] = {"tokens": new_tokens, "extra_shrink": round(1 - new_tokens / base["new_tokens"], 4),
                                            "lost": sorted(lost), "lost_pieces": len(lost_pieces),
                                            "saved_usd": round(saved, 3), "reread_usd": round(reread, 3), "gate": gate}
                print(f"  {variant}: {base['new_tokens']:,} -> {new_tokens:,} tokens "
                      f"({100 * (1 - new_tokens / base['new_tokens']):.1f}% smaller); lost {len(lost)} identifiers "
                      f"from {len(lost_pieces)} pieces; saves ~${saved:.2f} over {n_after} requests vs re-reads "
                      f"~${reread:.2f}" + (f"; gate {gate}" if gate else ""), flush=True)
                for l in sorted(lost)[:6]:
                    print(f"      lost {l}")
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
