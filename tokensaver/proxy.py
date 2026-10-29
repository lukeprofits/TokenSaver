"""The gateway: relays Claude Code's API traffic, rewrites only what is safe.

This is the one component that cannot fail open. The hooks sit beside the
session and a crash there costs a wasted invocation; this sits *in* the data
path, so a bug here is a failed turn. Two rules follow, and every design
choice below serves them:

1. **Relay first, observe second.** The response is fully flushed to the
   client before a single byte is parsed, costed or stored. Observation runs
   inside `try/except Exception`, so a defect in it can never surface as a
   broken request.
2. **Rewrite only what stays byte-stable.** With `rewrite=forward` (what setup
   configures; "off" relays every byte as it came), main-thread `/v1/messages`
   histories get `passes.FORWARD`, which changes a block only before it is
   first sent, and a background compaction at free points. Already-sent bytes
   are replayed from a ledger, so the cache holds and Claude's signed
   thinking stays valid (preserved thinking; see `_epoch`).
   `experiments/rewrite_determinism.py` is the gate. Any other request class
   is relayed untouched: the auto-mode classifier's own calls come through
   here, and a rewritten history is what made it deny commands. Any
   exception sends the original bytes.

The failure this exists to prevent is specific and silent: a gateway that
strips, repositions or reshapes `cache_control` makes the whole conversation
bill as uncached input on every turn - roughly 10x more, with no error raised.
`tokensaver proxy --stats` therefore flags any session whose main-thread
requests stop reading cache, and that check is a release gate, not a report.

Headers are forwarded as an open list: everything except hop-by-hop transport
headers goes through untouched, so a new `anthropic-beta` value or
`x-claude-code-*` header needs no code change here. The gateway spec is
explicit that a pinned allowlist breaks on the release that adds a capability.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import audit, background, config, machine, passes, pricing, prune, vault

# Settings come from `config`: the file, or env vars for a proxy started by
# hand. A launchd-run proxy sees no shell env, so the file is the source.
UPSTREAM = config.get("upstream")
PORT = config.get("port")

# Transport headers that describe this hop, not the request. Everything else
# is forwarded verbatim, including every anthropic-* and x-claude-code-* name.
HOP_BY_HOP = {
    "host", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
    "content-length",
    # Requested as identity so the relayed bytes stay transparent; a
    # content-encoding we had to decode would mean re-encoding on the way out.
    "accept-encoding",
}
# Never written to the vault.
SECRET_HEADERS = {"authorization", "x-api-key", "proxy-authorization", "cookie"}

CHUNK = 16 * 1024
# The tee keeps both ends of the response: `message_start` carries the input
# and cache token counts, `message_delta` at the very end carries the output
# count. A head-only copy would miss half the usage.
TEE_HEAD = TEE_TAIL = 96 * 1024
# Read once at start: changing it mid-session would change the bytes of every
# history already cached, so a new mode needs a restart.
REWRITE = config.get("rewrite")
# Request classes whose history is rewritten. Allowlist: an unknown or missing
# class (hint headers off) is relayed as is.
REWRITE_CLASSES = {"main"}


# Claude Code caches with a 1h TTL, refreshed on every hit (measured
# 2026-10-13: every cache write was ephemeral_1h). Past this gap the prefix
# is being rebuilt anyway, so changing it costs nothing extra. One minute of margin.
EXPIRED_S = 61 * 60
_STATE_LOCK = threading.Lock()
def _state_path(session: str):
    safe = "".join(c for c in session if c.isalnum() or c in "-_")[:80]
    return vault.VAULT_DIR / "proxy_state" / f"{safe}.json"


# ---- background compaction (config bg_compact; setup turns it on) ----------
#
# See `background`. The worker prepares a summary for a session that has gone
# quiet; the request path applies it after `BG_APPLY_AFTER_S` of quiet, then
# keeps applying the same bytes so the rebuilt cache holds.

BG_COMPACT = config.get("bg_compact")
BG_PREP_IDLE_S = config.get("bg_prep_idle_min") * 60   # start preparing after this much quiet
# Apply after this much quiet. Under the cache TTL the compacted prefix is a
# fresh cache write, which still pays: simulated on real sessions with the
# measured shrinks, 61 min 16.3% of the bill, 30 min 23.7%, 15 min 27.7%.
# Preparation (~1.5 min) starts at 10 min, so it is ready: no wait either way.
BG_APPLY_AFTER_S = config.get("bg_after_min") * 60
_LIVE: dict[str, dict] = {}     # session -> last request time and history
_BG_STARTED = False


def _bg_path(session: str):
    return _state_path(session).with_suffix(".bg.json")


def _bg_load(session: str) -> dict:
    try:
        return json.loads(_bg_path(session).read_text())
    except (OSError, ValueError):
        return {}


def _bg_save(session: str, st: dict) -> None:
    try:
        p = _bg_path(session)
        p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        p.write_text(json.dumps(st))
    except OSError:
        pass


LIVE_MAX = 20                   # sessions held in memory for the worker
LIVE_IDLE_DROP_S = 2 * 3600     # forget a session's history after this much quiet
BATTERY_MIN = 30                # percent; below this, on battery, prep waits
_battery_cache = {"t": 0.0, "ok": True}


def _on_low_battery() -> bool:
    """True only when we can tell we are unplugged and below BATTERY_MIN.
    Unknown counts as fine: never block compaction on a check that failed."""
    now = time.time()
    if now - _battery_cache["t"] < 60:
        return not _battery_cache["ok"]
    ok = True
    try:
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                                 timeout=3).stdout
            pct = re.search(r"(\d+)%", out)
            if "Battery Power" in out and pct and int(pct.group(1)) < BATTERY_MIN:
                ok = False
        else:
            for d in Path("/sys/class/power_supply").glob("BAT*"):
                status = (d / "status").read_text().strip()
                cap = int((d / "capacity").read_text().strip())
                if status == "Discharging" and cap < BATTERY_MIN:
                    ok = False
    except Exception:
        ok = True
    _battery_cache.update(t=now, ok=ok)
    return not ok


def _evict(now: float) -> None:
    """Bound memory: the worker holds whole histories. Prepared and applied
    compactions are on disk, so forgetting a quiet session loses nothing."""
    for session, live in list(_LIVE.items()):
        if now - live["last"] > LIVE_IDLE_DROP_S:
            _LIVE.pop(session, None)
    if len(_LIVE) > LIVE_MAX:
        for session, _ in sorted(_LIVE.items(), key=lambda kv: kv[1]["last"])[:len(_LIVE) - LIVE_MAX]:
            _LIVE.pop(session, None)


# Weekly (was monthly until 2026-10-29): a new user has enough history after
# about a week, and a monthly check left them on the defaults for 30 days.
# `tune` only moves a value on a meaningful difference, so this cannot jitter.
RETUNE_EVERY_S = 7 * 86400
_retune_checked = 0.0


def _last_tune_age(now: float) -> float | None:
    """Seconds since the last tune run (stored as UTC), None if never or unreadable."""
    last = config.section("tune").get("last_run")
    try:
        return now - calendar.timegm(time.strptime(last, "%Y-%m-%dT%H:%M:%SZ")) if last else None
    except (TypeError, ValueError):
        return None


def _maybe_retune(now: float) -> None:
    """Weekly, in its own thread: re-fit the idle threshold and the warm-swap
    payback limit to the user's latest habits. `tune` itself refuses to change
    anything until enough real data exists, and falls back to the default on
    thin evidence."""
    global _retune_checked
    if now - _retune_checked < 86400:
        return
    _retune_checked = now
    age = _last_tune_age(now)
    if age is not None and age < RETUNE_EVERY_S:
        return

    def run():
        try:
            from . import tune
            tune.run(apply=True)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def _bg_worker() -> None:
    """Prepare compactions for quiet sessions, one at a time, forever. Every
    failure is swallowed: this thread must never touch the relay.

    The compaction itself is rules only (no model, seconds, any machine). The
    optional prune stages use local models, which are large (the 35B ~28 GB in
    memory, Nimble ~16 GB), so with `bg_yield` on they wait whenever the
    machine strains (memory pressure, swap growth, heat, low battery; see
    machine.py), load only if they fit with headroom, and unload after each
    step. Without ollama the prune tier is off and nothing waits on it.
    It does NOT wait for the developer to stop using Claude: that is when the
    savings matter."""
    global BG_PREP_IDLE_S, BG_APPLY_AFTER_S
    while True:
        time.sleep(30)
        try:
            # Tuned or user-changed timings reach the running proxy here. A
            # prepared compaction is self-contained (cut, fingerprint, text),
            # so changing sizes between prepare and apply cannot break it.
            BG_PREP_IDLE_S = config.get("bg_prep_idle_min") * 60
            BG_APPLY_AFTER_S = config.get("bg_after_min") * 60
            background.refresh()
            now = time.time()
            _evict(now)
            _maybe_retune(now)
            # Working while the developer uses Claude is fine (that is when the
            # savings matter); straining the machine is not. With `bg_yield`
            # on (default), any sign of strain pauses the local models. The
            # rules-only prepare takes seconds and no model, so it still runs.
            bad = False
            if config.get("bg_yield"):
                bad, why = machine.strained()
                if not bad and _on_low_battery():          # also covers Linux
                    bad, why = True, f"battery under {BATTERY_MIN}%"
                if bad:
                    _note_strain(why)
            for session, live in list(_LIVE.items()):
                if now - live["last"] < BG_PREP_IDLE_S:
                    if not bad:
                        _prework(session, live)
                    continue
                if not bad and _prune_step(session, live):
                    break               # one heavy step per round
                if live.get("skip_until", 0) > live["last"]:
                    continue
                messages = live["messages"]
                with _STATE_LOCK:
                    st = _bg_load(session)
                applied = st.get("applied") or {}
                if not _prepare_due(st, messages):
                    continue            # nothing much new since the last one
                try:
                    prepared = background.prepare(messages)     # rules only: no model, seconds
                except Exception as exc:
                    live["skip_until"] = now
                    try:
                        vault.record("bg_compaction_error", tool="proxy", session_id=session,
                                     summary=f"prepare failed: {type(exc).__name__}: {exc}"[:500], index=False)
                    except Exception:
                        pass
                    continue
                if prepared is None or not _new_cut_ok(prepared, applied):
                    live["skip_until"] = now   # not eligible; look again after the next request
                    continue
                prepared["total_tokens"] = background.size_tokens(messages)
                if (prepared.get("gate") or {}).get("passed"):
                    prepared.update(_payback(prepared, applied, live))
                prepared["stage"] = "extract"
                with _STATE_LOCK:
                    st = _bg_load(session)
                    st["pending"] = prepared
                    _bg_save(session, st)
                try:
                    vault.record("bg_compaction_prepared", tool="proxy", session_id=session,
                                 summary=f"prepared: {prepared['old_tokens']:,} -> "
                                         f"{prepared['new_tokens']:,} tokens, gate "
                                         f"{ {k: v for k, v in prepared['gate'].items() if k != 'items'} }",
                                 meta={k: v for k, v in prepared.items() if k != "text"})
                except Exception:
                    pass
        except Exception:
            pass


# ---- staged pruning on top of a prepared compaction -------------------------
#
# extract (rules only, seconds, any machine) -> fast (overlap pairs judged by
# Nimble, minutes) -> deep (the local 35B spots restatements, Nimble judges)
# -> lines (the 35B picks lines to keep) -> one final check (the 35B). The
# stages build a candidate; only a finished, checked (or, on the fast tier,
# Nimble-only) candidate replaces the rules-only compaction, so a break that
# ends early applies rules-only: never worse than without the models.
# Verdicts are cached per session and pre-worked while the developer works
# (`_prework`), so most of the judging is done before the break.
# Measured on real breaks at p >= 0.8: fast 5.5%, fast+deep 7.1% smaller
# compacted text, 0 items lost that Claude used later.

_STRAIN: list[dict] = []


def _note_strain(why: str) -> None:
    _STRAIN.append({"t": time.time(), "why": why})
    del _STRAIN[:-50]


def _prune_cache(session: str) -> "prune.Cache":
    return prune.Cache(_bg_path(session).with_suffix(".prune.json"))


def _prework(session: str, live: dict) -> bool:
    """A little judging ahead of the next free point, while the session is in
    use. Never raises."""
    try:
        t, _ = prune.tier()
        if t == "off":
            return False
        if time.time() - live.get("prework_at", 0) < 60:
            return False
        live["prework_at"] = time.time()
        n = prune.prework(live["messages"], config.get("prune_judge"), _prune_cache(session),
                          machine.Yield(), budget_s=15)
        return n > 0
    except Exception:
        return False


def _ids_to_refs(old: list, drop) -> list:
    """Deleted content items as [message index, item index]: positions persist,
    object ids do not, and a later stage builds on the deletions in force."""
    refs = []
    for mi, m in enumerate(old):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            refs += [[mi, ii] for ii, x in enumerate(c) if id(x) in (drop or ())]
    return refs


def _refs_to_ids(messages: list, refs: list) -> set[int]:
    out = set()
    for mi, ii in refs:
        try:
            out.add(id(messages[mi]["content"][ii]))
        except (IndexError, KeyError, TypeError):
            continue
    return out


def _body(text: str) -> str:
    return text.split("\n\n## Identifiers from the earlier part\n\n")[0]


def _check_model() -> str | None:
    """The model that runs the final check, if this machine has it."""
    return background.LOCAL_MODEL if prune._name(background.LOCAL_MODEL, prune.installed()) else None


def _prune_step(session: str, live: dict) -> bool:
    """Run the next prune stage for this session's pending compaction, if any.
    Returns whether it did heavy work. Never raises.

    The stages build a candidate (`pending["pruned"]`) and never replace the
    applicable compaction on their way, so a break that ends mid-way applies
    the rules-only one (or an earlier finished result). After the tier's last
    stage the candidate gets ONE check, where the 35B exists: questions from
    the rules-only text of the stretches pruning changed, answered from the
    pruned text. Pass: it becomes the pending compaction; fail: rules-only
    stays. The fast tier (Nimble only, no 35B) is promoted unchecked: Nimble
    deletes a piece only at p >= prune_min_p."""
    try:
        t, _ = prune.tier()
        if t == "off":
            return False
        with _STATE_LOCK:
            st = _bg_load(session)
        pending = st.get("pending") or {}
        if not (pending.get("gate") or {}).get("passed") or pending.get("prune_done"):
            return False
        if pending.get("stage", "extract") != "extract" and "pruned" not in pending:
            return False                    # pruned and gated by the per-stage rule (before 2026-10-23)
        # The deep tier runs fast, deep (the 35B spots restatements), then
        # lines (the 35B picks lines of old output to keep; prune.line_picker).
        stages = ["fast"] if t == "fast" else ["fast", "deep", "lines"]
        cand = dict(pending.get("pruned") or {})
        done = cand.get("tried")
        nxt = stages.index(done) + 1 if done in stages else 0
        messages = live["messages"]
        cut = int(pending["cut"])
        if background.apply(messages, pending) is None:
            return False                    # the history moved on; a new prepare will come
        if nxt < len(stages):
            stage = stages[nxt]
            drop = _refs_to_ids(messages, cand.get("drop_refs") or [])
            before = int(cand.get("new_tokens") or pending["new_tokens"])
            if stage == "lines":
                ok, why = prune.can_load(background.LOCAL_MODEL)
                if not ok:
                    vault.record("bg_prune", tool="proxy", session_id=session, index=False,
                                 summary=f"lines stage paused: {why}", meta={"stage": stage, "stopped": why})
                    return True
                shrink, info = _line_picks(session, messages, cut)
                if info.get("stopped"):
                    prune.unload(background.LOCAL_MODEL)
                    vault.record("bg_prune", tool="proxy", session_id=session, index=False,
                                 summary=f"lines stage paused: {info['stopped']}", meta=info)
                    return True
                res, _ = background._extract(messages, cut, drop, shrink=shrink)
            else:
                drop, info = prune.run_stage(messages, cut, stage, judge_model=config.get("prune_judge"),
                                             min_p=config.get("prune_min_p"), cache=_prune_cache(session),
                                             yielder=machine.Yield(),
                                             protect=background.latest_doc_views(messages[:cut]))
                if drop is None:
                    vault.record("bg_prune", tool="proxy", session_id=session, index=False,
                                 summary=f"{stage} stage paused: {info.get('stopped')}", meta=info)
                    return True
                res, _ = background._extract(messages, cut, drop)
            better = res["new_tokens"] < before
            if better:
                cand = {"text": res["text"], "new_tokens": res["new_tokens"], "stage": stage, "info": info,
                        "drop_refs": _ids_to_refs(messages[:cut], drop), "lines": stage == "lines"}
            cand["tried"] = stage
            vault.record("bg_prune", tool="proxy", session_id=session, index=False,
                         summary=f"{stage} stage: {before:,} -> {res['new_tokens']:,} tokens, "
                                 f"{'kept for the check' if better else 'not smaller'} "
                                 f"({info.get('deleted')} pieces)",
                         meta=dict(info, stage=stage, before=before, after=res["new_tokens"], kept=better))
            if stage != stages[-1]:
                prune.unload(background.LOCAL_MODEL)
                _save_candidate(session, pending, cand, done=False)
                return True
        # Final: check (where the 35B exists) and promote, or keep rules-only.
        verdict, gate = "nothing smaller", {}
        if cand.get("text"):
            model = _check_model()
            if model is None:
                verdict, gate = "promoted unchecked", {"passed": True, "check": "none (fast tier, no check model)"}
            else:
                ok, why = prune.can_load(model)
                if not ok:
                    _save_candidate(session, pending, cand, done=False)
                    vault.record("bg_prune", tool="proxy", session_id=session, index=False,
                                 summary=f"final check paused: {why}", meta={"stage": "check", "stopped": why})
                    return True
                gate = _final_check(session, messages, cut, cand)
                verdict = "promoted (check passed)" if gate.get("passed") else "rules-only kept (check failed)"
            prune.unload(background.LOCAL_MODEL)
        promoted = verdict.startswith("promoted")
        _save_candidate(session, pending, cand, done=True, gate=gate if promoted else None)
        vault.record("bg_prune", tool="proxy", session_id=session, index=False,
                     summary=f"prune result: {pending['new_tokens']:,} -> {cand.get('new_tokens', pending['new_tokens']):,} "
                             f"tokens, {verdict}",
                     meta={"stage": "final", "before": int(pending["new_tokens"]),
                           "after": int(cand.get("new_tokens") or pending["new_tokens"]), "promoted": promoted,
                           "verdict": verdict, "gate": {k: v for k, v in gate.items() if k != "items"}})
        return True
    except Exception as exc:
        try:
            vault.record("bg_prune", tool="proxy", session_id=session, index=False,
                         summary=f"prune error: {type(exc).__name__}: {exc}"[:300])
        except Exception:
            pass
        return False


def _line_picks(session: str, messages: list, cut: int):
    current = "\n".join(background.render(messages[cut:], 300, 200))[-6_000:]
    lcache = prune.LineCache(_bg_path(session).with_suffix(".lines.json"))
    try:
        return prune.line_picker(current, background.mentioned(messages), lcache, machine.Yield())
    finally:
        lcache.save()


def _final_check(session: str, messages: list, cut: int, cand: dict) -> dict:
    """The one check on a pruned candidate: the standard questions, then the
    ones the current work might need, both written from the rules-only text
    of the stretches pruning changed and answered from the pruned text."""
    drop = _refs_to_ids(messages, cand.get("drop_refs") or [])
    shrink = None
    if cand.get("lines"):
        shrink, info = _line_picks(session, messages, cut)     # cached picks: no new model work
        if info.get("stopped"):
            return {"passed": False, "why": f"line picks unavailable: {info['stopped']}"}
    _, rules_for = background._extract(messages, cut)
    _, pruned_for = background._extract(messages, cut, drop, shrink=shrink)
    changed = lambda i, j: _body(rules_for(i, j)) != _body(pruned_for(i, j))
    current = "\n".join(background.render(messages[cut:], 300, 200))[-6_000:]
    old = messages[:cut]
    std = background.gate(old, pruned_for, original_for=rules_for, only=changed)
    if not std.get("passed"):
        return std
    cur = background.gate(old, pruned_for, current=current, original_for=rules_for, only=changed)
    return dict(std, current={k: v for k, v in cur.items() if k != "items"}, passed=bool(cur.get("passed")),
                check="final: pruned vs rules-only")


def _save_candidate(session: str, pending: dict, cand: dict, *, done: bool, gate: dict | None = None) -> None:
    """Store the candidate; with `gate`, promote it to the pending compaction."""
    with _STATE_LOCK:
        st = _bg_load(session)
        cur = st.get("pending") or {}
        if cur.get("cut") != pending["cut"] or cur.get("made") != pending.get("made"):
            return                          # replaced or applied meanwhile
        new = dict(cur, pruned=cand, prune_done=done)
        if gate is not None:
            new.update(text=cand["text"], new_tokens=cand["new_tokens"], gate=gate, stage=cand["stage"],
                       prune=cand.get("info"), drop_refs=cand.get("drop_refs"))
            new.update(_payback(new, st.get("applied") or {}, _LIVE.get(session) or {}))
        st["pending"] = new
        _bg_save(session, st)


def _prepare_due(st: dict, messages: list) -> bool:
    """Enough new history since the last compaction prepared OR applied.
    Measured: after an apply `pending` was empty, so the old check passed at
    once, re-prepared the same cut and applied it again on a warm cache for no
    saving ($1.92). The baseline is whichever is newer."""
    pending, applied = st.get("pending") or {}, st.get("applied") or {}
    base = max(int(pending.get("total_tokens") or 0), int(applied.get("total_tokens") or 0))
    return not base or background.size_tokens(messages) - base >= background.KEEP_TOKENS


def _new_cut_ok(prepared: dict, applied: dict) -> bool:
    """A new compaction must cut later than the one in force, or it is the
    same summary again."""
    return not applied or int(prepared.get("cut") or 0) > int(applied.get("cut") or 0)


def _payback(prepared: dict, applied: dict, live: dict) -> dict:
    """How many later requests a compaction needs to repay itself if applied
    while the cache is still warm.

    Applying on a warm cache writes the new prefix (1h write, 40x the read
    price on Opus 5.5) instead of reading the old one; every later request then
    reads fewer tokens. Measured in the first trial: a second compaction on top
    of the first saved only 68K tokens per request against what was already
    being sent, needed ~180 requests to repay and got ~40. Compared with what
    is sent NOW, never with the original history.

    `sent_new` is scaled from the local estimate. The worker has no
    credentials; the decision at the free point (`_decide`) counts real
    tokens and supersedes this whenever it can."""
    sent_now = int(live.get("sent_tokens") or 0)
    if not sent_now:
        return {"payback_requests": None, "payback_why": "current size unknown"}
    total = int(prepared.get("total_tokens") or 0)
    already = int(applied.get("old_tokens") or 0) - int(applied.get("new_tokens") or 0)
    current = total - max(already, 0)
    new = total - (int(prepared["old_tokens"]) - int(prepared["new_tokens"]))
    if current <= 0:
        return {"payback_requests": None, "payback_why": "no estimate"}
    sent_new = round(sent_now * new / current)
    p = pricing.price_for((live.get("skeleton") or {}).get("model") or "")
    w1, rd = (p.write_1h, p.read) if p else (8.0, 0.2)
    saved = sent_now - sent_new
    out = {"sent_now": sent_now, "sent_new": sent_new}
    if saved <= 0:
        return dict(out, payback_requests=None, payback_why="saves nothing against what is sent now")
    n = (sent_new * w1 - sent_now * rd) / (saved * rd)
    return dict(out, payback_requests=round(n, 1))


def _warm_ok(pending: dict) -> tuple[bool, str]:
    n = pending.get("payback_requests")
    limit = config.get("bg_warm_payback_max")
    if n is None:
        return False, pending.get("payback_why") or "no payback estimate"
    if n > limit:
        return False, f"payback {n:.0f} > {limit:.0f} requests"
    return True, f"payback {n:.0f} <= {limit:.0f} requests"


ESTIMATE_MARGIN = 0.2      # with no real count, compact only if the estimate saves 20% of the old part


def _estimate_margin_ok(pending: dict) -> bool:
    old, new = int(pending.get("old_tokens") or 0), int(pending.get("new_tokens") or 0)
    return old > 0 and old - new >= ESTIMATE_MARGIN * old


def _commit(session: str, st: dict, pending: dict, messages: list, alias_choice: dict | None) -> None:
    """Make `pending` the applied compaction from this request on. Caller holds _STATE_LOCK."""
    # The kept tail's thinking up to here is stripped (see background.apply);
    # later thinking is bound to the new prefix.
    st["applied"], st["pending"] = dict(pending, tail_end=len(messages)), None
    # A free point: the alias choice measured for this compaction (or none)
    # takes effect with it, and is frozen until the next.
    st["aliases"] = dict(alias_choice or {"mode": "off", "why": "no measurement"}, since="compaction")
    st["free_point"] = "compaction"
    _bg_save(session, st)


def _bg(messages: list, session: str | None, skeleton: dict | None = None,
        headers: dict | None = None, decided: bool = False) -> tuple[list, dict]:
    """Apply a prepared compaction where it is free and already decided."""
    global _BG_STARTED
    if not BG_COMPACT or not session:
        return messages, {}
    if not _BG_STARTED:
        _BG_STARTED = True
        threading.Thread(target=_bg_worker, daemon=True).start()
    now = time.time()
    prev = _LIVE.get(session)
    # skeleton (model, system, tools) stays in memory only. Auth headers are
    # NOT kept here (user decision, 2026-10-22): credentials live only for an
    # in-flight request and the audit jobs it queues. Real counts for a
    # free-point decision are made on the request that reaches it.
    _LIVE[session] = {"last": now, "messages": messages, "skeleton": skeleton or {}}
    with _STATE_LOCK:
        st = _bg_load(session)
        info: dict = {}
        pending = st.get("pending")
        # The previous request's time: from memory, else from the state file,
        # so a proxy restart does not lose the moment. A session never seen
        # counts as warm: never pay to rebuild a cache that may be live.
        last = prev["last"] if prev is not None else st.get("last")
        expired = last is not None and now - float(last) > BG_APPLY_AFTER_S
        if last is not None:
            info["gap_s"] = round(now - float(last))     # for the alias free-point check
        st["last"] = now
        _bg_save(session, st)
        cold = last is not None and now - float(last) > EXPIRED_S
        if expired and pending and (pending.get("gate") or {}).get("passed"):
            # Cold: the cache is being rebuilt anyway, so a smaller history is
            # strictly cheaper. Warm: only if it repays before the next free
            # point is likely (measured on this user's breaks; see _payback).
            if headers and audit.enabled() and not decided:
                # Real counts decide (`_decide`, on this request's own auth,
                # in `_rewrite`); nothing is committed here.
                info["bg_decide"] = {"cold": cold}
            else:
                ok, why = (True, "cache expired") if cold else _warm_ok(pending)
                if ok and cold and not _estimate_margin_ok(pending):
                    ok, why = False, "estimated saving under the 20% margin (no real count)"
                if not ok:
                    info["bg_skipped"] = why
                if ok and background.apply(messages, pending) is not None:
                    _commit(session, st, pending, messages, None)
                    info["bg_applied"] = True
        applied = st.get("applied")
        if not applied:
            return messages, info
        out = background.apply(messages, applied)
        if out is None:
            # The history no longer starts the same (Claude Code compacted or
            # rewound): this summary describes something else now.
            st["applied"] = None
            _bg_save(session, st)
            return messages, dict(info, bg_dropped=True)
        return out, dict(info, bg_cut=applied["cut"], bg_made=applied.get("made"),
                         bg_tokens=f"{applied['old_tokens']}->{applied['new_tokens']}",
                         bg_loaded_tools=background.loaded_tools(messages[:int(applied["cut"])]))


# ---- aliases: decided at free points, frozen between them -------------------
#
# Measured on real traffic: aliases defined as they go ("auto", removed) ADDED
# 2.4-3.3% tokens: `%name%` and its definition line tokenize worse than the
# paths unless a path recurs a lot. A plan picked in hindsight saved 1.8%. So a
# session's alias behaviour changes only when its cache is being rebuilt
# anyway, and is frozen in between so its bytes never change:
# - a compaction: real tokens are counted with and without a plan on the
#   request that applies it (`_decide`), and the cheaper one takes effect;
# - Claude Code rebuilt the history itself (/compact): the old plan
#   describes paths that are gone (measured: it cost 4.7% after a /compact),
#   so aliases go off until the next compaction measures again.
# A new session starts with "off".


def _hist_fp(messages: list) -> str:
    return background.fingerprint(_canon(messages[:2]))


# ---- forward passes: switched off where they measure as a net cost ----------
#
# The audit prices every forward pass on every counted request (one extra
# count with only that pass toggled). `pass_verdicts` reads the recent counts:
# a pass whose measured effect over at least PASS_MIN_EVIDENCE requests that it
# changed is a net cost goes off; back on only past the same margin the other
# way (hysteresis). Like aliases, a session adopts the verdict only at a free
# point, and even then only for blocks sent after it (the ledger replays the
# rest). path_aliases is decided on real counts at compactions (`_decide`).

PASS_MIN_EVIDENCE = 20
# Measured 2026-10-16: 40 counts from ONE session (right after a /compact, with
# the stale alias plan since fixed) switched dedup_noise and format off at
# -5% each; format had changed 13 characters of a request said to cost 6,200
# tokens. So evidence must span sessions, and a count whose token effect is
# larger than the characters toggled (plus slack for JSON escapes) is dropped.
PASS_MIN_SESSIONS = 2
PASS_PER_SESSION = PASS_MIN_EVIDENCE // 2    # no one session supplies over half
PASS_CHAR_SLACK = 64
PASS_MARGIN = 0.0005          # 0.05% of the tokens of the requests it touched
PASS_WINDOW = 300             # most recent counted requests considered
PASS_STALE_S = 14 * 86400     # evidence older than this is not evidence
_VERDICT: dict = {"at": 0.0, "off": (), "table": {}}


def pass_verdicts(now: float | None = None) -> tuple[str, ...]:
    """Forward passes to leave off at the next free point. Cached 10 min.

    Measuring is an enhancement, never a dependency: with no counts (audit
    off, refused, down), too few, or none in the last PASS_STALE_S, every pass
    stays on, the defaults the offline gates validated. Never raises."""
    import sqlite3
    now = time.time() if now is None else now
    if now - _VERDICT["at"] < 600:
        return _VERDICT["off"]
    _VERDICT["at"] = now
    stored = set(config.section("passes").get("off") or [])
    prev = stored
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - PASS_STALE_S))
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        rows = conn.execute("SELECT meta, session_id FROM events WHERE kind='token_audit' AND ts >= ? "
                            "AND json_extract(meta, '$.passes') IS NOT NULL "
                            # Counts from before 2026-10-22 kept thinking in the
                            # bodies, which made every toggle look like a cost
                            # (see audit.py); only thinking-free counts are evidence.
                            "AND json_extract(meta, '$.method') = 'no_thinking' "
                            "ORDER BY id DESC LIMIT ?", (since, PASS_WINDOW)).fetchall()
        conn.close()
    except Exception:
        _VERDICT.update(off=(), table={})
        return ()
    if not rows:
        prev = set()                       # nothing recent: back to the defaults
    table, seen = {}, {}
    for (m, sess) in rows:
        try:
            meta = json.loads(m or "{}")
        except ValueError:
            continue
        for name, p in (meta.get("passes") or {}).items():
            if name == "path_aliases" or not p.get("changed"):
                continue
            try:
                saving = int(p.get("saving") or 0)
                if p.get("chars") is not None and abs(saving) > int(p["chars"]) + PASS_CHAR_SLACK:
                    continue                   # impossible: a broken count
                sent = int(meta.get("tokens_sent") or 0)
            except (TypeError, ValueError):
                continue
            per = seen.setdefault(name, {})
            if per.get(sess, 0) >= PASS_PER_SESSION:
                continue                       # newest first: this session's say is used up
            per[sess] = per.get(sess, 0) + 1
            t = table.setdefault(name, [0, 0, 0])
            t[0] += 1
            t[1] += saving
            t[2] += sent
    off = set(prev)
    for name, (n, saving, total) in table.items():
        if n < PASS_MIN_EVIDENCE or not total or len(seen.get(name, ())) < PASS_MIN_SESSIONS:
            continue
        share = saving / total
        if share < -PASS_MARGIN:
            off.add(name)
        elif share > PASS_MARGIN:
            off.discard(name)
    result = tuple(sorted(off))
    _VERDICT.update(at=now, off=result, table=table)
    if off != stored:
        try:
            config.set_section("passes", {"off": list(result), "at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "table": table})
        except Exception:
            pass
    return result


def _session_setup(session: str | None, messages: list | None = None,
                   gap_s: float | None = None) -> tuple[object, tuple, dict]:
    """(aliases argument, passes to exclude, stats) for this request, frozen
    between free points. `messages` is the history as Claude Code sent it.

    Free points, where the session takes the current pass verdict:
    - its first request seen;
    - an expired cache (gap over the TTL);
    - a compaction applied (flagged by `_bg`);
    - Claude Code rebuilt the history (/compact: a new START). Aliases also go
      off then. A rewind keeps the start, and that prefix is still cached with
      the session's bytes, so it is not a free point.
    Never raises."""
    if not session:
        return "off", (), {}
    try:
        with _STATE_LOCK:
            st = _bg_load(session)
            a = st.get("aliases") or {}
            if a.get("mode") not in ("off", "plan"):
                a = {"mode": "off", "since": "first seen"}
            free = None
            if "passes_off" not in st:
                free = "first seen"
            elif gap_s is not None and gap_s > EXPIRED_S:
                free = "cache expired"
            if st.pop("free_point", None):
                free = "compaction"
            if isinstance(messages, list) and messages:
                fp = _hist_fp(messages)
                if st.get("hist_fp") not in (None, fp):
                    free = "history rebuilt"
                    if a.get("mode") != "off":
                        a = {"mode": "off", "since": "history rebuilt",
                             "why": "Claude Code replaced the history; the plan was for the old one"}
                st["hist_fp"] = fp
            if free:
                st["passes_off"] = list(pass_verdicts())
                st["passes_since"] = free
            st["aliases"], st["alias_last"] = a, time.time()
            _bg_save(session, st)
        arg = a.get("plan") if a.get("mode") == "plan" and a.get("plan") else "off"
        stats = {"aliases": a.get("mode")}
        if st.get("passes_off"):
            stats["passes_off"] = list(st["passes_off"])
        return arg, tuple(st.get("passes_off") or ()), stats
    except Exception:
        return "off", (), {}


# ---- automatic fallback --------------------------------------------------
#
# A rewrite that ever breaks the prompt cache bills the whole history as fresh
# input on every turn, silently. So the proxy watches for it and stops itself:
# a session whose cache stops being read while it should be warm, or whose
# rewrites keep failing, goes back to plain relay; two such sessions in a day
# turn rewriting off everywhere until a person clears it. Requests that are
# legitimately cold do not count: a session's first, one after a gap longer
# than the cache TTL, and the one that applies a compaction.

FALLBACK_ZERO_RUN = 3
FALLBACK_ERRORS = (3, 10)        # 3 rewrite errors within the last 10 requests
FALLBACK_GLOBAL = (2, 86400)     # 2 session fallbacks within 24h
_WATCH: dict[str, dict] = {}
_FELL_BACK: set[str] = set()


def _global_off() -> dict:
    st = config.section("fallback")
    return st if st.get("global_off") else {}


def _watch(session: str | None, request_class: str | None, path: str,
           usage: dict, stats: dict) -> None:
    """Called after the relay with the real usage. Never raises into it."""
    if not session or (request_class or "main") != "main" or path.split("?")[0] != "/v1/messages":
        return
    if not stats or session in _FELL_BACK:
        return
    now = time.time()
    w = _WATCH.setdefault(session, {"last": None, "zero": 0, "errors": []})
    warm = w["last"] is not None and now - w["last"] < EXPIRED_S - 60
    w["last"] = now
    w["errors"] = (w["errors"] + [bool(stats.get("error"))])[-FALLBACK_ERRORS[1]:]
    cold_ok = not warm or stats.get("bg_applied")
    if int(usage.get("cache_read_input_tokens") or 0) or cold_ok:
        w["zero"] = 0
    else:
        w["zero"] += 1
    why = None
    if w["zero"] >= FALLBACK_ZERO_RUN:
        why = f"{w['zero']} warm requests in a row read no cache"
    elif sum(w["errors"]) >= FALLBACK_ERRORS[0]:
        why = f"{sum(w['errors'])} rewrite errors in the last {len(w['errors'])} requests"
    if not why:
        return
    _FELL_BACK.add(session)
    st = config.section("fallback")
    recent = [e for e in st.get("events", []) if now - e.get("t", 0) < FALLBACK_GLOBAL[1]]
    recent.append({"t": now, "session": session, "why": why})
    st["events"] = recent[-20:]
    if len(recent) >= FALLBACK_GLOBAL[0] and not st.get("global_off"):
        st.update(global_off=True, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                  why=f"{len(recent)} sessions fell back within 24h; last: {why}")
    try:
        config.set_section("fallback", st)
        vault.record("rewrite_fallback", tool="proxy", session_id=session,
                     summary=f"rewriting stopped for this session: {why}"
                             + (" (and globally)" if st.get("global_off") else ""),
                     meta={"why": why, "global_off": bool(st.get("global_off"))}, index=False)
    except Exception:
        pass


# ---- quality after compaction ---------------------------------------------
#
# Tokens saved say nothing about whether Claude got worse. A compaction that
# hides something Claude needed shows up as behaviour: it reads a file again
# that it had already read, re-runs a command it already ran, or an edit fails.
# Each session is compared with itself: the rate of those over the 30 requests
# before a compaction against the 30 after. Local, no API calls.

QUALITY_WINDOW = 30
_Q: dict[str, dict] = {}


def _scan(new_messages: list, q: dict) -> tuple[int, int, int]:
    """(re-reads, re-runs, failed edits) in messages not seen before."""
    rereads = reruns = failed = 0
    for m in new_messages:
        items = m.get("content") if isinstance(m, dict) else None
        if not isinstance(items, list):
            continue
        for x in items:
            if not isinstance(x, dict):
                continue
            if x.get("type") == "tool_use":
                inp = x.get("input") or {}
                if x.get("name") == "Read" and inp.get("file_path"):
                    rereads += inp["file_path"] in q["reads"]
                    q["reads"].add(inp["file_path"])
                elif x.get("name") == "Bash" and isinstance(inp.get("command"), str):
                    reruns += inp["command"] in q["cmds"]
                    q["cmds"].add(inp["command"])
                if x.get("name") in ("Edit", "MultiEdit", "Write"):
                    q["edits"].add(x.get("id"))
            elif x.get("type") == "tool_result" and x.get("is_error") and x.get("tool_use_id") in q["edits"]:
                failed += 1
    return rereads, reruns, failed


def _quality(session: str | None, messages: list, stats: dict) -> None:
    if not session or not isinstance(messages, list):
        return
    q = _Q.setdefault(session, {"seen": 0, "reads": set(), "cmds": set(), "edits": set(),
                                "window": [], "after": None})
    if len(messages) < q["seen"]:        # history replaced: start over
        q.update(seen=0, reads=set(), cmds=set(), edits=set(), window=[], after=None)
    counts = _scan(messages[q["seen"]:], q)
    q["seen"] = len(messages)
    if stats.get("bg_applied"):
        before = [sum(c[i] for c in q["window"]) for i in range(3)]
        q["after"] = {"left": QUALITY_WINDOW, "before": before, "before_n": len(q["window"]),
                      "sums": [0, 0, 0], "n": 0}
    elif q["after"] is not None:
        a = q["after"]
        a["sums"] = [a["sums"][i] + counts[i] for i in range(3)]
        a["n"] += 1
        a["left"] -= 1
        if a["left"] <= 0:
            names = ("rereads", "reruns", "failed_edits")
            meta = {"before_requests": a["before_n"], "after_requests": a["n"],
                    **{f"before_{n}": a["before"][i] for i, n in enumerate(names)},
                    **{f"after_{n}": a["sums"][i] for i, n in enumerate(names)}}
            vault.record("bg_followup", tool="proxy", session_id=session, meta=meta, index=False,
                         summary=f"after compaction: re-reads {a['before'][0]}->{a['sums'][0]}, "
                                 f"re-runs {a['before'][1]}->{a['sums'][1]}, failed edits "
                                 f"{a['before'][2]}->{a['sums'][2]} over {a['before_n']}/{a['n']} requests")
            q["after"] = None
            _quality_verdict()
    q["window"] = (q["window"] + [counts])[-QUALITY_WINDOW:]
    if len(_Q) > 50:                     # bound memory
        for k in list(_Q)[:len(_Q) - 50]:
            _Q.pop(k, None)


# The safety net for the unchecked rules-only compaction: if, across sessions,
# Claude re-reads, re-runs or fails edits clearly more after compactions than
# before them, compaction turns itself off and says why. Not back on by
# itself, and never over a value the user set by hand (setup's own is fine).
QUALITY_MIN_FOLLOWUPS = 5
QUALITY_MAX_RISE = 0.1          # extra (re-reads + re-runs + failed edits) per request


def quality_rise(follow: list[dict]) -> float | None:
    """After-rate minus before-rate per request, or None with too few."""
    if len(follow) < QUALITY_MIN_FOLLOWUPS:
        return None
    names = ("rereads", "reruns", "failed_edits")
    bn = sum(f.get("before_requests") or 0 for f in follow)
    an = sum(f.get("after_requests") or 0 for f in follow)
    b = sum(f.get(f"before_{k}") or 0 for f in follow for k in names)
    a = sum(f.get(f"after_{k}") or 0 for f in follow for k in names)
    return a / max(an, 1) - b / max(bn, 1)


def _quality_verdict() -> None:
    global BG_COMPACT
    try:
        import sqlite3
        conn = sqlite3.connect(vault.DB_PATH)
        try:
            follow = [json.loads(m or "{}") for (m,) in conn.execute(
                "SELECT meta FROM events WHERE kind='bg_followup' ORDER BY id")]
        finally:
            conn.close()
        rise = quality_rise(follow)
        if rise is None or rise < QUALITY_MAX_RISE:
            return
        value, source = config.resolve("bg_compact")
        slot = ((config.load().get("settings") or {}).get("bg_compact") or {}).get("user") or {}
        if not value or source == "env" or (source == "user" and slot.get("why") not in (None, "", "setup")):
            return
        why = (f"quality signals: {rise:+.2f} re-reads/re-runs/failed edits per request after "
               f"compactions ({len(follow)} compactions)")
        config.set_value("bg_compact", False, "user", why=why, evidence={"rise": round(rise, 3),
                                                                        "followups": len(follow)})
        BG_COMPACT = False
        vault.record("bg_quality_off", tool="proxy", summary=f"background compaction turned OFF: {why}",
                     meta={"rise": rise, "followups": len(follow)}, index=False)
    except Exception:
        pass


# Claude Code also sends `auxiliary` requests that carry the whole conversation
# plus one instruction (prompt suggestions, session recaps). Sent unrewritten,
# their bytes differ from the rewritten main history, so they cannot share its
# cache: measured, one wrote a separate 387K-token cache. So an auxiliary
# request whose history is exactly the session's latest main history (minus
# its final instruction) gets the same rewrite, read-only: it never moves the
# idle timer, the alias state or anything else. Any other auxiliary request
# (quota probes, the auto-mode classifier) is relayed untouched.
_MAIN_HIST: dict[str, list] = {}


AUX_EXTRA_MAX = 4      # main history + the reply it got + an instruction, with room to spare
# Request classes that may carry the main history and then get its bytes, read-only.
SIDE_CLASSES = {"auxiliary", "compaction"}


def _canon(x):
    """A message in a form that ignores two differences that change no content:
    - cache_control markers: Claude Code moves its breakpoint between requests
      (measured: the only difference between a real prompt-suggestion request
      and the main history before it), and a marker says where to cache, not
      what the content is;
    - string content vs one text block: Claude Code sends the newest message
      as `[{"type": "text", "text": s, "cache_control": ...}]` in the main
      request but as the plain string `s` in a recap or suggestion. Measured
      in the first real trial: 3 of 6 side requests in a day failed to match on
      this alone, and each wrote its own 455-560K-token cache ($3.5-4.4)."""
    if isinstance(x, dict):
        out = {k: _canon(v) for k, v in x.items() if k != "cache_control"}
        if isinstance(out.get("content"), str) and "role" in out:
            out["content"] = [{"type": "text", "text": out["content"]}]
        return out
    if isinstance(x, list):
        return [_canon(v) for v in x]
    return x


def _follows_main(session: str | None, messages: list) -> bool:
    """The session's latest main history, except its newest message, is a
    prefix of `messages`, with at most AUX_EXTRA_MAX messages after it.

    The newest main message is excluded because Claude Code changes it for
    side requests: Claude Code's own compaction request (class `compaction`)
    is the same history with the instruction added to that message, so it has
    exactly as many messages as main (measured: 523 and 523, 522 identical).
    Relayed raw, it could not share the main cache and its thinking failed the
    prefix check (240 blocks dropped in one /compact)."""
    main = _MAIN_HIST.get(session or "")
    if not main or len(main) < 2 or not 0 <= len(messages) - len(main) <= AUX_EXTRA_MAX:
        return False
    k = len(main) - 1
    return (json.dumps(_canon(messages[:k]), sort_keys=True)
            == json.dumps(_canon(main[:k]), sort_keys=True))


# ---- sent-bytes ledger: replay what was sent, never recompute it ------------
#
# The forward rewrite is recomputed from the whole history on every request,
# and the cache holds only if every recomputation is byte-identical. Measured:
# once it was not (hash-seed set order after a restart: a 392K rewrite, ~$3).
# So each session keeps the messages it last sent; for the prefix that is
# unchanged since then (same messages from Claude Code, compared without
# cache_control), the stored bytes are sent again, with this request's
# cache_control placement. A recomputation that differs is recorded as drift
# (the cache is protected either way). The ledger restarts at free points,
# where the bytes are meant to change (epoch: compaction cut, alias plan,
# passes off). On disk too, so a restarted proxy replays the same bytes.

_SENT: dict[str, dict] = {}


def _ledger_path(session: str):
    return _bg_path(session).with_suffix(".sent.json")


def _msg_fp(m) -> str:
    return hashlib.sha1(json.dumps(_canon(m), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _epoch_v1(bg_cut, aliases, exclude) -> str:
    return hashlib.sha1(json.dumps([bg_cut, aliases, sorted(exclude or ())], sort_keys=True,
                                   default=str).encode()).hexdigest()[:16]


def _epoch(bg_cut, made=None) -> str:
    """The ledger's epoch: only a compaction starts a new one. Preserved
    thinking binds every thinking block to the exact bytes before it, so
    already-sent bytes never change between compactions; a pass verdict or
    alias change at a free point applies to blocks sent after it only (the
    ledger replays the rest). A compaction strips the kept tail's thinking, so
    its re-render is safe."""
    return "v2:" + hashlib.sha1(json.dumps([bg_cut, made], default=str).encode()).hexdigest()[:16]


def _ledger(session: str) -> dict | None:
    led = _SENT.get(session)
    if led is None:
        try:
            led = json.loads(_ledger_path(session).read_text())
        except (OSError, ValueError):
            return None
        _SENT[session] = led
    return led


def _transplant(stored, cur):
    """`stored` with `cur`'s cache_control placement, or None if their shapes
    differ (then the message is not replayed)."""
    if isinstance(stored, dict) and isinstance(cur, dict):
        if stored.get("type") != cur.get("type") or stored.get("role") != cur.get("role"):
            return None
        out = {k: v for k, v in stored.items() if k != "cache_control"}
        if "cache_control" in cur:
            out["cache_control"] = cur["cache_control"]
        sc, cc = stored.get("content"), cur.get("content")
        if isinstance(sc, list) or isinstance(cc, list):
            if not (isinstance(sc, list) and isinstance(cc, list)) or len(sc) != len(cc):
                return None
            items = [_transplant(s, c) for s, c in zip(sc, cc)]
            if any(x is None for x in items):
                return None
            out["content"] = items
        return out
    return stored if type(stored) is type(cur) else None


def _adopt_reason(session: str | None, messages: list) -> str | None:
    """Why this request's history holds messages Anthropic saw WITHOUT our
    rewrite, or None. Call before `_bg` (which creates the session's state).

    Preserved thinking binds each thinking block to the exact bytes before it,
    so history that went to Anthropic raw must stay raw. Measured 2026-10-23:
    a Claude session already open when routing was turned on (and again when
    the watchdog restored routing after a bypass) had all its earlier thinking
    rejected, 160 blocks on the first proxied request, and every later request
    of that session reported them again.
    - first_seen: the proxy has no ledger or state for the session, yet the
      history already has a reply from Claude (it started direct);
    - gap: routing was (re)enabled in settings.json after this session's last
      proxied request, so requests in between may have gone direct."""
    if not session or not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
        return None
    led = _ledger(session)
    if led is None:
        return None if _bg_path(session).exists() else "first_seen"
    try:
        from . import routing
        rs = routing.state()
        if not (rs.get("managed") and rs.get("on")):
            return None
        t = led.get("t")
        if t is None:
            try:
                t = _ledger_path(session).stat().st_mtime
            except OSError:
                t = 0
        return "gap" if float(rs.get("at") or 0) > float(t) else None
    except Exception:
        return None


def _last_assistant(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
            return i
    return -1


def _replay(session: str, inp: list, out: list, epoch: str, update: bool = True,
            legacy: str | None = None, raw_until: int = 0) -> tuple[list, dict]:
    """`out` with the unchanged prefix replayed from the ledger (see above).
    `update=False` for side requests: they read the main ledger, never write it.
    `legacy`: the pre-v2 epoch for this request; a ledger written under it is
    the same epoch, so a restart onto this code keeps the sent bytes.
    `raw_until`: messages after the replayed prefix and before this index were
    seen by Anthropic as Claude Code sent them (`_adopt_reason`): they go out
    as `inp`, and the ledger keeps them so, for every later request."""
    fps = [_msg_fp(m) for m in inp]
    led = _ledger(session)
    info: dict = {}
    k = 0
    if led and led.get("epoch") in (epoch, legacy):
        old = led.get("fp") or []
        while k < min(len(fps), len(old), len(out)) and fps[k] == old[k]:
            k += 1
        res, drift, skipped = list(out), 0, 0
        for i in range(k):
            stored = led["sent"][i]
            if json.dumps(_canon(stored), sort_keys=True) != json.dumps(_canon(out[i]), sort_keys=True):
                drift += 1
            t = _transplant(stored, out[i])
            if t is None:
                skipped += 1
            else:
                res[i] = t
        out = res
        info = {"replayed": k}
        if drift:
            info["drift"] = drift
        if skipped:
            info["replay_skipped"] = skipped
    if raw_until > k:
        out = list(out)
        for i in range(k, min(raw_until, len(out), len(inp))):
            out[i] = inp[i]
        info["adopted"] = min(raw_until, len(out)) - k
    if update:
        led = {"epoch": epoch, "fp": fps, "sent": out, "t": time.time()}
        _SENT[session] = led
        if len(_SENT) > 20:
            _SENT.pop(next(iter(_SENT)))
        try:
            p = _ledger_path(session)
            p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(led, ensure_ascii=False))
            os.replace(tmp, p)
        except OSError:
            pass
    if info.get("drift"):
        try:
            vault.record("rewrite_drift", tool="proxy", session_id=session, index=False,
                         summary=f"{info['drift']} message(s) recomputed differently; stored bytes sent",
                         meta=info)
        except Exception:
            pass
    return out, info


# ---- "if you hit enter now": the status line's estimate ---------------------
#
# What the next request would save: what the last one saved (forward passes,
# the compaction in force), plus a prepared, gated compaction if it would apply
# right now (quiet long enough, and cold or within the payback limit). So the
# number grows while the developer is away: the prune stages make the pending
# compaction smaller, and the idle time crosses the threshold. An estimate in
# bytes, from local state only; the real tokens are in `--stats`.

_NOW: dict[str, dict] = {}


_FACTOR = {"at": 0.0, "value": None}


def sim_factor() -> float:
    """Real tokens per estimated token a compaction removes (`sim.token_factor`,
    refit from this user's own counted compactions), cached for 10 minutes."""
    if time.time() - _FACTOR["at"] > 600:
        try:
            from . import sim
            _FACTOR["value"] = sim.token_factor()[0]
        except Exception:
            _FACTOR["value"] = 0.92
        _FACTOR["at"] = time.time()
    return _FACTOR["value"] or 0.92


def estimate(session: str) -> dict:
    """{"pct": saved if sent now, "next_pct"/"in_s": a bigger saving that a
    longer pause would unlock, "done": nothing left that waiting could improve},
    or {} when this session is unknown. Never raises."""
    try:
        n = _NOW.get(session)
        if not n or not n["in"]:
            return {}
        out, base = float(n["out"]), float(n["in"])
        res = {"pct": round(100 * (1 - out / base)), "done": True, "basis": "bytes"}
        # Real tokens when the session has a free count (it lags one request:
        # the count runs after the reply). Bytes overstate the saving: what is
        # removed is partly text that tokenizes cheaply.
        real = audit.LATEST.get(session)
        if real and real["orig"]:
            res["pct"], res["basis"] = round(100 * (1 - real["sent"] / real["orig"])), "tokens"
            to_real = lambda est_tokens: est_tokens * sim_factor()   # an estimate in real tokens
            out, base = float(real["sent"]), float(real["orig"])
        else:
            to_real = lambda est_tokens: est_tokens * background.CHARS_PER_TOKEN   # in bytes
        if not BG_COMPACT:
            return res
        st = _bg_load(session)
        pending, applied = st.get("pending") or {}, st.get("applied") or {}
        if not (pending.get("gate") or {}).get("passed"):
            # Nothing ready: still improving if the worker has a compaction to make.
            live = _LIVE.get(session) or {}
            msgs = live.get("messages")
            if msgs and background.worth_compacting(msgs) and _prepare_due(st, msgs) \
                    and live.get("skip_until", 0) <= live.get("last", 0):
                res["done"] = False
            return res
        if prune.tier()[0] != "off" and not pending.get("prune_done") \
                and not (pending.get("stage", "extract") != "extract" and "pruned" not in pending):
            res["done"] = False             # prune stages (or their final check) still to run
        red = lambda c: int(c.get("old_tokens") or 0) - int(c.get("new_tokens") or 0)
        extra = red(pending) - (red(applied) if applied else 0)
        if extra <= 0:
            return res
        nxt = round(100 * (1 - max(out - to_real(extra), 0) / base))
        gap = time.time() - n["t"]
        if gap > BG_APPLY_AFTER_S and (gap > EXPIRED_S or _warm_ok(pending)[0]):
            res["pct"] = nxt
        elif gap <= BG_APPLY_AFTER_S:
            res.update(next_pct=nxt, in_s=round(BG_APPLY_AFTER_S - gap), done=False)
        elif gap <= EXPIRED_S:
            res["done"] = False             # warm and not worth it yet; the cache expiring will make it free
        return res
    except Exception:
        return {}


def _headline() -> list[str]:
    """Tokens and dollars together: a token figure alone hides cache rebuilds
    (measured: Headroom's recompressing mode sent 11% fewer tokens and cost 37%
    more on the same ticket). Never raises."""
    import sqlite3
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        rows = [json.loads(m or "{}") for (m,) in conn.execute(
            "SELECT meta FROM events WHERE kind='token_audit' AND json_extract(meta, '$.reason')='every_n'"
            " AND json_extract(meta, '$.method') = 'no_thinking'")]
        conn.close()
        o = sum(r.get("tokens_original") or 0 for r in rows)
        t = sum(r.get("tokens_sent") or 0 for r in rows)
        from . import bill
        b = bill.compute()
    except Exception:
        return []
    parts = []
    if o:
        parts.append(f"{100 * (1 - t / o):.1f}% fewer tokens sent than plain Claude ({len(rows):,} counted requests"
                     " since the thinking-fair count of 2026-10-22)")
    if b.plain:
        parts.append(f"{100 * b.saved / b.plain:+.1f}% on the bill (${b.saved:+,.2f})")
    return ["Headline: " + "; ".join(parts), ""] if parts else []


def _keep_loaded(request: dict, stats: dict) -> None:
    """Tools that the compacted part loaded through tool search stay loaded:
    sent undeferred while the compaction is in force (see
    `background.loaded_tools`). Changes the tools only from the request that
    applies the compaction on, where the prefix is rewritten anyway."""
    names = stats.pop("bg_loaded_tools", None)
    tools = background.undefer(request.get("tools"), names)
    if tools is not None:
        request["tools"] = tools
        stats["bg_undeferred"] = len(names)


def _bg_peek(messages: list, session: str) -> tuple[list, dict]:
    """The session's applied compaction, if any, with no state change."""
    if not BG_COMPACT:
        return messages, {}
    applied = _bg_load(session).get("applied")
    out = background.apply(messages, applied) if applied else None
    if out is None:
        return messages, {}
    return out, {"bg_cut": applied["cut"], "bg_made": applied.get("made"),
                 "bg_loaded_tools": background.loaded_tools(messages[:int(applied["cut"])])}


def _aliases_peek(session: str) -> object:
    a = _bg_load(session).get("aliases") or {}
    if a.get("mode") == "plan" and a.get("plan"):
        return a["plan"]
    return "off"


# ---- the free-point decision, on real counts ------------------------------
#
# At the first request after a break with a gated compaction ready, count what
# would be billed with and without it (free count_tokens, this request's own
# auth, in parallel, capped at DECIDE_TIMEOUT_S), and with and without a path
# alias plan. Then: cold cache, apply if smaller; warm, only if it repays
# within `bg_warm_payback_max` requests. Real counts replaced byte estimates
# here because the estimate once said "smaller" for a compaction that billed
# ~23K tokens MORE (thinking signatures had been priced as text).

DECIDE_TIMEOUT_S = 2.5


def _count_all(bodies: dict, headers: dict, path: str) -> dict:
    hdrs = {k.lower(): v for k, v in (headers or {}).items() if k.lower() in audit.HEADER_FIELDS}
    got: dict = {}

    def one(name, req):
        try:
            body = audit.payload(json.dumps(req).encode())
            n, _, _ = audit._count(path, hdrs, body) if body else (None, 0, {})
            if n:
                got[name] = n
        except Exception:
            pass
    threads = [threading.Thread(target=one, args=(k, v), daemon=True) for k, v in bodies.items()]
    for t in threads:
        t.start()
    end = time.time() + DECIDE_TIMEOUT_S
    for t in threads:
        t.join(max(0.0, end - time.time()))
    return dict(got)


def _decide(session: str, request: dict, original: list, cold: bool, headers: dict,
            path: str) -> dict:
    """{"apply": bool, "info": {...}}; commits the compaction when it applies."""
    st = _bg_load(session)
    pending = st.get("pending") or {}
    compacted = background.apply(original, dict(pending, tail_end=len(original)))
    if compacted is None:
        return {"apply": False, "info": {"bg_skipped": "history changed since the compaction was made"}}
    exclude = pass_verdicts()
    plan = passes.hindsight_plan(compacted)
    tools = background.undefer(request.get("tools"),
                               background.loaded_tools(original[:int(pending["cut"])]))
    cand = dict(request, **({"tools": tools} if tools is not None else {}))
    bodies = {"now": request,
              "off": dict(cand, messages=passes.rewrite_messages(compacted, aliases="off", exclude=exclude))}
    if plan:
        bodies["plan"] = dict(cand, messages=passes.rewrite_messages(compacted, aliases=plan, exclude=exclude))
    t0 = time.time()
    n = _count_all(bodies, headers, path)
    info: dict = {"bg_counts": n, "bg_count_s": round(time.time() - t0, 2)}
    if "now" not in n or "off" not in n:
        ok, why = (True, "cache expired") if cold else _warm_ok(pending)
        if ok and cold and not _estimate_margin_ok(pending):
            ok, why = False, "estimated saving under the 20% margin (count unavailable)"
        mode = {"mode": "off", "why": "count unavailable"}
    else:
        use_plan = "plan" in n and n["off"] - n["plan"] > 0.002 * n["off"]
        best = n["plan"] if use_plan else n["off"]
        mode = ({"mode": "plan", "plan": plan, "tokens_off": n["off"], "tokens_plan": n["plan"]}
                if use_plan else {"mode": "off", "tokens_off": n["off"], "tokens_plan": n.get("plan")})
        p = pricing.price_for(request.get("model") or "")
        w1, rd = (p.write_1h, p.read) if p else (8.0, 0.2)
        if best >= n["now"]:
            ok, why = False, f"counted: {best:,} tokens with it vs {n['now']:,} without"
        elif cold:
            ok, why = True, f"cold cache; counted {n['now']:,} -> {best:,}"
        else:
            reqs = (best * w1 - n["now"] * rd) / ((n["now"] - best) * rd)
            limit = config.get("bg_warm_payback_max")
            ok = reqs <= limit
            why = f"warm; counted {n['now']:,} -> {best:,}, repays in {reqs:.0f} requests (limit {limit:.0f})"
    with _STATE_LOCK:
        st = _bg_load(session)
        if ok and st.get("pending", {}).get("made") == pending.get("made"):
            _commit(session, st, pending, original, mode)
        else:
            ok = False
    info["bg_applied" if ok else "bg_skipped"] = True if ok else why
    info["bg_why"] = why
    try:
        vault.record("bg_decision", tool="proxy", session_id=session, index=False,
                     summary=f"compaction {'applied' if ok else 'skipped'}: {why}",
                     meta=dict(info, cold=cold, cut=pending.get("cut")))
    except Exception:
        pass
    return {"apply": ok, "info": info}


def _rewrite(body: bytes | None, path: str, request_class: str | None,
             session: str | None = None, headers: dict | None = None,
             _decided: bool = False) -> tuple[bytes | None, dict]:
    """(bytes to send, stats). The original bytes unless every check passes."""
    if REWRITE != "forward" or not body or path.split("?")[0] != "/v1/messages":
        return body, {}
    if request_class not in REWRITE_CLASSES:
        if request_class not in SIDE_CLASSES or REWRITE != "forward" or not session:
            return body, {}
        try:
            request = json.loads(body)
            messages = request.get("messages") if isinstance(request, dict) else None
            if not isinstance(messages, list) or not _follows_main(session, messages):
                return body, {}
            if session in _FELL_BACK or _global_off():
                return body, {}
            messages, stats = _bg_peek(messages, session)
            _keep_loaded(request, stats)
            off = tuple(_bg_load(session).get("passes_off") or ())
            aliases = _aliases_peek(session)
            out = passes.rewrite_messages(messages, aliases=aliases, exclude=off)
            out, rinfo = _replay(session, messages, out, _epoch(stats.get("bg_cut"), stats.get("bg_made")),
                                 update=False, legacy=_epoch_v1(stats.get("bg_cut"), aliases, off))
            stats.update(rinfo)
            request["messages"] = out
            out = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
            return out, dict(stats, auxiliary=True, bytes_in=len(body), bytes_out=len(out))
        except Exception as exc:
            return body, {"error": f"{type(exc).__name__}: {exc}"}
    if session in _FELL_BACK:
        return body, {"fallback": "session"}
    if _global_off():
        return body, {"fallback": "global"}
    try:
        request = json.loads(body)
        messages = request.get("messages") if isinstance(request, dict) else None
        if not isinstance(messages, list):
            return body, {}
        if session:
            _MAIN_HIST[session] = messages
            if len(_MAIN_HIST) > 20:
                _MAIN_HIST.pop(next(iter(_MAIN_HIST)))
        skeleton = {k: request[k] for k in ("model", "system", "tools", "tool_choice", "thinking")
                    if k in request}
        original = messages
        adopt = _adopt_reason(session, messages) if not _decided else None
        messages, stats = _bg(messages, session, skeleton, headers, decided=_decided)
        decide = stats.pop("bg_decide", None)
        _keep_loaded(request, stats)
        alias_arg, exclude, astats = _session_setup(session, original, stats.get("gap_s"))
        stats.update(astats)
        out = passes.rewrite_messages(messages, aliases=alias_arg, exclude=exclude)
        raw_until = _last_assistant(messages) + 1 if adopt and not stats.get("bg_applied") else 0
        if raw_until and alias_arg not in ("off", None) and \
                any(passes.ALIAS_NOTE_MARK in json.dumps(m, ensure_ascii=False) for m in out[:raw_until]):
            # The plan's definition line would go out only in the part sent raw:
            # without it later aliases would mean nothing. No aliases this request.
            out = passes.rewrite_messages(messages, aliases="off", exclude=exclude)
        if session:
            out, rinfo = _replay(session, messages, out, _epoch(stats.get("bg_cut"), stats.get("bg_made")),
                                 legacy=_epoch_v1(stats.get("bg_cut"), alias_arg, exclude), raw_until=raw_until)
            stats.update(rinfo)
            if rinfo.get("adopted"):
                stats["adopted_why"] = adopt
        request["messages"] = out
        if decide and session:
            d = _decide(session, dict(request), original, decide["cold"], headers or {}, path)
            if d["apply"]:
                # Committed: build this request again, now compacted.
                sent2, st2 = _rewrite(body, path, request_class, session, headers, _decided=True)
                return sent2, dict(st2, **d["info"])
            stats.update(d["info"])
        stats["_base"] = (messages, alias_arg, exclude)
        out = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        if session:
            _NOW[session] = {"in": len(body), "out": len(out), "t": time.time()}
            if len(_NOW) > 50:
                _NOW.pop(next(iter(_NOW)))
        return out, dict(stats, bytes_in=len(body), bytes_out=len(out))
    except Exception as exc:
        return body, {"error": f"{type(exc).__name__}: {exc}"}


# ---- preserved thinking: the safety net ------------------------------------
#
# Opus 5.5 binds every thinking block to the exact bytes before it (docs:
# build-with-claude/preserved-thinking). Old accounts drop a block that no
# longer matches; accounts from 2026-08-31 reject the request with a 400. Our
# bytes are built never to cause that (the ledger, the tail strip at a
# compaction), and this catches what slips through: a binding 400 is retried
# once with `block_binding.prefix_mismatch_behavior: "drop_block"`, which the
# API documents for exactly this, and the session keeps that setting from then
# on so its bytes stay stable. Sending the client's original bytes instead
# would not help: its thinking was made on our bytes, so the original is the
# mismatched history (checked 2026-10-22: the original 400s, ours is 200).

BINDING_BETA = "thinking-binding-controls-2026-08-01"
_BINDING_ERR = ("bound to a different conversation", "prefix_mismatch_behavior")
_DROP_BLOCK: set[str] = set()


def _drop_block_on(session: str | None) -> bool:
    if not session:
        return False
    if session in _DROP_BLOCK:
        return True
    try:
        if _bg_load(session).get("drop_block"):
            _DROP_BLOCK.add(session)
            return True
    except Exception:
        pass
    return False


def _set_drop_block(session: str | None, why: str) -> None:
    if not session:
        return
    _DROP_BLOCK.add(session)
    try:
        with _STATE_LOCK:
            st = _bg_load(session)
            st["drop_block"] = {"since": time.time(), "why": why[:300]}
            _bg_save(session, st)
        vault.record("thinking_binding", tool="proxy", session_id=session, index=False,
                     summary="thinking blocks failed the prefix check; session now sends drop_block",
                     meta={"why": why[:300]})
    except Exception:
        pass


def _with_drop_block(raw: bytes | None) -> bytes | None:
    """`raw` with drop_block set on its thinking config, or None (no thinking)."""
    try:
        req = json.loads(raw)
        th = req.get("thinking") if isinstance(req, dict) else None
        if not isinstance(th, dict) or th.get("type") in (None, "disabled"):
            return None
        if (th.get("block_binding") or {}).get("prefix_mismatch_behavior") == "drop_block":
            return raw
        req["thinking"] = dict(th, block_binding={"prefix_mismatch_behavior": "drop_block"})
        return json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode()
    except Exception:
        return None


def _add_beta(headers: dict, beta: str = BINDING_BETA) -> dict:
    out = dict(headers)
    key = next((k for k in out if k.lower() == "anthropic-beta"), None)
    cur = out.get(key, "") if key else ""
    if beta in [x.strip() for x in cur.split(",")]:
        return out
    out[key or "anthropic-beta"] = ",".join(x for x in (cur, beta) if x)
    return out


def _transformations(text: str) -> list | None:
    """`input_transformations` from a relayed response (SSE or JSON), or None."""
    if "input_transformations" not in text:
        return None
    found: list = []
    for line in text.splitlines():
        line = line[5:].strip() if line.startswith("data:") else line.strip()
        if "input_transformations" not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        for holder in (event, event.get("message"), event.get("delta")):
            if isinstance(holder, dict) and isinstance(holder.get("input_transformations"), list):
                found.extend(holder["input_transformations"])
    return found


class _Buffered:
    """An already-read error response, relayed as it came."""

    def __init__(self, exc, payload: bytes) -> None:
        self.headers, self._payload, self.status = exc.headers, payload, exc.code

    def read(self, n: int = -1) -> bytes:
        out, self._payload = self._payload, b""
        return out

    read1 = read

    def close(self) -> None:
        pass


class _Tee:
    """Keeps the head and tail of a stream, bounded, without buffering it."""

    def __init__(self, head: int = TEE_HEAD, tail: int = TEE_TAIL) -> None:
        self._head = bytearray()
        self._tail = bytearray()
        self._head_max, self._tail_max = head, tail
        self.total = 0

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        if len(self._head) < self._head_max:
            self._head.extend(data[: self._head_max - len(self._head)])
        self._tail.extend(data)
        if len(self._tail) > self._tail_max:
            del self._tail[: len(self._tail) - self._tail_max]

    def text(self) -> str:
        if self.total <= self._head_max:
            return self._head.decode("utf-8", "replace")
        return (self._head.decode("utf-8", "replace") + "\n"
                + self._tail.decode("utf-8", "replace"))


def _usage_from_stream(text: str) -> dict:
    """Pull the usage numbers out of a relayed SSE body.

    Streaming splits them: `message_start` has input, cache-read and
    cache-creation counts; the final `message_delta` has output. Both ends of
    the tee are scanned and the results merged.
    """
    usage: dict = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except ValueError:
            continue
        found = None
        if isinstance(event.get("message"), dict):
            found = event["message"].get("usage")
        if found is None:
            found = event.get("usage")
        if isinstance(found, dict):
            for key, value in found.items():
                if isinstance(value, (int, float)) and value:
                    usage[key] = value
                elif key not in usage:
                    usage[key] = value
    return usage


def _usage_from_json(text: str) -> dict:
    try:
        body = json.loads(text)
    except ValueError:
        return {}
    usage = body.get("usage")
    return usage if isinstance(usage, dict) else {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "tokensaver"

    # Access logging goes to the vault, not to stderr; a chatty proxy buries
    # the one line that matters.
    def log_message(self, fmt: str, *args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/_everytoken/"):      # clients from before the rename
            self.path = "/_tokensaver/" + self.path[len("/_everytoken/"):]
        if self.path == "/_tokensaver/health":
            # Answered here, never relayed: the launcher asks this before
            # routing Claude Code through us, so it must not depend on the API.
            from . import __version__
            payload = json.dumps({"ok": True, "rewrite": REWRITE or "off",
                                  "bg_compact": BG_COMPACT, "audit": audit.status(),
                                  "fallback": bool(_global_off()), "version": __version__,
                                  "pid": os.getpid(), "started": _STARTED,
                                  "inflight": _INFLIGHT[0], "code": _CODE_ID, "estimate": True,
                                  "strain": _STRAIN[-5:] + machine.recent_events()[-5:]}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass      # the status-bar check gives up after 0.3 s; that is fine
            return
        if self.path.startswith("/_tokensaver/session/"):
            payload = json.dumps(estimate(self.path.rsplit("/", 1)[-1])).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._relay("GET")

    def do_POST(self) -> None:
        self._relay("POST")

    def do_PUT(self) -> None:
        self._relay("PUT")

    def do_DELETE(self) -> None:
        self._relay("DELETE")

    def do_HEAD(self) -> None:
        self._relay("HEAD")

    def do_PATCH(self) -> None:
        self._relay("PATCH")

    def _forward_headers(self) -> dict:
        out = {}
        for name, value in self.headers.items():
            if name.lower() in HOP_BY_HOP:
                continue
            out[name] = value
        out["Accept-Encoding"] = "identity"
        return out

    def _relay(self, method: str) -> None:
        with _INFLIGHT_LOCK:
            _INFLIGHT[0] += 1
        try:
            self._relay_counted(method)
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT[0] -= 1

    def _relay_counted(self, method: str) -> None:
        started = time.time()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        sent, self._rewrite_stats = _rewrite(
            body, self.path, self.headers.get("x-claude-code-request-class"),
            self.headers.get("x-claude-code-session-id"), dict(self.headers.items()))
        # In memory only (a plan is large): what the forward passes started from,
        # so the audit can price aliases on the request as it was really built.
        self._audit_base = self._rewrite_stats.pop("_base", None)
        session = self.headers.get("x-claude-code-session-id")
        headers = self._forward_headers()
        messages_call = method == "POST" and self.path.split("?")[0] == "/v1/messages" and bool(sent)
        if messages_call and REWRITE == "forward":
            if config.get("binding_monitor"):
                headers = _add_beta(headers)
            if _drop_block_on(session):
                db = _with_drop_block(sent)
                if db is not None:
                    sent, headers = db, _add_beta(headers)
                    self._rewrite_stats["drop_block"] = True

        def send(data: bytes | None, hdrs: dict):
            return urllib.request.urlopen(urllib.request.Request(
                UPSTREAM.rstrip("/") + self.path, data=data, headers=hdrs, method=method), timeout=900)

        try:
            response = send(sent, headers)      # same bytes in, same bytes out, unless opted in
            status = response.status
        except urllib.error.HTTPError as exc:
            # An HTTPError *is* the response. Relay it untouched: Claude Code's
            # capability-rejection recovery matches on the upstream's wording,
            # so wrapping or rewriting an error breaks the retry path. The one
            # exception is a preserved-thinking 400 (see `_BINDING_ERR`).
            response, status = exc, exc.code
            if status == 400 and messages_call and REWRITE == "forward":
                try:
                    payload = exc.read()
                except Exception:
                    payload = b""
                response = _Buffered(exc, payload)
                text = payload.decode("utf-8", "replace")
                retry = _with_drop_block(sent) if any(m in text for m in _BINDING_ERR) else None
                if retry is not None and retry != sent:
                    _set_drop_block(session, text)
                    self._rewrite_stats["binding_retry"] = True
                    sent = retry
                    try:
                        response = send(retry, _add_beta(headers))
                        status = response.status
                    except urllib.error.HTTPError as exc2:
                        response, status = exc2, exc2.code
                    except Exception as exc2:
                        self._fail(exc2)
                        return
        except Exception as exc:
            self._fail(exc)
            return

        tee = _Tee()
        content_type = response.headers.get("Content-Type", "")
        streaming = "text/event-stream" in content_type.lower()
        try:
            if streaming:
                self._relay_stream(status, response, tee)
            else:
                self._relay_whole(status, response, tee, head_only=method == "HEAD")
        except (BrokenPipeError, ConnectionResetError):
            return                           # client went away mid-response
        finally:
            try:
                response.close()
            except Exception:
                pass

        # Hot path is done. Nothing below may raise into the request.
        try:
            self._observe(method, body, tee, status, streaming, time.time() - started)
        except Exception:
            pass
        try:
            # Queue only; the count happens on audit's own thread, later.
            audit.offer(path=self.path, request_class=self.headers.get("x-claude-code-request-class"),
                        session=self.headers.get("x-claude-code-session-id"),
                        headers=dict(self.headers.items()), body=body, sent=sent,
                        rstats=getattr(self, "_rewrite_stats", {}), status=status,
                        usage=getattr(self, "_usage", {}), base=getattr(self, "_audit_base", None))
        except Exception:
            pass

    def _relay_stream(self, status: int, response, tee: _Tee) -> None:
        """Chunked relay, flushed per read.

        Buffering here stalls the UI, and dropping the upstream's SSE `ping`
        events trips Claude Code's streaming idle watchdog during a long
        thinking pause - so every chunk goes out exactly as it arrives.
        """
        self.send_response(status)
        for name, value in response.headers.items():
            if name.lower() in HOP_BY_HOP or name.lower() == "content-encoding":
                continue
            self.send_header(name, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            # read1, not read: `read(n)` on a chunked response blocks until it
            # has n bytes or the stream ends, which buffers the entire reply
            # and stalls the UI - measured, this was the bug the conformance
            # suite's inter-event timing check was written to catch. `read1`
            # returns whatever one syscall yields, so an SSE frame goes out as
            # soon as it arrives.
            chunk = response.read1(CHUNK)
            if not chunk:
                break
            tee.feed(chunk)
            self.wfile.write(f"{len(chunk):X}\r\n".encode())
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _relay_whole(self, status: int, response, tee: _Tee,
                     head_only: bool = False) -> None:
        payload = b"" if head_only else response.read()
        tee.feed(payload)
        self.send_response(status)
        for name, value in response.headers.items():
            if name.lower() in HOP_BY_HOP or name.lower() == "content-encoding":
                continue
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)
        self.wfile.flush()

    def _fail(self, exc: Exception) -> None:
        """Upstream unreachable. Shape the error like the API's own.

        Claude Code matches on error wording, so an unfamiliar envelope is
        worse than a familiar one; `502` with an `error` object is the closest
        honest approximation of "the hop failed".
        """
        payload = json.dumps({
            "type": "error",
            "error": {"type": "api_error",
                      "message": f"TokenSaver proxy could not reach upstream: "
                                 f"{type(exc).__name__}: {exc}"},
        }).encode()
        try:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except Exception:
            pass
        try:
            vault.record("proxy_error", content=f"{type(exc).__name__}: {exc}",
                         summary=f"upstream unreachable: {self.path}")
        except Exception:
            pass

    def _observe(self, method: str, body: bytes | None, tee: _Tee,
                 status: int, streaming: bool, seconds: float) -> None:
        if not body:
            return
        raw = body.decode("utf-8", "replace")
        try:
            request = json.loads(raw)
        except ValueError:
            return
        if not isinstance(request, dict):
            return

        usage = (_usage_from_stream(tee.text()) if streaming
                 else _usage_from_json(tee.text()))
        self._usage = usage
        model = request.get("model") or ""
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in SECRET_HEADERS}

        dropped = _transformations(tee.text()) if "/v1/messages" in self.path else None

        meta = {
            "path": self.path,
            "method": method,
            "status": status,
            "streaming": streaming,
            "seconds": round(seconds, 2),
            "model": model,
            "usage": usage,
            "request_class": self.headers.get("x-claude-code-request-class"),
            "agent_id": self.headers.get("x-claude-code-agent-id"),
            "compaction": self.headers.get("x-claude-code-compaction"),
            "context_compacted": self.headers.get("x-claude-code-context-compacted"),
            "prompt_id": self.headers.get("x-claude-code-prompt-id"),
            "headers": headers,
            "rewrite": getattr(self, "_rewrite_stats", {}),
        }
        if dropped:
            meta["input_transformations"] = dropped[:20]
            meta["thinking_dropped"] = len(dropped)
        context = (int(usage.get("input_tokens") or 0)
                   + int(usage.get("cache_read_input_tokens") or 0)
                   + int(usage.get("cache_creation_input_tokens") or 0))
        summary = (f"{method} {self.path} {status} {model} "
                   f"ctx={context:,} "
                   f"cls={self.headers.get('x-claude-code-request-class') or '-'}")
        vault.record("api_request", content=raw, tool="proxy",
                     session_id=self.headers.get("x-claude-code-session-id"),
                     summary=summary, meta=meta, index=False)
        _watch(self.headers.get("x-claude-code-session-id"),
               self.headers.get("x-claude-code-request-class"), self.path,
               usage, getattr(self, "_rewrite_stats", {}))
        if (self.path.split("?")[0] == "/v1/messages"
                and (self.headers.get("x-claude-code-request-class") or "main") == "main"):
            live = _LIVE.get(self.headers.get("x-claude-code-session-id") or "")
            if live is not None and context:
                live["sent_tokens"] = context      # what this session costs per request now
            _quality(self.headers.get("x-claude-code-session-id"), request.get("messages"),
                     getattr(self, "_rewrite_stats", {}))
        _maybe_prune()


PRUNE_EVERY_S = 3600
# First pass 2 min after start, not on the first request: run then, its
# connection and the request's own vault write both create the schema at once,
# one gets "database is locked", and record() drops that row (measured: the
# conformance suite lost rows this way).
_last_prune = time.time() - PRUNE_EVERY_S + 120


def _maybe_prune() -> None:
    """At most hourly, in its own thread: an always-on proxy stores every
    request body (measured: 25 MB for 364 requests), so retention is what
    keeps "set and forget" from filling the disk."""
    global _last_prune
    now = time.time()
    if now - _last_prune < PRUNE_EVERY_S:
        return
    _last_prune = now

    def run():
        try:
            done = vault.prune(config.get("vault_keep_days"), config.get("vault_max_mb") * 1024 * 1024)
            if done.get("pruned") or done.get("error"):
                vault.record("vault_pruned", tool="proxy", summary=f"retention: {done}", meta=done,
                             index=False)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


_INFLIGHT = [0]
_INFLIGHT_LOCK = threading.Lock()
_STARTED = time.time()
LOG_MAX_BYTES = 10 * 1024 * 1024


def code_id() -> str:
    """Fingerprint of the code on disk, so setup and doctor can tell when the
    running proxy is older than the checkout (after a `git pull`)."""
    import hashlib as _h
    root = Path(__file__).resolve().parent
    h = _h.sha256()
    for f in sorted(root.glob("*.py")):
        try:
            h.update(f.name.encode() + str(f.stat().st_mtime_ns).encode())
        except OSError:
            pass
    return h.hexdigest()[:12]


_CODE_ID = code_id()


def _rotate_log() -> None:
    log = vault.VAULT_DIR / "proxy.log"
    try:
        if log.stat().st_size > LOG_MAX_BYTES:
            log.replace(log.with_suffix(".log.1"))
    except OSError:
        pass


def serve(port: int = PORT, upstream: str = UPSTREAM, service: bool = False) -> int:
    """Run the proxy. `service=True` is how launchd runs it: a busy port is
    retried quietly instead of crashing (a crash loop is worse than waiting),
    and SIGTERM lets in-flight requests finish before exiting, so a restart
    never cuts a turn off mid-stream."""
    global UPSTREAM
    UPSTREAM = upstream
    if service:
        _rotate_log()
    logged = False
    while True:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError as exc:
            if not service:
                raise
            if not logged:
                print(f"TokenSaver: port {port} busy ({exc}); retrying every 15s", flush=True)
                logged = True
            time.sleep(15)
    server.daemon_threads = True
    base = f"http://127.0.0.1:{port}"
    if service:
        import signal

        def stop(*_):
            def drain():
                deadline = time.time() + 20
                while _INFLIGHT[0] > 0 and time.time() < deadline:
                    time.sleep(0.2)
                server.shutdown()
            threading.Thread(target=drain, daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        print(f"TokenSaver proxy (service) on {base} -> {upstream}; rewrite={REWRITE}, "
              f"bg_compact={BG_COMPACT}, audit={config.get('audit')}, code={_CODE_ID}", flush=True)
    else:
        print(f"TokenSaver proxy on {base} -> {upstream}")
        print("relay only; no request is modified." if REWRITE != "forward" else
              f"REWRITE={REWRITE}: main-thread /v1/messages histories are rewritten "
              f"(needs the hint headers below; everything else is relayed as is).")
        print("In the shell you want to use it from:\n")
        print(f"  export ANTHROPIC_BASE_URL={base}")
        print("  export CLAUDE_CODE_GATEWAY_HINT_HEADERS=1\n")
        print("Then `tokensaver proxy --stats` for what it saw. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    if service:
        print("TokenSaver proxy stopped (drained)", flush=True)
    return 0


def _prune_lines() -> list[str]:
    """Staged smart prune: what each stage achieved, and why it paused."""
    import sqlite3
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        rows = [json.loads(m or "{}") for (m,) in conn.execute(
            "SELECT meta FROM events WHERE kind='bg_prune' ORDER BY id")]
        conn.close()
    except Exception:
        return []
    t, why = prune.tier()
    out = [f"Smart prune: tier {t} ({why}); delete at p >= {config.get('prune_min_p')}; "
           f"ease off when strained: {'on' if config.get('bg_yield') else 'OFF (never pauses)'}"]
    if rows:
        # From 2026-10-23 a "final" row per compaction says whether the pruned
        # candidate replaced the rules-only one; older rows were gated per stage.
        finals = [r for r in rows if r.get("stage") == "final"]
        legacy = [r for r in rows if "after" in r and "kept" not in r and r.get("stage") != "final"]
        done = [r for r in rows if "after" in r and r.get("stage") != "final"]
        kept = [r for r in finals if r.get("promoted") and r["after"] < r["before"]] + \
            [r for r in legacy if r.get("gate", {}).get("passed") and r["after"] < r["before"]]
        saved = sum(r["before"] - r["after"] for r in kept)
        if finals:
            failed = sum(1 for r in finals if "check failed" in str(r.get("verdict")))
            unchecked = sum(1 for r in finals if "unchecked" in str(r.get("verdict")))
            out.append(f"  results {len(finals)}: promoted {sum(1 for r in finals if r.get('promoted'))} "
                       f"({unchecked} unchecked, fast tier), final check failed {failed}")
        paused: dict = {}
        for r in rows:
            if r.get("stopped"):
                k = str(r["stopped"]).split(":")[0]
                paused[k] = paused.get(k, 0) + 1
        out.append(f"  stages run {len(done)}, kept {len(kept)}, compacted text {saved:,} tokens smaller"
                   + (f"; paused: {', '.join(f'{k} x{v}' for k, v in paused.items())}" if paused else ""))
    out.append("")
    return out


def _token_lines() -> list[str]:
    """Real tokens from count_tokens samples, and behaviour after compaction."""
    import sqlite3
    out: list[str] = []
    try:
        conn = sqlite3.connect(vault.DB_PATH)
        everything = [json.loads(m or "{}") for (m,) in conn.execute(
            "SELECT meta FROM events WHERE kind='token_audit' ORDER BY id")]
        # Counts before 2026-10-22 kept thinking in both bodies; the original's
        # thinking was dropped (bound to our bytes), so they understate plain
        # Claude. Shown apart, never mixed in.
        audits = [a for a in everything if a.get("method") == "no_thinking"]
        legacy = len(everything) - len(audits)
        follow = [json.loads(m or "{}") for (m,) in conn.execute(
            "SELECT meta FROM events WHERE kind='bg_followup' ORDER BY id")]
        errors = conn.execute("SELECT count(*) FROM events WHERE kind='token_audit_error'").fetchone()[0]
        conn.close()
    except Exception:
        return out
    if legacy:
        out.append(f"({legacy} older counts left out: made before the thinking-fair count, they priced plain "
                   "Claude too low)")
    if audits:
        out.append(f"Real tokens (count_tokens, {len(audits)} sampled requests; {errors} errors):")
        allrows = [a for a in audits if a.get("reason") == "every_n"]
        if allrows:
            o = sum(a.get("tokens_original") or 0 for a in allrows)
            t = sum(a.get("tokens_sent") or 0 for a in allrows)
            every = int(config.get('audit_every'))
            out.append(f"  OVERALL ({'every request' if every == 1 else f'every {every}th request'}, changed or not): "
                       f"{100 * (1 - t / max(o, 1)):.1f}% fewer tokens over {len(allrows)} requests")
        for reason in ("first_changed", "every_n", "compaction_applied"):
            rows = [a for a in audits if a.get("reason") == reason]
            if not rows:
                continue
            o = sum(a.get("tokens_original") or 0 for a in rows)
            t = sum(a.get("tokens_sent") or 0 for a in rows)
            b_in = sum(a.get("bytes_in") or 0 for a in rows)
            b_out = sum(a.get("bytes_out") or 0 for a in rows)
            out.append(f"  {reason:20s} {len(rows):4d}  tokens {o:,} -> {t:,} "
                       f"({100 * (1 - t / max(o, 1)):.1f}% saved; bytes {100 * (1 - b_out / max(b_in, 1)):.1f}%)")
        att = [a for a in audits if (a.get("without") or {}).get("path_aliases")]
        # Older records priced the removed "auto" mode against the original;
        # newer ones price the plan in force on the request as really built.
        for label, part in (("path aliases (plan in force)",
                             [a for a in att if a.get("attribution_base") == "as_built"]),
                            ("path aliases (old auto mode, removed)",
                             [a for a in att if a.get("attribution_base") != "as_built"])):
            if not part:
                continue
            with_ = sum(a.get("tokens_sent") or 0 for a in part)
            without = sum(a["without"]["path_aliases"] for a in part)
            out.append(f"  {label}, {len(part)} requests: {without - with_:+,} tokens "
                       f"({100 * (without - with_) / max(without, 1):+.2f}%; positive = aliases save)")
        per = {}
        for a in audits:
            for name, pp in (a.get("passes") or {}).items():
                t = per.setdefault(name, [0, 0, 0, 0])
                t[0] += 1
                if pp.get("changed"):
                    t[1] += 1
                    t[2] += int(pp.get("saving") or 0)
                    t[3] += int(a.get("tokens_sent") or 0)
        if per:
            off = set(config.section("passes").get("off") or [])
            out.append("  each pass, toggled on its own (saving > 0 = it saves; "
                       f"off below -{100 * PASS_MARGIN:.2f}% after {PASS_MIN_EVIDENCE} requests it changed, "
                       f"at most {PASS_PER_SESSION} from any one session):")
            for name in audit.ATTRIBUTE:
                if name not in per:
                    continue
                n, ch, sav, tot = per[name]
                state = ("OFF (measured cost)" if name in off else "on") if name != "path_aliases" else \
                    "decided at compactions"
                out.append(f"    {name:13s} changed {ch:4d}/{n:<4d} saving {sav:+,} tokens "
                           f"({100 * sav / max(tot, 1):+.2f}% of those requests)  {state}")
        out.append("  (each sampling reason is shown apart: compaction requests are oversampled)")
        out.append("")
    if follow:
        keys = ("rereads", "reruns", "failed_edits")
        bn = sum(f.get("before_requests") or 0 for f in follow)
        an = sum(f.get("after_requests") or 0 for f in follow)
        parts = []
        for k in keys:
            b = sum(f.get(f"before_{k}") or 0 for f in follow)
            a = sum(f.get(f"after_{k}") or 0 for f in follow)
            parts.append(f"{k} {b / max(bn, 1):.2f} -> {a / max(an, 1):.2f}")
        out.append(f"After compaction ({len(follow)} compactions; per request, before -> after): "
                   + ", ".join(parts))
        if len(follow) < 10:
            out.append("  (fewer than 10 compactions: too few to call a difference)")
        rise = quality_rise(follow)
        if rise is not None:
            out.append(f"  rise {rise:+.2f} per request (compaction turns itself off at "
                       f"+{QUALITY_MAX_RISE:.2f}, from {QUALITY_MIN_FOLLOWUPS} compactions)")
    if not config.get("bg_compact"):
        slot = ((config.load().get("settings") or {}).get("bg_compact") or {}).get("user") or {}
        if str(slot.get("why", "")).startswith("quality signals"):
            out.append(f"  COMPACTION TURNED ITSELF OFF ({slot.get('at')}): {slot['why']}. "
                       "`tokensaver config set bg_compact true` to turn it back on")
        out.append("")
    return out


def stats(limit: int = 12) -> str:
    """What the relay saw, and whether caching survived it.

    The cache check is the point of this command. A gateway that breaks
    `cache_control` raises no error and returns correct answers; the only
    symptom is that cache reads stop and the bill rises ~10x.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(vault.DB_PATH)
        rows = conn.execute(
            "SELECT id, session_id, meta FROM events WHERE kind='api_request'"
            " ORDER BY id").fetchall()
        conn.close()
    except Exception as exc:
        return f"TokenSaver: could not read the vault ({exc})"
    if not rows:
        return ("TokenSaver: no proxied requests recorded yet.\n"
                "Start it with `tokensaver proxy`, then set "
                "ANTHROPIC_BASE_URL in the shell you want to measure.")

    sessions: dict[str, dict] = {}
    for _id, session_id, meta_json in rows:
        try:
            meta = json.loads(meta_json or "{}")
        except ValueError:
            continue
        if meta.get("_meta_blob"):
            meta = vault.read_meta(_id)
        key = session_id or "unknown"
        acc = sessions.setdefault(key, {
            "n": 0, "main": 0, "cached": 0, "uncached": 0, "ctx": 0,
            "zero_run": 0, "worst_zero_run": 0,
            "models": set(), "errors": 0,
        })
        usage = meta.get("usage") or {}
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        acc["n"] += 1
        if meta.get("model"):
            acc["models"].add(meta["model"])
        if int(meta.get("status") or 0) >= 400:
            acc["errors"] += 1
        acc["ctx"] += (int(usage.get("input_tokens") or 0) + cache_read
                       + int(usage.get("cache_creation_input_tokens") or 0))
        # Only real turns: count_tokens and other paths never read the cache,
        # and counting them raised false alarms.
        if ((meta.get("request_class") or "main") == "main"
                and str(meta.get("path") or "").split("?")[0] == "/v1/messages"):
            acc["main"] += 1
            if cache_read:
                acc["cached"] += 1
                acc["zero_run"] = 0
            else:
                acc["uncached"] += 1
                acc["zero_run"] += 1
                acc["worst_zero_run"] = max(acc["worst_zero_run"], acc["zero_run"])
        rw = meta.get("rewrite") or {}
        if rw.get("bytes_in"):
            acc["rw_in"] = acc.get("rw_in", 0) + int(rw["bytes_in"])
            acc["rw_out"] = acc.get("rw_out", 0) + int(rw.get("bytes_out") or rw["bytes_in"])
        if rw.get("bg_applied"):
            acc["bg"] = acc.get("bg", 0) + 1
        if rw.get("error"):
            acc["rw_err"] = acc.get("rw_err", 0) + 1
        if rw.get("drift"):
            acc["drift"] = acc.get("drift", 0) + int(rw["drift"])
        if meta.get("thinking_dropped"):
            acc["dropped"] = acc.get("dropped", 0) + int(meta["thinking_dropped"])
        if rw.get("binding_retry"):
            acc["binding_retry"] = acc.get("binding_retry", 0) + 1

    lines = [f"Proxied requests: {len(rows):,} across {len(sessions)} session(s)", ""]
    lines += _headline()
    try:
        from . import bill
        lines += bill.lines()
    except Exception:
        pass
    lines += _prune_lines()
    lines += _token_lines()
    fb = config.section("fallback")
    if fb.get("global_off"):
        lines += [f"*** REWRITING IS OFF (automatic fallback, {fb.get('at')}): {fb.get('why')}",
                  "    Requests are relayed unchanged. Clear with `tokensaver proxy --reset-fallback`.", ""]
    elif fb.get("events"):
        lines += [f"Session fallbacks in the last day: {len(fb['events'])} "
                  f"(last: {fb['events'][-1].get('why')})", ""]
    rw_in = sum(a.get("rw_in", 0) for a in sessions.values())
    if rw_in:
        rw_out = sum(a.get("rw_out", 0) for a in sessions.values())
        lines.append(f"Rewritten (what was actually sent): {rw_in:,} -> {rw_out:,} bytes, "
                     f"{100 * (1 - rw_out / rw_in):.1f}% smaller; background compactions applied: "
                     f"{sum(a.get('bg', 0) for a in sessions.values())}; rewrite errors (sent "
                     f"unchanged): {sum(a.get('rw_err', 0) for a in sessions.values())}; "
                     f"recomputed differently (stored bytes sent instead): "
                     f"{sum(a.get('drift', 0) for a in sessions.values())}")
        dropped = sum(a.get("dropped", 0) for a in sessions.values())
        retried = sum(a.get("binding_retry", 0) for a in sessions.values())
        lines.append(f"Preserved thinking: blocks the API reported as failing the prefix check: {dropped} "
                     f"(must be 0); 400s retried with drop_block: {retried}"
                     + ("   *** CHECK: our bytes broke Claude's thinking" if dropped or retried else ""))
        lines.append("")
    lines.append(f"  {'session':14s} {'reqs':>5s} {'main':>5s} {'cache hit':>9s} "
                 f"{'mean ctx':>9s} {'shrink':>7s} {'err':>4s}")
    alarms = []
    for key, acc in sorted(sessions.items(), key=lambda kv: -kv[1]["n"])[:limit]:
        hit = acc["cached"] / acc["main"] if acc["main"] else 0.0
        shrink = 1 - acc["rw_out"] / acc["rw_in"] if acc.get("rw_in") else 0.0
        lines.append(
            f"  {key[:14]:14s} {acc['n']:5d} {acc['main']:5d} {hit * 100:8.0f}% "
            f"{acc['ctx'] // max(acc['n'], 1):9,} {shrink * 100:6.1f}% "
            f"{acc['errors']:4d}")
        # One uncached main request is the first turn. Three in a row is the
        # signature of a gateway that has broken prompt caching.
        if acc["worst_zero_run"] >= 3:
            alarms.append(f"  session {key[:14]}: {acc['worst_zero_run']} consecutive "
                          f"main requests with no cache read")

    lines.append("")
    if alarms:
        lines.append("  CACHE ALARM - stop using the proxy and investigate:")
        lines.extend(alarms)
        lines.append("  This is the ~10x failure mode: caching broken, no error raised.")
    else:
        lines.append("  cache check: no session shows 3+ consecutive uncached main "
                     "requests (the ~10x failure signature)")
    lines.append("  shrink: bytes the rewrite removed from what was sent (forward passes + compaction).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--stats" in args:
        print(stats())
        return 0
    if "--reset-fallback" in args:
        config.set_section("fallback", {})
        print("fallback cleared: rewriting resumes on the next proxy start")
        return 0
    try:
        return serve()
    except OSError as exc:
        print(f"TokenSaver: cannot bind port {PORT}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
