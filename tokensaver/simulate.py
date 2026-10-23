"""Replay real sessions under a compaction threshold and price the result.

This is not a parametric model. It walks each session's actual per-request
context trajectory, so the growth rate, the session length, and the session
floor are all measured rather than assumed. The only estimated inputs are the
three listed in Assumptions below; each is surfaced in the output so a reader
can see what the number rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pricing import resolve
from .transcripts import Session

# Measured from the only two real compactions in this dataset (2026-09-29,
# 2026-09-30): 167,399 -> 8,806 and 167,475 -> 12,832 tokens, ~119s each.
MEASURED_POST_TOKENS = 10_819
MEASURED_DURATION_S = 119.0
# Tokens the summarizing request itself generates. Both observed summaries were
# 7,958 and 2,488 characters of message content; ~1.5k tokens is the midpoint.
SUMMARY_OUTPUT_TOKENS = 1_500


@dataclass
class Result:
    threshold: int
    cost: float
    cost_read: float
    cost_write: float
    cost_out: float
    compactions: int
    mean_context: float
    added_seconds: float
    sessions_affected: int


def simulate(sessions: list[Session], threshold: int | None,
             post_tokens: int = MEASURED_POST_TOKENS,
             reprice_as: str | None = None,
             rebuild_as: str = "write") -> Result:
    """`threshold=None` reproduces the measured baseline (no added compaction).

    `rebuild_as` decides how the resumed prefix is billed after a compaction.
    This is the single biggest unknown in the model and it is not settled by
    the data: the only compactions on disk come from a local-model session,
    where the resumed request billed as fully uncached input. For an Opus
    session Claude Code should re-cache it, which costs the 1h write rate -
    higher than input on every current model. "write" is the pessimistic
    reading, "input" the optimistic one. Step 0 of the plan measures which.
    """
    if rebuild_as not in ("write", "input"):
        raise SystemExit("TokenSaver: --rebuild-as must be 'write' or 'input'")
    override = resolve(reprice_as).price if reprice_as else None
    if reprice_as and override is None:
        raise SystemExit(f"TokenSaver: no verified prices for model {reprice_as!r}")

    c_read = c_write = c_out = 0.0
    compactions = 0
    ctx_sum = 0
    requests = 0
    affected = 0

    for sess in sessions:
        p0 = override or resolve(sess.primary_model).price
        # Skip anything with no Anthropic rate (local models bill elsewhere).
        if p0 is None or p0.read == 0:
            continue
        floor = sess.floor
        sim_ctx = floor
        prev_real = None
        hit = False

        for r in sess.requests:
            p = override or resolve(r.model).price or p0
            growth = 0 if prev_real is None else max(0, r.context - prev_real)
            prev_real = r.context
            sim_ctx += growth

            if threshold and sim_ctx > threshold and sim_ctx > floor + post_tokens:
                # Summarizing request reads the pre-compaction prefix, writes nothing.
                c_read += sim_ctx * p.read / 1e6
                c_out += SUMMARY_OUTPUT_TOKENS * p.out / 1e6
                compactions += 1
                hit = True
                sim_ctx = floor + post_tokens
                # The resumed prefix is paid for from scratch, one way or another.
                rate = p.write_1h if rebuild_as == "write" else p.inp
                c_write += sim_ctx * rate / 1e6

            write = r.write
            read = max(0, sim_ctx - write)
            c_read += read * p.read / 1e6
            c_write += write * p.write_1h / 1e6
            c_out += r.out * p.out / 1e6
            ctx_sum += sim_ctx
            requests += 1

        if hit:
            affected += 1

    return Result(
        threshold=threshold or 0,
        cost=c_read + c_write + c_out,
        cost_read=c_read, cost_write=c_write, cost_out=c_out,
        compactions=compactions,
        mean_context=ctx_sum / requests if requests else 0.0,
        added_seconds=compactions * MEASURED_DURATION_S,
        sessions_affected=affected,
    )


def render(sessions: list[Session], thresholds: list[int],
           reprice_as: str | None = None,
           post_tokens: int = MEASURED_POST_TOKENS,
           rebuild_as: str = "write") -> str:
    base = simulate(sessions, None, post_tokens, reprice_as, rebuild_as)
    label = f"repriced as {reprice_as}" if reprice_as else "as billed"
    L = [f"Threshold replay  ({label})",
         f"  replayed {len(sessions)} sessions against their real context trajectories",
         f"  assumptions: post-compaction seed {post_tokens:,} tok, "
         f"{MEASURED_DURATION_S:.0f}s per compaction, {SUMMARY_OUTPUT_TOKENS:,} summary output tok,"
         f" rebuild billed as {rebuild_as}",
         "",
         f"  {'autoCompactWindow':>18s} {'mean ctx':>9s} {'compactions':>11s} "
         f"{'sessions':>8s} {'added wait':>10s} {'cost':>9s} {'saving':>7s}",
         f"  {'(baseline)':>18s} {base.mean_context:9,.0f} {0:11d} {0:8d} "
         f"{'-':>10s} {base.cost:9,.2f} {'-':>7s}"]
    for t in thresholds:
        r = simulate(sessions, t, post_tokens, reprice_as, rebuild_as)
        wait = f"{r.added_seconds / 60:.0f} min"
        save = f"{100 * (1 - r.cost / base.cost):.0f}%" if base.cost else "-"
        L.append(f"  {t:18,} {r.mean_context:9,.0f} {r.compactions:11d} "
                 f"{r.sessions_affected:8d} {wait:>10s} {r.cost:9,.2f} {save:>7s}")
    L.append("")
    L.append("  Not modelled: rework. Compacting more often may cost extra turns")
    L.append("  (re-reads, re-derivation), which lands in cache write and output -")
    L.append("  neither of which compaction reduces. Only a live week measures it.")
    return "\n".join(L)
