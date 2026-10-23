"""Headroom on the same history, measured exactly like `tokensaver-cli benchmark`.

Uses `tokensaver.benchmark.replay` with Headroom (Apache-2.0) in place of
tokensaver's rewrite, so both tools go through the same replay, the same free
token counts, the same thinking factor and the same cost model. Headroom runs
like its proxy: messages already sent are frozen and keep the bytes it
returned for them; only new ones are compressed, once. (Its token mode, which
recompresses the whole history every request, measured 37.5% MORE expensive
on the trial ticket and is left out.) Its retrieve markers are counted; a
fetch would be an extra full request, not charged here.

Profiles: `coding` (its proxy default, lossless) and `agent-90` (its ML text
compressor, Kompress). Needs a Python with headroom-ai installed (see
~/.tokensaver/experiments/headroom_benchmark/README.md), run with
HEADROOM_KOMPRESS_BACKEND=onnx_cpu HEADROOM_COMPRESSION_DEADLINE_MS=600000.

Usage: <hr-venv>/bin/python experiments/replay_history.py coding|agent-90 [step]
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokensaver import names  # noqa: E402
from tokensaver import benchmark as bm, config, sim  # noqa: E402
from headroom_compare import dump, warm_up  # noqa: E402

OUT = names.home() / "experiments" / "replay_history.jsonl"


def headroom_transform(profile: str, stats: dict):
    from headroom import compress
    state = {"raw": None, "out": None}

    def transform(k: int, hist: list) -> list:
        prev_raw, prev_out = state["raw"], state["out"]
        frozen = 0
        if prev_raw is not None:
            # Slices of one message list share objects: "unchanged" is identity.
            while frozen < min(len(hist), len(prev_raw)) and hist[frozen] is prev_raw[frozen]:
                frozen += 1
        start = (prev_out[:frozen] if prev_out is not None else []) + json.loads(json.dumps(hist[frozen:]))
        try:
            r = compress(start, model="claude-opus-4-20250514", model_limit=1_000_000,
                         savings_profile=profile, frozen_message_count=frozen)
            res = r.messages
            stats["ml"] += any(x.startswith(("router:text:", "router:mixed:")) for x in r.transforms_applied)
        except Exception as exc:
            stats["errors"] += 1
            if stats["errors"] <= 3:
                print(f"    headroom error: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            res = start
        stats["markers"] += sum(dump(m).count("headroom_retrieve") + dump(m).count("<<ccr:") for m in res[frozen:])
        state["raw"], state["out"] = hist, res
        return res
    return transform


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    profile = args[0] if args else "coding"
    step = int(args[1]) if len(args) > 1 else 8
    data = [d for d in sim.sessions() if all(r.model.startswith("claude") for r in d.requests)]
    cred = bm.credentials()
    if cred is None:
        print("no credentials")
        return 2
    count = bm.Counter(cred[0])
    if profile == "agent-90":
        from headroom import compress
        from headroom.transforms import kompress_compressor as kc
        # Headroom's own background loader never finished here (752 s, twice),
        # and until it does it passes text through silently. Loading the model
        # directly takes 3 s. Then refuse to run unless it really compresses.
        kc._load_kompress(kc.HF_MODEL_ID, "auto")
        ready = warm_up(compress, wait_s=120)
        print(f"kompress ready: {ready}", flush=True)
        if not ready:
            return 4
    factors = bm.thinking_factors(data, count)
    after_s = config.SPECS["bg_after_min"].default * 60
    payback = config.SPECS["bg_warm_payback_max"].default
    rows, t0 = [], time.time()
    for i, d in enumerate(sorted(data, key=lambda d: len(d.requests)), 1):
        stats = {"ml": 0, "markers": 0, "errors": 0}
        f = factors.get(d.session.primary_model, (bm.DEFAULT_SIG_FACTOR, "assumed"))[0]
        t = time.time()
        r = bm.replay(d, count, step, after_s, payback, f, transform=headroom_transform(profile, stats))
        r.update(session=hashlib.sha256(d.stem.encode()).hexdigest()[:8], **stats)
        rows.append(r)
        if "skipped" not in r:
            print(f"  [{i}/{len(data)}] {r['requests']:5d} req  {bm._pct(r['ours_tokens'], r['plain_tokens']):5.1f}% tokens  "
                  f"{bm._pct(r['ours_cost'], r['plain_cost']):5.1f}% cost  ML on {stats['ml']} requests, "
                  f"{stats['markers']} retrieve markers  ({time.time() - t:.0f}s)", flush=True)
    lines = [f"headroom/{profile} (same replay, counts and cost model as tokensaver-cli benchmark):"]
    lines += bm.summarize(rows)
    lines.append(f" token counts: {count.calls:,} calls, {count.failed} failed; took {time.time() - t0:.0f}s")
    print("\n".join(lines))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        fh.write(json.dumps({"at": time.time(), "variant": f"headroom:{profile}", "step": step,
                             "summary": lines, "sessions": rows}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
