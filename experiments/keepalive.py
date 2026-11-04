"""Would cache keep-alive bumps have paid on this machine's own history? Local, $0.

Replays every real session (Claude Code's transcripts, forks excluded) with the
rule TokenSaver ships (tokensaver/keepalive.py): in a big or returned-to
session, a bump every BUMP_AFTER_S after a request, the first FREE_BUMPS at any
hour, the rest (to MAX_BUMPS) only before the learned day end
(`keepalive.stop_hour`, from earlier days only: no lookahead); the daily cap
ignored. A bump costs a cache read of the context; a return while the bumps
kept the cache alive saves a rewrite at the 1-hour price minus a read.

Not modelled: the proxy compacting on a warm return when that pays (it then
pays a write of the smaller context anyway: the break counts as saved here,
so this is slightly optimistic); the `claude`-process check (it only stops
more bumps). Bumps never change what is sent otherwise.
Usage: python3 experiments/keepalive.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import keepalive as ka, pricing, sim, transcripts  # noqa: E402

H = 3600


def main() -> int:
    fk = sim.forks()
    ss = [s for s in transcripts.load() if s.requests and s.path.stem not in fk]
    times = [sim._ts(r.timestamp) for s in ss for r in s.requests]
    ends = ka.day_ends(times)
    now = max(times)
    bill = saved = spent = wasted = 0.0
    kept = futile = moments = 0
    for s in ss:
        p = pricing.price_for(s.primary_model)
        if not p:
            continue
        rq = sorted(s.requests, key=lambda r: sim._ts(r.timestamp))
        for r in rq:
            bill += (r.inp * p.inp + r.read * p.read + r.write_1h * p.write_1h + r.write_5m * p.write_5m
                     + r.out * p.out) / 1e6
        returned = False
        for k, r in enumerate(rq):
            t = sim._ts(r.timestamp)
            g = sim._ts(rq[k + 1].timestamp) - t if k + 1 < len(rq) else None
            big = k + 1 >= ka.BIG_REQUESTS or returned or r.context >= ka.BIG_TOKENS
            if g is not None and g > H:
                returned = True
            if (g is None and now - t < 12 * H) or (g is not None and g <= H) or r.context * (p.write_1h - p.read) / 1e6 < ka.MIN_SAVING_USD:
                continue
            moments += 1
            if not big:
                continue
            stop = ka.stop_hour(ends, t)
            b, last = 0, t
            while b < ka.MAX_BUMPS:
                bt = last + ka.BUMP_AFTER_S
                if g is not None and bt >= t + g:
                    break
                if b >= ka.FREE_BUMPS and (stop is None or ka._day(bt) != ka._day(t) or ka._hour(bt) >= stop):
                    break
                b, last = b + 1, bt
            cost = b * r.context * p.read / 1e6
            spent += cost
            if b and g is not None and t + g <= last + H:
                saved += r.context * (p.write_1h - p.read) / 1e6
                kept += 1
            elif b:
                wasted += cost
                futile += 1
    net = saved - spent
    print(f"{len(ss)} real sessions, {len(ends)} days, {moments} idle moments over 1 h, bill ${bill:,.2f}")
    print(f"bumps would have kept {kept} breaks warm and bumped {futile} breaks for nothing")
    print(f"saved ${saved:,.2f}, spent ${spent:,.2f} (${wasted:,.2f} of it for nothing): "
          f"net {'+' if net >= 0 else '-'}${abs(net):,.2f} = {100 * net / bill:+.1f}% of the bill")
    return 0


if __name__ == "__main__":
    sys.exit(main())
