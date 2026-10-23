"""Cache-safety gate across processes: the same history must rewrite to the
same bytes whatever Python's hash seed is.

rewrite_determinism.py runs in one process, so it cannot see output that
depends on set iteration order, which Python randomises per process. That bug
shipped once: a tie in a length-only sort let alias definitions come out in a
different order after every proxy restart, and a live resume rewrote 392K
tokens of cache. This replays a frozen copy of each session under several
seeds in separate processes and requires identical hashes.

Local, free. Usage: python3 experiments/seed_determinism.py [n_sessions]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHILD = r'''
import sys, json, hashlib
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/experiments")
from pathlib import Path
from rewrite_determinism import conversation
from tokensaver import passes
h = hashlib.sha256()
for p in sorted(Path(sys.argv[2]).glob("*.jsonl")):
    m, _ = conversation(p)
    for mode in ("off",):
        h.update(json.dumps(passes.rewrite_messages(m, aliases=mode), ensure_ascii=False, sort_keys=True).encode())
    plan = passes.hindsight_plan(m)
    h.update(json.dumps(passes.rewrite_messages(m, aliases=plan), ensure_ascii=False, sort_keys=True).encode())
print(h.hexdigest())
'''


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    repo = str(Path(__file__).resolve().parent.parent)
    frozen = Path(tempfile.mkdtemp())
    files = sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: p.stat().st_size)[-n:]
    for p in files:                        # frozen: live sessions grow between runs
        shutil.copy(p, frozen / p.name)
    hashes = []
    for seed in ("1", "2", "3"):
        r = subprocess.run([sys.executable, "-c", CHILD, repo, str(frozen)], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=seed), timeout=1800)
        hashes.append(r.stdout.strip() or r.stderr[-300:])
    ok = len(set(hashes)) == 1
    print(f"{len(files)} sessions x 2 alias modes (off, plan) under hash seeds 1/2/3: "
          + ("identical" if ok else "DIFFERENT (cross-process nondeterminism)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
