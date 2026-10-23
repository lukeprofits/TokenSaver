#!/bin/sh
# TokenSaver vs Headroom on this machine's own history, then one table.
# Free (token counts only); about 20-40 minutes. Needs Headroom in
# ~/.tokensaver/experiments/hr-venv (see headroom_benchmark/README.md).
set -e
cd "$(dirname "$0")/.."
HR=~/.tokensaver/experiments/hr-venv/bin/python
echo "== 1/3 tokensaver"; ./tokensaver-cli benchmark --yes > /dev/null
echo "== 2/3 Headroom (default profile)"; HEADROOM_KOMPRESS_BACKEND=onnx_cpu "$HR" experiments/replay_history.py coding > /dev/null 2>&1
echo "== 3/3 Headroom (ML profile)"; HEADROOM_KOMPRESS_BACKEND=onnx_cpu HEADROOM_COMPRESSION_DEADLINE_MS=600000 "$HR" experiments/replay_history.py agent-90 > /dev/null 2>&1
echo; python3 experiments/compare_report.py
