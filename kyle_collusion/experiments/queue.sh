#!/bin/bash
# Runs sweeps in order once no other experiment process is running.
# Idempotent: finished runs are skipped and unfinished ones resume from their
# checkpoints, so just rerun it after any interruption. It never starts a run
# that is already running, because it waits for all experiment processes first.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
while pgrep -f "python3 -m kylecollusion.run" > /dev/null; do sleep 30; done
for g in "$@"; do
  python3 experiments/sweep.py "experiments/grids/$g.json" --workers 4
done
