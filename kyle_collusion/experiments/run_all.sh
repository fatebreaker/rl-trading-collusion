#!/bin/bash
# Restart point for the whole backlog after an interruption: every sweep skips
# finished runs and resumes unfinished ones from their checkpoints.
cd "$(dirname "$0")/.."
setsid nohup ./experiments/queue.sh exp5_controls exp7_dou exp9_interventions exp10_shared exp8_deep exp6_alpha \
  >> results/queue.log 2>&1 < /dev/null &
echo "queue started (pid $!)"
