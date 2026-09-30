#!/bin/bash
# Restart point for the whole backlog after an interruption: every sweep skips
# finished runs and resumes unfinished ones from their checkpoints.
cd "$(dirname "$0")/.."
setsid nohup ./experiments/queue.sh exp5_controls exp7_dou exp6_alpha exp9_interventions exp8_deep \
  >> results/queue.log 2>&1 < /dev/null &
echo "queue started (pid $!)"
