#!/bin/bash
# Dou et al. protocol with their stopping rule: each session learns until its
# greedy strategies are unchanged for 1e6 consecutive periods (cap 2e9).
# Runs one configuration at a time with all cores; resumable.
cd "$(dirname "$0")/.."
export PYTHONPATH=src NUMBA_NUM_THREADS=4
COMMON="--algo q --engine numba --sessions 100 --steps 2000000000 --eval-steps 20000
  --n-values 10 --n-actions 15 --grid-mode bracket --price-bins dou --n-price-bins 31
  --xi 500 --sigma-u 0.1 --impulse-reps 40 --shock-devs 0.05,0.25,1
  --log-every 100000000 --checkpoint-every 100000000"
KW='{"alpha": 0.01, "beta_decay": 5e-7, "explore_by_value": true, "stop_unchanged": 1000000}'
for run in "price_g095:--memory price" "value_g095:--memory value"; do
  name=${run%%:*}; extra=${run#*:}
  out=results/exp17_stoprule/$name.json
  [ -f "$out" ] && continue
  python3 -m kylecollusion.run $COMMON $extra --agent-kwargs "$KW" \
    --out "$out" --checkpoint checkpoints/exp17_stoprule/$name.pkl >> results/exp17_stoprule/$name.log 2>&1
done
