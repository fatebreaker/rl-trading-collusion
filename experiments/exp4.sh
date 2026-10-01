#!/bin/bash
# exp4: perfect monitoring (memory=orders) vs residual vs memoryless control.
# Resumable: rerunning this script continues each run from its checkpoint.
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
common="--algo q --sessions 100 --steps 15000000 --eval-steps 20000 --impulse-reps 40
        --log-every 1000000 --checkpoint-every 1000000 --agent-kwargs {\"beta_decay\":4e-7}"
run() {  # name, extra args
  [ -f results/exp4/$1.json ] && return
  setsid nohup python3 -m kylecollusion.run $common ${@:2} \
    --checkpoint checkpoints/exp4_$1.pkl --out results/exp4/$1.json \
    >> results/exp4/$1.log 2>&1 < /dev/null &
}
run orders_s0 --memory orders --seed 0
run orders_s1 --memory orders --seed 1
run residual_s0 --memory residual --seed 0
run none_s0 --memory none --seed 0
