#!/bin/bash
# LLM-trader pilot: one open-weight model, four conditions, two GPUs.
# Usage: experiments/llm_pilot.sh [model] [tag]   (skips finished runs)
cd "$(dirname "$0")/.."
MODEL=${1:-Qwen/Qwen3-8B}
TAG=${2:-qwen3_8b}
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_pilot
mkdir -p $OUT
run() {  # gpu condition...
  gpu=$1; shift
  for c in "$@"; do
    f=$OUT/${TAG}_$c.json
    [ -f "$f" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu $PY experiments/llm_pilot.py --condition $c --model $MODEL \
      --sessions 40 --periods 200 --dev-events 3 $EXTRA --out $f > $OUT/${TAG}_$c.log 2>&1
  done
}
run ${GPU_A:-2} duopoly solo &
run ${GPU_B:-3} monitor myopic &
wait
