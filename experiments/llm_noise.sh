#!/bin/bash
# Noise sweep: does in-context trading intensity scale with sigma_u / sigma_v,
# as it must for a trader that understands price impact? Skips finished runs.
# Usage: experiments/llm_noise.sh [model] [tag] [gpu]
cd "$(dirname "$0")/.."
MODEL=${1:-Qwen/Qwen3-8B}
TAG=${2:-qwen3_8b}
GPU=${3:-3}
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_pilot
for su in 0.5 2; do
  for c in solo duopoly; do
    f=$OUT/${TAG}_${c}_su${su}.json
    [ -f "$f" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $c --model $MODEL \
      --sigma-u $su --sessions 40 --periods 200 --dev-events 3 $EXTRA --out $f \
      > $OUT/${TAG}_${c}_su${su}.log 2>&1
  done
done
# Framing x presence of a rival (sigma_u = 1): same vague prompt, with and without a rival.
for c in duopoly_vague solo_vague; do
  f=$OUT/${TAG}_${c}.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $c --model $MODEL \
    --sessions 40 --periods 200 --dev-events 3 $EXTRA --out $f > $OUT/${TAG}_${c}.log 2>&1
done
