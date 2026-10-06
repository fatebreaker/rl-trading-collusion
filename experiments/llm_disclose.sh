#!/bin/bash
# Depth test with the market maker's rule disclosed (P = lambda x total flow):
# separates failing to estimate the price impact from failing to use it.
# Usage: experiments/llm_disclose.sh <gpu> <hf model> <tag> [extra llm_pilot args]
cd "$(dirname "$0")/.."
GPU=$1; MODEL=$2; TAG=$3; shift 3; EXTRA="$@"
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_disclose
mkdir -p $OUT
for su in 0.5 2; do
  f=$OUT/${TAG}_solo_su$su.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition solo --model $MODEL --sigma-u $su \
    --disclose-rule --sessions 40 --periods 200 --save-raw $EXTRA --out $f > ${f%.json}.log 2>&1
done
