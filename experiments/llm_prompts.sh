#!/bin/bash
# Prompt sensitivity of depth-blindness: a lone trader at sigma_u 0.5 and 2 under
# two more system-prompt wordings (c: market-microstructure register, d: terse).
# Usage: experiments/llm_prompts.sh <gpu> <hf model> <tag> [extra llm_pilot args]
cd "$(dirname "$0")/.."
GPU=$1; MODEL=$2; TAG=$3; shift 3; EXTRA="$@"
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_prompts
mkdir -p $OUT
for spec in "c 0.5" "c 2" "d 0.5" "d 2"; do
  set -- $spec
  f=$OUT/${TAG}_${1}_solo_su$2.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition solo --model $MODEL \
    --prompt-variant $1 --sigma-u $2 --sessions 40 --periods 200 --save-raw $EXTRA \
    --out $f > ${f%.json}.log 2>&1
done
