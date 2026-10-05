#!/bin/bash
# Is depth-blindness a sampling artefact? Qwen3-8B alone at sigma_u 0.5 and 2
# with greedy decoding (temperature 0).
# Usage: experiments/llm_greedy.sh <gpu> [extra llm_pilot args]
cd "$(dirname "$0")/.."
GPU=$1; shift; EXTRA="$@"
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_prompts
mkdir -p $OUT
for su in 0.5 2; do
  f=$OUT/qwen3_8b_greedy_solo_su$su.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition solo --model Qwen/Qwen3-8B \
    --temperature 0 --sigma-u $su --sessions 40 --periods 200 --save-raw $EXTRA \
    --out $f > ${f%.json}.log 2>&1
done
