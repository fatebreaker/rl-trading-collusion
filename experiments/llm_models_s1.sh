#!/bin/bash
# Third market depth (sigma_u = 1) for one open-weight model: solo and duopoly,
# so that the depth sweep has three points (as for Qwen3-4B and Qwen3-8B).
# Usage: experiments/llm_models_s1.sh <gpu> <hf model> <tag> [extra llm_pilot args]
cd "$(dirname "$0")/.."
GPU=$1; MODEL=$2; TAG=$3; shift 3; EXTRA="$@"
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_models
for cond in solo duopoly; do
  f=$OUT/${TAG}_${cond}_su1.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $cond --model $MODEL \
    --sigma-u 1 --sessions 40 --periods 200 --dev-events 3 --save-raw $EXTRA \
    --out $f > ${f%.json}.log 2>&1
done
