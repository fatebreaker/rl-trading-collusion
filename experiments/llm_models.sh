#!/bin/bash
# Model coverage for the NLP version: the depth test (solo and duopoly at
# sigma_u 0.5 and 2, with deviation tests) for one open-weight model.
# Usage: experiments/llm_models.sh <gpu> <hf model> <tag> [extra llm_pilot args]
cd "$(dirname "$0")/.."
GPU=$1; MODEL=$2; TAG=$3; shift 3; EXTRA="$@"
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_models
mkdir -p $OUT
for spec in "solo 0.5" "solo 2" "duopoly 2" "duopoly 0.5"; do
  set -- $spec
  f=$OUT/${TAG}_${1}_su$2.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $1 --model $MODEL \
    --sigma-u $2 --sessions 40 --periods 200 --dev-events 3 --save-raw $EXTRA \
    --out $f > ${f%.json}.log 2>&1
done
