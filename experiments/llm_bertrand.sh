#!/bin/bash
# LLM pricing agents in the logit-Bertrand duopoly (second environment):
# duopoly at currency scales 1 and 10 (scale sweep), the myopic placebo, and
# optionally the instructed trigger strategy (positive control).
# Usage: experiments/llm_bertrand.sh <gpu> <hf model> <tag> "<conditions>" [extra llm_bertrand args]
#   conditions: space-separated from "duopoly:1 duopoly:10 myopic:1 trigger:1"
cd "$(dirname "$0")/.."
GPU=$1; MODEL=$2; TAG=$3; CONDS=$4; shift 4; EXTRA="$@"
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_bertrand
mkdir -p $OUT
for spec in $CONDS; do
  cond=${spec%%:*}; k=${spec##*:}
  f=$OUT/${TAG}_${cond}_k$k.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_bertrand.py --condition $cond --model $MODEL \
    --scale $k --sessions 40 --periods 100 --dev-events 3 --dev-horizon 6 --save-raw $EXTRA \
    --out $f > ${f%.json}.log 2>&1
done
