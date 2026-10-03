#!/bin/bash
# Untrained model in the GRPO training format (no notes, 8-period history,
# temperature 1.0): the "before RL" reference for audited adapters.
# Usage: experiments/llm_base_grpo_format.sh <model> <tag> <gpu> [extra]
cd "$(dirname "$0")/.."
MODEL=$1; TAG=$2; GPU=$3; shift 3; EXTRA="$@"
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_base
mkdir -p $OUT
for su in 0.5 2; do
  for c in duopoly solo; do
    f=$OUT/${TAG}_${c}_su${su}.json
    [ -f "$f" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $c --model $MODEL \
      --no-notes --history 8 --temperature 1.0 --max-tokens 64 --sigma-u $su \
      --sessions 40 --periods 100 --dev-events 3 --gpu-mem 0.5 $EXTRA --out $f > ${f%.json}.log 2>&1
  done
done
