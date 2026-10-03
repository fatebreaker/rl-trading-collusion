#!/bin/bash
# Escalation under transparency, session level: duopoly vs perfect monitoring at
# sigma_u = 1, second seed, with raw paths saved. Usage: experiments/llm_escalation.sh <gpu>
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
for c in monitor duopoly; do
  f=results/llm_pilot/qwen3_8b_${c}_s1.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $c --model Qwen/Qwen3-8B \
    --seed 1 --sessions 40 --periods 200 --dev-events 3 --gpu-mem 0.75 --out $f > ${f%.json}.log 2>&1
done
