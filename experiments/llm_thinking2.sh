#!/bin/bash
# Qwen3-8B thinking duopoly at sigma_u = 0.5 (pairs with the sigma_u = 2 run).
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
f=results/llm_pilot/qwen3_8b_think_duopoly_su0.5.json
[ -f "$f" ] || CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition duopoly --model Qwen/Qwen3-8B \
  --thinking --sigma-u 0.5 --sessions 20 --periods 100 --dev-events 2 --dev-horizon 5 \
  --temperature 0.6 --max-tokens 8192 --gpu-mem 0.75 --out $f > ${f%.json}.log 2>&1
