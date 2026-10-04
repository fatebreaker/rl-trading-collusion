#!/bin/bash
# Round 2 (referee-proofing): adaptive market maker, three traders, second seeds.
# Usage: experiments/llm_round2.sh <gpu> <spec>...   spec = name:condition:sigma_u:seed:sessions:extra
cd "$(dirname "$0")/.."
GPU=$1; shift
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
for spec in "$@"; do
  IFS=: read name cond su seed sess extra <<< "$spec"
  f=results/llm_pilot/qwen3_8b_${name}.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $cond --model Qwen/Qwen3-8B \
    --sigma-u $su --seed $seed --sessions $sess --periods 200 --dev-events 3 --save-raw \
    ${extra//,/ } --out $f > ${f%.json}.log 2>&1
done
