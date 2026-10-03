#!/bin/bash
# Does reasoning fix price-impact naivety? Qwen3-8B with thinking on: noise test
# (solo at sigma_u 0.5 and 2) and the sigma_u = 2 duopoly with deviation tests.
cd "$(dirname "$0")/.."
GPU=${1:-3}
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_pilot
for spec in "solo 0.5" "solo 2" "duopoly 2"; do
  set -- $spec
  f=$OUT/qwen3_8b_think_${1}_su$2.json
  [ -f "$f" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $1 --model Qwen/Qwen3-8B \
    --thinking --sigma-u $2 --sessions 20 --periods 100 --dev-events 2 --dev-horizon 5 \
    --temperature 0.6 --max-tokens 8192 --out $f > ${f%.json}.log 2>&1
done
