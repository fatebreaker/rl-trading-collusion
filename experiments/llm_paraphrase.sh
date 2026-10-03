#!/bin/bash
# Prompt robustness: the depth exhibit (solo and duopoly at sigma_u 0.5 and 2)
# under the paraphrased prompt (variant b). Usage: experiments/llm_paraphrase.sh [gpu]
cd "$(dirname "$0")/.."
GPU=${1:-2}
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
OUT=results/llm_pilot
for su in 0.5 2; do
  for c in solo duopoly; do
    f=$OUT/qwen3_8b_pb_${c}_su${su}.json
    [ -f "$f" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $c --model Qwen/Qwen3-8B \
      --prompt-variant b --sigma-u $su --sessions 40 --periods 200 --dev-events 3 \
      --out $f > ${f%.json}.log 2>&1
  done
done
