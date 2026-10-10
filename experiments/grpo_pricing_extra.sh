#!/bin/bash
# Extra deviation tests of a GRPO-trained pricing adapter paired with itself, at its training temperature and
# history, without notes (as grpo_pricing_audit.sh). Usage:
#   experiments/grpo_pricing_extra.sh <run dir|base:MODEL> <adapter|-> <gpu> <tag> [llm_bertrand args...]
# Writes results/grpo_pricing_audit/<run>_<adapter>_<tag>.json (base:MODEL -> base_<model>_<tag>.json).
cd "$(dirname "$0")/.."
RUN=$1; AD=$2; GPU=$3; TAG=$4; shift 4
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
if [[ $RUN == base:* ]]; then
  MODEL=${RUN#base:}; HIST=8; TEMP=1.0; LORA=""
  OUT=results/grpo_pricing_audit/base_$(basename $MODEL | tr 'A-Z.-' 'a-zpp')_$TAG.json
else
  read MODEL HIST TEMP <<< $($PY -c "
import json; c = json.load(open('$RUN/config.json'))
print(c['model'], c['history'], c['temperature'])")
  LORA="--lora $RUN/$AD"
  OUT=results/grpo_pricing_audit/$(basename $RUN)_${AD}_$TAG.json
fi
[ -s $OUT ] && exit 0
mkdir -p results/grpo_pricing_audit
CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_bertrand.py --condition duopoly --model $MODEL $LORA \
  --no-notes --history $HIST --temperature $TEMP --max-tokens 64 --gpu-mem 0.3 "$@" --out $OUT > ${OUT%.json}.log 2>&1
