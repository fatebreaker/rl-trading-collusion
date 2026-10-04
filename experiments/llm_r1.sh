#!/bin/bash
# DeepSeek-R1-Distill-Qwen-14B depth test. Its reasoning runs 1.6k-8.6k tokens
# (a 6000-token cap truncated a third of answers), so the cap is 12000 and the
# design is smaller than for the other models: 10 sessions x 60 periods.
# Usage: experiments/llm_r1.sh <gpus> <condition> <sigma_u> [extra llm_pilot args]
cd "$(dirname "$0")/.."
GPUS=$1; COND=$2; SU=$3; shift 3
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
f=results/llm_models/r1d14b_${COND}_su$SU.json
[ -f "$f" ] && exit 0
CUDA_VISIBLE_DEVICES=$GPUS $PY experiments/llm_pilot.py --condition $COND \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --sigma-u $SU --thinking --temperature 0.6 \
  --max-tokens 12000 --max-model-len 16384 --sessions 10 --periods 60 \
  --dev-events 3 --dev-horizon 5 --save-raw --tp 2 "$@" --out $f > ${f%.json}.log 2>&1
