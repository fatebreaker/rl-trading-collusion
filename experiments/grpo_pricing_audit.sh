#!/bin/bash
# Audit a GRPO-trained pricing adapter with the in-context pricing protocol: the duopoly with
# both deviation tests (best response and a visible 10% cut), and the same duopoly with all
# prices x10 (scale sweep), at the training temperature and history, without notes.
# Usage: experiments/grpo_pricing_audit.sh <run dir> <gpu> [adapter name] [extra args]
cd "$(dirname "$0")/.."
RUN=$1; GPU=$2; ADAPTER=$3
[ -z "$ADAPTER" ] && ADAPTER=$(ls -d $RUN/adapter_[0-9]* | tail -1 | xargs basename)
shift $(( $# < 3 ? $# : 3 )); EXTRA="$@"  # e.g. --dtype half --attention-backend triton_attn
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
read MODEL HIST TEMP <<< $($PY -c "
import json; c = json.load(open('$RUN/config.json'))
print(c['model'], c['history'], c['temperature'])")
OUT=results/grpo_pricing_audit/$(basename $RUN)_$ADAPTER
mkdir -p results/grpo_pricing_audit
for K in 1 10; do
  F=${OUT}_duopoly_k$K.json
  [ -f $F ] && continue
  DEV="--dev-events 3 --dev-horizon 8 --dev-cut 0.1"
  [ $K = 10 ] && DEV="--dev-events 0 --dev-cut 0"
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_bertrand.py --condition duopoly --model $MODEL --lora $RUN/$ADAPTER \
    --scale $K --no-notes --history $HIST --temperature $TEMP --max-tokens 64 --sessions 40 --periods 100 \
    $DEV --gpu-mem 0.3 --save-raw $EXTRA --out $F > ${F%.json}.log 2>&1
done
