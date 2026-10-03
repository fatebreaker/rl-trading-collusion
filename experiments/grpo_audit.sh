#!/bin/bash
# Audit a GRPO-trained adapter with the in-context protocol: duopoly (with both
# deviation tests) and the solo placebo, at the run's training noise level and
# temperature, without notes and with the training history length.
# Usage: experiments/grpo_audit.sh <run dir> <gpu> [adapter name] [extra args]
cd "$(dirname "$0")/.."
RUN=$1; GPU=$2; ADAPTER=$3
[ -z "$ADAPTER" ] && ADAPTER=$(ls -d $RUN/adapter_[0-9]* | tail -1 | xargs basename)
shift $(( $# < 3 ? $# : 3 )); EXTRA="$@"  # e.g. --dtype half --attention-backend triton_attn
PY=${PY:-$(command -v python)}
export PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=$(dirname $PY):$PATH
read MODEL SU HIST TEMP RANK <<< $($PY -c "
import json; c = json.load(open('$RUN/config.json'))
print(c['model'], c['sigma_u'], c['history'], c['temperature'], c['lora_rank'])")
OUT=results/grpo_audit/$(basename $RUN)_$ADAPTER
mkdir -p results/grpo_audit
for c in duopoly solo; do
  [ -f ${OUT}_$c.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/llm_pilot.py --condition $c --model $MODEL \
    --lora $RUN/$ADAPTER --no-notes --history $HIST --temperature $TEMP --max-tokens 64 \
    --sigma-u $SU --sessions 40 --periods 100 --dev-events 3 --gpu-mem 0.3 $EXTRA \
    --out ${OUT}_$c.json > ${OUT}_$c.log 2>&1
done
