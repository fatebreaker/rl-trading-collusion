#!/bin/bash
# GPT-5.5 (frontier model) depth test: solo and duopoly at sigma_u 0.5 and 2.
# Flex processing (half price, about $0.007 per call) and a smaller design than
# the GPT-5.4 runs to bound cost:
# 10 sessions x 60 periods, low reasoning effort, best-response deviation test
# only (two events, four-period horizon). Needs OPENAI_API_KEY.
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python)}
export PYTHONPATH=src
OUT=results/llm_api
for spec in "solo 2 8" "solo 0.5 8" "duopoly 2 20" "duopoly 0.5 20"; do
  set -- $spec
  f=$OUT/gpt-5p5_low_${1}_su$2.json
  [ -f "$f" ] && continue
  $PY experiments/llm_pilot.py --condition $1 --backend openai --model gpt-5.5-2026-04-23 \
    --sigma-u $2 --sessions 10 --periods 60 --reasoning-effort low --service-tier flex --dev-events 2 \
    --dev-horizon 4 --dev-shift 0 --save-raw --run-budget $3 --total-budget 140 \
    --out $f > ${f%.json}.log 2>&1
done
