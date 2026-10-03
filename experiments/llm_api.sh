#!/bin/bash
# Hosted models: noise test (solo) and duopoly with deviation tests at sigma_u 0.5 and 2.
# Spending is capped per run and over all runs (results/openai_spend.jsonl).
# Usage: experiments/llm_api.sh <model> <reasoning effort: none|low|...> [run budget USD]
cd "$(dirname "$0")/.."
MODEL=$1; EFF=$2; BUDGET=${3:-6}
PY=${PY:-$(command -v python)}
export PYTHONPATH=src OPENAI_API_KEY=$(tr -d '\n' < ~/.config/openai/key)
OUT=results/llm_api
mkdir -p $OUT
TAG=${MODEL//./p}_$EFF
for spec in "solo 0.5" "solo 2" "duopoly 2" "duopoly 0.5"; do
  set -- $spec
  f=$OUT/${TAG}_${1}_su$2.json
  [ -f "$f" ] && continue
  $PY experiments/llm_pilot.py --backend openai --model $MODEL --reasoning-effort $EFF \
    --condition $1 --sigma-u $2 --sessions 20 --periods 100 --dev-events 2 --dev-horizon 6 \
    --run-budget $BUDGET --total-budget 50 --out $f > ${f%.json}.log 2>&1 || echo "FAILED $f"
done
python3 -c "import json; print('ledger total: \$%.2f' % sum(json.loads(l)['cost'] for l in open('$OUT/../openai_spend.jsonl')))"
