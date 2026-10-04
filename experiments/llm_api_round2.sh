#!/bin/bash
# Round 2 API runs: pinned snapshots, raw responses saved, total spend capped at $100.
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python)}
export PYTHONPATH=src OPENAI_API_KEY=$(tr -d '\n' < ~/.config/openai/key)
NANO=gpt-5.4-nano-2026-03-17; MINI=gpt-5.4-mini-2026-03-17
run() {  # name model condition sigma_u seed budget [extra]
  f=results/llm_api/$1.json; [ -f "$f" ] && return
  $PY experiments/llm_pilot.py --backend openai --model $2 --reasoning-effort low --condition $3 \
    --sigma-u $4 --seed $5 --sessions 20 --periods 100 --dev-events 2 --dev-horizon 6 --save-raw \
    --run-budget $6 --total-budget 100 $7 --out $f > ${f%.json}.log 2>&1 || echo "FAILED $f"
  python3 -c "import json; print('$1 done; ledger \$%.2f' % sum(json.loads(l)['cost'] for l in open('results/openai_spend.jsonl')))"
}
run nano-snap_low_triopoly_su2      $NANO triopoly 2   0 12
run nano-snap_low_amm_duopoly_su2   $NANO duopoly  2   0 9 --adaptive-mm
run nano-snap_low_duopoly_su2_s1    $NANO duopoly  2   1 9
run nano-snap_low_duopoly_su0.5_s1  $NANO duopoly  0.5 1 9
run mini-snap_low_duopoly_su0.5     $MINI duopoly  0.5 0 14
